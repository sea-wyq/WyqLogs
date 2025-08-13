import math
import re
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# --------------------------
# 1. 分布式环境初始化（适配 torchrun）
# --------------------------
def setup():
    """通过 torchrun 自动设置的环境变量初始化分布式环境"""
    # torchrun 会自动设置以下环境变量：
    # - RANK：全局进程编号（0 为主节点）
    # - WORLD_SIZE：总进程数（所有节点的 GPU 总数）
    # - MASTER_ADDR：主节点 IP（由 torchrun 协调）
    # - MASTER_PORT：主节点端口（由 torchrun 协调）
    dist.init_process_group(
        backend="nccl",  # GPU 通信后端
        init_method="env://"  # 从环境变量读取配置
    )
    # 当前进程的本地 GPU 编号（每个节点内的 GPU 序号，0,1,2...）
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


# --------------------------
# 2. 数据集定义（不变）
# --------------------------
class BERTDataset(Dataset):
    def __init__(self, sentences, token_list, word_dict, maxlen, max_pred):
        self.sentences = sentences
        self.token_list = token_list
        self.word_dict = word_dict
        self.maxlen = maxlen
        self.max_pred = max_pred
        self.vocab_size = len(word_dict)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens_a_index = random.randint(0, len(self.token_list) - 1)
        tokens_b_index = random.randint(0, len(self.token_list) - 1)
        tokens_a, tokens_b = self.token_list[tokens_a_index], self.token_list[tokens_b_index]

        input_ids = [self.word_dict['[CLS]']] + tokens_a + [self.word_dict['[SEP]']] + tokens_b + [self.word_dict['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        n_pred = min(self.max_pred, max(1, int(round(len(input_ids) * 0.15))))
        cand_masked_pos = [i for i, token in enumerate(input_ids)
                          if token != self.word_dict['[CLS]'] and token != self.word_dict['[SEP]']]
        random.shuffle(cand_masked_pos)
        masked_tokens, masked_pos = [], []
        
        for pos in cand_masked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random.random() < 0.8:
                input_ids[pos] = self.word_dict['[MASK]']
            elif random.random() < 0.5:
                input_ids[pos] = self.word_dict[list(self.word_dict.keys())[random.randint(4, self.vocab_size - 1)]]

        n_pad = self.maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        is_next = 1 if (tokens_a_index + 1 == tokens_b_index) else 0

        return (
            torch.LongTensor(input_ids),
            torch.LongTensor(segment_ids),
            torch.LongTensor(masked_tokens),
            torch.LongTensor(masked_pos),
            torch.LongTensor([is_next])
        )


# --------------------------
# 3. 模型参数配置（不变）
# --------------------------
maxlen = 30
batch_size = 6  # 单卡 batch_size
max_pred = 5
n_layers = 6
n_heads = 12
d_model = 768
d_ff = 768 * 4
d_k = d_v = 64
n_segments = 2
epochs = 100
save_path = "./bert_torchrun_model.pt"  # 建议保存在共享存储


# --------------------------
# 4. 注意力工具函数与模型定义（不变）
# --------------------------
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, maxlen, n_segments):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(maxlen, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        return self.norm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, maxlen, n_segments, n_layers, n_heads, d_k, d_v, d_ff):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size, d_model, maxlen, n_segments)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_k, d_v, d_ff) for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        
        h_pooled = self.activ1(self.fc(output[:, 0]))
        logits_clsf = self.classifier(h_pooled)

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias

        return logits_lm, logits_clsf


# --------------------------
# 5. 训练函数（适配 torchrun）
# --------------------------
def train(local_rank, dataset, vocab_size):
    # 初始化分布式环境
    device = torch.device(f"cuda:{local_rank}")

    # 分布式采样器
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=2
    )

    # 初始化模型
    model = BERT(
        vocab_size=vocab_size,
        d_model=d_model,
        maxlen=maxlen,
        n_segments=n_segments,
        n_layers=n_layers,
        n_heads=n_heads,
        d_k=d_k,
        d_v=d_v,
        d_ff=d_ff
    ).to(device)

    # DDP 包装
    ddp_model = DDP(model, device_ids=[local_rank])

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # 每个 epoch 打乱数据
        ddp_model.train()
        total_loss = 0.0

        for batch in dataloader:
            input_ids = batch[0].to(device)
            segment_ids = batch[1].to(device)
            masked_tokens = batch[2].to(device)
            masked_pos = batch[3].to(device)
            is_next = batch[4].squeeze().to(device)

            optimizer.zero_grad()
            logits_lm, logits_clsf = ddp_model(input_ids, segment_ids, masked_pos)
            
            loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)
            loss_clsf = criterion(logits_clsf, is_next)
            loss = loss_lm + loss_clsf
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # 仅主进程（rank=0）打印日志
        if dist.get_rank() == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.6f}")

    # 仅主进程保存模型
    if dist.get_rank() == 0:
        torch.save(ddp_model.module.state_dict(), save_path)
        print(f"模型已保存至 {save_path}")

    cleanup()


# --------------------------
# 6. 主函数（torchrun 入口）
# --------------------------
def main():
    # 初始化分布式环境，获取本地 GPU 编号
    local_rank = setup()

    # 数据准备（所有节点必须同步数据）
    text = (
        'Hello, how are you? I am Romeo.\n'
        'Hello, Romeo My name is Juliet. Nice to meet you.\n'
        'Nice meet you too. How are you today?\n'
        'Great. My baseball team won the competition.\n'
        'Oh Congratulations, Juliet\n'
        'Thanks you Romeo'
    )
    expanded_text = text * 100

    sentences = re.sub("[.,!?\\-]", '', expanded_text.lower()).split('\n')
    word_list = list(set(" ".join(sentences).split()))
    word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    for i, w in enumerate(word_list):
        word_dict[w] = i + 4
    vocab_size = len(word_dict)

    token_list = []
    for sentence in sentences:
        arr = [word_dict[s] for s in sentence.split()]
        token_list.append(arr)

    dataset = BERTDataset(
        sentences=sentences,
        token_list=token_list,
        word_dict=word_dict,
        maxlen=maxlen,
        max_pred=max_pred
    )

    # 启动训练
    train(local_rank, dataset, vocab_size)


if __name__ == "__main__":
    # 固定随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    main()