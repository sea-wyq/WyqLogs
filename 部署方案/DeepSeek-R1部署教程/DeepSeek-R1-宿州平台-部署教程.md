DeepSeek-R1: 基于Ascend 910ProB环境部署并对话
===================================

DeepSeek-R1 是一款由深度求索推出的推理模型，通过引入强化学习（RL）前的冷启动数据训练，有效解决了重复生成、可读性差及多语言混杂等问题。该模型在数学、代码与逻辑推理任务中表现卓越，性能对标 OpenAI-o1。其创新训练框架突破了传统 RL 训练的局限性，为复杂推理任务提供了高效解决方案。

本文以 [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) 为例，申请`910ProB`计算卡资源并在`JupyterLab`开发环境中部署模型并进行对话。主要内容包括：

- [DeepSeek-R1: 基于Ascend 910ProB环境部署并对话](#deepseek-r1-基于ascend-910prob环境部署并对话)
  - [1. 创建训练项目](#1-创建训练项目)
  - [2. 创建jupyter任务](#2-创建jupyter任务)
  - [3. 模型部署](#3-模型部署)

## 1. 创建训练项目

在工作台中，创建新的训练任务。

1. 选择训练项目，点击创建训练项目按钮，然后填写项目名称、编程语言和算法框架。
2. 选择引用的模型，DeepSeek模型在公开模型栏目下。
3. 查看模型挂载路径，后续推理命令需要指定模型的挂载路径（/model/systemuser/DeepSeek-R1-Distill-Qwen）。
![image](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image.png)
![image-3](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-3.png)
![image-2](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-2.png)
![image-9](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-9.png)
![image-10](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-10.png)

点击创建项目，等待项目创建完成。

## 2. 创建jupyter任务

进入刚创建的训练项目,选择创建Jupyter任务。

1. 选择jupyter。
2. 填写镜像，算力套餐和队列。

> 注： 镜像选择`mindie1.0-cann8.0-py310-ubuntu20.05`

![image-5](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-5.png)
![image-6](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-6.png)

点击确认后，等待开启调试按钮处于可点击状态。
![image-7](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-7.png)

## 3. 模型部署

部署流程如下：

1. 点击`Terminal`并执行`bash`命令，激活环境并注入环境变量。![image-8](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-8.png)
2. 执行下面推理命令;

```bash
python3 -m examples.run_pa \
   --model_path /model/systemuser/DeepSeek-R1-Distill-Qwen \
   --input_text ["中国的四大名著是什么?"] \
   --max_output_length 2048
```

- examples.run_pa：推理脚本。
- model_path：项目模型挂载的目录。
- input_text：请求的问题文本。
- max_output_length：输出的最大文本长度。

推理结果展示如下：

```bash
(Python310) root@deepseek-8470-task1-0:/usr/local/Ascend/atb-models# python3 -m examples.run_pa --model_path /model/systemuser/DeepSeek-R1-Distill-Qwen --input_text ["中国的四大名著是什么?"] --max_output_length 2048
...
[2025-02-10 10:55:23,153] [2687] [281473644918960] [llm] [INFO][logging.py-227] : <<<<<<< ori k_caches[0].shape=torch.Size([24, 32, 128, 16])
[2025-02-10 10:55:23,166] [2687] [281473644918960] [llm] [INFO][flash_causal_qwen2.py-435] : <<<<<<<after transdata k_caches[0].shape=torch.Size([24, 32, 128, 16])
[2025-02-10 10:55:23,168] [2687] [281473644918960] [llm] [INFO][logging.py-227] : >>>>>>id of kcache is 281473433199184 id of vcache is 281473433198944
[2025-02-10 10:55:26,025] [2687] [281473644918960] [llm] [INFO][logging.py-227] : warmup_memory(GB):  1.18
[2025-02-10 10:55:26,025] [2687] [281473644918960] [llm] [INFO][logging.py-227] : ---------------end warm_up---------------
[2025-02-10 10:55:26,026] [2687] [281473644918960] [llm] [INFO][logging.py-227] : ---------------begin inference---------------
[2025-02-10 10:55:26,098] [2687] [281473644918960] [llm] [INFO][logging.py-227] : ------total req num: 1, infer start--------
[2025-02-10 10:55:43,969] [2687] [281473644918960] [llm] [INFO][logging.py-227] : ---------------end inference---------------
[2025-02-10 10:55:43,970] [2687] [281473644918960] [llm] [INFO][logging.py-227] : Answer[0]: <think>
嗯，我现在要回答关于中国四大名著的问题。首先，我得回忆一下四大名著都包括哪些书。我记得四大名著是中国古代四大经典的合称，分别是《水浒传》、《西游记》、《红楼梦》和《三国演义》。对吗？让我再确认一下。

《水浒传》主要讲的是梁山好汉聚义的故事，里面有108位好汉，对吧？他们因为各种原因聚在一起，反抗朝廷。我记得里面有宋江、李逵这些人物，他们后来被朝廷所杀，对吧？

然后是《西游记》，这是一部讲述唐僧取经的故事。里面有很多妖怪和猪八戒、孙悟空这些角色。我记得孙悟空是齐天大圣，猪八戒是猪的八戒，他们都是取经路上的帮手。唐僧虽然有点呆头呆脑，但很善良，对吧？

接下来是《红楼梦》，这是一部描绘了封建家族制度和人情世故的长篇小说。里面有很多复杂的角色，比如林黛玉、贾宝玉、贾政这些。我记得林黛玉是贾府的女主人，后来因为贾宝玉的去世而 RIP，对吧？还有三字一句的诗，比如“贾珠贾粉黛，贾家三春雪”，这些是不是出自《红楼梦》？

最后是《三国演义》，这是一部讲述三国时期各个英雄的故事。里面有刘备、关羽、张飞这些武将，还有曹操、孙权这些主要人物。故事主要围绕着刘备联合孙权对抗曹操，后来又分道扬镳，各自发展。我记得诸葛亮是智慧的代表，帮助刘备和蜀国，对吧？

哦，对了，四大名著都是中国古典文学的瑰宝，影响深远，至今仍在世界文学经典中占有重要地位。它们不仅在中国文化中占有重要地位，还被翻译成多种语言，传播到世界各地。

不过，我是不是漏掉了什么？比如，四大名著是不是还有其他书？我记得好像还有《三国志》之类的，但那是不是四大名著之一呢？不，四大名著指的是四部小说，而不是历史书。所以，没错，四大名著就是这四部。

再想想，这些书的作者是谁？《水浒传》是施耐庵写的，对吧？《西游记》是吴承恩写的，对吗？《红楼梦》是曹雪芹写的，而《三国演义》是罗贯中的作品。对的，这些作者都是明朝的著名作家。

总结一下，四大名著包括《水浒传》、《西游记》、《红楼梦》和《三国演义》。它们各自讲述了不同的故事，涉及不同的主题，但都是中国文学史上的经典之作。这些书不仅在中国广为流传，还对世界文学的发展产生了深远的影响。

嗯，我觉得这样回答应该没问题了。如果有哪里不对，可能需要再查证一下，但总体上我记得这些是正确的。
</think>

中国的四大名著是指《水浒传》、《西游记》、《红楼梦》和《三国演义》。这些经典文学作品不仅是中国文化的瑰宝，也对世界文学的发展产生了深远影响。以下是它们的简要介绍：

1. **《水浒传》**：作者是施耐庵，讲述了梁山好汉聚义反抗朝廷的故事，涉及108位好汉，最终被朝廷所杀。

2. **《西游记》**：作者是吴承恩，讲述了唐僧取经的传奇故事，其中的孙悟空、猪八戒等角色广为人知。

3. **《红楼梦》**：作者是曹雪芹，是一部描绘封建家族制度和人情世故的长篇小说，其中林黛玉、贾宝玉等角色深受读者喜爱。

4. **《三国演义》**：作者是罗贯中，讲述了三国时期的英雄故事，涉及刘备、关羽、张飞等武将，以及曹操、孙权等主要人物。

这些作品不仅是古代中国的经典，也被翻译成多种语言，成为世界文学的重要组成部分。<｜end▁of▁sentence｜>
```
