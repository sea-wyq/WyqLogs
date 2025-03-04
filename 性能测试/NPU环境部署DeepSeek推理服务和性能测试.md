# DeepSeek服务部署



## 参考文档
- [手动部署推理服务_AI开发平台ModelArts_华为云](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_ds_infer_0006.html)
- [服务化接口-MindIE Service开发指南-服务化集成部署-MindIE1.0.RC3开发文档-昇腾社区](https://www.hiascend.com/document/detail/zh/mindie/10RC3/envdeployment/instg/mindie_instg_0006.html)
- [工具编译-HCCL性能测试工具-训练&推理开发-开发工具-CANN商用版8.0.RC3开发文档-昇腾社区](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/devtools/hccltool/HCCLpertest_16_0003.html)
- [State_Cloud/DeepSeek-R1-W8A8 | 魔乐社区](https://modelers.cn/models/State_Cloud/DeepSeek-R1-W8A8)
- [性能测试-快速开始-MindIE Service开发指南-服务化集成部署-MindIE1.0.0开发文档-昇腾社区](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0012.html)
- [MindIE Benchmark](https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0012.html)

## 实验环境 
NPU卡
- 910 ProB (更新驱动到24.1.rc3可完成32B 70B 服务部署和推理)
- 910B2C（所有参数模型都可部署）

镜像
```bash
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.0.T3-800I-A2-py311-openeuler24.03-lts
swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:1.0.0-800I-A2-py311-openeuler24.03-lts
```

模型权重
- State_Cloud/DeepSeek-R1-Distill-Llama-70B
- State_Cloud/DeepSeek-R1-Distill-Qwen-32B

模型权重来源：魔乐社区（下载速度比huggingface和魔塔更快）
```bash
export HUB_WHITE_LIST_PATHS=/DATA/disk1/DeepSeek-R1-Distill-Qwen-32B
cat << EOF > down.py
from openmind_hub import snapshot_download
snapshot_download(repo_id="State_Cloud/DeepSeek-R1-Distill-Qwen-32B", token="f3b5f3047ee1eec6816cd03405d96b9c74032fb5", repo_type="model",local_dir="/DATA/disk1/DeepSeek-R1-Distill-Qwen-32B",local_dir_use_symlinks=False ,resume_download=True)
EOF
```
## 启动推理容器
```bash
docker run -it -d \
    -p 40000:1025 \  #容器端口映射到宿主机端口
    --name wyq-ds \
    --shm-size=100g \
    --ipc=host \
    --privileged \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /home/aicc:/home/aicc \  # 挂载的数据集位置
    69f30d0c15be bash
```
或者
```bash
docker run -it -d \
    --name wyq-ds \
    --shm-size=100g \
    --net=host \  #使用宿主机网络
    --ipc=host \
      --privileged \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    --device=/dev/davinci0 \
    --device=/dev/davinci1 \
    --device=/dev/davinci2 \
    --device=/dev/davinci3 \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci6 \
    --device=/dev/davinci7 \
    --device=/dev/davinci8 \
    --device=/dev/davinci9 \
    --device=/dev/davinci10 \
    --device=/dev/davinci11 \
    --device=/dev/davinci12 \
    --device=/dev/davinci13 \
    --device=/dev/davinci14 \
    --device=/dev/davinci15 \
    --device=/dev/davinci16 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    -v /DATA/disk1:/home \    # 数据集挂载路径
    4949ab22a9cb bash
``` 
## 服务配置修改
```bash
vim  /usr/local/Ascend/mindie/latest/mindie-service/conf/config.json

{
    "Version" : "1.0.0",
    "LogConfig" :
    {
        "logLevel" : "Info",
        "logFileSize" : 20,
        "logFileNum" : 20,
        "logPath" : "logs/mindie-server.log"
    },

    "ServerConfig" :
    {
        "ipAddress" : "0.0.0.0",  # 访问IP地址
        "managementIpAddress" : "0.0.0.0",
        "port" : 1025,
        "managementPort" : 1026,
        "metricsPort" : 1027,
        "allowAllZeroIpListening" : true,
        "maxLinkNum" : 1000,
        "httpsEnabled" : false,  # 禁用https
        "fullTextEnabled" : false,
        "tlsCaPath" : "security/ca/",
        "tlsCaFile" : ["ca.pem"],
        "tlsCert" : "security/certs/server.pem",
        "tlsPk" : "security/keys/server.key.pem",
        "tlsPkPwd" : "security/pass/key_pwd.txt",
        "tlsCrlPath" : "security/certs/",
        "tlsCrlFiles" : ["server_crl.pem"],
        "managementTlsCaFile" : ["management_ca.pem"],
        "managementTlsCert" : "security/certs/management/server.pem",
        "managementTlsPk" : "security/keys/management/server.key.pem",
        "managementTlsPkPwd" : "security/pass/management/key_pwd.txt",
        "managementTlsCrlPath" : "security/management/certs/",
        "managementTlsCrlFiles" : ["server_crl.pem"],
        "kmcKsfMaster" : "tools/pmt/master/ksfa",
        "kmcKsfStandby" : "tools/pmt/standby/ksfb",
        "inferMode" : "standard",
        "interCommTLSEnabled" : true,
        "interCommPort" : 1121,
        "interCommTlsCaPath" : "security/grpc/ca/",
        "interCommTlsCaFiles" : ["ca.pem"],
        "interCommTlsCert" : "security/grpc/certs/server.pem",
        "interCommPk" : "security/grpc/keys/server.key.pem",
        "interCommPkPwd" : "security/grpc/pass/key_pwd.txt",
        "interCommTlsCrlPath" : "security/grpc/certs/",
        "interCommTlsCrlFiles" : ["server_crl.pem"],
        "openAiSupport" : "vllm"
    },

    "BackendConfig" : {
        "backendName" : "mindieservice_llm_engine",
        "modelInstanceNumber" : 1,
        "npuDeviceIds" : [[0,1,2,3]],  # 使用的计算卡id
        "tokenizerProcessNumber" : 8,
        "multiNodesInferEnabled" : false,
        "multiNodesInferPort" : 1120,
        "interNodeTLSEnabled" : true,
        "interNodeTlsCaPath" : "security/grpc/ca/",
        "interNodeTlsCaFiles" : ["ca.pem"],
        "interNodeTlsCert" : "security/grpc/certs/server.pem",
        "interNodeTlsPk" : "security/grpc/keys/server.key.pem",
        "interNodeTlsPkPwd" : "security/grpc/pass/mindie_server_key_pwd.txt",
        "interNodeTlsCrlPath" : "security/grpc/certs/",
        "interNodeTlsCrlFiles" : ["server_crl.pem"],
        "interNodeKmcKsfMaster" : "tools/pmt/master/ksfa",
        "interNodeKmcKsfStandby" : "tools/pmt/standby/ksfb",
        "ModelDeployConfig" :
        {
            "maxSeqLen" : 8192,
            "maxInputTokenLen" : 8192,
            "truncation" : false, #输入截断
            "ModelConfig" : [
                {
                    "modelInstanceType" : "Standard",
                    "modelName" : "32b", #  部署的服务模型名称
                    "modelWeightPath" : "/home/aicc/32B/DeepSeek-R1-Distill-Qwen-32B-W8A8",
                    "worldSize" : 4,  # 计算卡数量
                    "cpuMemSize" : 5,
                    "npuMemSize" : -1,
                    "backendType" : "atb",
                    "trustRemoteCode" : false
                }
            ]
        },

        "ScheduleConfig" :
        {
            "templateType" : "Standard",
            "templateName" : "Standard_LLM",
            "cacheBlockSize" : 128,

            "maxPrefillBatchSize" : 50,
            "maxPrefillTokens" : 8192,
            "prefillTimeMsPerReq" : 150,
            "prefillPolicyType" : 0,

            "decodeTimeMsPerReq" : 50,
            "decodePolicyType" : 0,

            "maxBatchSize" : 200,
            "maxIterTimes" : 8192,
            "maxPreemptCount" : 0,
            "supportSelectBatch" : false,
            "maxQueueDelayMicroseconds" : 5000
        }
    }
}
```
## 启动推理服务
Api接口模式
```bash
# 以下命令需在所有机器上同时执行（多机）
# 解决权重加载过慢问题
export OMP_NUM_THREADS=1
export NPU_MEMORY_FRACTION=0.95
cd /usr/local/Ascend/mindie/latest/mindie-service/
./bin/mindieservice_daemon
```

脚本推理模式
```bash
torchrun --nproc_per_node 8  --master_port 20038   -m examples.run_pa  --model_path /model/systemuser/DS-R1-Distill-Llama-70B --input_text [ --is_chat --max_output_length 128
```
接口请求示例
```bash
# 服务推理
curl 127.0.0.1:1025/generate -d '{
"prompt": "你是谁？",
"max_tokens": 32,
"stream": false,
"do_sample":true,
"repetition_penalty": 1.00,
"temperature": 0.01,
"top_p": 0.001,
"top_k": 1,
"model": "32B"
}'

curl 192.168.103.39:40000/generate -d '{
"prompt": "你是谁？",
"max_tokens": 32,
"stream": false,
"do_sample":true,
"repetition_penalty": 1.00,
"temperature": 0.01,
"top_p": 0.001,
"top_k": 1,
"model": "32B"
}'

curl localhost:1025/v1/models  # 获取服务模型名称
curl 120.92.208.206:1025/v1/models 

curl http://120.92.208.206:1025/v1/chat/completions -d '{   # 适配openai推理接口
 "model": "DeepseekR1",
 "messages": [{
     "role": "system",
     "content": "你是一个有才华的人"
    },
    {
     "role": "user", 
     "content": "生成一个有文采的1000字作文"
    }
],
 "max_tokens": 2000,
 "presence_penalty": 1.03,
 "frequency_penalty": 1.0,
 "seed": null,
 "temperature": 0.5,
 "top_p": 0.95,
 "stream": true
}'

curl -s  http://120.92.208.206:1025/v1/completions \
-H "Content-Type: application/json" \
-d '{
         "model": "70B",
          "prompt": "解释一下量子计算<think>\n",
          "max_tokens": 1024,
          "temperature": 0.1,
           "stream": true
}'
```


查看容器IP
```bash
docker inspect wyq-ds | grep IP   # 查看容器IP
```
量化脚本如下：BF16 转INT8（W8A8）
```bash
python3 quant_llama.py --model_path /home/aicc/70B/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/ --save_directory /home/aicc/70B-W8A8 --calib_file ./home/aicc/msit/msmodelslim/example/common/boolq.jsonl --w_bit 8 --a_bit 8 --device_type npu  
```

# 性能测试
使用 mindIE BenchMark 进行性能测试

注：--Concurrency 并发参数设置失效，无法测试不同并发情况。
使用合成数据（synthetic）进行性能测试样例

```bash
export MINDIE_LOG_TO_STDOUT="benchmark:1; client:1"

benchmark \
--DatasetType "synthetic" \
--ModelName DeepseekR1 \
--ModelPath "/home/deepseek-r1-w8a8" \
--TestType engine \
--Http http://localhost:1025 \
--ManagementHttp https://0.0.0.0:1026 \
--Concurrency 1 \
--MaxInputLen 1024 \
--MaxOutputLen 1024 \
--TaskKind stream \
--Tokenizer True \
--SyntheticConfigPath /usr/local/lib/python3.11/site-packages/mindiebenchmark/config/synthetic_config.json
```
使用 vllm benchmarks 测试mindie 推理服务接口

仓库地址：[vllm/benchmarks/benchmark_serving.py](https://github.com/vllm-project/vllm/blob/main/benchmarks/benchmark_serving.py)
```bash
python3 benchmarks/benchmark_serving.py \
--backend vllm \
--model cognitivecomputations/DeepSeek-R1-bf16 \
--served-model-name DeepseekR1 \  # 模型名称需和服务的模型名称对应
--host 120.92.208.206 \
--port 1025 \
--dataset-name random \
--random-input 1024 \
--random-output 1024 \
--max-concurrency 16 \
--num-prompts 20
```
