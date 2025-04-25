
**huggingface数据拉取**

```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com 
huggingface-cli download --resume-download unsloth/DeepSeek-R1-BF16 --local-dir /model
```

[下载接口 | 文档 | 魔乐社区](https://modelers.cn/docs/zh/openmind-hub-client/0.9/api_reference/download_api.html#om-hub-download)
类似huggingface 模型下载速度快

**huggingface 脚本下载数据集和模型*
```bash
from huggingface_hub import snapshot_download

# 模型下载
snapshot_download(
    "unsloth/DeepSeek-R1-BF16",
    revision="main",
    local_dir="/model",
    local_dir_use_symlinks=False,
    max_workers=8,
    allow_patterns=["*.safetensors", "*.json", "*.py"],
)

## 数据集下载
snapshot_download(
    "regisss/scrolls_gov_report_preprocessed_mlperf_2",
    revision="21ff1233ee3e87bc780ab719c755170148aba1cb",
    allow_patterns="*.parquet",
    local_dir=args.data_dir,
    local_dir_use_symlinks=False,
    max_workers=16,
    repo_type="dataset",
)
```

**脚本下载模型数据集，进程退出重新拉起**

```bash
#!/bin/bash

while true; do
    python3 down.py
    EXIT_STATUS=$?
    if [ $EXIT_STATUS -eq 0 ]; then
        echo "Program exited normally. Exiting monitoring."
        break
    else
        echo "Program exited with status $EXIT_STATUS. Restarting..."
    fi
done
```