# NPU 环境安装和配置

## 环境配置和下载

python 版本：3.10

显卡型号: 910b3

driver: Ascend-hdk-910b-npu-driver_23.0.3_linux-aarch64.run
firmware: Ascend-hdk-910b-npu-firmware_7.1.0.5.220.run
cann: Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run
nnae: Ascend-cann-nnae_8.1.RC1_linux-aarch64.run

下载地址：<https://www.hiascend.com/developer/download/community/result?module=speed+pt+tf+cann&product=4&model=26>

下载地址：<https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=26&cann=8.1.RC1&driver=Ascend+HDK+23.0.0>

![image-1-2025-07-28](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-1-2025-07-28.png)

![image-2025-07-31](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-2025-07-31.png)

## 卸载旧驱动，固件和 cann toookit ,nnae（24.1.0 版本）

```bash
# 卸载旧驱动
cd /usr/local/Ascend/driver/script
bash uninstall.sh

# 卸载旧固件
cd /usr/local/Ascend/firmware/script
bash uninstall.sh

# 卸载旧cann toolkit
cd /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/script
bash uninstall.sh
```

## 安装驱动和固件

以 root 用户登录，将驱动和固件包上传。创建驱动运行用户 HwHiAiUser。

```bash
groupadd -g 1000 HwHiAiUser
useradd -g HwHiAiUser -u 1000 -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash

chmod +x Ascend-hdk-910-npu-driver_23.0.rc3_linux-aarch64.run
chmod +x Ascend-hdk-910-npu-firmware_7.0.0.5.242.run
chmod +x Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run
chmod +x Ascend-cann-nnae_8.1.RC1_linux-aarch64.run

./Ascend-hdk-910-npu-driver_23.0.rc3_linux-aarch64.run --full --install-for-all
# 出现类似如下回显信息，说明安装成功。Driver package installed successfully!
# 还可以通过执行npu-smi info命令检查驱动是否加载成功。

./Ascend-hdk-910-npu-firmware_7.0.0.5.242.run --full

#出现类似如下回显信息，说明安装成功。 SFirmware package installed successfully! Reboot now or after driver installation for the installation/upgrade to take effect

./Ascend-cann-toolkit_8.1.RC1_linux-aarch64.run --install
# 安装过程中输入Y同意协议。安装完成后，若显示如下信息，则说明软件安装成功。 Ascend-cann-toolkit install success

# 配置CANN环境变量，将下面这句话加入.bashrc:

source /usr/local/Ascend/ascend-toolkit/set_env.sh

reboot

```

## 镜像信息

docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.1.rc1-910b-ubuntu22.04-py3.10

通过 docker 容器进行训练验证

```bash
docker run -it --rm \
    --shm-size=10g \
    --privileged \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \ten
    --device=/dev/davinci0 \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/sbin:/usr/local/sbin:ro \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.1.rc1-910b-ubuntu22.04-py3.10 bash
```

## Pyotrch 训练验证

实验镜像：[Dockerfile](./Dockerfile-Pytorch)

参考链接：<https://gitee.com/ascend/pytorch/releases/tag/v7.0.0-pytorch2.1.0>

实验脚本

![验证脚本](./pytorch.py)

执行结果：

python3 train.py

```bash
root@18059736c839:/home# python3 train.py 
使用NPU设备: 0
开始在NPU上训练...
Train Epoch: 1 [0/60000 (0%)]   Loss: 2.305514
Train Epoch: 1 [6400/60000 (11%)]       Loss: 2.307951
Train Epoch: 1 [12800/60000 (21%)]      Loss: 2.304825
Train Epoch: 1 [19200/60000 (32%)]      Loss: 2.302143
Train Epoch: 1 [25600/60000 (43%)]      Loss: 2.303910
Train Epoch: 1 [32000/60000 (53%)]      Loss: 2.299968
Train Epoch: 1 [38400/60000 (64%)]      Loss: 2.304023
Train Epoch: 1 [44800/60000 (75%)]      Loss: 2.302255
Train Epoch: 1 [51200/60000 (85%)]      Loss: 2.304232
Train Epoch: 1 [57600/60000 (96%)]      Loss: 2.306597
Epoch 1, Train Loss: 2.302971, Train Accuracy: 5975/60000 (9.96%)
...
```

pytortch 训练正常

## MindSpore 训练验证

实验镜像： [Dockerfile](./Dockerfile-Mindspore)

实验脚本：

![验证脚本](./mindspore.py)

实验结果：

```bash
root@8ce9e7e867ca:/home# python3 train.py
...

file_sizes: 100%|██████████████████████████| 10.8M/10.8M [00:00<00:00, 30.9MB/s]
Extracting zip file...
Successfully downloaded / unzipped to ./
Shape of image [N, C, H, W]: (64, 1, 28, 28) Float32
Shape of label: (64,) Int32
Shape of image [N, C, H, W]: (64, 1, 28, 28) Float32
Shape of label: (64,) Int32
Network(
  (flatten): Flatten()
  (dense_relu_sequential): SequentialCell(
    (0): Dense(input_channels=784, output_channels=512, has_bias=True)
    (1): ReLU()
    (2): Dense(input_channels=512, output_channels=512, has_bias=True)
    (3): ReLU()
    (4): Dense(input_channels=512, output_channels=10, has_bias=True)
  )
)
...

Epoch 3
-------------------------------
loss: 0.325300  [  0/938]
loss: 0.738744  [100/938]
loss: 0.300107  [200/938]
loss: 0.278535  [300/938]
loss: 0.184577  [400/938]
loss: 0.160165  [500/938]
loss: 0.262306  [600/938]
loss: 0.263122  [700/938]
loss: 0.114557  [800/938]
loss: 0.250729  [900/938]
Test:
 Accuracy: 94.0%, Avg loss: 0.209832

Done!
```

安装链接：<https://www.mindspore.cn/install/>
参考文档：<https://gitee.com/mindspore/docs/blob/r2.6.0/tutorials/source_zh_cn/beginner/quick_start.ipynb>

## DeepSpeed 训练验证

实验镜像：[Dockerfile](./Dockerfile-DeepSpeed)

验证脚本：

![验证脚本](./deepspeed.py)

验证命令：

```bash
deepspeed --include localhost:0,1 train.py 单机双卡
deepspeed --include localhost:0 train.py 单机单卡
```

验证结果：

```bash
[2025-07-29 08:49:05,492] [INFO] [fused_optimizer.py:345:_update_scale] Reducing dynamic loss scale from 4096.0 to 2048.0
[2025-07-29 08:49:06,775] [INFO] [logging.py:96:log_dist] [Rank 0] step=2000, skipped=6, lr=[0.001], mom=[[0.8, 0.999]]
[2025-07-29 08:49:06,776] [INFO] [timer.py:260:stop] epoch=0/micro_step=2000/global_step=2000, RunningAvgSamplesPerSec=1755.8142279325314, CurrSamplesPerSec=1681.6313929886987, MemAllocated=0.0GB, MaxMemAllocated=0.0GB
[ 1,  2000] loss:  1.681
```

## Tensorflow 训练验证

1. 使用华为镜像仓库中提供的 tensorflow 镜像，tensorflow 版本是 2.6.5

   ```bash
   docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/ascend-tensorflow:24.0.RC1-A1-ubuntu20.04
   ```

   > tesorflow 适配器只适配 tensorflow1.5._和 2.6._，其他版本都不是行。

2. 安装适配 cann8.1.rc1 的 TensorFlow Adapter 版本

   下载链接：<https://gitee.com/ascend/tensorflow/releases/tag/tfa_v0.0.36_8.1.RC1>

   ```bash
   wget https://gitee.com/ascend/tensorflow/releases/download/tfa_v0.0.36_8.1.RC1/npu_device-2.6.5-py3-none-manylinux2014_aarch64.whl

   pip3 install npu_device-2.6.5-py3-none-manylinux2014_aarch64.whl
   ```

3. 容器验证

   ```bash
   ## 起容器
   docker run -it --rm \
       --shm-size=10g \
       --privileged \
       --device=/dev/davinci_manager \
       --device=/dev/hisi_hdc \
       --device=/dev/devmm_svm \
       --device=/dev/davinci0 \
       -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
       -v /usr/local/sbin:/usr/local/sbin:ro \
       swr.cn-south-1.myhuaweicloud.com/ascendhub/ascend-tensorflow:24.0.RC1-A1-ubuntu20.04 bash

   ## 加载需要的环境变量
   source /usr/local/Ascend/ascend-toolkit/set_env.sh

   ```

4. 验证脚本

    ![验证脚本](./tensorflow.py)

5. 结果

   ```bash
   HwHiAiUser@7baf9d6fd8c6:~$ python3 train.py
   ...
   2025-07-31 02:47:08.942350: I core/npu_device_register.cpp:106] Npu device instance /job:localhost/replica:0/task:0/device:NPU:0 created
   使用NPU设备: /job:localhost/replica:0/task:0/device:NPU:0
   模型权重设备: /job:localhost/replica:0/task:0/device:NPU:0

   开始NPU训练...
   ...
   Epoch 5/5
   训练损失: 2.3026, 训练准确率: 2.3026
   测试损失: 2.3027, 测试准确率: 0.0000
   ```

## PaddlePaddle 训练验证

以容器保存为镜像构建镜像: [Dockerfile](./Dockerfile-PaddlePaddle)

PaddleCustomDevice 适配了很多厂商的卡

PaddleCustomDevice 2.6分支 支持910A芯片
PaddleCustomDevice develop分支 支持 910B 芯片，

参考链接：<https://github.com/PaddlePaddle/PaddleCustomDevice/blob/develop/backends/npu/README.md>

```bash

# 1. pull PaddlePaddle Ascend NPU development docker image
# dockerfile of the image is in tools/dockerfile directory
# Ascend 910B - check with the output of 'lspci | grep d802'
registry.baidubce.com/device/paddle-npu:cann80RC1-ubuntu20-x86_64-gcc84-py310
registry.baidubce.com/device/paddle-npu:cann80RC1-ubuntu20-aarch64-gcc84-py310

# 2. refer to the following commands to start docker container
docker run -it --rm \
    --privileged --shm-size=10G \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -e ASCEND_RT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
    registry.baidubce.com/device/paddle-npu:cann80RC1-ubuntu20-aarch64-gcc84-py310 /bin/bash

# 3. clone the source code
git clone https://github.com/PaddlePaddle/PaddleCustomDevice
cd PaddleCustomDevice/backends/npu

# 2. please ensure the PaddlePaddle cpu whl package is already installed
# the development docker image NOT have PaddlePaddle cpu whl installed by default
# you may download and install the nightly built cpu whl package with links below
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/


# 3. compile options, whether to compile with unit testing, default is ON
export WITH_TESTING=OFF

# 4. execute compile script - submodules will be synced on demand when compile
bash tools/compile.sh

# 5. install the generated whl package, which is under build/dist directory
pip install build/dist/paddle_custom_npu*.whl

# 1. list available custom backends
python -c "import paddle; print(paddle.device.get_all_custom_device_type())"
# output as following
# ['npu']

# 2. check installed custom npu version
python -c "import paddle_custom_device; paddle_custom_device.npu.version()"
# output as following
# version: 0.0.0
# commit: 9bfc65a7f11072699d0c5af160cf7597720531ea
# cann: 8.0.RC1

# 3. health check
python -c "import paddle; paddle.utils.run_check()"
# output as following
# Running verify PaddlePaddle program ...
# PaddlePaddle works well on 1 npu.
# PaddlePaddle works well on 8 npus.
# PaddlePaddle is installed successfully! Let's start deep learning with PaddlePaddle now.

```

脚本文件

![验证脚本](./paddlepaddle.py)

python3 train.py

（完全按照上面流程执行）结果：

![image-1-2025-07-31](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-1-2025-07-31.png)

## Caffe 训练验证

官方仓库的最新更新都是在 7 年起了，无法适配 npu 的训练。

## ColossalAI 训练验证

实验镜像构建： [Dockerfile](./Dockerfile-ColossalAI)

脚本文件：

![验证脚本](./colossal-ai.py)

结果

![image-3-2025-07-31](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-3-2025-07-31.png)

```bash
root@b01de8e78883:/home/train# colossalai run --nproc_per_node 2  train.py
...
[07/30/25 03:19:30] INFO     colossalai - colossalai - INFO:
                             /usr/local/python3.10.17/lib/python3.10/site-packag
                             es/colossalai/initialize.py:69 launch
                    INFO     colossalai - colossalai - INFO: Distributed
                             environment is initialized, world size: 2
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
Files already downloaded and verified
Epoch [1/80]: 100%|██████████| 250/250 [00:34<00:00,  7.34it/s, loss=1.44]
Epoch [2/80]:   8%|▊         | 20/250 [00:02<00:25,  8.95it/s, loss=1.38]
```

## 框架验证结论

| 框架         | 结果                                   |
| ------------ | -------------------------------------- |
| Pytorch      | 基于当前环境可以训练                   |
| MindSpore    | 基于当前环境可以训练                   |
| DeepSpeed    | 基于当前环境可以训练                   |
| TensorFlow   | 基于当前环境可以训练，存在版本限制       |
| PaddlePaddle | 基于当前环境可以训练                   |
| Caffe        | 无法训练                             |
| ColossalAI   | 基于当前环境可以训练                   |

镜像列表

172.16.147.11:5000/training/pytorch2.1:8.1.rc1-910b-ubuntu22.04-py3.10
172.16.147.11:5000/training/tensorflow2.6.5:8.1.rc1-910b
172.16.147.11:5000/training/mindspore2.6.0:8.1.rc1-910b-ubuntu22.04-py3.10
172.16.147.11:5000/training/deepspeed0.13.3:8.1.rc1-910b-ubuntu22.04-py3.10
172.16.147.11:5000/training/colossalai0.4.2:8.1.rc1-910b-ubuntu22.04-py3.10
172.16.147.11:5000/training/paddle-npu:cann80RC1-ubuntu20-aarch64-gcc84-py310
