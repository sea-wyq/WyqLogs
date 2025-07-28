
# NPU环境安装和配置

## 1. NPU 环境卸载

### 卸载旧驱动，固件和cann toookit

```bash
# 卸载旧驱动
cd /usr/local/Ascend/driver
./install.sh uninstall

# 卸载旧固件
cd /usr/local/Ascend/driver
./install.sh uninstall_firmware

# 卸载旧cann toolkit
cd /usr/local/Ascend/ascend-toolkit/latest
./install.sh uninstall
```

## 2. NPU训练环境配置

需要配置满足mindspore，tensorflow 和 deepspeed 的训练环境

### 软件链接和cann

安装地址：<https://www.hiascend.com/developer/download/community?module=cann&product=4&model=10>

![image-1-2025-07-28](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-1-2025-07-28.png)

CANN 配置

```bash
Ascend-cann-toolkit_8.1.RC1_linux-x86_64.run
```

参考链接：<https://www.hiascend.com/developer/download/community/result?module=speed+pt+tf+cann&product=4&model=10>

MindSpore 配置

```bash
git clone -b 2.0.0_core_r0.8.0 <https://gitee.com/ascend/MindSpeed.git>
pip install -e MindSpeed
```

参考链接：<https://gitee.com/ascend/MindSpeed/tree/2.0.0_core_r0.8.0/>

pytorch-npu & pytorch 配置

```bash
pytorch== 2.1.0
torch_npu-2.1.0.post12-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

参考链接：<https://gitee.com/ascend/pytorch/releases/tag/v7.0.0-pytorch2.1.0>

TensorFlow Adapter  安装

```bash
tensorflow==2.6.5

git clone <https://gitee.com/ascend/tensorflow.git>
cd tensorflow/tf_adapter_2.x
mkdir build
cd build
cmake ..
make -j8
make install **.whl
```

参考链接：<https://gitee.com/ascend/tensorflow/releases/tag/tfa_v0.0.36_8.1.RC1>

### Dirver和Firmware

```bash
Ascend-hdk-910-npu-driver_23.0.rc3_linux-aarch64.run
Ascend-hdk-910-npu-firmware_7.0.0.5.242.run
```

参考链接：<https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=10&cann=7.0.0.alpha003&driver=1.0.21.alpha>

## 3. 安装驱动和固件

下载链接 <https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=10&cann=7.0.0.alpha003&driver=1.0.21.alpha>
下载链接:<https://www.hiascend.com/developer/download/community/result?module=cann&product=4&model=10>

![image-2025-07-28](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-2025-07-28.png)

以root用户登录，将驱动和固件包上传。创建驱动运行用户HwHiAiUser。

```bash
groupadd -g 1000 HwHiAiUser
useradd -g HwHiAiUser -u 1000 -d /home/HwHiAiUser -m HwHiAiUser -s /bin/bash

chmod +x Ascend-hdk-910-npu-driver_23.0.rc3_linux-aarch64.run
chmod +x Ascend-hdk-910-npu-firmware_7.0.0.5.242.run

./ Ascend-hdk-910-npu-driver_23.0.rc3_linux-aarch64.run --full --install-for-all
# 出现类似如下回显信息，说明安装成功。Driver package installed successfully!
# 还可以通过执行npu-smi info命令检查驱动是否加载成功。

./ Ascend-hdk-910-npu-firmware_7.0.0.5.242.run --full

#出现类似如下回显信息，说明安装成功。 SFirmware package installed successfully! Reboot now or after driver installation for the installation/upgrade to take effect


chmod +x Ascend-cann-toolkit_7.0.0.alpha003_linux-aarch64.run

./Ascend-cann-toolkit_7.0.0.alpha001_linux-x86_64.run --install
# 安装过程中输入Y同意协议。安装完成后，若显示如下信息，则说明软件安装成功。 Ascend-cann-toolkit install success

# 配置CANN环境变量，将下面这句话加入.bashrc:

source /usr/local/Ascend/ascend-toolkit/set_env.sh

reboot

```

CANN对python版本有要求，同时需要安装一些python包，可以通过下面的脚本检查是否满足：

```bash
cd /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/bin
bash prereq_check.bash
```

## 4. 验证

```bash
python3 -c "import torch;import torch_npu;print(torch_npu.npu.is_available())"
```
