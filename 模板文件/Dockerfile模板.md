# Dockerfile 模版文件

```dockerfile
# FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM registry.bitahub.com:5000/dockerhub/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04


USER root

# 配置 ubuntu apt 源为中科大镜像源
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

# 安装系统依赖、常用工具、修改时区、修改字符编码
ENV LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV TZ=Asia/Shanghai
ENV SHELL=/bin/bash

RUN rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        apt-utils build-essential ca-certificates software-properties-common \
        wget curl vim git openssh-server tmux htop iputils-ping iproute2 net-tools unzip tzdata locales && \
    ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo ${TZ} > /etc/timezone && \
    locale-gen en_US.UTF-8 && \
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# 用 miniconda 安装 Python
RUN wget --quiet -O ~/miniconda.sh https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py311_24.5.0-0-Linux-x86_64.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm -f ~/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH

# 设置清华 Anaconda 镜像
# 参考 https://mirror.tuna.tsinghua.edu.cn/help/anaconda/
RUN conda config --add channels defaults && \
    conda config --set show_channel_urls yes && \
    conda config --append default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
    conda config --append default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r && \
    conda config --append default_channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2 && \
    conda config --set custom_channels.conda-forge https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --set custom_channels.msys2 https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --set custom_channels.bioconda https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --set custom_channels.menpo https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --set custom_channels.pytorch https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --set custom_channels.pytorch-lts https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --set custom_channels.simpleitk https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud && \
    conda config --set custom_channels.deepmodeling https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/

# 设置清华 PyPI 镜像
# 安装 JupyterLab
# 最后解决 /opt/conda/lib/libtinfo.so.6: no version information available (required by bash)
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    python -m pip --no-cache-dir install jupyterlab && \
    conda install -c conda-forge ncurses -y && \
    rm -rf /root/.cache/pip /tmp/*

# 解决 SSH 登录后无法获得 Dockerfile ENV 指令定义的环境变量
RUN printenv | \
    grep -vE '^(SHELL|PWD|LOGNAME|MOTD_SHOWN|LANG|LS_COLORS|LC_|USER|SSH_|SHLVL|HOME|TERM|OLDPWD|_|NVIDIA_|KUBERNETES_|HOSTNAME|POD_IP)' | \
    awk '{ print "export " $1 }' > /etc/profile.d/dockerfile-env.sh
```
