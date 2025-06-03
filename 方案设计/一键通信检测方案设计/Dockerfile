FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

USER root

RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

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


COPY nccl-tests nccl-tests

RUN cd  nccl-tests && \
    make -j40

COPY MLNX_OFED_LINUX-5.8-6.0.4.2-ubuntu22.04-x86_64.tgz MLNX_OFED_LINUX-5.8-6.0.4.2-ubuntu22.04-x86_64.tgz

RUN tar -zxvf MLNX_OFED_LINUX-5.8-6.0.4.2-ubuntu22.04-x86_64.tgz && \
    cd MLNX_OFED_LINUX-5.8-6.0.4.2-ubuntu22.04-x86_64 &&  \
    ./mlnxofedinstall --user-space-only --without-fw-update --force