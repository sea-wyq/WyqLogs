FROM xj-registry.bitahub.com:5000/autotest0304/nvidia-pytorch:24.01-py3-rdma

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
RUN cd  nccl-tests && make -j40