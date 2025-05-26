FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends infiniband-diags iproute2 net-tools \
    openssh-server wget vim perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 libnccl2=2.23.4-1+cuda12.7 libnccl-dev=2.23.4-1+cuda12.7

# install openmpi
RUN apt install -y openmpi-bin libopenmpi-dev

# install nccl-tests
RUN cd ~ && wget https://github.com/NVIDIA/nccl-tests/archive/refs/tags/v2.13.11.tar.gz && tar -zxvf v2.13.11.tar.gz && cd nccl-tests-2.13.11 && make -j40 MPI=1 MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi CUDA_HOME=/usr/local/cuda NCCL_HOME=/usr/lib/x86_64-linux-gnu