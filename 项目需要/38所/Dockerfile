FROM swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:8.1.rc1-910b-ubuntu22.04-py3.10

RUN pip3 install torch torch_npu 
RUN pip3 install deepspeed

ENV ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest 
ENV LD_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH
ENV PATH=$ASCEND_HOME_PATH/bin:$PATH
ENV PYTHONPATH=$ASCEND_HOME_PATH/python/site-packages:$PYTHONPATH
ENV ASCEND_RUNTIME_PATH=/usr/local/Ascend/ascend-toolkit/latest/runtime

# example env

RUN apt update && apt install -y libopenmpi-dev && pip3 install mpi4py
