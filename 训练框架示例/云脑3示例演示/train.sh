#!/bin/bash

# 公共环境变量设置
export HCCL_CONNECT_TIMEOUT=600
export RANK_TABLE_FILE=/user/config/jobstart_hccl.json
export RANK_SIZE=256
export DEPLOY_MODE=0
export GE_USE_STATIC_MEMORY=1

# 循环处理每个设备
for i in {0..7}; do
    # 计算CPU核心范围（每个设备分配40个核心）
    cpu_start=$((i * 40))
    cpu_end=$((cpu_start + 39))
    
    # 设置当前设备的环境变量
    export DEVICE_ID=$i
    export RANK_ID=$i
    
    # 创建日志目录并复制文件
    rm -rf "LOG$i"
    mkdir -p "./LOG$i/ms_log"
    cp *.py "./LOG$i"
    env > "./LOG$i/env.log"
    
    # 设置日志路径
    export GLOG_log_dir="/data/bert/LOG$i/ms_log"
    export GLOG_logtostderr=0
    
    # 切换到日志目录并启动训练进程
    cd "/data/bert/LOG$i"
    taskset -c $cpu_start-$cpu_end python /data/bert/run_pretrain.py \
        --distribute=true \
        --epoch_size=10 \
        --enable_save_ckpt=false \
        --do_shuffle=false \
        --enable_data_sink=true \
        --data_sink_steps=100 \
        --accumulation_steps=1 \
        --allreduce_post_accumulation=true \
        --save_checkpoint_path=./ \
        --save_checkpoint_num=1 \
        --config_path=/data/bert/pretrain_config_Ascend_Boost.yaml \
        --data_url=/data/bert_large/bert_large/new_train_data/ \
        --eval_data_dir=/data/bert_large/bert_large/new_eval_data/ \
        --load_checkpoint_path=/data/bert_large/bert_large/msdata/new_ckpt.ckpt \
        --device_id=$i \
        --device_num=8 &
    
    # 返回原目录并输出信息
    cd -
    echo "run with rank_id=$i device_id=$i logic_id=$i"
done

wait