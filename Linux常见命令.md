# Linux常见命令

## 检查 libnvidia-ml.so.1 库文件是否存在：使用以下命令检查 libnvidia-ml.so.1 库文件是否存在于系统中

```bash
find / -name libnvidia-ml.so.1
```

## 地址数量计算

10.96.0.0/24 整个地址范围下有多少地址?
10.96.0.0/24地址范围下有256个地址。在CIDR表示法中，/24表示子网掩码为255.255.255.0，其中有8位用于主机地址，因此有2^8=256个可用地址。
10.96.0.0/16 整个地址范围下有多少地址?
10.96.0.0/16地址范围下有65536个地址。在CIDR表示法中，/16表示子网掩码为255.255.0.0，其中有16位用于主机地址，因此有2^16=65536个可用地址。
10.96.0.0/20 整个地址范围下有多少地址?
10.96.0.0/16地址范围下有4094个地址。在CIDR表示法中，/20表示子网掩码为255.16.0.0，其中有12位用于主机地址，因此有2^12=4094个可用地址。

## CPU的计量单位

CPU的计量单位叫毫核。集群中的每一个节点可以通过操作系统确认本节点的CPU内核数量，将这个数量乘以1000，得到的就是节点总的CPU总数量。
如，一个节点有两个核，那么该节点的CPU总量为2000m。如果你要使用单核的十分之一，则你要求的是100m。　　
即，这个m的单位，是将一个cpu内核抽象化，分成1000等份。每一份即为一个millicore，即千分之一个核，或毫核。跟米与毫米的关系是一样的。

## ansible 多机命令同时执行

## 覆盖docker镜像默认设置的entrypoint命令

```bash
docker run -it --rm -v /root/tmp/minist:/home/HwHiAiUser/minist  --entrypoint  bash  5f29bfb81121
```

## 每秒执行一下命令

```bash
watch -n 1 nvidia-smi
```

## 拉取gitlab仓库代码镜像配置

```bash
From registry.cnbita.com:5000/golangci/golang:1.21

ENV GITLAB_ACCESS_TOKEN=********

RUN echo "machine gitlab.bitahub.com login oauth2accesstoken password ${GITLAB_ACCESS_TOKEN}" > ~/.netrc

RUN go env -w GOPROXY=<https://goproxy.cn,direct>

RUN go env -w GOPRIVATE=gitlab.bitahub.com/hero-os/hero-os-util
```

## cpu 信息获取

lscpu

## 磁盘信息读取

sudo smartctl -a /dev/sda

## 硬盘信息获取

sudo fdisk -l

## pyotrch 设置日志等级

```bash
export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
```

## 使用镜像本身的cuda而不是通过device-plugin 挂载的cuda ?

```bash
ls -l /usr/lib/x86_64-linux-gnu/libcuda.so*
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.470.74 /usr/lib/x86_64-linux-gnu/libcuda.so.1
registry.cnbita.com:5000/yangpengcheng/llama-recipes:nightly-20240426
```

## docker 启动镜像和pod 命令启动镜像的区别？（可执行文件找不到，例如pip3）

观察~/.bashrc 文件 （环境的配置和环境变量的设置）
示例Dockerfile

```bash
FROM swr.cn-central-221.ovaijisuan.com/wh-aicc-fae/mindie:910A-ascend_24.1.rc3-cann_8.0.t63-py_3.10-ubuntu_20.04-aarch64-mindie_1.0.T71.02

ENV PATH=/root/miniconda3/envs/Python310/bin:$PATH
ENV LD_LIBRARY_PATH=/root/miniconda3/envs/Python310/lib:$LD_LIBRARY_PATH

RUN pip3 install jupyterlab
```

## torchrun 启动命令和参数设置

```bash
export NCCL_DEBUG=INFO
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

torchrun --nnodes=2 --nproc_per_node=1 \
    --master_addr=10.244.58.79 \
    --master_port=21000 \
    /app/checker.py

torchrun --nnodes=2 --nproc_per_node=1 \
    --rdzv_id=demo \  
    --local_addr=10.244.58.77 \ # 防止域名解析失败
    --rdzv_backend=c10d \
    --rdzv_endpoint=10.244.58.77:21000 \
     /app/checker.py
```

## 查看节点端口是否在被监听

```bash
netstat -tuln | grep 1025
```

## IP联通性验证流程

```bash
验证 IP 层可达性
ping <目标IP>  # ICMP协议，验证网络层是否通（可能被防火墙禁用）

验证端口是否开放（TCP/UDP）
telnet <目标IP> <端口>  # 经典工具，成功则显示「Connected」
nc -zv <目标IP> <端口>   # 更轻量，支持脚本化，输出端口状态
nc -zuv <目标IP> <端口>  # 发送 UDP 探测包，需服务端响应

验证 DNS 解析（若用域名访问）
nslookup <域名>  # 检查域名是否正确解析为目标 IP
ping <域名>       # 确认解析后的 IP 是否正确

验证arp协议的ip mac地址是否正常同步了
arp 172.17.0.7

查看路由表
ip route

通用路由追踪（ICMP/UDP，可能被防火墙过滤）
traceroute <目标IP>  # Linux 默认用 UDP

tcpdump 抓包确认是否收到数据
tcpdump -i any host <目标IP> and port <端口> -nn -vv  # 监听所有网卡
```

## Conda配置下载源

```bash
conda config --add channels <https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/>
conda config --add channels <https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/>
conda config --add channels <https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud//pytorch/>
conda config --add channels <https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/>
conda config --set show_channel_urls yes
```

## K8S-默认存储类StorageClass 设置

```bash
eclass local-path -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```

## find 路径 -name 文件名    在指定路径下按照文件名寻找文件

```bash
find / -name .bashrc                     #在根目录下寻找.bashrc文件
```

## SCP 通过 SSH 协议安全地将文件复制到远程系统和从远程系统复制文件到本地

```bash
scp network-snapshot-001600.pkl yqwu18@172.31.94.45:/home/yqwu18/hucheng_s61948944  #复制文件
scp -r models/ yqwu18@172.31.94.45:/home/yqwu18/hucheng_s61948944  #复制文件夹

-C - 这会在复制过程中压缩文件或目录。
-P - 如果默认 SSH 端口不是 22，则使用此选项指定 SSH 端口。
-r - 此选项递归复制目录及其内容。
-p - 保留文件的访问和修改时间。
```

## 查看系统目录的空间占用

```bash
dh -h   
```

## 修改文件权限

```bash
chown 777 -R models/     #递归修改文件夹的权限
chmod +x  文件名          # 给文件添加执行权限
```

## 查看端口是否占用

```bash
lsof -i:端口号
```

查看文档的行数

```bash
 wc 文件名                     # 方法1
 cat 文件名 | grep -c ""       # 方法2
```

## 查看当下目录的所占内存

```bash
du -sh
```

## 查看文件所占用的内存

```bash
du -sh file  
```

## 查看文件大小

```bash
wc -c 文件名
```

## MD5加密算法

```bash
md5sum models.tar.gz
```

## 通过域名查找ip地址

```bash
nslookup 域名
```

## 查看linux系统的版本是ubunut还是centos

```bash
cat /etc/os-release
```

## 查看系统架构

```bash
arch
```

## 获取主机IP

```bash
hostname -I
```

## linux换源

```bash
sudo vim /etc/apt/sources.list
deb <https://mirrors.ustc.edu.cn/ubuntu/> focal main restricted universe multiverse
deb-src <https://mirrors.ustc.edu.cn/ubuntu/> focal main restricted universe multiverse
deb <https://mirrors.ustc.edu.cn/ubuntu/> focal-updates main restricted universe multiverse
deb-src <https://mirrors.ustc.edu.cn/ubuntu/> focal-updates main restricted universe multiverse
deb <https://mirrors.ustc.edu.cn/ubuntu/> focal-backports main restricted universe multiverse
deb-src <https://mirrors.ustc.edu.cn/ubuntu/> focal-backports main restricted universe multiverse
deb <https://mirrors.ustc.edu.cn/ubuntu/> focal-security main restricted universe multiverse
deb-src <https://mirrors.ustc.edu.cn/ubuntu/> focal-security main restricted universe multiverse
deb <https://mirrors.ustc.edu.cn/ubuntu/> focal-proposed main restricted universe multiverse
deb-src <https://mirrors.ustc.edu.cn/ubuntu/> focal-proposed main restricted universe multiverse
```

## 运行sh脚本提示syntax error unexpected end of file

vim 打开文件，输入:set ff=unix

## 修改系统变量

1、在/etc/profile文件中添加变量【对所有用户生效（永久的）】
2、在用户目录下的.bash_profile文件中增加变量【对单一用户生效（永久的）】
3、直接运行export命令定义变量【只对当前shell（BASH）有效（临时的）】

## 模拟系统负载较高时的场景

```bash
sudo apt install stress

消耗 CPU 资源
$ stress -c 4
消耗内存资源
$ stress --vm 2 --vm-bytes 300M --vm-keep
```

## 可以删除所有python进程

```bash
pkill -f python
```
