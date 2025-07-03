
## 安装c++环境

```bash

sudo apt-get install build-essential

```

## 安装go环境

```bash
#  下载安装包
sudo wget https://golang.google.cn/dl/go1.23.5.linux-amd64.tar.gz
sudo wget https://golang.google.cn/dl/go1.23.5.linux-arm64.tar.gz
sudo rm -rf /usr/local/go && sudo tar -C /usr/local -xzf go1.23.5.linux-amd64.tar.gz

# sudo vim 打开 $HOME/.profile 或者/etc/profile文件，追加导出命令
export PATH=$PATH:/usr/local/go/bin
export GOPROXY=https://goproxy.io,direct
export GO111MODULE=on


source /etc/profile

# 验证是否生效

 go version

```

# 安装miniconda

```bash
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py311_23.11.0-1-Linux-x86_64.sh
chmod 777 Miniconda3-py311_23.11.0-1-Linux-x86_64.sh
bash Miniconda3-py311_23.11.0-1-Linux-x86_64.sh
一直yes
bash

# 检查是否安装成功
conda --help

# 配置conda镜像

conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes


# 创建环境

#创建conda小环境 - -n 用来指定环境的名称
conda create -n myenv
#指定环境中需要带的python的版本
conda create -n myenv python=3.8.5
# 启动小环境
conda activate myenv
#退出小环境
conda deactivate
#查看共有多少个小环境
conda env list  / conda info --env
#删除conda小环境
conda remove -n python --all
 
更新软件：conda update 软件名
卸载软件：conda remove 软件名
删除环境：conda remove -n 环境名
克隆环境：conda create –n 新环境名 –clone 旧环境名

```
