# 常用配置

## 常用命令

使用SSH连接远程主机（需要登录密码验证）

```bash
ssh wyq@192.168.106.137 -p 2222 (默认端口22)
```

使用ssh在远程主机执行一条命令并显示到本地, 然后继续本地工作

```bash
ssh wyq@192.168.106.137 ls -l
```

通过跳板机登录目的主机

```bash
ssh -J wyq@192.168.106.137 root@192.168.106.142
```

## 节点间ssh免密（两个节点都需要操作）

ssh-keygen -t rsa -b 4096 -C "<wuyiqiang@example.com>"
ssh-copy-id root@10.1.30.43     或者 手动将直接将公钥文件内容拷贝到服务器上，vim ~/.ssh/authorized_keys

vim ~/.ssh/config

Host a100-44
    HostName 10.1.30.44
    User root
    IdentityFile ~/.ssh/id_rsa

## 构建ssh镜像

```bash
FROM ubuntu:16.04
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:wyq' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
```

## SSH Socks5代理

目标： 将所有ssh服务通过代理的方式暴露出去，减少网络的攻击面。

代理脚本实现：

```go
package main

import (
 "context"
 "log"

 "github.com/armon/go-socks5"
)

func main() {
 conf := &socks5.Config{
  Rules: &PermitDestPort{
   Ports: []int{22},
  },
 }
 log.Println("rule set: permit port 22")
 server, err := socks5.New(conf)
 if err != nil {
  panic(err)
 }
 log.Println("socks proxy server listen on :9527...")
 if err := server.ListenAndServe("tcp", ":9527"); err != nil {
  panic(err)
 }
}

// PermitDestPort is an implementation of the RuleSet which
// filtering supported ports
type PermitDestPort struct {
 Ports []int
}

func (p *PermitDestPort) Allow(ctx context.Context, req *socks5.Request) (context.Context, bool) {
 for _, port := range p.Ports {
  if req.DestAddr.Port == port {
   return ctx, true
  }
 }
 return ctx, false
}

```

服务部署

```bash
apiVersion: apps/v1
kind: Deployment
metadata:



  labels:
    app.kubernetes.io/instance: socks-proxy-stg61
  name: test

spec:
  
  replicas: 1

  selector:
    matchLabels:
      app: socks-proxy
      app.kubernetes.io/instance: socks-proxy-stg61
      app.kubernetes.io/name: socks-proxy
 
  template:
    metadata:
      creationTimestamp: null
      labels:
        app: socks-proxy
        app.kubernetes.io/instance: socks-proxy-stg61
        app.kubernetes.io/name: socks-proxy
    spec:
     
      containers:
      - image: registry.cnbita.com:5000/hero-archive/www/socks-proxy:v4.1.0
        imagePullPolicy: IfNotPresent
        name: socks-proxy

      dnsConfig:
        nameservers:
        - 10.96.0.10
        options:
        - name: ndots
          value: "5"
        searches:
        - hero-user.svc.cluster.local
        - svc.cluster.local
      dnsPolicy: None
      nodeSelector:
        node-role.kubernetes.io/hero-system: "true"
      restartPolicy: Always
      schedulerName: default-scheduler

```

## SSH连接 远程主机 报错 Permission denied (publickey)

```bash
vim /etc/ssh/sshd_config

# PasswordAuthentication no --> PasswordAuthentication yes
# PermitRootLogin without-password -->PermitRootLogin yes

sudo service ssh restart
```

## 报错ssh: Could not resolve hostname zook2: Name or service not known

检测IP地址是否可以ping通，
检测域名解析是否成功。

解决方案：修改etc/hosts中的文件，加入所有主机的地址映射

## SSH启动报错：Missing privilege separation directory: /run/sshd

解决办法： 创建一个目录 mkdir /var/run/sshd
