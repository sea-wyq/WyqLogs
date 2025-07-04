
# gitea LFS 功能验证

目的：验证使用gitea lfs功能上传大文件。

教程地址：[使用 Docker 安装 | Gitea Documentation](https://docs.gitea.com/zh-cn/installation/install-with-docker#%E5%90%AF%E5%8A%A8)
LFS 配置地址：[Git Large File Storage setup | Gitea Documentation](https://docs.gitea.com/next/usage/git-lfs-setup)

docker compose 二进制包在线安装

```bash
curl -L https://github.com/docker/compose/releases/download/1.25.0/docker-compose-`uname -s`-`uname -m` -o /usr/local/bin/docker-compose 
sudo chmod  +x /usr/local/bin/docker-compose
```

gitea安装部署

gitea部署文件

```bash
version: "3"

networks:
  gitea:
    external: false

services:
  server:
    image: registry.cnbita.com:5000/wuyiqiang/gitea:1.21.1
    container_name: gitea
    environment:
      - USER_UID=1000
      - USER_GID=1000
      - GITEA__database__DB_TYPE=mysql
      - GITEA__database__HOST=db:3306
      - GITEA__database__NAME=gitea
      - GITEA__database__USER=gitea
      - GITEA__database__PASSWD=gitea
    restart: always
    networks:
      - gitea
    volumes:
      - ./gitea:/data
      - /etc/timezone:/etc/timezone:ro
      - /etc/localtime:/etc/localtime:ro
    ports:
      - "3000:3000"
      - "222:22"
    depends_on:
      - db

  db:
    image: registry.cnbita.com:5000/wuyiqiang/mysql:8
    restart: always
    environment:
      - MYSQL_ROOT_PASSWORD=gitea
      - MYSQL_USER=gitea
      - MYSQL_PASSWORD=gitea
      - MYSQL_DATABASE=gitea
    networks:
      - gitea
    volumes:
      - ./mysql:/var/lib/mysql
```

服务部署

```bash
docker-compose up -d
```

使用docker-compose部署后访问 <http://server-ip:3000> 并遵循安装向导。

验证大文件lfs上传功能
（1）3.9G文件成功上传。
![image-2025-07-04](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-2025-07-04.png)

![image-1-2025-07-04](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-1-2025-07-04.png)

（1）7.5G文件成功上传。
![image-2-2025-07-04](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-2-2025-07-04.png)

api接口功能验证
详情看：API 使用指南：[Gitea API. | Gitea Documentation](https://docs.gitea.com/zh-cn/api/1.20/#tag/admin/operation/adminCronRun)

gitea 配置minio s3存储
(1) j进入部署得gitea容器，在gitea得配置文件/data/gitea/conf/app.ini 中添加下面参数

```bash
[lfs]
#PATH = /opt/gitea/data/lfs
STORAGE_TYPE=minio
MINIO_ACCESS_KEY_ID=LEINAOYUNOS
MINIO_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYLEINAOYUNKEY
MINIO_BUCKET=gitea
MINIO_LOCATION=us-east-1
MINIO_USE_SSL=true
SERVE_DIRECT=false
MINIO_ENDPOINT=hero-dev-miniogw.cnbita.com
```

(2) 在minio中创建gitea存储桶

git lfs 配置本地存储、minio s3、杉岩 s3 存储性能对比
本地存储
无需修改任何配置。
上传文件大小：97M
![image-3-2025-07-04](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-3-2025-07-04.png)
![image-4-2025-07-04](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-4-2025-07-04.png)

上传文件大小：7.7G
![image-5-2025-07-04](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-5-2025-07-04.png)
![image-6-2025-07-04](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-6-2025-07-04.png)

Minio S3存储
上传文件大小：97m
![image-7-2025-07-04](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-7-2025-07-04.png)
![image-8-2025-07-04](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-8-2025-07-04.png)

上传文件大小：7.7G
![image-9-2025-07-04](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-9-2025-07-04.png)
![image-10-2025-07-04](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-10-2025-07-04.png)

杉岩 S3存储
(1) 在启动得gitea容器中，在配置文件/data/gitea/conf/app.ini 中添加下面参数

```bash
[lfs]   # 只针对lfs文件得到存储
STORAGE_TYPE = sanyan

[storage.sanyan]   # 该配置针对仓库中得所有文件
STORAGE_TYPE = minio

MINIO_ENDPOINT = 10.0.104.11:8080

MINIO_ACCESS_KEY_ID = leinao

MINIO_SECRET_ACCESS_KEY = leinao

MINIO_BUCKET = model-repositories

MINIO_LOCATION = us-east-1

MINIO_USE_SSL = false

MINIO_INSECURE_SKIP_VERIFY = false  
```

(2) 或者在gitea得配置文件/data/gitea/conf/app.ini 中添加下面参数

```bash
[lfs]   # 只针对lfs文件得到存储        
...

STORAGE_TYPE = minio

MINIO_ENDPOINT = 10.0.104.11:8080

MINIO_ACCESS_KEY_ID = leinao

MINIO_SECRET_ACCESS_KEY = leinao

MINIO_BUCKET = model-repositories

MINIO_LOCATION = us-east-1

MINIO_USE_SSL = false

MINIO_INSECURE_SKIP_VERIFY = false  
```

上面两种配置都可实现添加杉岩s3 对象存储。
上传文件大小：44m
![image-11-2025-07-04](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-11-2025-07-04.png)

上传文件大小：4.1G
![image-12-2025-07-04](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-12-2025-07-04.png)

上传文件大小：7.7G
![image-13-2025-07-04](https://fourt-wyq.oss-cn-shanghai.aliyuncs.com/images/image-13-2025-07-04.png)

性能对比表

                       44M                              7.7G 
Local       upload : 5.4s , download : 3.4supload : 2.4m , download : 2.6m
S3.         upload : 3.42s,download : 4.6supload : 6.3m,download : 3.2m
SanYan.     upload : 12.6s , download : 2.4supload : 4.2m,download : 3.5m

参考文献
[配置说明 | Gitea Documentation](https://docs.gitea.com/zh-cn/administration/config-cheat-sheet)
