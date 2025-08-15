
# docker 命令

## 将宿主机下载的工具挂载到容器

```bash
-v /usr/local/sbin:/usr/local/sbin
```

## 多架构镜像构建

```bash
docker buildx build -f Dockerfile -t registry.cnbita.com:5000/cluster-images/model-repository:v2.1  --platform=linux/arm64,linux/amd64 --push .
```

## 批量删除镜像

```bash
docker images | grep node-exporter-infiniband | awk '{print $1 ":" $2}' | xargs docker rmi
```

## 拉取对应架构的镜像

```bash
docker pull golang:1.19-alpine3.18  --platform=linux/arm64   
```

## 根据不同集群架构环境拉取对应架构的镜像

```bash
export DOCKER_CLI_EXPERIMENTAL=enabled

# 构建 amd64 平台的镜像
docker buildx build -t myapp:1.0-amd64 --platform linux/amd64 .
# 构建 arm64 平台的镜像
docker buildx build -t myapp:1.0-arm64 --platform linux/arm64 .

docker push myapp:1.0-amd64
docker push myapp:1.0-arm64

docker manifest create myapp:1.0 \
  myapp:1.0-amd64 \
  myapp:1.0-arm64
    
docker manifest push myapp:1.0
```

## 镜像更名

```bash
docker tag  grafana/mimir:2.15.0  registry.cnbita.com:5000/wuyiqiang/grafana_mimir:2.15.0
```

## 查看镜像和容器休息

```bash
docker inspect 镜像名称
docker inspect 容器名称
```

/etc/docker/daemon.json   docker 配置文件地址

docker inspect 容器名称
docker inspect 镜像名称

docker info 查看docker相关配置信息

## 临时修改容器配置，使用 runc 运行时（以 Docker 为例）

docker run --rm --runtime=runc registry.hub.com:5000/38suo/pytorch-train-38:1.2
