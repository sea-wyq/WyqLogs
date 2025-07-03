# 常用命令

正常下载

```bash
wget <https://golang.google.cn/dl/go1.16.15.linux-amd64.tar.gz>
```

安静模式

```bash
wget -q <https://golang.google.cn/dl/go1.16.15.linux-amd64.tar.gz>
```

断点续传

```bash
wget -c <https://golang.google.cn/dl/go1.16.15.linux-amd64.tar.gz>
```

重命名

```bash
wget -O test.tar.gz  <https://golang.google.cn/dl/go1.16.15.linux-amd64.tar.gz>
```

后台执行

```bash
wget -b <https://golang.google.cn/dl/go1.16.15.linux-amd64.tar.gz>
```
