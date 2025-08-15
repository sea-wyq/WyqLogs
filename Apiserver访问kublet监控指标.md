
# kubernetes API

```bash
#查看kubernetes 的API
kubectl get --raw /
#使用URL 查询Kubernetes 资源
#查询Node 资源
kubectl get --raw /api/v1/nodes|python -m json.tool
#查询service 资源
kubectl get --raw /api/v1/services|python -m json.tool
#查询namespace 资源
kubectl get --raw /api/v1/namespaces|python -m json.tool
#查询Pod 资源
kubectl get --raw /api/v1/pods|python -m json.tool
#查询某个namespace 的Deployment资源
kubectl get --raw /apis/apps/v1/namespaces/${namespace}/deployments|python -m json.tool
#查询某个namespace 的Pod资源
kubectl get --raw /apis/apps/v1/namespaces/${namespace}
```

## 通过kubectl 查看kubelet的cadvisor监控指标

```bash
kubectl get --raw=/api/v1/nodes/{nodeName}/proxy/metrics/cadvisor
```
