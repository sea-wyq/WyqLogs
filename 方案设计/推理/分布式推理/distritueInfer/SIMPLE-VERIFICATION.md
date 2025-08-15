# KEDA StatefulSet 弹性伸缩验证

## 项目简化说明

本项目已简化为最小化验证 KEDA 控制 StatefulSet 数量弹性伸缩的核心功能。

### 核心功能

- **Service-level Scaling**: `spec.replicas` 控制创建多少个独立的 StatefulSet
- **KEDA 集成**: 通过 scale subresource 自动调整 `spec.replicas`
- **简化实现**: 使用 nginx 镜像作为测试容器，每个 StatefulSet 一个 Pod

## 快速验证步骤

### 1. 生成和部署 CRD

```bash
# 生成 CRD 和 RBAC
make manifests

# 应用 CRD
kubectl apply -f config/crd/bases/

# 部署控制器 (本地测试)
make run
```

### 2. 创建测试资源

```bash
# 创建 VLLMCluster
kubectl apply -f config/samples/simple-test.yaml

# 验证创建的 StatefulSets
kubectl get statefulsets -l app=vllm-cluster
kubectl get pods -l app=vllm-cluster

# 查看 VLLMCluster 状态
kubectl get vllmclusters
```

预期结果：

```
NAME          REPLICAS   READY   PHASE         AGE
simple-test   2          2       Ready         1m

# StatefulSets
NAME            READY   AGE
simple-test-0   1/1     1m
simple-test-1   1/1     1m

# Services
NAME            TYPE        CLUSTER-IP      PORT(S)
simple-test-0   ClusterIP   10.96.1.100     80/TCP
simple-test-1   ClusterIP   10.96.1.101     80/TCP
```

### 3. 测试手动扩缩容

```bash
# 手动扩容到 3 个 StatefulSet
kubectl patch vllmcluster simple-test -p '{"spec":{"replicas":3}}'

# 验证新建的 StatefulSet
kubectl get statefulsets -l app=vllm-cluster

# 手动缩容到 1 个 StatefulSet  
kubectl patch vllmcluster simple-test -p '{"spec":{"replicas":1}}'

# 验证删除的 StatefulSet
kubectl get statefulsets -l app=vllm-cluster
```

### 4. 验证 Scale API

```bash
# 测试 scale subresource
kubectl scale vllmcluster simple-test --replicas=4

# 查看结果
kubectl get vllmclusters simple-test -o yaml | grep -A5 status:
```

### 5. 部署 KEDA (如果未安装)

```bash
kubectl apply -f https://github.com/kedacore/keda/releases/download/v2.12.0/keda-2.12.0.yaml

# 等待 KEDA 就绪
kubectl get pods -n keda-system
```

### 6. 创建 KEDA ScaledObject

```bash
# 部署 ScaledObject
kubectl apply -f config/samples/keda-simple.yaml

# 验证 ScaledObject
kubectl get scaledobjects

# 查看自动创建的 HPA
kubectl get hpa
```

### 7. 验证 KEDA 自动扩缩容

```bash
# 给 Pod 增加 CPU 负载触发扩容
kubectl exec -it simple-test-0-0 -- sh -c "yes > /dev/null &"

# 观察扩容过程 (几分钟后)
watch kubectl get vllmclusters
watch kubectl get statefulsets -l app=vllm-cluster

# 停止负载，观察缩容
kubectl exec -it simple-test-0-0 -- pkill yes
```

## 验证要点

### 1. StatefulSet 命名规则

- `simple-test-0` (端口 8080)
- `simple-test-1` (端口 8081)
- `simple-test-2` (端口 8082)

### 2. 每个 StatefulSet 只有 1 个 Pod

```yaml
spec:
  replicas: 1  # 固定为 1
```

### 3. KEDA 控制的是 StatefulSet 数量

- `minReplicaCount: 1` → 最少 1 个 StatefulSet
- `maxReplicaCount: 5` → 最多 5 个 StatefulSet
- KEDA 修改 `VLLMCluster.spec.replicas` 触发控制器创建/删除 StatefulSet

### 4. 访问测试

```bash
# 测试每个独立服务
kubectl port-forward svc/simple-test-0 8080:80 &
curl http://localhost:8080
# 应该看到: "StatefulSet 0 is running on port 8080"

kubectl port-forward svc/simple-test-1 8081:80 &
curl http://localhost:8081  
# 应该看到: "StatefulSet 1 is running on port 8081"
```

## 清理资源

```bash
# 删除测试资源
kubectl delete -f config/samples/keda-simple.yaml
kubectl delete -f config/samples/simple-test.yaml

# 停止控制器 (Ctrl+C)
```

## 验证成功标准

✅ **创建多个 StatefulSet**: `spec.replicas=3` 创建 3 个独立的 StatefulSet  
✅ **手动扩缩容**: `kubectl scale` 命令能正确增减 StatefulSet 数量  
✅ **KEDA 自动扩缩容**: CPU 负载触发 StatefulSet 数量变化  
✅ **资源隔离**: 每个 StatefulSet 使用不同端口，完全独立  

这样就验证了 KEDA 可以通过 scale API 控制自定义资源的 StatefulSet 数量弹性伸缩。
