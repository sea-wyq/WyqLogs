# VLLMCluster - KEDA StatefulSet 弹性伸缩验证项目

## 项目目标

验证 KEDA 能否实现对 StatefulSet **数量**（而非Pod数量）的弹性伸缩控制。

## 核心概念

### Service-level Scaling vs Pod-level Scaling

| 扩缩容类型 | spec.replicas=3 的效果 | 适用场景 |
|-----------|----------------------|---------|
| **Pod-level** (传统) | 1个StatefulSet，3个Pod | 单服务高并发 |
| **Service-level** (本项目) | 3个StatefulSet，各1个Pod | 多服务独立部署 |

### 实现原理

```yaml
# VLLMCluster CR
apiVersion: training.distributed-inference.io/v1alpha1
kind: VLLMCluster
metadata:
  name: test-cluster
spec:
  replicas: 3  # ← KEDA 控制此字段

# 创建的资源
StatefulSets:
- test-cluster-0 (1 Pod, 端口 8080)
- test-cluster-1 (1 Pod, 端口 8081)  
- test-cluster-2 (1 Pod, 端口 8082)

Services:
- test-cluster-0, test-cluster-1, test-cluster-2
```

## 项目结构

```
├── api/v1alpha1/
│   └── vllmcluster_types.go          # 简化的CRD定义
├── internal/controller/
│   └── vllmcluster_controller.go     # Service-level扩缩容控制器
├── config/samples/
│   ├── simple-test.yaml              # 基础测试用例
│   └── keda-simple.yaml              # KEDA ScaledObject配置
├── SIMPLE-VERIFICATION.md            # 详细验证步骤
└── README.md                         # 本文档
```

## 快速开始

### 1. 部署控制器

```bash
# 生成CRD和RBAC
make manifests

# 应用CRD
kubectl apply -f config/crd/bases/

# 本地运行控制器
make run
```

### 2. 创建测试资源

```bash
# 创建VLLMCluster (2个StatefulSet)
kubectl apply -f config/samples/simple-test.yaml

# 验证结果
kubectl get vllmclusters
kubectl get statefulsets -l app=vllm-cluster
```

预期输出：
```
NAME          REPLICAS   READY   PHASE   AGE
simple-test   2          2       Ready   30s

NAME            READY   AGE
simple-test-0   1/1     30s
simple-test-1   1/1     30s
```

### 3. 测试手动扩缩容

```bash
# 扩容到4个StatefulSet
kubectl patch vllmcluster simple-test -p '{"spec":{"replicas":4}}'

# 缩容到1个StatefulSet
kubectl scale vllmcluster simple-test --replicas=1
```

### 4. 集成KEDA自动扩缩容

```bash
# 安装KEDA
kubectl apply -f https://github.com/kedacore/keda/releases/download/v2.12.0/keda-2.12.0.yaml

# 部署ScaledObject
kubectl apply -f config/samples/keda-simple.yaml

# 验证KEDA接管扩缩容
kubectl get scaledobjects
kubectl get hpa
```

## 核心特性验证

### ✅ StatefulSet数量控制

- `spec.replicas=1` → 1个StatefulSet
- `spec.replicas=5` → 5个StatefulSet
- 每个StatefulSet独立运行，使用不同端口

### ✅ KEDA Scale API集成

- KEDA通过scale subresource修改`spec.replicas`
- 控制器监听变更，自动创建/删除StatefulSet
- 支持基于CPU、内存、HTTP等指标的自动扩缩容

### ✅ 资源隔离

- 每个StatefulSet独立的Service和端口
- 支持独立的存储和配置
- 完全的服务级别隔离

## 技术实现

### Scale Subresource配置

```go
// +kubebuilder:subresource:scale:specpath=.spec.replicas,statuspath=.status.replicas,selectorpath=.status.selector
type VLLMCluster struct {
    Spec   VLLMClusterSpec
    Status VLLMClusterStatus
}
```

### 控制器逻辑

1. **监听`spec.replicas`变化**
2. **计算需要的StatefulSet数量**  
3. **创建/删除StatefulSet和Service**
4. **聚合状态到`status.replicas`**

### KEDA集成

KEDA ScaledObject → Scale API → VLLMCluster.spec.replicas → 控制器 → StatefulSet数量变化

## 使用场景

此模式适用于：

- **多租户服务**: 每个租户独立的服务实例
- **A/B测试**: 不同版本的独立部署  
- **模型服务**: 不同模型的独立推理服务
- **微服务拆分**: 服务粒度的弹性伸缩

## 清理

```bash
# 删除测试资源
kubectl delete -f config/samples/keda-simple.yaml
kubectl delete -f config/samples/simple-test.yaml

# 删除CRD
kubectl delete -f config/crd/bases/

# 停止控制器 (Ctrl+C)
```

## 项目价值

通过此项目验证了：

1. **KEDA可以控制自定义资源的任意字段**（不仅限于Pod副本数）
2. **Service-level弹性伸缩是可行的技术方案**
3. **Kubernetes Controller模式的强大扩展性**

这为构建更复杂的分布式系统弹性伸缩方案提供了基础验证。