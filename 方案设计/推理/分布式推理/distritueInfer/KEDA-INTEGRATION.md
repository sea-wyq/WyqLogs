# KEDA 与 VLLMCluster 集成详解

## 核心映射关系

KEDA ScaledObject 的 `minReplicaCount` 和 `maxReplicaCount` 字段控制的是 VLLMCluster CR 中的 **`spec.replicas`** 字段。

### 字段映射

```yaml
# KEDA ScaledObject
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
spec:
  scaleTargetRef:
    apiVersion: training.distributed-inference.io/v1alpha1
    kind: VLLMCluster
    name: my-vllm-cluster
  minReplicaCount: 1    # 控制 VLLMCluster.spec.replicas 的最小值
  maxReplicaCount: 10   # 控制 VLLMCluster.spec.replicas 的最大值

---
# VLLMCluster CR  
apiVersion: training.distributed-inference.io/v1alpha1
kind: VLLMCluster
spec:
  replicas: 2  # ← 这个字段被 KEDA 控制
  # ... 其他配置
```

## 实现原理

### 1. Scale Subresource 配置

在 VLLMCluster 的类型定义中，我们通过 kubebuilder 注释配置了 scale subresource：

```go
// +kubebuilder:subresource:scale:specpath=.spec.replicas,statuspath=.status.replicas,selectorpath=.status.selector
type VLLMCluster struct {
    // ...
}
```

这个注释告诉 Kubernetes API Server：
- **specpath=.spec.replicas**: scale 操作修改 `spec.replicas` 字段
- **statuspath=.status.replicas**: 当前副本数状态存储在 `status.replicas`
- **selectorpath=.status.selector**: Pod 选择器存储在 `status.selector`

### 2. VLLMCluster 结构定义

```go
type VLLMClusterSpec struct {
    // Replicas specifies the number of StatefulSet replicas
    // +kubebuilder:default:=1
    // +kubebuilder:validation:Minimum:=1
    Replicas *int32 `json:"replicas,omitempty"`
    // ... 其他字段
}

type VLLMClusterStatus struct {
    // Replicas is the current number of replicas
    Replicas int32 `json:"replicas,omitempty"`
    
    // ReadyReplicas is the number of ready replicas  
    ReadyReplicas int32 `json:"readyReplicas,omitempty"`
    
    // Selector is the label selector for pods
    Selector string `json:"selector,omitempty"`
    // ... 其他字段
}
```

### 3. 控制器实现逻辑

在控制器的 `updateStatus` 方法中，我们更新状态字段：

```go
// updateStatus updates the VLLMCluster status
func (r *VLLMClusterReconciler) updateStatus(ctx context.Context, vllmCluster *trainingv1alpha1.VLLMCluster) error {
    // 从 StatefulSet 获取当前状态
    if vllmCluster.Status.StatefulSetName != "" {
        statefulSet := &appsv1.StatefulSet{}
        // ... 获取 StatefulSet
        
        // 更新副本状态
        vllmCluster.Status.Replicas = statefulSet.Status.Replicas
        vllmCluster.Status.ReadyReplicas = statefulSet.Status.ReadyReplicas
    }
    
    // 更新 selector 用于 scale subresource
    vllmCluster.Status.Selector = fmt.Sprintf("app=vllm-cluster,cluster=%s,component=vllm-server", vllmCluster.Name)
    
    // 更新状态
    return r.Status().Update(ctx, vllmCluster)
}
```

### 4. StatefulSet 创建逻辑

控制器根据 `spec.replicas` 创建 StatefulSet：

```go
func (r *VLLMClusterReconciler) createStatefulSetForVLLMCluster(vllmCluster *trainingv1alpha1.VLLMCluster) *appsv1.StatefulSet {
    statefulSet := &appsv1.StatefulSet{
        // ...
        Spec: appsv1.StatefulSetSpec{
            Replicas: vllmCluster.Spec.Replicas,  // ← 使用 spec.replicas
            // ...
        },
    }
    return statefulSet
}
```

## 工作流程

1. **KEDA 监控指标**：根据配置的 triggers 监控各种指标（CPU、内存、队列长度等）

2. **计算目标副本数**：KEDA 根据指标计算出期望的副本数

3. **调用 Scale API**：KEDA 调用 VLLMCluster 的 scale subresource API

4. **更新 spec.replicas**：Kubernetes API Server 更新 VLLMCluster 的 `spec.replicas` 字段

5. **控制器响应变更**：VLLMCluster 控制器检测到 spec 变更，触发 reconcile

6. **更新 StatefulSet**：控制器更新 StatefulSet 的副本数

7. **Pod 扩缩容**：StatefulSet 控制器创建或删除 Pod

8. **状态同步**：VLLMCluster 控制器更新 status 字段反映当前状态

## KEDA 触发器示例

### Prometheus 触发器
```yaml
triggers:
- type: prometheus
  metadata:
    serverAddress: http://prometheus:9090
    metricName: vllm_request_queue_size
    threshold: '10'
    query: avg(vllm_request_queue_size{job="vllm-cluster"})
```

### HTTP 触发器
```yaml
triggers:
- type: http
  metadata:
    targetValue: '100'  # 目标响应时间 100ms
    url: http://vllm-cluster:8000/v1/models
```

### CPU/内存触发器
```yaml
triggers:
- type: cpu
  metadata:
    type: Utilization
    value: '70'
- type: memory  
  metadata:
    type: Utilization
    value: '80'
```

## 生成的 Scale Subresource

在生成的 CRD 中会包含 scale subresource 配置：

```yaml
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: vllmclusters.training.distributed-inference.io
spec:
  # ...
  versions:
  - name: v1alpha1
    # ...
    subresources:
      status: {}
      scale:
        specReplicasPath: .spec.replicas
        statusReplicasPath: .status.replicas
        labelSelectorPath: .status.selector
```

## 验证扩缩容

```bash
# 查看当前副本数
kubectl get vllmcluster my-cluster -o jsonpath='{.spec.replicas}'

# 查看 KEDA ScaledObject 状态
kubectl get scaledobject my-cluster-scaler -o yaml

# 查看 HPA（KEDA 内部使用）
kubectl get hpa

# 手动测试 scale API
kubectl scale vllmcluster my-cluster --replicas=3
```

这样，KEDA 就可以通过标准的 Kubernetes scale API 来控制 VLLMCluster 的副本数，实现基于各种指标的自动扩缩容。

