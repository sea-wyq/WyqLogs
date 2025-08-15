# VLLMCluster 部署指南

## 概述

VLLMCluster 是一个自定义 Kubernetes 控制器，用于部署和管理大模型 (LLM) 的分布式推理服务。它基于 VLLM 引擎，支持 StatefulSet 部署和 KEDA 自动扩缩容。

## 功能特性

- ✅ **自动化部署**: 通过声明式 API 自动创建 StatefulSet、Service 等资源
- ✅ **分布式推理**: 支持张量并行和流水线并行
- ✅ **自动扩缩容**: 集成 KEDA，支持多种触发器类型
- ✅ **资源管理**: 支持 GPU、CPU、内存资源调度
- ✅ **存储管理**: 支持模型存储和缓存卷配置
- ✅ **监控集成**: 内置健康检查和指标收集

## 架构图

\`\`\`mermaid
graph TB
    A[VLLMCluster CR] --> B[VLLMCluster Controller]
    B --> C[StatefulSet]
    B --> D[Service]
    B --> E[KEDA ScaledObject]
    
    C --> F[VLLM Pod 1]
    C --> G[VLLM Pod 2]
    C --> H[VLLM Pod N]
    
    E --> I[Prometheus Metrics]
    E --> J[HTTP Requests]
    E --> K[Queue Length]
    
    F --> L[GPU Node 1]
    G --> M[GPU Node 2]
    H --> N[GPU Node N]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style E fill:#fff3e0
    style L fill:#e8f5e8
    style M fill:#e8f5e8
    style N fill:#e8f5e8
\`\`\`

## 前置要求

### 1. Kubernetes 集群
- Kubernetes 1.19+
- 支持 CRD 和 Controller
- 配置 GPU 节点（可选）

### 2. 依赖组件
\`\`\`bash
# 安装 KEDA (可选，如需自动扩缩容)
kubectl apply -f https://github.com/kedacore/keda/releases/download/v2.12.0/keda-2.12.0.yaml

# 安装 Prometheus (可选，如需监控)
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack
\`\`\`

## 安装控制器

### 1. 克隆代码
\`\`\`bash
git clone <repository-url>
cd distributed-inference
\`\`\`

### 2. 生成和应用 CRD
\`\`\`bash
make manifests
kubectl apply -f config/crd/bases/
\`\`\`

### 3. 部署控制器
\`\`\`bash
# 构建镜像 (如果需要)
make docker-build IMG=your-registry/vllm-controller:latest
make docker-push IMG=your-registry/vllm-controller:latest

# 部署控制器
make deploy IMG=your-registry/vllm-controller:latest
\`\`\`

## 使用示例

### 基础部署

\`\`\`yaml
apiVersion: training.distributed-inference.io/v1alpha1
kind: VLLMCluster
metadata:
  name: my-vllm-cluster
spec:
  model:
    modelPath: "microsoft/DialoGPT-medium"
    maxModelLength: 2048
  vllm:
    tensorParallelSize: 1
    pipelineParallelSize: 1
    servePort: 8000
  replicas: 1
  resources:
    resources:
      requests:
        memory: "4Gi"
        cpu: "2"
      limits:
        memory: "8Gi" 
        cpu: "4"
\`\`\`

### 高级部署 (带GPU和扩缩容)

\`\`\`yaml
apiVersion: training.distributed-inference.io/v1alpha1
kind: VLLMCluster
metadata:
  name: advanced-vllm-cluster
spec:
  model:
    modelPath: "meta-llama/Llama-2-7b-chat-hf"
    quantizationMethod: "awq"
  vllm:
    tensorParallelSize: 2
    pipelineParallelSize: 1
    gpuMemoryUtilization: "0.9"
  replicas: 2
  resources:
    resources:
      requests:
        nvidia.com/gpu: "2"
        memory: "16Gi"
      limits:
        nvidia.com/gpu: "2"
        memory: "32Gi"
    nodeSelector:
      accelerator: "nvidia-tesla-v100"
    tolerations:
    - key: "nvidia.com/gpu"
      operator: "Equal"
      value: "present"
      effect: "NoSchedule"
  autoScaling:
    enabled: true
    minReplicas: 1
    maxReplicas: 5
    triggers:
    - type: "prometheus"
      metadata:
        serverAddress: "http://prometheus:9090"
        metricName: "vllm_request_queue_size"
        threshold: "10"
\`\`\`

## KEDA 集成

### 支持的触发器类型

1. **Prometheus 指标**
   - 请求队列长度
   - GPU 利用率
   - 响应时间

2. **HTTP 请求**
   - 端点响应时间
   - 并发连接数

3. **消息队列**
   - RabbitMQ 队列长度
   - Redis List 长度
   - Kafka 消费滞后

4. **系统资源**
   - CPU 使用率
   - 内存使用率

### 独立 KEDA 配置

\`\`\`yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-cluster-scaler
spec:
  scaleTargetRef:
    apiVersion: training.distributed-inference.io/v1alpha1
    kind: VLLMCluster
    name: my-vllm-cluster
  minReplicaCount: 1
  maxReplicaCount: 10
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus:9090
      metricName: vllm_queue_size
      threshold: '5'
      query: avg(vllm_request_queue_size{job="vllm"})
\`\`\`

## 监控和诊断

### 检查状态
\`\`\`bash
# 查看 VLLMCluster 状态
kubectl get vllmclusters

# 详细状态信息
kubectl describe vllmcluster my-vllm-cluster

# 查看 Pod 状态
kubectl get pods -l app=vllm-cluster,cluster=my-vllm-cluster

# 查看日志
kubectl logs -l app=vllm-cluster,cluster=my-vllm-cluster
\`\`\`

### 健康检查端点
- `http://<service-ip>:8000/health` - 服务健康状态
- `http://<service-ip>:8000/v1/models` - 可用模型列表
- `http://<service-ip>:8000/metrics` - Prometheus 指标

### 常见问题排查

1. **Pod 启动失败**
   - 检查镜像是否存在
   - 验证资源需求
   - 查看节点资源可用性

2. **模型加载失败**
   - 检查模型路径是否正确
   - 验证存储卷配置
   - 查看内存是否足够

3. **GPU 不可用**
   - 确认节点有 GPU 资源
   - 检查 GPU 驱动和运行时
   - 验证资源请求配置

## 最佳实践

### 资源配置
1. **内存**: 模型大小的 1.5-2 倍
2. **GPU**: 根据模型大小选择合适的 GPU 类型
3. **存储**: 使用高性能 SSD 存储模型和缓存

### 扩缩容配置  
1. **最小副本**: 至少保持 1 个副本运行
2. **最大副本**: 根据集群资源限制设置
3. **指标阈值**: 根据业务 SLA 调整触发器阈值
4. **稳定窗口**: 设置适当的冷却时间避免抖动

### 安全配置
1. 使用专用的 ServiceAccount
2. 配置 NetworkPolicy 限制网络访问
3. 使用 Secret 管理敏感配置
4. 启用 RBAC 权限控制

## 故障排除

### 控制器日志
\`\`\`bash
kubectl logs -n system deployment/controller-manager
\`\`\`

### 资源事件
\`\`\`bash
kubectl get events --sort-by='.lastTimestamp'
\`\`\`

### 调试模式
\`\`\`bash
# 启用调试日志
kubectl patch deployment controller-manager -p '{"spec":{"template":{"spec":{"containers":[{"name":"manager","args":["--zap-log-level=debug"]}]}}}}'
\`\`\`

## 升级指南

### 控制器升级
1. 备份当前配置
2. 更新 CRD 定义
3. 滚动更新控制器
4. 验证功能正常

### 模型版本更新
1. 更新 VLLMCluster CR 中的 model.revision
2. 控制器将自动处理滚动更新
3. 监控更新过程和服务可用性

## 社区和支持

- 项目仓库: [GitHub Repository]
- 问题反馈: [GitHub Issues]
- 文档: [项目文档]
- 社区讨论: [Discord/Slack]

## 许可证

本项目采用 Apache License 2.0 许可证。

