
## 新部署服务inference-controller

代码仓库：<https://gitlab.bitahub.com/hero-os/infra-nexus-schedule/inference-controller/-/tree/test?ref_type=heads>
分支：test
镜像构建：docker build --platform linux/amd64 -t registry.cnbita.com:5000/heros-dev/inference-controller:v1.2.4 .
helm包地址：<https://gitlab.bitahub.com/hero-os/infra-nexus-schedule/inference-controller/-/tree/test/helm/inference-controller>
部署namespce: hero-system
部署位置：成员集群
注：需要在成员集群nacos中，新增inference-controller nacos配置 inference-controller-config.yaml， 配置文件如下：

```bash
EnableWebhooks: "true"
LicenseServiceURL: http://license-auth.monitoring:8000
Namespace: hero-user
MaxPendingDuration: 5
#ModelRepositoryHFEndpoint: http://10.0.101.71
#ModelRepositoryHFEndpoint: http://${GLOBAL_GIT.URL}
ModelRepositoryHFDownloadTimeout: 6000
Prometheus: http://prometheus-k8s.monitoring.svc.cluster.local:9090
HostSuffix: .nip.io
```

## 更新服务：mcluster-interpreter

代码仓库：<https://gitlab.bitahub.com/hero-os/infra-cluster-matrix/mlcluster-interpreter>
分支：test
部署位置：主控集群
注：  
（1）需要在karmada 集群中部署 inferenceservice crd文件。
kubectl apply -f <https://gitlab.bitahub.com/hero-os/infra-cluster-matrix/mlcluster-interpreter/-/blob/test/crd/infer/system.hero.ai_inferenceservices.yaml>
（2）在主控集群的clusterpropagationpolicies 新增InferenceService字段信息。对需要下发该资源的子中心进行新增。
示例如下：kubectl edit clusterpropagationpolicies.policy.karmada.io a15155090705084416176536

```bash
- apiVersion: system.hero.ai/v1alpha1
    kind: InferenceService
    labelSelector:
      matchLabels:
        cluster.hero.ai/cluster: a15155090705084416176536
```

（3） 需要在karmada 集群中部署 CustomResourceDefinition inferenceservice 文件。
kubectl apply -f <https://gitlab.bitahub.com/hero-os/infra-cluster-matrix/mlcluster-interpreter/-/blob/test/customization/inferenceservice.yaml>

## 更新服务 resource-manager

仓库地址：<https://gitlab.bitahub.com/hero-os/infra-nexus-resource-manager/resource-manager>
分支：test
部署位置: 主控和成员集群
