# npu 监控指标支持

```bash
"job_gpu_utilization":     ` npu_chip_info_utilization *on ( vdie_id) group_left npu_container_info{containerName=~".*$2.*"}`,
"job_gpu_mem_utilization": `npu_chip_info_hbm_used_memory  *on ( vdie_id) group_left npu_container_info{containerName=~".*$2.*"} /npu_chip_info_hbm_total_memory *on ( vdie_id) group_left npu_container_info{containerName=~".*$2.*"} * 100`,
"job_gpu_mem_used":        `npu_chip_info_hbm_used_memory *on ( vdie_id) group_left npu_container_info{containerName=~"$2.*"}`,
"job_gpu_card_used":       `kube_pod_container_resource_requests{resource=~"huawei_com.*"}* on (pod) group_left () (kube_pod_status_phase{namespace="hero-user",pod=~"$2.*",phase="Running"} > 0)`,
"job_gpu_mem_all":         `npu_chip_info_hbm_total_memory *on ( vdie_id) group_left npu_container_info{containerName=~".*$2.*"}`,
"cluster_occupied_npu_num": `sum(kube_pod_status_phase{phase="Running"} * on (pod) group_left sum(kube_pod_container_resource_requests{resource=~"huawei_com.*"}) by (pod, resource))`,


npu_chip_info_utilization *on ( vdie_id) group_left npu_container_info{pod_name="$pod"}
(npu_chip_info_hbm_used_memory / npu_chip_info_hbm_total_memory*100)*on ( vdie_id) group_left npu_container_info{pod_name="$pod"}
(npu_chip_info_hbm_total_memory-npu_chip_info_hbm_used_memory) *on ( vdie_id) group_left npu_container_info{pod_name="$pod"}
npu_chip_info_hbm_used_memory *on ( vdie_id) group_left npu_container_info{pod_name="$pod"}
npu_chip_info_hbm_total_memory *on ( vdie_id) group_left npu_container_info{pod_name="$pod"}

machine_npu_nums *on (pod) group_left(node) kube_pod_info{node="$node"}
kube_pod_container_resource_requests{resource=~"nvidia_com.*|huawei_com.*"}  * on (pod) group_left () sum by (pod) (kube_pod_status_phase{phase=~"Pending|Running"} != 0) 
kube_node_status_capacity{resource=~"nvidia_com.*|huawei_com.*"}

```

## 安装部署

```bash
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: npu-exporter
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: npu-exporter
  template:
    metadata:
      annotations:
        seccomp.security.alpha.kubernetes.io/pod: runtime/default
      labels:
        app: npu-exporter
    spec:
      nodeSelector:
        node-role.kubernetes.io/npu: "true"
      containers:
      - name: npu-exporter
        image: registry.hub.com:5000/cluster-images/npu-exporter:v5.0.0
        resources:
          requests:
              memory: 1000Mi
              cpu: 1000m
          limits:
              memory: 1000Mi
              cpu: 1000m
        imagePullPolicy: Never
        command: [ "/bin/bash", "-c", "--"]
        # pair firstly
        args: [ "umask 027;npu-exporter -port=8082 -ip=0.0.0.0  -updateTime=5
                 -logFile=/var/log/mindx-dl/npu-exporter/npu-exporter.log -logLevel=0 -containerMode=docker" ]
        securityContext:
          privileged: true
          readOnlyRootFilesystem: true
          runAsUser: 0
          runAsGroup: 0
        ports:
          - name: http
            containerPort: 8082
            protocol: TCP
        volumeMounts:
          - name: log-npu-exporter
            mountPath: /var/log/mindx-dl/npu-exporter
          - name: localtime
            mountPath: /etc/localtime
            readOnly: true
          - name: ascend-driver
            mountPath: /usr/local/Ascend/driver
            readOnly: true
          - name: ascend-dcmi
            mountPath: /usr/local/dcmi
            readOnly: true
          - name: sys
            mountPath: /sys
            readOnly: true
          - name: docker-shim  # delete when only use containerd or isula
            mountPath: /var/run/dockershim.sock
            readOnly: true
          - name: docker  # delete when only use containerd or isula
            mountPath: /var/run/docker
            readOnly: true
          - name: cri-dockerd  # reserve when k8s version is 1.24+ and the container runtime is docker
            mountPath: /var/run/cri-dockerd.sock
            readOnly: true
          - name: containerd  # delete when only use isula
            mountPath: /run/containerd
            readOnly: true
          - name: isulad  # delete when use containerd or docker
            mountPath: /run/isulad.sock
            readOnly: true
          - name: tmp
            mountPath: /tmp
      volumes:
        - name: log-npu-exporter
          hostPath:
            path: /var/log/mindx-dl/npu-exporter
            type: Directory
        - name: localtime
          hostPath:
            path: /etc/localtime
        - name: ascend-driver
          hostPath:
            path: /usr/local/Ascend/driver
        - name: ascend-dcmi
          hostPath:
            path: /usr/local/dcmi
        - name: sys
          hostPath:
            path: /sys
        - name: docker-shim # delete when only use containerd or isula
          hostPath:
            path: /var/run/dockershim.sock
        - name: docker  # delete when only use containerd or isula
          hostPath:
            path: /var/run/docker
        - name: cri-dockerd # reserve when k8s version is 1.24+ and the container runtime is docker
          hostPath:
            path: /var/run/cri-dockerd.sock
        - name: containerd  # delete when only use isula
          hostPath:
            path: /run/containerd
        - name: isulad  # delete when use containerd or docker
          hostPath:
            path: /run/isulad.sock
        - name: tmp
          hostPath:
            path: /tmp

---
apiVersion: v1
kind: Service
metadata:
  name: npu-exporter-service
  namespace: monitoring
  labels:
    app.kubernetes.io/component: exporter
    app.kubernetes.io/name: npu-exporter
    app.kubernetes.io/part-of: kube-prometheus
spec:
  ports:
  - name: metrics
    port: 8082
    protocol: TCP
    targetPort: 8082
  selector:
    app: npu-exporter
  type: NodePort

---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  labels:
    app.kubernetes.io/component: exporter
    app.kubernetes.io/name: npu-exporter
    app.kubernetes.io/part-of: kube-prometheus
  name: npu-exporter
  namespace: monitoring
spec:
  endpoints:
  - interval: 10s
    port: metrics
  selector:
    matchLabels:
      app.kubernetes.io/component: exporter
      app.kubernetes.io/name: npu-exporter
      app.kubernetes.io/part-of: kube-prometheus
```
