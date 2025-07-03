# 模版文件

```bash
apiVersion: v1
kind: Endpoints
metadata:
  name: wangs-ds-standalone
  namespace: monitoring
  labels:
    app: wangs-ds-standalone
subsets:
- addresses:
  - ip: 10.1.30.4
  ports:
  - name: http
    port: 12345
    protocol: TCP
---
apiVersion: v1
kind: Service
metadata:
  name: wangs-ds-standalone
  namespace: monitoring
  labels:
    app: wangs-ds-standalone
spec:
  ports:
  - name: http
    protocol: TCP
    port: 12345
    targetPort: 12345
  sessionAffinity: None
  type: ClusterIP
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  labels:
    app: wangs-ds-standalone
  name: wangs-ds-standalone
  namespace: monitoring
spec:
  endpoints:
  - interval: 30s
    path: /metrics
    port: http
  namespaceSelector:
    matchNames:
    - monitoring
  selector:
    matchLabels:
      app: wangs-ds-standalone

```
