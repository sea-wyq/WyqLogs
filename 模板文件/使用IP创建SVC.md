# 模版文件

```bash
apiVersion: v1
kind: Endpoints
metadata:
  name: sd-32b
  namespace: bita-user
  labels:
    app: sd-32b
subsets:
- addresses:
  - ip: 172.17.0.2
  ports:
  - name: http
    port: 1025
    protocol: TCP
---
apiVersion: v1
kind: Service
metadata:
  name: sd-32b
  namespace: bita-user
  labels:
    app: sd-32b
spec:
  ports:
  - name: http
    protocol: TCP
    port: 1025
    targetPort: 1025
  sessionAffinity: None
  type: ClusterIP


```
