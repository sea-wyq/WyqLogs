使用token 对kubelet 进行接口访问


```bash

---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: temp-role
rules:
- apiGroups:
  - ""
  resources:
  - nodes/metrics
  - nodes/stats
  - nodes/proxy
  verbs:
  - get

---

apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: temp-rolebinding
  namespace: default
subjects:
  - kind: ServiceAccount
    name: temp-sa
    namespace: default
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: temp-role

---

apiVersion: v1
kind: ServiceAccount
metadata:
  name: temp-sa
  namespace: default

---

apiVersion: v1
kind: Secret
metadata:
  name: temp-sa-secret
  namespace: default
  annotations:
    kubernetes.io/service-account.name: "temp-sa"
type: kubernetes.io/service-account-token
```



curl -s -k -H "Authorization: Bearer eyJhbGciOiJSUzI1NiIsImtpZCI6ImV5NjJva3kyaUNWaUxQVTh2b2RZRGUtMVFEc3oxQUJOYUxjVlNudE9HSEUifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJkZWZhdWx0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6InRlbXAtc2Etc2VjcmV0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQubmFtZSI6InRlbXAtc2EiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC51aWQiOiI3MTcyNmI4My1jZDljLTQ0MzEtYWQxMC00YjViMzBjNzdkZWIiLCJzdWIiOiJzeXN0ZW06c2VydmljZWFjY291bnQ6ZGVmYXVsdDp0ZW1wLXNhIn0.wzT0k9eQmxrfhShOzA1JuXGvkFFOPVXTbs1inxkGhYh9bifXQ17m9F9AH1Y80hrzQST2Y6kM2DWksjzTrOmFIYftur6_CEoNee7i7Xp3Edo_gquCvz-TCyzMQY5rjzmN6IENPGECDwYtoVheddCewfAxRlT2W3L4ttsPeRGfIj1cuJHgv-qdMxTkepRDTuen-ouW4yJo8lFmqu6nDBzN8UmAPDj_iXOuAYGiBCbAXW07ipflB-mIyDwPz8Q7nuaQrX2q8aGpondFwQiYKs98D32Vp0aDvACWmbo2Bvk6d7v-6WYz6JRCWHWpxjydiEAug7eyQVqZSjlKR7vcejZhvA" https://10.0.102.61:10250/pods/fluid-system | jq


结果正常访问