```bash
---

apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: hero-role
rules:
  - apiGroups: ["monitoring.hero.ai"]
    resources: ["*"]
    verbs: ["*"]

---

apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: hero-rolebinding
subjects:
  - kind: ServiceAccount
    name: hero-sa
    namespace: heros-controllers-system
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: hero-role

---

apiVersion: v1
kind: ServiceAccount
metadata:
  name: hero-sa
  namespace: heros-controllers-system

---

apiVersion: v1
kind: Secret
metadata:
  name: hero-sa-secret
  namespace: heros-controllers-system
  annotations:
    kubernetes.io/service-account.name: "hero-sa"
type: kubernetes.io/service-account-token

```