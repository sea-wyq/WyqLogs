/*
Copyright 2025.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controller

import (
	"context"
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	trainingv1alpha1 "github.com/wuyiqiang/distributed-inference/api/v1alpha1"
)

// VLLMClusterReconciler reconciles a VLLMCluster object
type VLLMClusterReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=training.distributed-inference.io,resources=vllmclusters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=training.distributed-inference.io,resources=vllmclusters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=training.distributed-inference.io,resources=vllmclusters/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=statefulsets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=services,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop
func (r *VLLMClusterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	// Fetch the VLLMCluster instance
	vllmCluster := &trainingv1alpha1.VLLMCluster{}
	err := r.Get(ctx, req.NamespacedName, vllmCluster)
	if err != nil {
		if errors.IsNotFound(err) {
			log.Info("VLLMCluster resource not found. Ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Set defaults
	r.setDefaults(vllmCluster)

	// Handle multiple StatefulSets (Service-level scaling)
	if err := r.reconcileStatefulSets(ctx, vllmCluster); err != nil {
		log.Error(err, "Failed to reconcile StatefulSets")
		return ctrl.Result{}, err
	}

	// Handle Services
	if err := r.reconcileServices(ctx, vllmCluster); err != nil {
		log.Error(err, "Failed to reconcile Services")
		return ctrl.Result{}, err
	}

	// Update status
	if err := r.updateStatus(ctx, vllmCluster); err != nil {
		log.Error(err, "Failed to update VLLMCluster status")
		return ctrl.Result{}, err
	}

	return ctrl.Result{}, nil
}

// setDefaults sets default values
func (r *VLLMClusterReconciler) setDefaults(vllmCluster *trainingv1alpha1.VLLMCluster) {
	if vllmCluster.Spec.Replicas == nil {
		replicas := int32(1)
		vllmCluster.Spec.Replicas = &replicas
	}
	if vllmCluster.Spec.Image == nil {
		image := "nginx:latest"
		vllmCluster.Spec.Image = &image
	}
	if vllmCluster.Spec.Port == nil {
		port := int32(8080)
		vllmCluster.Spec.Port = &port
	}
}

// reconcileStatefulSets manages multiple StatefulSets based on spec.replicas
func (r *VLLMClusterReconciler) reconcileStatefulSets(ctx context.Context, vllmCluster *trainingv1alpha1.VLLMCluster) error {
	log := logf.FromContext(ctx)

	desiredReplicas := *vllmCluster.Spec.Replicas

	// List existing StatefulSets
	existingStatefulSets := &appsv1.StatefulSetList{}
	listOpts := []client.ListOption{
		client.InNamespace(vllmCluster.Namespace),
		client.MatchingLabels{
			"app":     "vllm-cluster",
			"cluster": vllmCluster.Name,
		},
	}

	err := r.List(ctx, existingStatefulSets, listOpts...)
	if err != nil {
		return err
	}

	currentCount := len(existingStatefulSets.Items)
	log.Info("StatefulSet scaling", "current", currentCount, "desired", desiredReplicas)

	// Create new StatefulSets if needed
	if int32(currentCount) < desiredReplicas {
		for i := int32(currentCount); i < desiredReplicas; i++ {
			statefulSet := r.createStatefulSet(vllmCluster, i)
			log.Info("Creating StatefulSet", "name", statefulSet.Name)
			if err := r.Create(ctx, statefulSet); err != nil {
				return err
			}
		}
	}

	// Delete excess StatefulSets if needed
	if int32(currentCount) > desiredReplicas {
		for i := int32(currentCount) - 1; i >= desiredReplicas; i-- {
			statefulSetName := fmt.Sprintf("%s-%d", vllmCluster.Name, i)
			statefulSet := &appsv1.StatefulSet{}
			namespacedName := types.NamespacedName{Name: statefulSetName, Namespace: vllmCluster.Namespace}

			err := r.Get(ctx, namespacedName, statefulSet)
			if err == nil {
				log.Info("Deleting StatefulSet", "name", statefulSetName)
				if err := r.Delete(ctx, statefulSet); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

// createStatefulSet creates a StatefulSet for specific index
func (r *VLLMClusterReconciler) createStatefulSet(vllmCluster *trainingv1alpha1.VLLMCluster, index int32) *appsv1.StatefulSet {
	// Each StatefulSet has only 1 replica
	replicas := int32(1)
	basePort := *vllmCluster.Spec.Port
	servicePort := basePort + index

	labels := map[string]string{
		"app":     "vllm-cluster",
		"cluster": vllmCluster.Name,
		"index":   fmt.Sprintf("%d", index),
	}

	container := corev1.Container{
		Name:  "app",
		Image: *vllmCluster.Spec.Image,
		Ports: []corev1.ContainerPort{
			{
				Name:          "http",
				ContainerPort: servicePort,
			},
		},
		// Simple nginx welcome page to verify functionality
		Command: []string{"sh", "-c"},
		Args:    []string{fmt.Sprintf("echo 'StatefulSet %d is running on port %d' > /usr/share/nginx/html/index.html && nginx -g 'daemon off;'", index, servicePort)},
		// Add resource requests for CPU metrics
		Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("100m"),
				corev1.ResourceMemory: resource.MustParse("128Mi"),
			},
		},
	}

	statefulSet := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-%d", vllmCluster.Name, index),
			Namespace: vllmCluster.Namespace,
			Labels:    labels,
		},
		Spec: appsv1.StatefulSetSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: labels,
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{container},
				},
			},
		},
	}

	controllerutil.SetControllerReference(vllmCluster, statefulSet, r.Scheme)
	return statefulSet
}

// reconcileServices manages Services for each StatefulSet
func (r *VLLMClusterReconciler) reconcileServices(ctx context.Context, vllmCluster *trainingv1alpha1.VLLMCluster) error {
	log := logf.FromContext(ctx)

	desiredReplicas := *vllmCluster.Spec.Replicas
	basePort := *vllmCluster.Spec.Port

	for i := int32(0); i < desiredReplicas; i++ {
		servicePort := basePort + i
		service := r.createService(vllmCluster, i, servicePort)
		serviceName := fmt.Sprintf("%s-%d", vllmCluster.Name, i)
		namespacedName := types.NamespacedName{Name: serviceName, Namespace: vllmCluster.Namespace}

		existingService := &corev1.Service{}
		err := r.Get(ctx, namespacedName, existingService)
		if err != nil && errors.IsNotFound(err) {
			log.Info("Creating Service", "name", serviceName)
			if err := r.Create(ctx, service); err != nil {
				return err
			}
		}
	}

	return nil
}

// createService creates a Service for specific index
func (r *VLLMClusterReconciler) createService(vllmCluster *trainingv1alpha1.VLLMCluster, index int32, port int32) *corev1.Service {
	labels := map[string]string{
		"app":     "vllm-cluster",
		"cluster": vllmCluster.Name,
		"index":   fmt.Sprintf("%d", index),
	}

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s-%d", vllmCluster.Name, index),
			Namespace: vllmCluster.Namespace,
			Labels:    labels,
		},
		Spec: corev1.ServiceSpec{
			Selector: labels,
			Ports: []corev1.ServicePort{
				{
					Name:       "http",
					Port:       80,
					TargetPort: intstr.FromInt32(port),
				},
			},
		},
	}

	controllerutil.SetControllerReference(vllmCluster, service, r.Scheme)
	return service
}

// updateStatus updates the VLLMCluster status
func (r *VLLMClusterReconciler) updateStatus(ctx context.Context, vllmCluster *trainingv1alpha1.VLLMCluster) error {
	// Get all StatefulSets
	statefulSets := &appsv1.StatefulSetList{}
	listOpts := []client.ListOption{
		client.InNamespace(vllmCluster.Namespace),
		client.MatchingLabels{
			"app":     "vllm-cluster",
			"cluster": vllmCluster.Name,
		},
	}

	err := r.List(ctx, statefulSets, listOpts...)
	if err != nil {
		return err
	}

	// Aggregate status
	totalReplicas := int32(0)
	totalReady := int32(0)

	for _, sts := range statefulSets.Items {
		totalReplicas += sts.Status.Replicas
		totalReady += sts.Status.ReadyReplicas
	}

	vllmCluster.Status.Replicas = totalReplicas
	vllmCluster.Status.ReadyReplicas = totalReady

	if totalReady == *vllmCluster.Spec.Replicas {
		vllmCluster.Status.Phase = "Ready"
	} else if totalReplicas > 0 {
		vllmCluster.Status.Phase = "Progressing"
	} else {
		vllmCluster.Status.Phase = "Pending"
	}

	// Set selector for scale subresource
	vllmCluster.Status.Selector = fmt.Sprintf("app=vllm-cluster,cluster=%s", vllmCluster.Name)

	return r.Status().Update(ctx, vllmCluster)
}

// SetupWithManager sets up the controller with the Manager
func (r *VLLMClusterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&trainingv1alpha1.VLLMCluster{}).
		Owns(&appsv1.StatefulSet{}).
		Owns(&corev1.Service{}).
		Complete(r)
}
