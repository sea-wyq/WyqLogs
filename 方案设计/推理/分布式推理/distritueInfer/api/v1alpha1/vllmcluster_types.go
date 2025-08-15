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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// VLLMClusterSpec defines the desired state of VLLMCluster.
type VLLMClusterSpec struct {
	// Replicas specifies the number of StatefulSet replicas (distributed instances)
	// +kubebuilder:default:=1
	// +kubebuilder:validation:Minimum:=1
	Replicas *int32 `json:"replicas,omitempty"`

	// Image specifies the container image
	// +kubebuilder:default:="nginx:latest"
	Image *string `json:"image,omitempty"`

	// Port for the service
	// +kubebuilder:default:=8080
	Port *int32 `json:"port,omitempty"`
}

// VLLMClusterStatus defines the observed state of VLLMCluster.
type VLLMClusterStatus struct {
	// Replicas is the current number of replicas
	Replicas int32 `json:"replicas,omitempty"`

	// ReadyReplicas is the number of ready replicas
	ReadyReplicas int32 `json:"readyReplicas,omitempty"`

	// Phase indicates the current phase
	Phase string `json:"phase,omitempty"`

	// Selector is the label selector for the pods managed by this VLLMCluster
	// Used by the scale subresource
	Selector string `json:"selector,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:subresource:scale:specpath=.spec.replicas,statuspath=.status.replicas,selectorpath=.status.selector
// +kubebuilder:printcolumn:name="Replicas",type="integer",JSONPath=".spec.replicas"
// +kubebuilder:printcolumn:name="Ready",type="integer",JSONPath=".status.readyReplicas"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// VLLMCluster is the Schema for the vllmclusters API.
type VLLMCluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   VLLMClusterSpec   `json:"spec,omitempty"`
	Status VLLMClusterStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// VLLMClusterList contains a list of VLLMCluster.
type VLLMClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []VLLMCluster `json:"items"`
}

func init() {
	SchemeBuilder.Register(&VLLMCluster{}, &VLLMClusterList{})
}
