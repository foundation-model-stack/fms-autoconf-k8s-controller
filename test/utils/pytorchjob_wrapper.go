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

package utils

import (
	kubeflowv1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

// PyTorchJobWrapper wraps a PyTorchJob for easier test construction.
type PyTorchJobWrapper struct {
	kubeflowv1.PyTorchJob
}

// MakePyTorchJob creates a wrapper for a PyTorchJob with a default configuration.
func MakePyTorchJob(name, namespace string) *PyTorchJobWrapper {
	return &PyTorchJobWrapper{
		PyTorchJob: kubeflowv1.PyTorchJob{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "kubeflow.org/v1",
				Kind:       "PyTorchJob",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:        name,
				Namespace:   namespace,
				Labels:      make(map[string]string),
				Annotations: make(map[string]string),
			},
			Spec: kubeflowv1.PyTorchJobSpec{
				PyTorchReplicaSpecs: make(map[kubeflowv1.ReplicaType]*kubeflowv1.ReplicaSpec),
			},
		},
	}
}

// Obj returns the inner PyTorchJob.
func (w *PyTorchJobWrapper) Obj() *kubeflowv1.PyTorchJob {
	return &w.PyTorchJob
}

// Clone returns a deep copy of the wrapper.
func (w *PyTorchJobWrapper) Clone() *PyTorchJobWrapper {
	return &PyTorchJobWrapper{
		PyTorchJob: *w.PyTorchJob.DeepCopy(),
	}
}

// Label sets a label on the PyTorchJob.
func (w *PyTorchJobWrapper) Label(key, value string) *PyTorchJobWrapper {
	w.Labels[key] = value
	return w
}

// Annotation sets an annotation on the PyTorchJob.
func (w *PyTorchJobWrapper) Annotation(key, value string) *PyTorchJobWrapper {
	if w.Annotations == nil {
		w.Annotations = make(map[string]string)
	}
	w.Annotations[key] = value
	return w
}

// Suspend sets the suspend field on the PyTorchJob.
func (w *PyTorchJobWrapper) Suspend(suspend bool) *PyTorchJobWrapper {
	if w.Spec.RunPolicy.Suspend == nil {
		w.Spec.RunPolicy.Suspend = ptr.To(suspend)
	} else {
		*w.Spec.RunPolicy.Suspend = suspend
	}
	return w
}

// MasterReplicaSpec creates or updates the Master replica spec.
func (w *PyTorchJobWrapper) MasterReplicaSpec(replicas int32) *ReplicaSpecWrapper {
	if w.Spec.PyTorchReplicaSpecs == nil {
		w.Spec.PyTorchReplicaSpecs = make(map[kubeflowv1.ReplicaType]*kubeflowv1.ReplicaSpec)
	}
	if w.Spec.PyTorchReplicaSpecs["Master"] == nil {
		w.Spec.PyTorchReplicaSpecs["Master"] = &kubeflowv1.ReplicaSpec{
			Replicas: ptr.To(replicas),
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "pytorch",
							Image: "pytorch:latest",
						},
					},
				},
			},
		}
	} else {
		w.Spec.PyTorchReplicaSpecs["Master"].Replicas = ptr.To(replicas)
	}
	return &ReplicaSpecWrapper{
		job:         w,
		replicaType: "Master",
	}
}

// WorkerReplicaSpec creates or updates the Worker replica spec.
func (w *PyTorchJobWrapper) WorkerReplicaSpec(replicas int32) *ReplicaSpecWrapper {
	if w.Spec.PyTorchReplicaSpecs == nil {
		w.Spec.PyTorchReplicaSpecs = make(map[kubeflowv1.ReplicaType]*kubeflowv1.ReplicaSpec)
	}
	if w.Spec.PyTorchReplicaSpecs["Worker"] == nil {
		w.Spec.PyTorchReplicaSpecs["Worker"] = &kubeflowv1.ReplicaSpec{
			Replicas: ptr.To(replicas),
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "pytorch",
							Image: "pytorch:latest",
						},
					},
				},
			},
		}
	} else {
		w.Spec.PyTorchReplicaSpecs["Worker"].Replicas = ptr.To(replicas)
	}
	return &ReplicaSpecWrapper{
		job:         w,
		replicaType: "Worker",
	}
}

// ReplicaSpecWrapper wraps a replica spec for easier configuration.
type ReplicaSpecWrapper struct {
	job         *PyTorchJobWrapper
	replicaType kubeflowv1.ReplicaType
}

// Container configures the container in the replica spec.
func (r *ReplicaSpecWrapper) Container() *ContainerWrapper {
	spec := r.job.Spec.PyTorchReplicaSpecs[r.replicaType]
	if len(spec.Template.Spec.Containers) == 0 {
		spec.Template.Spec.Containers = []corev1.Container{
			{
				Name:  "pytorch",
				Image: "pytorch:latest",
			},
		}
	}
	return &ContainerWrapper{
		replicaSpec: r,
		container:   &spec.Template.Spec.Containers[0],
	}
}

// PodSpec returns a wrapper for the pod spec.
func (r *ReplicaSpecWrapper) PodSpec() *PodSpecWrapper {
	spec := r.job.Spec.PyTorchReplicaSpecs[r.replicaType]
	return &PodSpecWrapper{
		replicaSpec: r,
		podSpec:     &spec.Template.Spec,
	}
}

// Done returns to the PyTorchJobWrapper.
func (r *ReplicaSpecWrapper) Done() *PyTorchJobWrapper {
	return r.job
}

// ContainerWrapper wraps a container for easier configuration.
type ContainerWrapper struct {
	replicaSpec *ReplicaSpecWrapper
	container   *corev1.Container
}

// Image sets the container image.
func (c *ContainerWrapper) Image(image string) *ContainerWrapper {
	c.container.Image = image
	return c
}

// Command sets the container command.
func (c *ContainerWrapper) Command(command ...string) *ContainerWrapper {
	c.container.Command = command
	return c
}

// Args sets the container args.
func (c *ContainerWrapper) Args(args ...string) *ContainerWrapper {
	c.container.Args = args
	return c
}

// Env adds an environment variable to the container.
func (c *ContainerWrapper) Env(name, value string) *ContainerWrapper {
	c.container.Env = append(c.container.Env, corev1.EnvVar{
		Name:  name,
		Value: value,
	})
	return c
}

// GPURequest sets the GPU resource request and limit.
func (c *ContainerWrapper) GPURequest(count int64) *ContainerWrapper {
	if c.container.Resources.Requests == nil {
		c.container.Resources.Requests = corev1.ResourceList{}
	}
	if c.container.Resources.Limits == nil {
		c.container.Resources.Limits = corev1.ResourceList{}
	}
	qty := resource.MustParse(string(rune(count)))
	c.container.Resources.Requests["nvidia.com/gpu"] = qty
	c.container.Resources.Limits["nvidia.com/gpu"] = qty
	return c
}

// CPURequest sets the CPU resource request and limit.
func (c *ContainerWrapper) CPURequest(cpu string) *ContainerWrapper {
	if c.container.Resources.Requests == nil {
		c.container.Resources.Requests = corev1.ResourceList{}
	}
	if c.container.Resources.Limits == nil {
		c.container.Resources.Limits = corev1.ResourceList{}
	}
	qty := resource.MustParse(cpu)
	c.container.Resources.Requests[corev1.ResourceCPU] = qty
	c.container.Resources.Limits[corev1.ResourceCPU] = qty
	return c
}

// MemoryRequest sets the memory resource request and limit.
func (c *ContainerWrapper) MemoryRequest(memory string) *ContainerWrapper {
	if c.container.Resources.Requests == nil {
		c.container.Resources.Requests = corev1.ResourceList{}
	}
	if c.container.Resources.Limits == nil {
		c.container.Resources.Limits = corev1.ResourceList{}
	}
	qty := resource.MustParse(memory)
	c.container.Resources.Requests[corev1.ResourceMemory] = qty
	c.container.Resources.Limits[corev1.ResourceMemory] = qty
	return c
}

// Done returns to the ReplicaSpecWrapper.
func (c *ContainerWrapper) Done() *ReplicaSpecWrapper {
	return c.replicaSpec
}

// PodSpecWrapper wraps a pod spec for easier configuration.
type PodSpecWrapper struct {
	replicaSpec *ReplicaSpecWrapper
	podSpec     *corev1.PodSpec
}

// NodeSelector sets a node selector on the pod spec.
func (p *PodSpecWrapper) NodeSelector(key, value string) *PodSpecWrapper {
	if p.podSpec.NodeSelector == nil {
		p.podSpec.NodeSelector = make(map[string]string)
	}
	p.podSpec.NodeSelector[key] = value
	return p
}

// NodeAffinity sets node affinity on the pod spec.
func (p *PodSpecWrapper) NodeAffinity(key string, operator corev1.NodeSelectorOperator, values ...string) *PodSpecWrapper {
	if p.podSpec.Affinity == nil {
		p.podSpec.Affinity = &corev1.Affinity{}
	}
	if p.podSpec.Affinity.NodeAffinity == nil {
		p.podSpec.Affinity.NodeAffinity = &corev1.NodeAffinity{}
	}
	if p.podSpec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
		p.podSpec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution = &corev1.NodeSelector{}
	}

	term := corev1.NodeSelectorTerm{
		MatchExpressions: []corev1.NodeSelectorRequirement{
			{
				Key:      key,
				Operator: operator,
				Values:   values,
			},
		},
	}

	p.podSpec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms = append(
		p.podSpec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms,
		term,
	)
	return p
}

// Done returns to the ReplicaSpecWrapper.
func (p *PodSpecWrapper) Done() *ReplicaSpecWrapper {
	return p.replicaSpec
}

// WithAccelerateLaunchCommand is a helper to set up a typical accelerate launch command.
func (c *ContainerWrapper) WithAccelerateLaunchCommand(modelName, method string, batchSize, tokensPerSample int) *ContainerWrapper {
	peftFlag := ""
	if method == "lora" {
		peftFlag = "--peft_method lora"
	}

	command := "accelerate launch --num_processes=1 --num_machines=1 " +
		"-m tuning.sft_trainer " +
		"--model_name_or_path " + modelName + " " +
		"--per_device_train_batch_size " + string(rune(batchSize)) + " " +
		"--max_seq_length " + string(rune(tokensPerSample)) + " " +
		peftFlag

	return c.Command("sh", "-c").Args(command)
}
