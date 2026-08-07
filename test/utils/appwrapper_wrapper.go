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
	"encoding/json"

	kubeflowv1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	awv1beta2 "github.com/project-codeflare/appwrapper/api/v1beta2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// AppWrapperWrapper wraps an AppWrapper for building test objects
type AppWrapperWrapper struct {
	awv1beta2.AppWrapper
}

// MakeAppWrapper creates a new AppWrapperWrapper
func MakeAppWrapper(name, namespace string) *AppWrapperWrapper {
	return &AppWrapperWrapper{
		AppWrapper: awv1beta2.AppWrapper{
			ObjectMeta: metav1.ObjectMeta{
				Name:        name,
				Namespace:   namespace,
				Labels:      make(map[string]string),
				Annotations: make(map[string]string),
			},
			Spec: awv1beta2.AppWrapperSpec{
				Components: []awv1beta2.AppWrapperComponent{},
			},
		},
	}
}

// Label adds a label to the AppWrapper
func (w *AppWrapperWrapper) Label(key, value string) *AppWrapperWrapper {
	w.AppWrapper.Labels[key] = value
	return w
}

// Annotation adds an annotation to the AppWrapper
func (w *AppWrapperWrapper) Annotation(key, value string) *AppWrapperWrapper {
	w.AppWrapper.Annotations[key] = value
	return w
}

// PyTorchJobComponent adds a PyTorchJob component to the AppWrapper
func (w *AppWrapperWrapper) PyTorchJobComponent(job *kubeflowv1.PyTorchJob) *AppWrapperWrapper {
	// Convert PyTorchJob to unstructured
	jobBytes, err := json.Marshal(job)
	if err != nil {
		panic(err)
	}

	component := awv1beta2.AppWrapperComponent{
		Template: runtime.RawExtension{
			Raw: jobBytes,
		},
	}

	w.AppWrapper.Spec.Components = append(w.AppWrapper.Spec.Components, component)
	return w
}

// Obj returns the underlying AppWrapper object
func (w *AppWrapperWrapper) Obj() *awv1beta2.AppWrapper {
	return &w.AppWrapper
}

// Clone creates a deep copy of the wrapper
func (w *AppWrapperWrapper) Clone() *AppWrapperWrapper {
	return &AppWrapperWrapper{
		AppWrapper: *w.AppWrapper.DeepCopy(),
	}
}
