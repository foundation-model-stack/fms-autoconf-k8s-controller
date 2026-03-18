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

const (
	// KueueAdmissionGatedByAnnotation is the annotation key used by Kueue to gate admission of a Job.
	// While this annotation is present, Kueue will temporarily consider the Job as inadmissible.
	// When the annotation is removed or contains an empty value, Kueue will resume its normal
	// admission check phases and the Job will be considered for admission.
	// This feature is available in Kueue v0.17
	KueueAdmissionGatedByAnnotation = "kueue.x-k8s.io/admission-gated-by"
)
