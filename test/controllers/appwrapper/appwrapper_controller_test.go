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

package appwrapper

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/foundation-model-stack/fms-autoconf-k8s-controller/internal/controller"
	testutilpkg "github.com/foundation-model-stack/fms-autoconf-k8s-controller/test/testutil"
	"github.com/foundation-model-stack/fms-autoconf-k8s-controller/test/utils"
	kubeflowv1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	awv1beta2 "github.com/project-codeflare/appwrapper/api/v1beta2"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func TestAppWrapperReconciler_WithImmediateRecommender(t *testing.T) {
	const (
		watchLabelKey     = "autoconf-test"
		watchLabelValue   = "enabled"
		recommendationKey = "test.ibm.com/recommendation"
	)

	// Setup scheme
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = kubeflowv1.AddToScheme(s)
	_ = awv1beta2.AddToScheme(s)

	// Create a base PyTorchJob to wrap
	basePyTorchJob := utils.MakePyTorchJob("test-job", "default").
		MasterReplicaSpec(1).
		Container().
		Command("sh", "-c").
		Args("accelerate launch --num_processes=1 -m tuning.sft_trainer --model_name_or_path ibm-granite/granite-8b-code-base-4k --per_device_train_batch_size 8 --max_seq_length 8192 --peft_method lora").
		Env("AUTOCONF_GPU_MODEL", "NVIDIA-A100-SXM4-80GB").
		Done().
		Done().
		Obj()

	baseAppWrapper := utils.MakeAppWrapper("test-aw", "default").
		Label(watchLabelKey, watchLabelValue).
		PyTorchJobComponent(basePyTorchJob)

	tests := map[string]struct {
		aw                 *awv1beta2.AppWrapper
		mockResult         *controller.RecommendationResult
		mockError          error
		wantRequeue        bool
		wantError          error
		wantDoneLabel      bool
		wantRecommendation bool
		wantDerivedCreated bool
	}{
		"successful recommendation": {
			aw: baseAppWrapper.Clone().Obj(),
			mockResult: &controller.RecommendationResult{
				Requirements: &controller.ResourceRequirements{
					Workers:      2,
					GPUs:         1,
					CanRecommend: true,
				},
				CanRecommend: true,
			},
			mockError:          nil,
			wantRequeue:        false,
			wantError:          nil,
			wantDoneLabel:      true,
			wantRecommendation: true,
			wantDerivedCreated: true,
		},
		"cannot recommend": {
			aw: baseAppWrapper.Clone().
				Annotation(controller.KueueAdmissionGatedByAnnotation, "test-gate").
				Obj(),
			mockResult: &controller.RecommendationResult{
				Requirements: nil,
				CanRecommend: false,
			},
			mockError:          nil,
			wantRequeue:        false,
			wantError:          nil,
			wantDoneLabel:      true,
			wantRecommendation: true,
			wantDerivedCreated: false,
		},
		"recommender error": {
			aw:                 baseAppWrapper.Clone().Obj(),
			mockResult:         nil,
			mockError:          reconcile.TerminalError(nil),
			wantRequeue:        false,
			wantError:          reconcile.TerminalError(nil),
			wantDoneLabel:      false,
			wantRecommendation: false,
			wantDerivedCreated: false,
		},
		"appwrapper without pytorchjob": {
			aw: utils.MakeAppWrapper("no-ptj", "default").
				Label(watchLabelKey, watchLabelValue).
				Obj(),
			mockResult:         nil,
			mockError:          nil,
			wantRequeue:        false,
			wantError:          fmt.Errorf("expected exactly 1 PyTorchJob object but found 0"),
			wantDoneLabel:      false,
			wantRecommendation: false,
			wantDerivedCreated: false,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			// Create fake client with the AppWrapper
			fakeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(tc.aw).
				Build()

			// Create mock recommender
			mockRecommender := testutilpkg.NewMockImmediateRecommender()
			if tc.mockResult != nil {
				mockRecommender.SetResult(tc.mockResult)
			}
			if tc.mockError != nil {
				mockRecommender.SetError(tc.mockError)
			}

			// Create reconciler
			reconciler := &controller.AppWrapperReconciler{
				Client:   fakeClient,
				Scheme:   s,
				Recorder: record.NewFakeRecorder(100),
				PatchingInstructions: controller.PatchingInstructions{
					DoneLabelKey:                controller.DefaultAutoconfDoneLabelKey,
					DoneLabelValue:              controller.DefaultAutoconfDoneLabelValue,
					WatchLabelKey:               watchLabelKey,
					UnsuspendDerivedJobs:        false,
					WaitingForAdoRequestIDLabel: "waiting-for-request",
					PatchCPURequest:             true,
					DefaultGPUModel:             "NVIDIA-A100-SXM4-80GB",
					AutoconfModelVersion:        "3.1.0",
					RecommendationAnnotationKey: recommendationKey,
					ImmediateRecommender:        mockRecommender,
				},
			}

			// Reconcile
			req := reconcile.Request{
				NamespacedName: types.NamespacedName{
					Name:      tc.aw.Name,
					Namespace: tc.aw.Namespace,
				},
			}

			result, err := reconciler.Reconcile(context.Background(), req)

			// Check error - compare error strings since error objects have different addresses
			var wantErrStr, gotErrStr string
			if tc.wantError != nil {
				wantErrStr = tc.wantError.Error()
			}
			if err != nil {
				gotErrStr = err.Error()
			}
			if wantErrStr != gotErrStr {
				t.Errorf("error mismatch:\nwant: %q\ngot:  %q", wantErrStr, gotErrStr)
			}

			// Check requeue
			if result.Requeue != tc.wantRequeue {
				t.Errorf("requeue: got %v, want %v", result.Requeue, tc.wantRequeue)
			}

			// Get updated AppWrapper
			updatedAW := &awv1beta2.AppWrapper{}
			err = fakeClient.Get(context.Background(), req.NamespacedName, updatedAW)
			if err != nil {
				t.Fatalf("failed to get updated AppWrapper: %v", err)
			}

			// Check done label
			if tc.wantDoneLabel {
				if updatedAW.Labels[controller.DefaultAutoconfDoneLabelKey] != controller.DefaultAutoconfDoneLabelValue {
					t.Errorf("done label: got %v, want %v", updatedAW.Labels[controller.DefaultAutoconfDoneLabelKey], controller.DefaultAutoconfDoneLabelValue)
				}
			} else {
				if _, exists := updatedAW.Labels[controller.DefaultAutoconfDoneLabelKey]; exists {
					t.Error("done label should not exist")
				}
			}

			// Check recommendation annotation
			if tc.wantRecommendation {
				if _, exists := updatedAW.Annotations[recommendationKey]; !exists {
					t.Error("recommendation annotation should exist")
				}
			} else {
				if _, exists := updatedAW.Annotations[recommendationKey]; exists {
					t.Error("recommendation annotation should not exist")
				}
			}

			// Check if derived AppWrapper was created
			if tc.wantDerivedCreated {
				awList := &awv1beta2.AppWrapperList{}
				err = fakeClient.List(context.Background(), awList)
				if err != nil {
					t.Fatalf("failed to list AppWrappers: %v", err)
				}

				// Should have 2 AppWrappers: original + derived
				if len(awList.Items) != 2 {
					t.Errorf("expected 2 AppWrappers (original + derived), got %d", len(awList.Items))
				}

				// Find the derived AppWrapper
				var derived *awv1beta2.AppWrapper
				for i := range awList.Items {
					if awList.Items[i].Name != tc.aw.Name {
						derived = &awList.Items[i]
						break
					}
				}

				if derived == nil {
					t.Fatal("derived AppWrapper not found")
				}

				// Check that derived has owner reference
				if len(derived.OwnerReferences) != 1 {
					t.Errorf("derived AppWrapper should have 1 owner reference, got %d", len(derived.OwnerReferences))
				}

				// Check that derived has done label
				if derived.Labels[controller.DefaultAutoconfDoneLabelKey] != controller.DefaultAutoconfDoneLabelValue {
					t.Errorf("derived done label: got %v, want %v", derived.Labels[controller.DefaultAutoconfDoneLabelKey], controller.DefaultAutoconfDoneLabelValue)
				}

				// Check that derived has recommendation annotation
				if _, exists := derived.Annotations[recommendationKey]; !exists {
					t.Error("derived recommendation annotation should exist")
				}

				// Check that derived does not have admission gate
				if _, exists := derived.Annotations[controller.KueueAdmissionGatedByAnnotation]; exists {
					t.Error("derived should not have admission gate annotation")
				}

				// Verify the PyTorchJob component was updated with recommendations
				if len(derived.Spec.Components) != 1 {
					t.Fatalf("derived should have 1 component, got %d", len(derived.Spec.Components))
				}

				// Decode the PyTorchJob from the component
				var derivedJob kubeflowv1.PyTorchJob
				err = json.Unmarshal(derived.Spec.Components[0].Template.Raw, &derivedJob)
				if err != nil {
					t.Fatalf("failed to unmarshal derived PyTorchJob: %v", err)
				}

				// Check that GPU resources were set
				container := derivedJob.Spec.PyTorchReplicaSpecs[controller.PrimaryPyTorchReplica].Template.Spec.Containers[0]
				gpuLimit := container.Resources.Limits[controller.GPUResourceRequirement]
				if gpuLimit.Value() != int64(tc.mockResult.Requirements.GPUs) {
					t.Errorf("GPU limit: got %d, want %d", gpuLimit.Value(), tc.mockResult.Requirements.GPUs)
				}
			}
		})
	}
}

func TestAppWrapperReconciler_WithDeferredRecommender(t *testing.T) {
	const (
		watchLabelKey     = "autoconf-test"
		watchLabelValue   = "enabled"
		requestIDLabel    = "waiting-for-request"
		recommendationKey = "test.ibm.com/recommendation"
	)

	// Setup scheme
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = kubeflowv1.AddToScheme(s)
	_ = awv1beta2.AddToScheme(s)

	// Create a base PyTorchJob to wrap
	basePyTorchJob := utils.MakePyTorchJob("test-job", "default").
		MasterReplicaSpec(1).
		Container().
		Command("sh", "-c").
		Args("accelerate launch --num_processes=1 -m tuning.sft_trainer --model_name_or_path ibm-granite/granite-8b-code-base-4k --per_device_train_batch_size 8 --max_seq_length 8192 --peft_method lora").
		Env("AUTOCONF_GPU_MODEL", "NVIDIA-A100-SXM4-80GB").
		Done().
		Done().
		Obj()

	baseAppWrapper := utils.MakeAppWrapper("test-aw", "default").
		Label(watchLabelKey, watchLabelValue).
		PyTorchJobComponent(basePyTorchJob)

	tests := map[string]struct {
		aw                 *awv1beta2.AppWrapper
		pendingChecks      int
		mockResult         *controller.RecommendationResult
		wantRequeue        bool
		wantRequestIDLabel bool
		wantDoneLabel      bool
	}{
		"first reconcile - initiate request": {
			aw:            baseAppWrapper.Clone().Obj(),
			pendingChecks: 2,
			mockResult:    nil, // Will be pending
			wantRequeue:        true,
			wantRequestIDLabel: true,
			wantDoneLabel:      false,
		},
		"second reconcile - still pending": {
			aw: baseAppWrapper.Clone().
				Label(requestIDLabel, "mock-request-1").
				Obj(),
			pendingChecks:      2,
			mockResult:         nil, // Still pending
			wantRequeue:        true,
			wantRequestIDLabel: true,
			wantDoneLabel:      false,
		},
		"third reconcile - result ready": {
			aw: baseAppWrapper.Clone().
				Label(requestIDLabel, "mock-request-1").
				Obj(),
			pendingChecks: 0, // Result is ready immediately (simulating 3rd check after 2 pending)
			mockResult: &controller.RecommendationResult{
				Requirements: &controller.ResourceRequirements{
					Workers:      2,
					GPUs:         1,
					CanRecommend: true,
				},
				CanRecommend: true,
			},
			wantRequeue:        false,
			wantRequestIDLabel: true,
			wantDoneLabel:      true,
		},
		"second reconcile - result ready on 2nd check": {
			aw: baseAppWrapper.Clone().
				Label(requestIDLabel, "mock-request-1").
				Obj(),
			pendingChecks: 1, // 1 pending check, then ready
			mockResult: &controller.RecommendationResult{
				Requirements: &controller.ResourceRequirements{
					Workers:      3,
					GPUs:         2,
					CanRecommend: true,
				},
				CanRecommend: true,
			},
			wantRequeue:        false,
			wantRequestIDLabel: true,
			wantDoneLabel:      true,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			// Create fake client with the AppWrapper
			fakeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(tc.aw).
				Build()

			// Create mock recommender
			mockRecommender := testutilpkg.NewMockDeferredRecommender()
			mockRecommender.SetPendingChecks(tc.pendingChecks)
			if tc.mockResult != nil {
				mockRecommender.SetResult(tc.mockResult)
			}

			// Create reconciler
			reconciler := &controller.AppWrapperReconciler{
				Client:   fakeClient,
				Scheme:   s,
				Recorder: record.NewFakeRecorder(100),
				PatchingInstructions: controller.PatchingInstructions{
					DoneLabelKey:                controller.DefaultAutoconfDoneLabelKey,
					DoneLabelValue:              controller.DefaultAutoconfDoneLabelValue,
					WatchLabelKey:               watchLabelKey,
					UnsuspendDerivedJobs:        false,
					WaitingForAdoRequestIDLabel: requestIDLabel,
					PatchCPURequest:             true,
					DefaultGPUModel:             "NVIDIA-A100-SXM4-80GB",
					AutoconfModelVersion:        "3.1.0",
					RecommendationAnnotationKey: recommendationKey,
					DeferredRecommender:         mockRecommender,
				},
			}

			// Reconcile
			req := reconcile.Request{
				NamespacedName: types.NamespacedName{
					Name:      tc.aw.Name,
					Namespace: tc.aw.Namespace,
				},
			}

			result, err := reconciler.Reconcile(context.Background(), req)

			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			// Check requeue
			if result.Requeue != tc.wantRequeue {
				t.Errorf("requeue: got %v, want %v", result.Requeue, tc.wantRequeue)
			}

			// Get updated AppWrapper
			updatedAW := &awv1beta2.AppWrapper{}
			err = fakeClient.Get(context.Background(), req.NamespacedName, updatedAW)
			if err != nil {
				t.Fatalf("failed to get updated AppWrapper: %v", err)
			}

			// Check request ID label
			if tc.wantRequestIDLabel {
				if _, exists := updatedAW.Labels[requestIDLabel]; !exists {
					t.Error("request ID label should exist")
				}
			}

			// Check done label
			if tc.wantDoneLabel {
				if updatedAW.Labels[controller.DefaultAutoconfDoneLabelKey] != controller.DefaultAutoconfDoneLabelValue {
					t.Errorf("done label: got %v, want %v", updatedAW.Labels[controller.DefaultAutoconfDoneLabelKey], controller.DefaultAutoconfDoneLabelValue)
				}
			}
		})
	}
}

func TestAppWrapperWrapper(t *testing.T) {
	job := utils.MakePyTorchJob("test-job", "default").
		MasterReplicaSpec(1).
		Container().
		Command("echo", "hello").
		Done().
		Done().
		Obj()

	aw := utils.MakeAppWrapper("test-aw", "default").
		Label("key1", "value1").
		Annotation("anno1", "value1").
		PyTorchJobComponent(job).
		Obj()

	if aw.Name != "test-aw" {
		t.Errorf("name: got %s, want test-aw", aw.Name)
	}

	if aw.Namespace != "default" {
		t.Errorf("namespace: got %s, want default", aw.Namespace)
	}

	if aw.Labels["key1"] != "value1" {
		t.Errorf("label: got %s, want value1", aw.Labels["key1"])
	}

	if aw.Annotations["anno1"] != "value1" {
		t.Errorf("annotation: got %s, want value1", aw.Annotations["anno1"])
	}

	if len(aw.Spec.Components) != 1 {
		t.Errorf("components: got %d, want 1", len(aw.Spec.Components))
	}

	// Verify the component contains a PyTorchJob
	var decodedJob kubeflowv1.PyTorchJob
	err := json.Unmarshal(aw.Spec.Components[0].Template.Raw, &decodedJob)
	if err != nil {
		t.Fatalf("failed to unmarshal PyTorchJob: %v", err)
	}

	if decodedJob.Name != "test-job" {
		t.Errorf("job name: got %s, want test-job", decodedJob.Name)
	}
}
