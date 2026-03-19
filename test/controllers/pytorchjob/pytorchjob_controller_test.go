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

package pytorchjob

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/ibm/resource-requirements-appwrapper/internal/controller"
	testutilpkg "github.com/ibm/resource-requirements-appwrapper/test/testutil"
	"github.com/ibm/resource-requirements-appwrapper/test/utils"
	kubeflowv1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

// recordedEvent represents a single event that was recorded
type recordedEvent struct {
	objectName string
	objectKind string
	eventType  string
	reason     string
	message    string
}

// testRecorder is a test event recorder that tracks all events
type testRecorder struct {
	mu     sync.Mutex
	events []recordedEvent
}

func newTestRecorder() *testRecorder {
	return &testRecorder{
		events: make([]recordedEvent, 0),
	}
}

func (r *testRecorder) Event(object runtime.Object, eventtype, reason, message string) {
	r.recordEvent(object, eventtype, reason, message)
}

func (r *testRecorder) Eventf(object runtime.Object, eventtype, reason, messageFmt string, args ...interface{}) {
	// For simplicity, we'll just store the format string without formatting
	r.recordEvent(object, eventtype, reason, messageFmt)
}

func (r *testRecorder) AnnotatedEventf(object runtime.Object, annotations map[string]string, eventtype, reason, messageFmt string, args ...interface{}) {
	r.recordEvent(object, eventtype, reason, messageFmt)
}

func (r *testRecorder) recordEvent(object runtime.Object, eventtype, reason, message string) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Extract object metadata
	var objectName, objectKind string
	if metaObj, ok := object.(metav1.Object); ok {
		objectName = metaObj.GetName()
	}
	if object != nil {
		objectKind = object.GetObjectKind().GroupVersionKind().Kind
	}

	r.events = append(r.events, recordedEvent{
		objectName: objectName,
		objectKind: objectKind,
		eventType:  eventtype,
		reason:     reason,
		message:    message,
	})
}

// HasEvent checks if an event matching the criteria was recorded
func (r *testRecorder) HasEvent(objectName, eventType, messageSubstring string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()

	for _, event := range r.events {
		if event.objectName == objectName &&
			event.eventType == eventType &&
			strings.Contains(event.message, messageSubstring) {
			return true
		}
	}
	return false
}

// GetEvents returns all recorded events (for debugging)
func (r *testRecorder) GetEvents() []recordedEvent {
	r.mu.Lock()
	defer r.mu.Unlock()

	eventsCopy := make([]recordedEvent, len(r.events))
	copy(eventsCopy, r.events)
	return eventsCopy
}

var _ record.EventRecorder = &testRecorder{}

func TestPyTorchJobReconciler_WithImmediateRecommender(t *testing.T) {
	const (
		watchLabelKey     = "autoconf-test"
		watchLabelValue   = "enabled"
		recommendationKey = "test.ibm.com/recommendation"
	)

	// Setup scheme
	s := runtime.NewScheme()
	_ = scheme.AddToScheme(s)
	_ = kubeflowv1.AddToScheme(s)

	baseJob := utils.MakePyTorchJob("test-job", "default").
		Label(watchLabelKey, watchLabelValue).
		MasterReplicaSpec(1).
		Container().
		Command("sh", "-c").
		Args("accelerate launch --num_processes=1 -m tuning.sft_trainer --model_name_or_path ibm-granite/granite-8b-code-base-4k --per_device_train_batch_size 8 --max_seq_length 8192 --peft_method lora").
		Env("AUTOCONF_GPU_MODEL", "NVIDIA-A100-SXM4-80GB").
		Done().
		Done()

	tests := map[string]struct {
		job                *kubeflowv1.PyTorchJob
		mockResult         *controller.RecommendationResult
		mockError          error
		wantRequeue        bool
		wantError          error // nil means no error expected
		wantDoneLabel      bool
		wantRecommendation bool
		wantGateRemoved    bool
	}{
		"successful recommendation": {
			job: baseJob.Clone().Obj(),
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
			wantGateRemoved:    false,
		},
		"cannot recommend": {
			job: baseJob.Clone().
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
			wantRecommendation: true, // Should have error JSON
			wantGateRemoved:    true,
		},
		"recommender error": {
			job:                baseJob.Clone().Obj(),
			mockResult:         nil,
			mockError:          reconcile.TerminalError(nil),
			wantRequeue:        false,
			wantError:          reconcile.TerminalError(nil),
			wantDoneLabel:      false,
			wantRecommendation: false,
			wantGateRemoved:    false,
		},
		"job without watch label": {
			job: utils.MakePyTorchJob("no-label-job", "default").
				MasterReplicaSpec(1).
				Container().
				Command("echo", "hello").
				Done().
				Done().
				Obj(),
			mockResult:         nil, // Should not be called
			mockError:          nil,
			wantRequeue:        false,
			wantError:          fmt.Errorf("cannot extract minimum resource requirements for PyTorch job no-label-job\nmissing required environment variables in primary PytorchReplicaSpec [AUTOCONF_MODEL_NAME AUTOCONF_TOKENS_PER_SAMPLE AUTOCONF_BATCH_SIZE]"),
			wantDoneLabel:      false,
			wantRecommendation: false,
			wantGateRemoved:    false,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			// Create fake client with the job
			fakeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(tc.job).
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
			reconciler := &controller.PyTorchJobReconciler{
				Client:   fakeClient,
				Scheme:   s,
				Recorder: newTestRecorder(),
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
					Name:      tc.job.Name,
					Namespace: tc.job.Namespace,
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

			// Get updated job
			updatedJob := &kubeflowv1.PyTorchJob{}
			err = fakeClient.Get(context.Background(), req.NamespacedName, updatedJob)
			if err != nil {
				t.Fatalf("failed to get updated job: %v", err)
			}

			// Check done label
			if tc.wantDoneLabel {
				if updatedJob.Labels[controller.DefaultAutoconfDoneLabelKey] != controller.DefaultAutoconfDoneLabelValue {
					t.Errorf("done label: got %v, want %v", updatedJob.Labels[controller.DefaultAutoconfDoneLabelKey], controller.DefaultAutoconfDoneLabelValue)
				}
			} else {
				if _, exists := updatedJob.Labels[controller.DefaultAutoconfDoneLabelKey]; exists {
					t.Error("done label should not exist")
				}
			}

			// Check recommendation annotation
			if tc.wantRecommendation {
				if _, exists := updatedJob.Annotations[recommendationKey]; !exists {
					t.Error("recommendation annotation should exist")
				}
			} else {
				if _, exists := updatedJob.Annotations[recommendationKey]; exists {
					t.Error("recommendation annotation should not exist")
				}
			}

			// Check gate removed
			if tc.wantGateRemoved {
				if _, exists := updatedJob.Annotations[controller.KueueAdmissionGatedByAnnotation]; exists {
					t.Error("admission gate should be removed")
				}
			}
		})
	}
}

func TestPyTorchJobReconciler_WithDeferredRecommender(t *testing.T) {
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

	baseJob := utils.MakePyTorchJob("test-deferred-job", "default").
		Label(watchLabelKey, watchLabelValue).
		MasterReplicaSpec(1).
		Container().
		Command("sh", "-c").
		Args("accelerate launch --num_processes=1 -m tuning.sft_trainer --model_name_or_path ibm-granite/granite-8b-code-base-4k --per_device_train_batch_size 8 --max_seq_length 8192").
		Env("AUTOCONF_GPU_MODEL", "NVIDIA-A100-SXM4-80GB").
		Done().
		Done()

	tests := map[string]struct {
		job                *kubeflowv1.PyTorchJob
		pendingChecks      int
		mockResult         *controller.RecommendationResult
		wantRequeue        bool
		wantRequestIDLabel bool
		wantDoneLabel      bool
	}{
		"first reconcile - initiate request": {
			job:                baseJob.Clone().Obj(),
			pendingChecks:      2,
			mockResult:         nil, // Will be pending
			wantRequeue:        true,
			wantRequestIDLabel: true,
			wantDoneLabel:      false,
		},
		"second reconcile - still pending": {
			job: baseJob.Clone().
				Label(requestIDLabel, "mock-request-1").
				Obj(),
			pendingChecks:      2,
			mockResult:         nil, // Still pending
			wantRequeue:        true,
			wantRequestIDLabel: true,
			wantDoneLabel:      false,
		},
		"third reconcile - result ready": {
			job: baseJob.Clone().
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
			job: baseJob.Clone().
				Label(requestIDLabel, "mock-request-1").
				Obj(),
			pendingChecks: 1, // Pending on 1st check (InitiateRecommendation), ready on 2nd check
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
			// Create fake client with the job
			fakeClient := fake.NewClientBuilder().
				WithScheme(s).
				WithObjects(tc.job).
				WithStatusSubresource(tc.job).
				Build()

			// Create mock recommender
			mockRecommender := testutilpkg.NewMockDeferredRecommender()
			mockRecommender.SetPendingChecks(tc.pendingChecks)
			if tc.mockResult != nil {
				mockRecommender.SetResult(tc.mockResult)
			}

			// Create reconciler
			reconciler := &controller.PyTorchJobReconciler{
				Client:   fakeClient,
				Scheme:   s,
				Recorder: newTestRecorder(),
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
					Name:      tc.job.Name,
					Namespace: tc.job.Namespace,
				},
			}

			result, err := reconciler.Reconcile(context.Background(), req)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			// Check requeue
			if result.Requeue != tc.wantRequeue {
				t.Errorf("requeue: got %v, want %v", result.Requeue, tc.wantRequeue)
			}

			// Get updated job
			updatedJob := &kubeflowv1.PyTorchJob{}
			err = fakeClient.Get(context.Background(), req.NamespacedName, updatedJob)
			if err != nil {
				t.Fatalf("failed to get updated job: %v", err)
			}

			// Check request ID label
			if tc.wantRequestIDLabel {
				if _, exists := updatedJob.Labels[requestIDLabel]; !exists {
					t.Error("request ID label should exist")
				}
			}

			// Check done label
			if tc.wantDoneLabel {
				if updatedJob.Labels[controller.DefaultAutoconfDoneLabelKey] != controller.DefaultAutoconfDoneLabelValue {
					t.Errorf("done label: got %v, want %v", updatedJob.Labels[controller.DefaultAutoconfDoneLabelKey], controller.DefaultAutoconfDoneLabelValue)
				}
			}
		})
	}
}

func TestPyTorchJobWrapper(t *testing.T) {
	job := utils.MakePyTorchJob("test", "ns").
		Label("key", "value").
		Annotation("anno", "val").
		MasterReplicaSpec(2).
		Container().
		Image("pytorch:latest").
		Command("python", "train.py").
		Env("VAR", "value").
		Done().
		Done().
		Obj()

	if job.Name != "test" {
		t.Errorf("name: got %v, want test", job.Name)
	}
	if job.Namespace != "ns" {
		t.Errorf("namespace: got %v, want ns", job.Namespace)
	}
	if job.Labels["key"] != "value" {
		t.Errorf("label: got %v, want value", job.Labels["key"])
	}
	if job.Annotations["anno"] != "val" {
		t.Errorf("annotation: got %v, want val", job.Annotations["anno"])
	}

	masterSpec := job.Spec.PyTorchReplicaSpecs["Master"]
	if masterSpec == nil {
		t.Fatal("master spec should exist")
	}
	if *masterSpec.Replicas != 2 {
		t.Errorf("replicas: got %v, want 2", *masterSpec.Replicas)
	}

	container := masterSpec.Template.Spec.Containers[0]
	if container.Image != "pytorch:latest" {
		t.Errorf("image: got %v, want pytorch:latest", container.Image)
	}
	if diff := cmp.Diff(container.Command, []string{"python", "train.py"}); diff != "" {
		t.Errorf("command mismatch (-want +got):\n%s", diff)
	}

	foundEnv := false
	for _, env := range container.Env {
		if env.Name == "VAR" && env.Value == "value" {
			foundEnv = true
			break
		}
	}
	if !foundEnv {
		t.Error("environment variable VAR=value not found")
	}
}
