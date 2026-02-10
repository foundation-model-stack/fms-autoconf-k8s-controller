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
	"strings"

	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/go-logr/logr"
	kubeflowv1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	awv1beta2 "github.com/project-codeflare/appwrapper/api/v1beta2"
)

// AppWrapperReconciler reconciles a AppWrapper object
type AppWrapperReconciler struct {
	client.Client
	PatchingInstructions
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
}

// Updates the AppWrapper if it wraps exactly 1 PytorchJob object
func (r *AppWrapperReconciler) UpdateAppWrapper(aw *awv1beta2.AppWrapper, log logr.Logger) (RecommenderRequest, error) {
	// VV: Only support up to 1 wrapped PyTorchJob objects.
	// This is for simplicity so that there is at most 1 RequestID associated with each AppWrapper object
	pytorchJobObjects := 0
	ptjIdx := -1
	job := kubeflowv1.PyTorchJob{}

	for idx := range aw.Spec.Components {
		obj := &unstructured.Unstructured{}

		if aw.Spec.Components[idx].Template.Object != nil {
			return RecommenderRequest{Pending: false}, fmt.Errorf("expected aw.Spec.Components[%d].Template.Object to be nil", idx)
		}

		if _, _, err := unstructured.UnstructuredJSONScheme.Decode(aw.Spec.Components[idx].Template.Raw, nil, obj); err != nil {
			log.Error(err, "Cannot decode component", "index", idx)
			return RecommenderRequest{Pending: false}, err
		}

		gvk := obj.GroupVersionKind()
		if gvk.Group != "kubeflow.org" || gvk.Kind != "PyTorchJob" {
			continue
		}

		pytorchJobObjects++
		ptjIdx = idx

		if pytorchJobObjects == 1 {
			// VV: We support AppWrappers that contain exactly 1 Ptj
			err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, &job)

			if err != nil {
				log.Error(err, "Cannot convert unstructured object to PyTorchJob", "index", idx)
				return RecommenderRequest{Pending: false}, err
			}
		}
	}

	if pytorchJobObjects != 1 {
		return RecommenderRequest{Pending: false}, fmt.Errorf("expected exactly 1 PyTorchJob object but found %d", pytorchJobObjects)
	}

	rr, raw, err := r.UpdatePyTorchJob(&job, aw.Labels[r.WaitingForAdoRequestIDLabel], log)

	if err != nil {
		log.Error(err, "Cannot update PyTorchJob component", "index", ptjIdx)
		return rr, err
	}

	if rr.AppliedRecommendation {
		aw.Spec.Components[ptjIdx].Template = runtime.RawExtension{
			Raw: raw,
		}
	}

	return rr, nil
}

// +kubebuilder:rbac:groups=workload.codeflare.dev,resources=appwrappers,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=workload.codeflare.dev,resources=appwrappers/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=workload.codeflare.dev,resources=appwrappers/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the AppWrapper object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.22.1/pkg/reconcile
func (r *AppWrapperReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	job := &awv1beta2.AppWrapper{}
	if err := r.Get(ctx, req.NamespacedName, job); err != nil {
		log.V(4).Info("unable to fetch AppWrapper", "err", err)
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if job.Labels[r.DoneLabelKey] == r.DoneLabelValue {
		log.Info("AppWrapper is already updated")
		return ctrl.Result{}, nil
	}

	log.Info("Reconciling AppWrapper")

	original := job.DeepCopy()

	rr, err := r.UpdateAppWrapper(job, log)

	if err != nil {
		return ctrl.Result{}, err
	}

	if rr.RecommendationJSON != "" {
		// VV: We got an answer from the recommender (either recommendation or error)
		// First, mark the original job as "done" so that we don't attempt to mutate it again

		job.Labels[r.DoneLabelKey] = r.DoneLabelValue

		if job.Annotations == nil {
			job.Annotations = make(map[string]string)
		}
		job.Annotations[r.RecommendationAnnotationKey] = rr.RecommendationJSON

		delete(original.Labels, r.WatchLabelKey)

		if err := r.Update(ctx, original); err != nil {
			return handleUpdateWrapperError(err, log)
		}

		if rr.AppliedRecommendation {
			// VV: appwrapper.spec.components[idx].template.raw are immutable so we have to create a new object.
			// We also need to patch the old one to remove the label that would trigger our plugin to handle it

			derived := job.DeepCopy()
			derived.Name = ""
			derived.GenerateName = strings.TrimSuffix(job.Name, "-") + "-"
			derived.ResourceVersion = ""

			derived.OwnerReferences = []v1.OwnerReference{
				{
					APIVersion: job.APIVersion,
					Kind:       job.Kind,
					Name:       job.Name,
					UID:        job.UID,
				},
			}

			if err := r.Create(ctx, derived); err != nil {
				log.Error(err, "unable to create a new AppWrapper")
				return ctrl.Result{}, err
			}

			log.Info("Created new AppWrapper with recommendations", "name", derived.Name)

			r.Recorder.Event(
				job,
				corev1.EventTypeNormal,
				"PatchedWithRecommendations",
				"Created new AppWrapper with recommended resource requirements",
			)
		} else {
			// No recommendation - mark original as done without creating new one
			log.Info("AppWrapper marked as processed - no recommendation available")

			r.Recorder.Event(
				job,
				corev1.EventTypeWarning,
				"NoRecommendationAvailable",
				"Recommendation engine could not generate recommendations; AppWrapper proceeding without modifications",
			)
		}

		return ctrl.Result{}, nil
	} else if rr.Pending {
		if job.Labels[r.WaitingForAdoRequestIDLabel] != rr.RequestID {
			job.Labels[r.WaitingForAdoRequestIDLabel] = rr.RequestID

			if err := r.Patch(ctx, job, client.MergeFrom(original)); err != nil {
				return handleUpdateWrapperError(err, log)
			}

			log.Info("Submitted ADO request for recommendations", "RequestID", rr.RequestID)

			r.Recorder.Eventf(
				job,
				corev1.EventTypeNormal,
				"ADORequestSubmitted",
				"Submitted ADO request %s for recommendations", rr.RequestID,
			)

			return ctrl.Result{Requeue: true}, nil
		} else {
			log.Info("The ADO request for resource requirements of PyTorch job is pending", "RequestID", rr.RequestID)

			r.Recorder.Eventf(
				job,
				corev1.EventTypeNormal,
				"ADORequestPending",
				"ADO request %s still pending; will requeue", rr.RequestID,
			)

			return ctrl.Result{Requeue: true}, nil
		}
	}

	return ctrl.Result{Requeue: true}, fmt.Errorf("unexpected flow RecommenderRequest: %#v", rr)
}

// SetupWithManager sets up the controller with the Manager.
func (r *AppWrapperReconciler) SetupWithManager(mgr ctrl.Manager) error {
	r.Recorder = mgr.GetEventRecorderFor("appwrapper")

	return ctrl.NewControllerManagedBy(mgr).
		Named("appwrapper").
		For(&awv1beta2.AppWrapper{}).
		Complete(r)
}
