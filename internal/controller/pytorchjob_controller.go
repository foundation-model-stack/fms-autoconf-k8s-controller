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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	v1api "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/tools/record"
	"k8s.io/utils/ptr"
	"mvdan.cc/sh/v3/syntax"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/go-logr/logr"
	kubeflowv1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
)

// VV: this is an assumption in AppWrapper
const (
	PrimaryPyTorchReplica kubeflowv1.ReplicaType = "Master"
	WorkerPyTorchReplica  kubeflowv1.ReplicaType = "Worker"
)

const GPUResourceRequirement corev1.ResourceName = "nvidia.com/gpu"

type RecommenderRequest struct {
	Pending               bool
	AppliedRecommendation bool
	RequestID             string
	RecommendationJSON    string
}

type PatchingInstructions struct {
	DoneLabelKey                string
	DoneLabelValue              string
	WatchLabelKey               string
	UnsuspendPatchedJobs        bool
	PathWrapperScript           string
	UrlAdo                      string
	WaitingForAdoRequestIDLabel string
	PatchCPURequest             bool
	DefaultGPUModel             string
	AutoconfModelVersion        string
	RecommendationAnnotationKey string
}

func (p *PatchingInstructions) Copy() PatchingInstructions {
	return PatchingInstructions{
		DoneLabelKey:                p.DoneLabelKey,
		DoneLabelValue:              p.DoneLabelValue,
		WatchLabelKey:               p.WatchLabelKey,
		UnsuspendPatchedJobs:        p.UnsuspendPatchedJobs,
		PathWrapperScript:           p.PathWrapperScript,
		UrlAdo:                      p.UrlAdo,
		WaitingForAdoRequestIDLabel: p.WaitingForAdoRequestIDLabel,
		PatchCPURequest:             p.PatchCPURequest,
		DefaultGPUModel:             p.DefaultGPUModel,
		AutoconfModelVersion:        p.AutoconfModelVersion,
		RecommendationAnnotationKey: p.RecommendationAnnotationKey,
	}
}

// PyTorchJobReconciler reconciles a PyTorchJob object
type PyTorchJobReconciler struct {
	client.Client
	PatchingInstructions
	Scheme   *runtime.Scheme
	Recorder record.EventRecorder
}

// +kubebuilder:rbac:groups=kubeflow.codeflare.dev,resources=pytorchjobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=kubeflow.codeflare.dev,resources=pytorchjobs/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=kubeflow.codeflare.dev,resources=pytorchjobs/finalizers,verbs=update

func (r *PatchingInstructions) PatchContainerCommandLine(container *v1.Container) error {
	shellType, allOtherargs, cmdline := GetCommandLine(append(container.Command, container.Args...))

	numAccelerateLaunchCommands := 0

	if cmdline != "" {
		reader := strings.NewReader(cmdline)
		parser := syntax.NewParser(syntax.Variant(syntax.LangBash))

		graph, err := parser.Parse(reader, "commandline")

		if err != nil {
			return err
		}

		// VV: Recursively walk the graph and patch the cmdline of `accelerate launch` "CallExpr" nodes
		syntax.Walk(graph, func(n syntax.Node) bool {
			// VV: Return `false` when we don't want to go any deeper into the tree
			// This is for cases in which we know that we are either not going to find any more accelerate launch commands
			// Or we've already encountered a problem
			call, ok := n.(*syntax.CallExpr)

			if !ok {
				return true
			}

			if !isAccelerateLaunch(call) {
				return false
			}

			argvTokens, err := WordsToQuotedArgs(call.Args)
			if err != nil {
				return false
			}

			newTokens, err := PatchAccelerateLaunchCommandline(argvTokens)
			if err != nil {
				return false
			}

			call.Args, err = StringsToWords(parser, newTokens)
			if err != nil {
				return false
			}

			numAccelerateLaunchCommands++
			return true
		})

		var out bytes.Buffer
		pr := syntax.NewPrinter(
			syntax.SpaceRedirects(true),
			syntax.Minify(true),
		)
		err = pr.Print(&out, graph)

		if err != nil {
			return err
		}

		cmdline = out.String()
	} else {
		if len(allOtherargs) > 2 && allOtherargs[0] == "accelerate" && allOtherargs[1] == "launch" {
			numAccelerateLaunchCommands++

			var err error
			allOtherargs, err = PatchAccelerateLaunchCommandline(allOtherargs)
			if err != nil {
				return err
			}
		}
	}

	if numAccelerateLaunchCommands != 1 {
		return fmt.Errorf("expected exactly one accelerate launch command, found %d", numAccelerateLaunchCommands)
	}

	if shellType != "" {
		container.Command = []string{shellType}
		container.Args = append(allOtherargs, "-c", cmdline)
	} else {
		container.Command = allOtherargs
		container.Args = []string{}
	}

	return nil
}

func (r *PatchingInstructions) PatchPytorchReplicaSpec(replica *kubeflowv1.ReplicaSpec, minGPURecommenderInput MinGPURecommenderInput, minResources ResourceRequirements) error {
	replica.Replicas = ptr.To(int32(minResources.Replicas))

	envVarNames := []string{
		"AUTOCONF_NUM_GPUS", "AUTOCONF_TOTAL_GPUS", "AUTOCONF_NUM_PROCESSES",
		"AUTOCONF_NUM_MACHINES", "AUTOCONF_PER_DEVICE_BATCH_SIZE",
		"AUTOCONF_TOKENS_PER_SAMPLE", "AUTOCONF_MODEL_NAME", "AUTOCONF_METHOD",
		"AUTOCONF_GPU_MODEL", "AUTOCONF_BATCH_SIZE",
	}
	envVars := []corev1.EnvVar{}

	container := &replica.Template.Spec.Containers[0]

	for _, v := range container.Env {
		if !slices.Contains(envVarNames, v.Name) {
			envVars = append(envVars, v)
		}
	}

	totalGPUs := minResources.Workers * minResources.GPUs
	totalProcesses := max(1, totalGPUs)

	envVars = append(envVars, corev1.EnvVar{Name: "AUTOCONF_NUM_GPUS", Value: fmt.Sprintf("%d", minResources.GPUs)})
	envVars = append(envVars, corev1.EnvVar{Name: "AUTOCONF_TOTAL_GPUS", Value: fmt.Sprintf("%d", totalGPUs)})
	envVars = append(envVars, corev1.EnvVar{Name: "AUTOCONF_NUM_PROCESSES", Value: fmt.Sprintf("%d", totalProcesses)})
	envVars = append(envVars, corev1.EnvVar{Name: "AUTOCONF_NUM_MACHINES", Value: fmt.Sprintf("%d", minResources.Workers)})
	envVars = append(envVars, corev1.EnvVar{Name: "AUTOCONF_PER_DEVICE_BATCH_SIZE", Value: fmt.Sprintf("%d", minGPURecommenderInput.BatchSize/totalProcesses)})
	envVars = append(envVars, corev1.EnvVar{Name: "AUTOCONF_TOKENS_PER_SAMPLE", Value: fmt.Sprintf("%d", minGPURecommenderInput.TokensPerSample)})
	envVars = append(envVars, corev1.EnvVar{Name: "AUTOCONF_MODEL_NAME", Value: minGPURecommenderInput.ModelName})
	envVars = append(envVars, corev1.EnvVar{Name: "AUTOCONF_METHOD", Value: minGPURecommenderInput.Method})
	envVars = append(envVars, corev1.EnvVar{Name: "AUTOCONF_GPU_MODEL", Value: minGPURecommenderInput.GPUModel})
	envVars = append(envVars, corev1.EnvVar{Name: "AUTOCONF_BATCH_SIZE", Value: fmt.Sprintf("%d", minGPURecommenderInput.BatchSize)})

	container.Env = envVars

	numGpus := *resource.NewQuantity(int64(minResources.GPUs), resource.DecimalSI)
	if container.Resources.Limits == nil {
		container.Resources.Limits = corev1.ResourceList{}
	}
	if container.Resources.Requests == nil {
		container.Resources.Requests = corev1.ResourceList{}
	}
	container.Resources.Limits[GPUResourceRequirement] = numGpus
	container.Resources.Requests[GPUResourceRequirement] = numGpus

	if r.PatchCPURequest {
		numCPUs := *resource.NewQuantity(int64(max(minResources.GPUs, 1))*2, resource.DecimalSI)
		container.Resources.Requests[corev1.ResourceCPU] = numCPUs
		container.Resources.Limits[corev1.ResourceCPU] = numCPUs
	}

	return r.PatchContainerCommandLine(container)
}

func (r *PatchingInstructions) UpdatePyTorchJob(job *kubeflowv1.PyTorchJob, requestID string, log logr.Logger) (RecommenderRequest, []byte, error) {
	if len(job.Spec.PyTorchReplicaSpecs) > 2 {
		return RecommenderRequest{RequestID: requestID, Pending: false}, nil, fmt.Errorf("cannot have more than 2 PyTorchReplicaSpecs but it has %d", len(job.Spec.PyTorchReplicaSpecs))
	}

	_, masterReplicaExists := job.Spec.PyTorchReplicaSpecs[PrimaryPyTorchReplica]

	if !masterReplicaExists {
		return RecommenderRequest{RequestID: requestID, Pending: false}, nil, fmt.Errorf("missing the Master PyTorchReplicaSpec")
	}

	replicaErrors := []error{}

	for nameReplica, replica := range job.Spec.PyTorchReplicaSpecs {
		if len(replica.Template.Spec.Containers) != 1 {
			replicaErrors = append(
				replicaErrors,
				fmt.Errorf("the PyTorchReplicaSpec %s does not have exactly 1 container, instead it has %d",
					nameReplica, len(replica.Template.Spec.Containers),
				),
			)
		}
	}

	if len(replicaErrors) > 0 {
		return RecommenderRequest{RequestID: requestID, Pending: false}, nil, errors.Join(replicaErrors...)
	}
	minGPURecommenderInput, err := ExtractMinGPURecommenderInput(job, r.DefaultGPUModel, r.AutoconfModelVersion, log)

	if err != nil {
		return RecommenderRequest{RequestID: requestID, Pending: false}, nil, errors.Join(fmt.Errorf("cannot extract minimum resource requirements for PyTorch job %s", job.Name), err)
	} else {
		log.V(1).Info("Requesting recommendation from recommender", "features", minGPURecommenderInput)
	}

	var recs *ResourceRequirements = nil

	if r.UrlAdo != "" {
		// VV: When using an ADO REST API to compute the resource requirements we update a PyTorchJob object in 2 steps:
		// 1. trigger the min_gpu_recommender custom_experiment to compute the resource requirements and receive a requestID
		// 2. poll the status of the requestID and when it's done read it to get the recommended resource requirements
		if requestID == "" {
			// VV: haven't triggered the custom_experiment yet
			requestID, err = SendRequestToCalcMinimumResourceRequirements(minGPURecommenderInput, r.UrlAdo, log)
			return RecommenderRequest{RequestID: requestID, Pending: true}, nil, err
		} else {
			// VV: check whether the custom_experiment has finished producing the recommended resource requirements
			recs, err = CheckRequestToCalcMinimumResourceRequirements(minGPURecommenderInput, r.UrlAdo, requestID, log)

			if err != nil {
				return RecommenderRequest{RequestID: requestID, Pending: true}, nil, errors.Join(fmt.Errorf("failed to check request %s for PyTorch job %s", requestID, job.Name), err)
			}
		}
	} else if r.PathWrapperScript != "" {
		rr := ResourceRequirements{}
		rr, err = RunPythonWrapperToCalcMinimumResourceRequirements(minGPURecommenderInput, r.PathWrapperScript, log)

		if err == nil {
			recs = &rr
		}

		if err != nil {
			return RecommenderRequest{RequestID: requestID, Pending: false}, nil, errors.Join(fmt.Errorf("cannot compute resource requirements for PyTorch job %s", job.Name), err)
		}
	}

	if recs == nil {
		// VV: The recommender is still working on it, try again later
		return RecommenderRequest{RequestID: requestID, Pending: true}, nil, nil
	}

	// Check if recommendation was possible
	if !recs.CanRecommend {
		// Cannot recommend - create error JSON, do NOT patch the job
		errorJSON, _ := json.Marshal(map[string]interface{}{
			"error": "No recommendation",
		})
		return RecommenderRequest{
			RequestID:             requestID,
			Pending:               false,
			AppliedRecommendation: false,
			RecommendationJSON:    string(errorJSON),
		}, nil, nil
	}

	log.Info("computed resource requirements", "recs", recs)

	// VV: the "Master" PyTorchReplica always has 1 Replica, the remaining ones go to the recsWorkers
	recsMain := ResourceRequirements{
		Workers:  recs.Workers,
		GPUs:     recs.GPUs,
		Replicas: 1,
	}

	recsWorkers := ResourceRequirements{
		Workers:  recs.Workers,
		GPUs:     recs.GPUs,
		Replicas: recs.Workers - 1,
	}

	if recsWorkers.Replicas < 1 {
		// VV: Need to drop the entire PyTorchReplicaSpec otherwise AppWrapper will reject the Job
		for key := range job.Spec.PyTorchReplicaSpecs {
			if key != PrimaryPyTorchReplica {
				delete(job.Spec.PyTorchReplicaSpecs, key)
			}
		}
	} else if len(job.Spec.PyTorchReplicaSpecs) == 1 {
		secondary := job.Spec.PyTorchReplicaSpecs[PrimaryPyTorchReplica].DeepCopy()
		job.Spec.PyTorchReplicaSpecs[WorkerPyTorchReplica] = secondary
	}

	// VV: This configures the PyTorchJob controller in KFTO to set WORLD_SIZE = number of pods participating in the distributed job
	job.Spec.NprocPerNode = ptr.To("auto")

	for name, spec := range job.Spec.PyTorchReplicaSpecs {
		if name == PrimaryPyTorchReplica {
			r.PatchPytorchReplicaSpec(spec, minGPURecommenderInput, recsMain)
		} else {
			r.PatchPytorchReplicaSpec(spec, minGPURecommenderInput, recsWorkers)
		}
	}

	if r.UnsuspendPatchedJobs {
		job.Spec.RunPolicy.Suspend = ptr.To(false)
	}

	// Create success JSON
	successJSON, _ := json.Marshal(map[string]interface{}{
		"recommendation": map[string]int{
			"workers": recs.Workers,
			"gpus":    recs.GPUs,
		},
	})

	data, err := json.Marshal(job)
	return RecommenderRequest{
		RequestID:             requestID,
		Pending:               false,
		AppliedRecommendation: true,
		RecommendationJSON:    string(successJSON),
	}, data, err
}

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
// TODO(user): Modify the Reconcile function to compare the state specified by
// the PyTorchJob object against the actual cluster state, and then
// perform operations to make the cluster state reflect the state specified by
// the user.
//
// For more details, check Reconcile and its Result here:
// - https://pkg.go.dev/sigs.k8s.io/controller-runtime@v0.22.1/pkg/reconcile
func (r *PyTorchJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)
	job := &kubeflowv1.PyTorchJob{}

	if err := r.Get(ctx, req.NamespacedName, job); err != nil {
		log.V(4).Info("unable to fetch PyTorchJob", "err", err)
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if job.Labels[r.DoneLabelKey] == r.DoneLabelValue {
		log.Info("PyTorchJob is already updated")
		return ctrl.Result{}, nil
	}

	log.Info("Reconciling PyTorchJob")

	original := job.DeepCopy()

	rr, _, err := r.UpdatePyTorchJob(job, job.Labels[r.WaitingForAdoRequestIDLabel], log)

	log.V(1).Info("Got back RecommenderRequest", "RecommenderRequest", rr)

	if err != nil {
		log.Error(err, "Cannot update PyTorchJob")

		r.Recorder.Eventf(
			job,
			corev1.EventTypeWarning,
			"RecommendationFailed",
			"Failed to update PyTorchJob: %v",
			err,
		)

		return ctrl.Result{}, err
	}

	if rr.RecommendationJSON != "" {
		// VV: We got an answer from the recommender (either recommendation or error)
		// Never look at either of these objects again
		delete(original.Labels, r.WatchLabelKey)
		// VV: Don't delete the label r.WaitingForAdoRequestIDLabel - users may want to check that request ID

		if r.DoneLabelKey != "kueue.x-k8s.io/queue-name" {
			// VV: The vpytorchjobs.kb.io webhook forbids mutating the Kueue label name
			original.Labels[r.DoneLabelKey] = r.DoneLabelValue
		}
		delete(job.Labels, r.WatchLabelKey)

		if err := r.Update(ctx, original); err != nil {
			return handleUpdateWrapperError(err, log)
		}

		// VV: Label and annotate the derived job
		job.Labels[r.DoneLabelKey] = r.DoneLabelValue

		if job.Annotations == nil {
			job.Annotations = make(map[string]string)
		}
		job.Annotations[r.RecommendationAnnotationKey] = rr.RecommendationJSON

		if rr.AppliedRecommendation {
			// VV: The vpytorchjobs.kb.io webhook forbids updating the kueue label so for integration with Kueue
			// We'll create a new object, just like we do with AppWrapper

			job.GenerateName = strings.TrimSuffix(original.Name, "-") + "-"
			job.Name = ""
			job.ResourceVersion = ""

			job.OwnerReferences = []v1api.OwnerReference{
				{
					APIVersion: original.APIVersion,
					Kind:       original.Kind,
					Name:       original.Name,
					UID:        original.UID,
				},
			}

			if err := r.Create(ctx, job); err != nil {
				log.Error(err, "unable to create a new PyTorchJob")
				return ctrl.Result{}, err
			}

			log.Info("Created new PyTorchJob with recommendations", "name", job.Name)

			r.Recorder.Event(
				original,
				corev1.EventTypeNormal,
				"PatchedWithRecommendations",
				"Created new PyTorchJob with recommended resource requirements",
			)
		} else {
			// No recommendation - mark original as done without creating new one
			log.Info("PyTorchJob marked as processed - no recommendation available")

			r.Recorder.Event(
				original,
				corev1.EventTypeWarning,
				"NoRecommendationAvailable",
				"Recommendation engine could not generate recommendations; PyTorchJob proceeding without modifications",
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
func (r *PyTorchJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	r.Recorder = mgr.GetEventRecorderFor("pytorchjob-controller")

	return ctrl.NewControllerManagedBy(mgr).
		Named("pytorchjob").
		For(&kubeflowv1.PyTorchJob{}).
		Complete(r)
}
