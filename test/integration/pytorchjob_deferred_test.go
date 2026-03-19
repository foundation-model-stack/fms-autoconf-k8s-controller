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

package integration

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	kubeflowv1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"

	"github.com/ibm/resource-requirements-appwrapper/internal/controller"
	"github.com/ibm/resource-requirements-appwrapper/test/utils"
)

func makeDefaultJob(name, namespace string) *kubeflowv1.PyTorchJob {
	return utils.MakePyTorchJob(name, namespace).
		Label(watchLabelKey, watchLabelValue).
		MasterReplicaSpec(1).
		Container().
		Command("sh", "-c").
		Args("accelerate launch --num_processes=1 -m tuning.sft_trainer --model_name_or_path ibm-granite/granite-8b-code-base-4k --per_device_train_batch_size 8 --max_seq_length 8192").
		Env("AUTOCONF_GPU_MODEL", "NVIDIA-A100-SXM4-80GB").
		Done().
		Done().
		Obj()
}

var _ = Describe("PyTorchJob Controller with Deferred Recommender", Serial, func() {
	const (
		timeout  = time.Second * 10
		interval = time.Millisecond * 250
	)

	var (
		namespace string
	)

	BeforeEach(func() {
		// Create a unique namespace for each test
		namespace = "test-" + utils.RandomString(8)
		ns := &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: namespace,
			},
		}
		Expect(k8sClient.Create(ctx, ns)).To(Succeed())

		// Reset the mock recommender state for each test
		mockRecommender.Reset()
	})

	AfterEach(func() {
		// Clean up the namespace
		ns := &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: namespace,
			},
		}
		Expect(k8sClient.Delete(ctx, ns)).To(Succeed())
	})

	Context("When reconciling a PyTorchJob with deferred recommender", func() {
		It("Should handle the full lifecycle: initiate, poll pending, get result", func() {
			By("Setting up mock deferred recommender with 2 pending checks")
			mockRecommender.SetPendingChecks(2)
			mockRecommender.SetResult(&controller.RecommendationResult{
				Requirements: &controller.ResourceRequirements{
					Workers:      2,
					GPUs:         1,
					CanRecommend: true,
				},
				CanRecommend: true,
			})

			By("Creating a PyTorchJob")
			job := makeDefaultJob("test-deferred-job", namespace)

			Expect(k8sClient.Create(ctx, job)).To(Succeed())

			jobKey := types.NamespacedName{Name: job.Name, Namespace: job.Namespace}

			By("Waiting for request ID label to be added")
			Eventually(func() bool {
				updatedJob := &kubeflowv1.PyTorchJob{}
				err := k8sClient.Get(ctx, jobKey, updatedJob)
				if err != nil {
					return false
				}
				_, exists := updatedJob.Labels[requestIDLabel]
				return exists
			}, timeout, interval).Should(BeTrue(), "Request ID label should be added")

			By("Verifying the mock was called multiple times (polling)")
			Eventually(func() int {
				return mockRecommender.CheckCallCount
			}, timeout, interval).Should(BeNumerically(">=", 2), "CheckRecommendation should be called at least twice")

			By("Waiting for job to be marked as done after pending checks complete")
			Eventually(func() string {
				updatedJob := &kubeflowv1.PyTorchJob{}
				err := k8sClient.Get(ctx, jobKey, updatedJob)
				if err != nil {
					return ""
				}
				return updatedJob.Labels[controller.DefaultAutoconfDoneLabelKey]
			}, timeout, interval).Should(Equal(controller.DefaultAutoconfDoneLabelValue), "Job should be marked as done")

			By("Verifying recommendation annotation was added")
			updatedJob := &kubeflowv1.PyTorchJob{}
			Expect(k8sClient.Get(ctx, jobKey, updatedJob)).To(Succeed())
			Expect(updatedJob.Annotations[recommendationKey]).NotTo(BeEmpty(), "Recommendation annotation should be added")

			By("Verifying watch label was removed")
			_, exists := updatedJob.Labels[watchLabelKey]
			Expect(exists).To(BeFalse(), "Watch label should be removed")

			By("Verifying job spec was updated with recommendations")
			masterSpec := updatedJob.Spec.PyTorchReplicaSpecs[controller.PrimaryPyTorchReplica]
			Expect(masterSpec).NotTo(BeNil())
			Expect(*masterSpec.Replicas).To(Equal(int32(1)), "Master should have 1 replica")

			workerSpec := updatedJob.Spec.PyTorchReplicaSpecs[controller.WorkerPyTorchReplica]
			Expect(workerSpec).NotTo(BeNil())
			Expect(*workerSpec.Replicas).To(Equal(int32(1)), "Worker should have 1 replica (2 workers - 1 master)")
		})

		It("Should handle result ready on second check", func() {
			By("Setting up mock deferred recommender with 1 pending check")
			mockRecommender.SetPendingChecks(1)
			mockRecommender.SetResult(&controller.RecommendationResult{
				Requirements: &controller.ResourceRequirements{
					Workers:      2,
					GPUs:         1,
					CanRecommend: true,
				},
				CanRecommend: true,
			})

			By("Creating a PyTorchJob")
			job := makeDefaultJob("test-quick-result", namespace)

			Expect(k8sClient.Create(ctx, job)).To(Succeed())

			jobKey := types.NamespacedName{Name: job.Name, Namespace: job.Namespace}

			By("Waiting for request ID label to be added")
			Eventually(func() bool {
				updatedJob := &kubeflowv1.PyTorchJob{}
				err := k8sClient.Get(ctx, jobKey, updatedJob)
				if err != nil {
					return false
				}
				_, exists := updatedJob.Labels[requestIDLabel]
				return exists
			}, timeout, interval).Should(BeTrue(), "Request ID label should be added")

			By("Waiting for job to be marked as done")
			Eventually(func() string {
				updatedJob := &kubeflowv1.PyTorchJob{}
				err := k8sClient.Get(ctx, jobKey, updatedJob)
				if err != nil {
					return ""
				}
				return updatedJob.Labels[controller.DefaultAutoconfDoneLabelKey]
			}, timeout, interval).Should(Equal(controller.DefaultAutoconfDoneLabelValue), "Job should be marked as done")

			By("Verifying recommendation annotation was added")
			updatedJob := &kubeflowv1.PyTorchJob{}
			Expect(k8sClient.Get(ctx, jobKey, updatedJob)).To(Succeed())
			Expect(updatedJob.Annotations[recommendationKey]).NotTo(BeEmpty(), "Recommendation annotation should be added")
		})

		It("Should handle immediate result (no pending)", func() {
			By("Setting up mock deferred recommender with 0 pending checks")
			mockRecommender.SetPendingChecks(0)
			mockRecommender.SetResult(&controller.RecommendationResult{
				Requirements: &controller.ResourceRequirements{
					Workers:      2,
					GPUs:         1,
					CanRecommend: true,
				},
				CanRecommend: true,
			})

			By("Creating a PyTorchJob")
			job := makeDefaultJob("test-immediate-result", namespace)

			Expect(k8sClient.Create(ctx, job)).To(Succeed())

			jobKey := types.NamespacedName{Name: job.Name, Namespace: job.Namespace}

			By("Waiting for job to be marked as done")
			Eventually(func() string {
				updatedJob := &kubeflowv1.PyTorchJob{}
				err := k8sClient.Get(ctx, jobKey, updatedJob)
				if err != nil {
					return ""
				}
				return updatedJob.Labels[controller.DefaultAutoconfDoneLabelKey]
			}, timeout, interval).Should(Equal(controller.DefaultAutoconfDoneLabelValue), "Job should be marked as done")

			By("Verifying recommendation annotation was added")
			updatedJob := &kubeflowv1.PyTorchJob{}
			Expect(k8sClient.Get(ctx, jobKey, updatedJob)).To(Succeed())
			Expect(updatedJob.Annotations[recommendationKey]).NotTo(BeEmpty(), "Recommendation annotation should be added")
		})
	})
})

// Made with Bob
