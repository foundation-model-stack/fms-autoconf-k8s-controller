// Copyright (c) IBM Corporation
// SPDX-License-Identifier: MIT

package controller

import (
	"context"
	"reflect"
	"testing"

	"github.com/go-logr/logr"
	kubeflowv1 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v1"
	v1 "k8s.io/api/core/v1"
)

func TestExtractPartialInformationFromCommandlineWithoutEnvVars(t *testing.T) {
	command := `some other command here

export X="something else here"

accelerate launch --use_fsdp --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP --fsdp_forward_prefetch=false \
--fsdp_offload_params=false --fsdp_sharding_strategy=HYBRID_SHARD --fsdp_state_dict_type=FULL_STATE_DICT \
--fsdp_cpu_ram_efficient_loading=true --fsdp_sync_module_states=true --dynamo_backend="no" --machine_rank="${RANK}" \
--main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --mixed_precision="no" \
--num_machines="${NUM_MACHINES}" --num_processes="2" --rdzv_backend="static" --same_network \
-m tuning.sft_trainer --log_level info --eval_strategy no --save_strategy no \
--learning_rate 1e-05 --weight_decay 0.0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
--logging_steps 1 --include_tokens_per_second True --packing False --response_template "\n### Response:" \
--dataset_text_field output --gradient_accumulation_steps 4 --gradient_checkpointing True --max_steps -1 \
--num_train_epochs 1.0 --model_name_or_path "ibm-granite/granite-8b-code-base-4k" \
--per_device_train_batch_size 8 --torch_dtype bfloat16 --max_seq_length '8192' \
--training_data_path /data/fms-hf-tuning/artificial-dataset/news-tokens-16384plus-entries-4096.jsonl \
--output_dir /tmp/output/ --peft_method lora  --r 4 --lora_alpha 16 --target_modules q_proj v_proj --use_flash_attn True

some other command here`

	job := kubeflowv1.PyTorchJob{
		Spec: kubeflowv1.PyTorchJobSpec{
			PyTorchReplicaSpecs: map[kubeflowv1.ReplicaType]*kubeflowv1.ReplicaSpec{
				PrimaryPyTorchReplica: &kubeflowv1.ReplicaSpec{
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Command: []string{
										"sh", "-c",
									},
									Args: []string{command},
								},
							},
						},
					},
				},
			},
		},
	}

	config, err := ExtractPartialInformationFromCommandlineAndEnvVars(&job)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := map[string]string{
		"AUTOCONF_MODEL_NAME":        "granite-8b-code-base",
		"AUTOCONF_METHOD":            "lora",
		"AUTOCONF_TOKENS_PER_SAMPLE": "8192",
		"AUTOCONF_BATCH_SIZE":        "16",
	}

	if !reflect.DeepEqual(config, expected) {
		t.Fatalf("expected %#v, got %#v", expected, config)
	}
}

func pyTorchJobWithCommandLineWithEnvVars() kubeflowv1.PyTorchJob {
	command := `some other command here

export X="something else here"

accelerate launch --use_fsdp --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP --fsdp_forward_prefetch=false \
--fsdp_offload_params=false --fsdp_sharding_strategy=HYBRID_SHARD --fsdp_state_dict_type=FULL_STATE_DICT \
--fsdp_cpu_ram_efficient_loading=true --fsdp_sync_module_states=true --dynamo_backend="no" --machine_rank="${RANK}" \
--main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}" --mixed_precision="no" \
--num_machines="${NUM_MACHINES}" --num_processes="${NUM_PROCESSES}" --rdzv_backend="static" --same_network \
-m tuning.sft_trainer --log_level info --eval_strategy no --save_strategy no \
--learning_rate 1e-05 --weight_decay 0.0 --warmup_ratio 0.03 --lr_scheduler_type cosine \
--logging_steps 1 --include_tokens_per_second True --packing False --response_template "\n### Response:" \
--dataset_text_field output --gradient_accumulation_steps 4 --gradient_checkpointing True --max_steps -1 \
--num_train_epochs 1.0 --model_name_or_path "ibm-granite/granite-8b-code-base-4k" \
--per_device_train_batch_size 8 --torch_dtype bfloat16 --max_seq_length $TOKENS_PER_SAMPLE \
--training_data_path /data/fms-hf-tuning/artificial-dataset/news-tokens-16384plus-entries-4096.jsonl \
--output_dir /tmp/output/ --peft_method lora  --r 4 --lora_alpha 16 --target_modules q_proj v_proj --use_flash_attn True

some other command here`

	return kubeflowv1.PyTorchJob{
		Spec: kubeflowv1.PyTorchJobSpec{
			PyTorchReplicaSpecs: map[kubeflowv1.ReplicaType]*kubeflowv1.ReplicaSpec{
				PrimaryPyTorchReplica: &kubeflowv1.ReplicaSpec{
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Command: []string{
										"sh", "-c", command,
									},
									// Args: []string{command},
									Env: []v1.EnvVar{
										{
											Name:  "TOKENS_PER_SAMPLE",
											Value: "8192",
										},
										{
											Name:  "NUM_PROCESSES",
											Value: "2",
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

}

func TestExtractPartialInformationFromCommandlineWithEnvVars(t *testing.T) {
	job := pyTorchJobWithCommandLineWithEnvVars()

	config, err := ExtractPartialInformationFromCommandlineAndEnvVars(&job)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := map[string]string{
		"AUTOCONF_MODEL_NAME":        "granite-8b-code-base",
		"AUTOCONF_METHOD":            "lora",
		"AUTOCONF_TOKENS_PER_SAMPLE": "8192",
		"AUTOCONF_BATCH_SIZE":        "16",
	}

	if !reflect.DeepEqual(config, expected) {
		t.Fatalf("expected %#v, got %#v", expected, config)
	}
}

func TestExtractMinGPURecommenderInputNoEnvVars(t *testing.T) {
	job := pyTorchJobWithCommandLineWithEnvVars()
	log, _ := logr.FromContext(context.TODO())

	minGpuRecommenderInput, err := ExtractMinGPURecommenderInput(&job, "NVIDIA-A100-SXM4-80GB", "2.0.0", log)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := MinGPURecommenderInput{
		ModelName:       "granite-8b-code-base",
		Method:          "lora",
		GPUModel:        "NVIDIA-A100-SXM4-80GB",
		TokensPerSample: 8192,
		BatchSize:       16,
		ModelVersion:    "2.0.0",
	}

	if !reflect.DeepEqual(minGpuRecommenderInput, expected) {
		t.Fatalf("expected %#v, got %#v", expected, minGpuRecommenderInput)
	}
}
