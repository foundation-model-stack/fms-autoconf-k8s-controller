// Copyright (c) IBM Corporation
// SPDX-License-Identifier: MIT

package controller

import (
	"testing"

	v1 "k8s.io/api/core/v1"
)

func TestExtractGPUModelFromNodeSelector(t *testing.T) {
	tests := []struct {
		name         string
		nodeSelector map[string]string
		expected     string
	}{
		{
			name:         "nil nodeSelector",
			nodeSelector: nil,
			expected:     "",
		},
		{
			name:         "empty nodeSelector",
			nodeSelector: map[string]string{},
			expected:     "",
		},
		{
			name: "nodeSelector with GPU product",
			nodeSelector: map[string]string{
				"nvidia.com/gpu.product": "NVIDIA-A100-80GB-PCIe",
			},
			expected: "NVIDIA-A100-80GB-PCIe",
		},
		{
			name: "nodeSelector with other labels but no GPU product",
			nodeSelector: map[string]string{
				"kubernetes.io/hostname": "node-1",
				"node-role":              "worker",
			},
			expected: "",
		},
		{
			name: "nodeSelector with GPU product and other labels",
			nodeSelector: map[string]string{
				"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3",
				"kubernetes.io/hostname": "node-1",
			},
			expected: "NVIDIA-H100-80GB-HBM3",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractGPUModelFromNodeSelector(tt.nodeSelector)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestExtractGPUModelFromNodeAffinity(t *testing.T) {
	tests := []struct {
		name         string
		nodeAffinity *v1.NodeAffinity
		expected     string
	}{
		{
			name:         "nil nodeAffinity",
			nodeAffinity: nil,
			expected:     "",
		},
		{
			name:         "empty nodeAffinity",
			nodeAffinity: &v1.NodeAffinity{},
			expected:     "",
		},
		{
			name: "nodeAffinity with GPU product and single value",
			nodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "nvidia.com/gpu.product",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"NVIDIA-A100-80GB-PCIe"},
								},
							},
						},
					},
				},
			},
			expected: "NVIDIA-A100-80GB-PCIe",
		},
		{
			name: "nodeAffinity with GPU product and multiple values",
			nodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "nvidia.com/gpu.product",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"NVIDIA-A100-80GB-PCIe", "NVIDIA-H100-80GB-HBM3"},
								},
							},
						},
					},
				},
			},
			expected: "",
		},
		{
			name: "nodeAffinity with NotIn operator (anti-affinity)",
			nodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "nvidia.com/gpu.product",
									Operator: v1.NodeSelectorOpNotIn,
									Values:   []string{"NVIDIA-A100-80GB-PCIe"},
								},
							},
						},
					},
				},
			},
			expected: "",
		},
		{
			name: "nodeAffinity with other labels but no GPU product",
			nodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "kubernetes.io/hostname",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"node-1"},
								},
							},
						},
					},
				},
			},
			expected: "",
		},
		{
			name: "nodeAffinity with GPU product in second term",
			nodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "kubernetes.io/hostname",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"node-1"},
								},
							},
						},
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "nvidia.com/gpu.product",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"NVIDIA-H100-80GB-HBM3"},
								},
							},
						},
					},
				},
			},
			expected: "NVIDIA-H100-80GB-HBM3",
		},
		{
			name: "nodeAffinity with Exists operator",
			nodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "nvidia.com/gpu.product",
									Operator: v1.NodeSelectorOpExists,
								},
							},
						},
					},
				},
			},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractGPUModelFromNodeAffinity(tt.nodeAffinity)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

func TestExtractGPUModelFromPodSpec(t *testing.T) {
	tests := []struct {
		name     string
		podSpec  *v1.PodSpec
		expected string
	}{
		{
			name: "GPU model from environment variable",
			podSpec: &v1.PodSpec{
				Containers: []v1.Container{
					{
						Env: []v1.EnvVar{
							{Name: "AUTOCONF_GPU_MODEL", Value: "NVIDIA-A100-80GB-PCIe"},
						},
					},
				},
			},
			expected: "NVIDIA-A100-80GB-PCIe",
		},
		{
			name: "GPU model from nodeSelector",
			podSpec: &v1.PodSpec{
				Containers: []v1.Container{
					{
						Env: []v1.EnvVar{},
					},
				},
				NodeSelector: map[string]string{
					"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3",
				},
			},
			expected: "NVIDIA-H100-80GB-HBM3",
		},
		{
			name: "GPU model from nodeAffinity",
			podSpec: &v1.PodSpec{
				Containers: []v1.Container{
					{
						Env: []v1.EnvVar{},
					},
				},
				Affinity: &v1.Affinity{
					NodeAffinity: &v1.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
							NodeSelectorTerms: []v1.NodeSelectorTerm{
								{
									MatchExpressions: []v1.NodeSelectorRequirement{
										{
											Key:      "nvidia.com/gpu.product",
											Operator: v1.NodeSelectorOpIn,
											Values:   []string{"NVIDIA-V100-32GB"},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: "NVIDIA-V100-32GB",
		},
		{
			name: "env var takes precedence over nodeSelector",
			podSpec: &v1.PodSpec{
				Containers: []v1.Container{
					{
						Env: []v1.EnvVar{
							{Name: "AUTOCONF_GPU_MODEL", Value: "NVIDIA-A100-80GB-PCIe"},
						},
					},
				},
				NodeSelector: map[string]string{
					"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3",
				},
			},
			expected: "NVIDIA-A100-80GB-PCIe",
		},
		{
			name: "nodeSelector takes precedence over nodeAffinity",
			podSpec: &v1.PodSpec{
				Containers: []v1.Container{
					{
						Env: []v1.EnvVar{},
					},
				},
				NodeSelector: map[string]string{
					"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3",
				},
				Affinity: &v1.Affinity{
					NodeAffinity: &v1.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
							NodeSelectorTerms: []v1.NodeSelectorTerm{
								{
									MatchExpressions: []v1.NodeSelectorRequirement{
										{
											Key:      "nvidia.com/gpu.product",
											Operator: v1.NodeSelectorOpIn,
											Values:   []string{"NVIDIA-V100-32GB"},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: "NVIDIA-H100-80GB-HBM3",
		},
		{
			name: "no GPU model found",
			podSpec: &v1.PodSpec{
				Containers: []v1.Container{
					{
						Env: []v1.EnvVar{
							{Name: "OTHER_VAR", Value: "value"},
						},
					},
				},
			},
			expected: "",
		},
		{
			name: "empty env var value is ignored",
			podSpec: &v1.PodSpec{
				Containers: []v1.Container{
					{
						Env: []v1.EnvVar{
							{Name: "AUTOCONF_GPU_MODEL", Value: ""},
						},
					},
				},
				NodeSelector: map[string]string{
					"nvidia.com/gpu.product": "NVIDIA-A100-80GB-PCIe",
				},
			},
			expected: "NVIDIA-A100-80GB-PCIe",
		},
		{
			name: "nodeAffinity with multiple values is ignored",
			podSpec: &v1.PodSpec{
				Containers: []v1.Container{
					{
						Env: []v1.EnvVar{},
					},
				},
				Affinity: &v1.Affinity{
					NodeAffinity: &v1.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
							NodeSelectorTerms: []v1.NodeSelectorTerm{
								{
									MatchExpressions: []v1.NodeSelectorRequirement{
										{
											Key:      "nvidia.com/gpu.product",
											Operator: v1.NodeSelectorOpIn,
											Values:   []string{"NVIDIA-A100-80GB-PCIe", "NVIDIA-H100-80GB-HBM3"},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractGPUModelFromPodSpec(tt.podSpec)
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}

// Made with Bob
