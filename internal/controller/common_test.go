// Copyright (c) IBM Corporation
// SPDX-License-Identifier: MIT

//go:build restapi
// +build restapi

package controller

import (
	"reflect"
	"testing"

	"github.com/go-logr/logr"
)

// TestSendRequestToCalcMininumResourceRequirements_Success verifies that a successful 200 response
// with a JSON body is handled without error. Since we don't assert on the concrete fields of
// ResourceRequirements (unknown here), we return an empty JSON object and assert that no error is returned.
func TestSendRequestToCalcMininumResourceRequirements_Success(t *testing.T) {

	in := MinGPURecommenderInput{
		ModelName:       "llama-7b",
		Method:          "lora",
		GPUModel:        "NVIDIA-A100-80GB-PCIe",
		TokensPerSample: 8192,
		BatchSize:       16,
		NumGPUs:         1,
	}

	got, err := SendRequestToCalcMinimumResourceRequirements(in, "https://ado-api-discovery-dev.apps.morrigan.accelerated-discovery.res.ibm.com", logr.Discard())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := ResourceRequirements{
		Workers: 1,
		GPUs:    2,
	}
	if !reflect.DeepEqual(got, expected) {
		t.Fatalf("expected %#v ResourceRequirements, got %#v", expected, got)
	}
}
