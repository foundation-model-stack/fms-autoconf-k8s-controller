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

	logf "sigs.k8s.io/controller-runtime/pkg/log"
)

// PythonScriptRecommender implements ImmediateRecommender by executing a Python
// script that computes resource recommendations synchronously.
type PythonScriptRecommender struct {
	scriptPath string
}

// NewPythonScriptRecommender creates a new ImmediateRecommender that executes
// the Python script at the given path.
func NewPythonScriptRecommender(scriptPath string) *PythonScriptRecommender {
	return &PythonScriptRecommender{
		scriptPath: scriptPath,
	}
}

// GetRecommendation computes and returns resource recommendations by executing
// the Python script synchronously.
func (r *PythonScriptRecommender) GetRecommendation(ctx context.Context, input MinGPURecommenderInput) (*RecommendationResult, error) {
	log := logf.FromContext(ctx)

	log.V(1).Info("Requesting recommendation from Python script", "features", input)

	// Use the existing function to run the Python wrapper script
	recs, err := RunPythonWrapperToCalcMinimumResourceRequirements(input, r.scriptPath, log)

	if err != nil {
		return nil, err
	}

	// Check if the script indicated it cannot recommend
	if recs.Error != "" {
		log.Info("Python script cannot recommend resources", "error", recs.Error)
		return &RecommendationResult{
			Requirements: nil,
			CanRecommend: false,
		}, nil
	}

	log.Info("Computed resource requirements from Python script", "recs", recs)

	return &RecommendationResult{
		Requirements: &recs,
		CanRecommend: true,
	}, nil
}
