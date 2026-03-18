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
)

// RecommendationResult wraps the resource requirements recommendation along with
// metadata about whether a recommendation could be generated.
type RecommendationResult struct {
	// Requirements contains the recommended resource configuration.
	// This will be nil if CanRecommend is false.
	Requirements *ResourceRequirements

	// CanRecommend indicates whether the recommendation engine was able to
	// generate recommendations for the given input. When false, the workload
	// should proceed without modifications.
	CanRecommend bool
}

// ImmediateRecommender provides synchronous resource recommendations.
// Implementations compute and return recommendations in a single call.
//
// Example: A local Python script that computes recommendations immediately.
type ImmediateRecommender interface {
	// GetRecommendation computes and returns resource recommendations immediately.
	//
	// Parameters:
	//   - ctx: Context for cancellation, timeouts, and logging (use logr.FromContextOrDiscard)
	//   - input: The workload parameters to get recommendations for
	//
	// Returns:
	//   - *RecommendationResult: The recommendation result (never nil on success)
	//   - error: Any error that occurred during computation
	//
	// The result's CanRecommend field indicates whether recommendations could be
	// generated. When false, the workload should proceed without modifications.
	GetRecommendation(ctx context.Context, input MinGPURecommenderInput) (*RecommendationResult, error)
}

// DeferredRecommender provides asynchronous resource recommendations via a
// request/poll pattern. Implementations initiate a recommendation computation
// and allow polling for results.
//
// Example: A REST API that processes recommendations asynchronously.
type DeferredRecommender interface {
	// InitiateRecommendation starts a recommendation computation and returns a
	// request ID for tracking.
	//
	// Parameters:
	//   - ctx: Context for cancellation, timeouts, and logging (use logr.FromContextOrDiscard)
	//   - input: The workload parameters to get recommendations for
	//
	// Returns:
	//   - requestID: A unique identifier for tracking this recommendation request
	//   - error: Any error that occurred during initiation
	InitiateRecommendation(ctx context.Context, input MinGPURecommenderInput) (requestID string, err error)

	// CheckRecommendation checks the status of a recommendation request.
	//
	// Parameters:
	//   - ctx: Context for cancellation, timeouts, and logging (use logr.FromContextOrDiscard)
	//   - requestID: The request ID returned by InitiateRecommendation
	//   - input: The original workload parameters (may be needed for validation)
	//
	// Returns:
	//   - *RecommendationResult: The recommendation if ready, nil if still pending
	//   - error: Any error that occurred during the check
	//
	// When the result is nil and error is nil, the recommendation is still being
	// computed and the caller should poll again later.
	//
	// When the result is non-nil, its CanRecommend field indicates whether
	// recommendations could be generated. When false, the workload should proceed
	// without modifications.
	CheckRecommendation(ctx context.Context, requestID string, input MinGPURecommenderInput) (*RecommendationResult, error)
}

// Made with Bob
