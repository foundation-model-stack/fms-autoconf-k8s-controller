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

// RestAPIRecommender implements DeferredRecommender by making requests to a
// REST API that processes recommendations asynchronously.
type RestAPIRecommender struct {
	baseURL string
}

// NewRestAPIRecommender creates a new DeferredRecommender that communicates
// with the REST API at the given base URL.
func NewRestAPIRecommender(baseURL string) *RestAPIRecommender {
	return &RestAPIRecommender{
		baseURL: baseURL,
	}
}

// InitiateRecommendation starts a recommendation computation by sending a
// request to the REST API and returns a request ID for tracking.
func (r *RestAPIRecommender) InitiateRecommendation(ctx context.Context, input MinGPURecommenderInput) (string, error) {
	log := logf.FromContext(ctx)

	log.V(1).Info("Initiating recommendation request to REST API", "features", input)

	// Use the existing function to send the request
	requestID, err := SendRequestToCalcMinimumResourceRequirements(input, r.baseURL, log)

	if err != nil {
		return "", err
	}

	log.Info("Submitted REST API request for recommendations", "RequestID", requestID)

	return requestID, nil
}

// CheckRecommendation checks the status of a recommendation request by polling
// the REST API. Returns nil result if the recommendation is still pending.
func (r *RestAPIRecommender) CheckRecommendation(ctx context.Context, requestID string, input MinGPURecommenderInput) (*RecommendationResult, error) {
	log := logf.FromContext(ctx)

	log.V(1).Info("Checking recommendation request status", "RequestID", requestID)

	// Use the existing function to check the request status
	recs, err := CheckRequestToCalcMinimumResourceRequirements(input, r.baseURL, requestID, log)

	if err != nil {
		return nil, err
	}

	// If recs is nil, the request is still pending
	if recs == nil {
		log.V(1).Info("Recommendation request still pending", "RequestID", requestID)
		return nil, nil
	}

	// Check if the API indicated it cannot recommend
	if !recs.CanRecommend {
		log.Info("REST API cannot recommend resources", "RequestID", requestID)
		return &RecommendationResult{
			Requirements: nil,
			CanRecommend: false,
		}, nil
	}

	log.Info("Obtained resource requirements from REST API", "RequestID", requestID, "recs", recs)

	return &RecommendationResult{
		Requirements: recs,
		CanRecommend: true,
	}, nil
}
