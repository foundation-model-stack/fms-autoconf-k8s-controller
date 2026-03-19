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

package testutil

import (
	"context"
	"fmt"
	"sync"

	"github.com/ibm/resource-requirements-appwrapper/internal/controller"
)

// MockImmediateRecommender is a mock implementation of ImmediateRecommender
// for testing purposes. It allows configuring responses and tracking calls.
type MockImmediateRecommender struct {
	mu sync.Mutex

	// Result to return from GetRecommendation
	Result *controller.RecommendationResult
	// Error to return from GetRecommendation
	Err error

	// Tracking
	CallCount int
	LastInput controller.MinGPURecommenderInput
}

// NewMockImmediateRecommender creates a new mock immediate recommender with
// default successful response.
func NewMockImmediateRecommender() *MockImmediateRecommender {
	return &MockImmediateRecommender{
		Result: &controller.RecommendationResult{
			Requirements: &controller.ResourceRequirements{
				Workers:      2,
				GPUs:         1,
				CanRecommend: true,
			},
			CanRecommend: true,
		},
	}
}

// GetRecommendation returns the configured result and error.
func (m *MockImmediateRecommender) GetRecommendation(ctx context.Context, input controller.MinGPURecommenderInput) (*controller.RecommendationResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.CallCount++
	m.LastInput = input

	return m.Result, m.Err
}

// SetResult configures the result to return.
func (m *MockImmediateRecommender) SetResult(result *controller.RecommendationResult) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Result = result
}

// SetError configures the error to return.
func (m *MockImmediateRecommender) SetError(err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Err = err
}

// Reset resets the mock state.
func (m *MockImmediateRecommender) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.CallCount = 0
	m.LastInput = controller.MinGPURecommenderInput{}
}

// MockDeferredRecommender is a mock implementation of DeferredRecommender
// for testing purposes. It simulates async behavior and allows configuring
// responses and tracking calls.
type MockDeferredRecommender struct {
	mu sync.Mutex

	// Configuration
	// Result to return from CheckRecommendation when ready
	Result *controller.RecommendationResult
	// Error to return from InitiateRecommendation
	InitiateErr error
	// Error to return from CheckRecommendation
	CheckErr error
	// Number of CheckRecommendation calls before returning result (simulates pending)
	PendingChecks int

	// State tracking
	InitiateCallCount int
	CheckCallCount    int
	LastInput         controller.MinGPURecommenderInput
	LastRequestID     string
	// Map of requestID -> number of times it's been checked
	checkCounts map[string]int
}

// NewMockDeferredRecommender creates a new mock deferred recommender with
// default successful response that returns immediately (no pending checks).
func NewMockDeferredRecommender() *MockDeferredRecommender {
	return &MockDeferredRecommender{
		Result: &controller.RecommendationResult{
			Requirements: &controller.ResourceRequirements{
				Workers:      2,
				GPUs:         1,
				CanRecommend: true,
			},
			CanRecommend: true,
		},
		PendingChecks: 0,
		checkCounts:   make(map[string]int),
	}
}

// InitiateRecommendation returns a mock request ID.
func (m *MockDeferredRecommender) InitiateRecommendation(ctx context.Context, input controller.MinGPURecommenderInput) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.InitiateCallCount++
	m.LastInput = input

	if m.InitiateErr != nil {
		return "", m.InitiateErr
	}

	requestID := fmt.Sprintf("mock-request-%d", m.InitiateCallCount)
	m.LastRequestID = requestID
	m.checkCounts[requestID] = 0

	return requestID, nil
}

// CheckRecommendation simulates async behavior by returning nil for the first
// PendingChecks calls, then returning the configured result.
func (m *MockDeferredRecommender) CheckRecommendation(ctx context.Context, requestID string, input controller.MinGPURecommenderInput) (*controller.RecommendationResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.CheckCallCount++
	m.LastInput = input
	m.LastRequestID = requestID

	if m.CheckErr != nil {
		return nil, m.CheckErr
	}

	// Track how many times this specific request has been checked
	m.checkCounts[requestID]++

	// Simulate pending state
	if m.checkCounts[requestID] <= m.PendingChecks {
		return nil, nil
	}

	return m.Result, nil
}

// SetResult configures the result to return.
func (m *MockDeferredRecommender) SetResult(result *controller.RecommendationResult) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.Result = result
}

// SetInitiateError configures the error to return from InitiateRecommendation.
func (m *MockDeferredRecommender) SetInitiateError(err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.InitiateErr = err
}

// SetCheckError configures the error to return from CheckRecommendation.
func (m *MockDeferredRecommender) SetCheckError(err error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.CheckErr = err
}

// SetPendingChecks configures how many CheckRecommendation calls should
// return nil (pending) before returning the result.
func (m *MockDeferredRecommender) SetPendingChecks(count int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.PendingChecks = count
}

// Reset resets the mock state.
func (m *MockDeferredRecommender) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.InitiateCallCount = 0
	m.CheckCallCount = 0
	m.LastInput = controller.MinGPURecommenderInput{}
	m.LastRequestID = ""
	m.checkCounts = make(map[string]int)
}
