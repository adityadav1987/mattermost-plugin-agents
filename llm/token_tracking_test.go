// Copyright (c) 2023-present Mattermost, Inc. All Rights Reserved.
// See LICENSE.txt for license information.

package llm

import (
	"testing"

	"github.com/mattermost/mattermost/server/public/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

// MockLLMetrics is a mock implementation of the LLMetrics interface
type MockLLMetrics struct {
	mock.Mock
}

func (m *MockLLMetrics) IncrementLLMRequests() {
	m.Called()
}

func (m *MockLLMetrics) IncrementInputTokens(userID, teamID string, count int) {
	m.Called(userID, teamID, count)
}

func (m *MockLLMetrics) IncrementOutputTokens(userID, teamID string, count int) {
	m.Called(userID, teamID, count)
}

// MockLanguageModel is a mock implementation of the LanguageModel interface
type MockLanguageModel struct {
	mock.Mock
}

func (m *MockLanguageModel) ChatCompletion(request CompletionRequest, opts ...LanguageModelOption) (*TextStreamResult, error) {
	args := m.Called(request, opts)
	return args.Get(0).(*TextStreamResult), args.Error(1)
}

func (m *MockLanguageModel) ChatCompletionNoStream(request CompletionRequest, opts ...LanguageModelOption) (string, error) {
	args := m.Called(request, opts)
	return args.String(0), args.Error(1)
}

func (m *MockLanguageModel) CountTokens(text string) int {
	args := m.Called(text)
	return args.Int(0)
}

func (m *MockLanguageModel) InputTokenLimit() int {
	args := m.Called()
	return args.Int(0)
}

func TestTokenTrackingWrapper_ChatCompletion(t *testing.T) {
	t.Run("tracks token usage from stream events", func(t *testing.T) {
		mockMetrics := &MockLLMetrics{}
		mockLLM := &MockLanguageModel{}
		wrapper := NewTokenTrackingWrapper(mockLLM, mockMetrics)

		// Create a mock stream with usage event
		mockStream := make(chan TextStreamEvent, 3)
		mockStream <- TextStreamEvent{Type: EventTypeText, Value: "Hello"}
		mockStream <- TextStreamEvent{Type: EventTypeUsage, Value: TokenUsage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15}}
		mockStream <- TextStreamEvent{Type: EventTypeEnd, Value: nil}
		close(mockStream)

		mockResult := &TextStreamResult{Stream: mockStream}
		mockLLM.On("ChatCompletion", mock.Anything, mock.Anything).Return(mockResult, nil)
		mockMetrics.On("IncrementInputTokens", "user123", "team456", 10).Return()
		mockMetrics.On("IncrementOutputTokens", "user123", "team456", 5).Return()

		request := CompletionRequest{
			Context: &Context{
				RequestingUser: &model.User{Id: "user123"},
				Team:           &model.Team{Id: "team456"},
			},
		}

		result, err := wrapper.ChatCompletion(request)
		require.NoError(t, err)
		require.NotNil(t, result)

		// Read all events from the intercepted stream
		var events []TextStreamEvent
		for event := range result.Stream {
			events = append(events, event)
		}

		// Should have forwarded text and end events, but not usage event
		require.Len(t, events, 2)
		assert.Equal(t, EventTypeText, events[0].Type)
		assert.Equal(t, "Hello", events[0].Value)
		assert.Equal(t, EventTypeEnd, events[1].Type)

		mockLLM.AssertExpectations(t)
		mockMetrics.AssertExpectations(t)
	})

	t.Run("handles nil metrics gracefully", func(t *testing.T) {
		mockLLM := &MockLanguageModel{}
		wrapper := NewTokenTrackingWrapper(mockLLM, nil)

		mockStream := make(chan TextStreamEvent, 2)
		mockStream <- TextStreamEvent{Type: EventTypeUsage, Value: TokenUsage{InputTokens: 10, OutputTokens: 5, TotalTokens: 15}}
		mockStream <- TextStreamEvent{Type: EventTypeEnd, Value: nil}
		close(mockStream)

		mockResult := &TextStreamResult{Stream: mockStream}
		mockLLM.On("ChatCompletion", mock.Anything, mock.Anything).Return(mockResult, nil)

		request := CompletionRequest{Context: &Context{}}
		result, err := wrapper.ChatCompletion(request)
		require.NoError(t, err)

		// Should complete without panic even with nil metrics
		var events []TextStreamEvent
		for event := range result.Stream {
			events = append(events, event)
		}

		assert.Len(t, events, 1) // Only end event forwarded
		assert.Equal(t, EventTypeEnd, events[0].Type)

		mockLLM.AssertExpectations(t)
	})

	t.Run("handles invalid usage event value", func(t *testing.T) {
		mockMetrics := &MockLLMetrics{}
		mockLLM := &MockLanguageModel{}
		wrapper := NewTokenTrackingWrapper(mockLLM, mockMetrics)

		mockStream := make(chan TextStreamEvent, 2)
		mockStream <- TextStreamEvent{Type: EventTypeUsage, Value: "invalid_value"}
		mockStream <- TextStreamEvent{Type: EventTypeEnd, Value: nil}
		close(mockStream)

		mockResult := &TextStreamResult{Stream: mockStream}
		mockLLM.On("ChatCompletion", mock.Anything, mock.Anything).Return(mockResult, nil)

		request := CompletionRequest{Context: &Context{}}
		result, err := wrapper.ChatCompletion(request)
		require.NoError(t, err)

		// Should complete without calling metrics (invalid value ignored)
		var events []TextStreamEvent
		for event := range result.Stream {
			events = append(events, event)
		}

		assert.Len(t, events, 1) // Only end event forwarded
		mockLLM.AssertExpectations(t)
		mockMetrics.AssertExpectations(t) // No calls expected
	})
}

func TestTokenTrackingWrapper_ChatCompletionNoStream(t *testing.T) {
	t.Run("delegates to streaming method", func(t *testing.T) {
		mockMetrics := &MockLLMetrics{}
		mockLLM := &MockLanguageModel{}
		wrapper := NewTokenTrackingWrapper(mockLLM, mockMetrics)

		mockStream := make(chan TextStreamEvent, 3)
		mockStream <- TextStreamEvent{Type: EventTypeText, Value: "Hello world"}
		mockStream <- TextStreamEvent{Type: EventTypeUsage, Value: TokenUsage{InputTokens: 5, OutputTokens: 10, TotalTokens: 15}}
		mockStream <- TextStreamEvent{Type: EventTypeEnd, Value: nil}
		close(mockStream)

		mockResult := &TextStreamResult{Stream: mockStream}
		mockLLM.On("ChatCompletion", mock.Anything, mock.Anything).Return(mockResult, nil)
		mockMetrics.On("IncrementInputTokens", "unknown", "unknown", 5).Return()
		mockMetrics.On("IncrementOutputTokens", "unknown", "unknown", 10).Return()

		request := CompletionRequest{Context: &Context{}}
		result, err := wrapper.ChatCompletionNoStream(request)
		require.NoError(t, err)
		assert.Equal(t, "Hello world", result)

		mockLLM.AssertExpectations(t)
		mockMetrics.AssertExpectations(t)
	})
}

func TestTokenTrackingWrapper_DelegatedMethods(t *testing.T) {
	mockMetrics := &MockLLMetrics{}
	mockLLM := &MockLanguageModel{}
	wrapper := NewTokenTrackingWrapper(mockLLM, mockMetrics)

	t.Run("CountTokens delegates to wrapped model", func(t *testing.T) {
		mockLLM.On("CountTokens", "test text").Return(42)

		result := wrapper.CountTokens("test text")
		assert.Equal(t, 42, result)

		mockLLM.AssertExpectations(t)
	})

	t.Run("InputTokenLimit delegates to wrapped model", func(t *testing.T) {
		mockLLM.On("InputTokenLimit").Return(4096)

		result := wrapper.InputTokenLimit()
		assert.Equal(t, 4096, result)

		mockLLM.AssertExpectations(t)
	})
}
