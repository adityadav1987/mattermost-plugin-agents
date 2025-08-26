// Copyright (c) 2023-present Mattermost, Inc. All Rights Reserved.
// See LICENSE.txt for license information.

package llm

import (
	"github.com/mattermost/mattermost-plugin-ai/metrics"
)

// TokenTrackingWrapper wraps a LanguageModel to track token usage in Prometheus metrics
type TokenTrackingWrapper struct {
	wrapped LanguageModel
	metrics metrics.LLMetrics
}

// NewTokenTrackingWrapper creates a new wrapper that tracks token usage
func NewTokenTrackingWrapper(wrapped LanguageModel, metrics metrics.LLMetrics) *TokenTrackingWrapper {
	return &TokenTrackingWrapper{
		wrapped: wrapped,
		metrics: metrics,
	}
}

// ChatCompletion intercepts the streaming response to extract and track token usage
func (w *TokenTrackingWrapper) ChatCompletion(request CompletionRequest, opts ...LanguageModelOption) (*TextStreamResult, error) {
	result, err := w.wrapped.ChatCompletion(request, opts...)
	if err != nil {
		return nil, err
	}

	// Create a new channel to intercept events
	interceptedStream := make(chan TextStreamEvent)

	go func() {
		defer close(interceptedStream)

		for event := range result.Stream {
			// Check for token usage events and track them (but don't forward)
			if event.Type == EventTypeUsage {
				if usage, ok := event.Value.(TokenUsage); ok {
					userID, teamID := w.extractUserContext(request.Context)
					if w.metrics != nil {
						w.metrics.IncrementInputTokens(userID, teamID, usage.InputTokens)
						w.metrics.IncrementOutputTokens(userID, teamID, usage.OutputTokens)
					}
				}
				// Don't forward usage events to consumers
				continue
			}

			// Forward all other events to the consumer
			interceptedStream <- event
		}
	}()

	return &TextStreamResult{Stream: interceptedStream}, nil
}

// ChatCompletionNoStream uses the streaming method internally, so token tracking
// happens automatically when ReadAll() processes the intercepted stream
func (w *TokenTrackingWrapper) ChatCompletionNoStream(request CompletionRequest, opts ...LanguageModelOption) (string, error) {
	result, err := w.ChatCompletion(request, opts...)
	if err != nil {
		return "", err
	}
	return result.ReadAll()
}

// CountTokens delegates to the wrapped model
func (w *TokenTrackingWrapper) CountTokens(text string) int {
	return w.wrapped.CountTokens(text)
}

// InputTokenLimit delegates to the wrapped model
func (w *TokenTrackingWrapper) InputTokenLimit() int {
	return w.wrapped.InputTokenLimit()
}

// extractUserContext extracts user and team IDs from the LLM context for metrics labeling
func (w *TokenTrackingWrapper) extractUserContext(context *Context) (userID, teamID string) {
	if context == nil {
		return "unknown", "unknown"
	}

	if context.RequestingUser != nil {
		userID = context.RequestingUser.Id
	} else {
		userID = "unknown"
	}

	if context.Team != nil {
		teamID = context.Team.Id
	} else {
		teamID = "unknown"
	}

	return userID, teamID
}
