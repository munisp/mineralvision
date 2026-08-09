// Package workflows provides Temporal workflow definitions
package workflows

import (
	"fmt"
	"time"

	"github.com/mineralvision/orchestrator/internal/journeys"
	"go.temporal.io/sdk/temporal"
	"go.temporal.io/sdk/workflow"
)

// JourneyInput contains the input for a journey workflow
type JourneyInput struct {
	JourneyID string                 `json:"journey_id"`
	ProjectID string                 `json:"project_id"`
	UserID    string                 `json:"user_id"`
	Inputs    map[string]interface{} `json:"inputs"`
	Steps     []journeys.Step        `json:"steps"`
}

// JourneyOutput contains the output of a journey workflow
type JourneyOutput struct {
	JourneyID   string                 `json:"journey_id"`
	Status      string                 `json:"status"`
	CompletedAt string                 `json:"completed_at"`
	Outputs     map[string]interface{} `json:"outputs"`
	Error       string                 `json:"error,omitempty"`
}

// StepResult contains the result of a single step
type StepResult struct {
	StepID      string                 `json:"step_id"`
	Status      string                 `json:"status"`
	Output      map[string]interface{} `json:"output"`
	Error       string                 `json:"error,omitempty"`
	CompletedAt string                 `json:"completed_at"`
}

// ApprovalSignal is sent when a human approves/rejects a step
type ApprovalSignal struct {
	StepID   string `json:"step_id"`
	Approved bool   `json:"approved"`
	Comment  string `json:"comment,omitempty"`
}

// JourneyWorkflow orchestrates the execution of a user journey
func JourneyWorkflow(ctx workflow.Context, input JourneyInput) (*JourneyOutput, error) {
	logger := workflow.GetLogger(ctx)
	logger.Info("Starting journey workflow", "journey_id", input.JourneyID)

	// Set up activity options with retries
	activityOptions := workflow.ActivityOptions{
		StartToCloseTimeout: 5 * time.Minute,
		RetryPolicy: &temporal.RetryPolicy{
			InitialInterval:    time.Second,
			BackoffCoefficient: 2.0,
			MaximumInterval:    time.Minute,
			MaximumAttempts:    3,
		},
	}
	ctx = workflow.WithActivityOptions(ctx, activityOptions)

	// Track outputs from each step
	stepOutputs := make(map[string]interface{})
	var lastError error

	// Set up signal channel for approvals
	approvalChan := workflow.GetSignalChannel(ctx, "step_approval")

	// Set up query handler for current step
	currentStep := ""
	err := workflow.SetQueryHandler(ctx, "current_step", func() (map[string]interface{}, error) {
		return map[string]interface{}{
			"step_id":    currentStep,
			"journey_id": input.JourneyID,
		}, nil
	})
	if err != nil {
		logger.Error("Failed to set query handler", "error", err)
	}

	// Execute each step in sequence
	for _, step := range input.Steps {
		currentStep = step.ID
		logger.Info("Executing step", "step_id", step.ID, "step_name", step.Name)

		// Check permission if required
		if step.PermissionCheck != "" {
			var hasPermission bool
			err := workflow.ExecuteActivity(ctx, "CheckPermission", input.UserID, step.PermissionCheck, input.ProjectID).Get(ctx, &hasPermission)
			if err != nil {
				logger.Error("Permission check failed", "error", err)
				lastError = err
				break
			}
			if !hasPermission {
				lastError = fmt.Errorf("user %s does not have permission %s", input.UserID, step.PermissionCheck)
				break
			}
		}

		// Handle human approval steps
		if step.RequiresApproval {
			logger.Info("Waiting for human approval", "step_id", step.ID)

			// Wait for approval signal with timeout
			var approval ApprovalSignal
			timeout := time.Duration(step.TimeoutSeconds) * time.Second
			if timeout == 0 {
				timeout = 24 * time.Hour
			}

			selector := workflow.NewSelector(ctx)
			timedOut := false

			selector.AddReceive(approvalChan, func(c workflow.ReceiveChannel, more bool) {
				c.Receive(ctx, &approval)
			})

			selector.AddFuture(workflow.NewTimer(ctx, timeout), func(f workflow.Future) {
				timedOut = true
			})

			selector.Select(ctx)

			if timedOut {
				lastError = fmt.Errorf("approval timeout for step %s", step.ID)
				break
			}

			if !approval.Approved {
				lastError = fmt.Errorf("step %s was rejected: %s", step.ID, approval.Comment)
				break
			}

			stepOutputs[step.ID] = map[string]interface{}{
				"approved": true,
				"comment":  approval.Comment,
			}
			continue
		}

		// Execute the step based on type
		var result StepResult
		stepTimeout := time.Duration(step.TimeoutSeconds) * time.Second
		if stepTimeout == 0 {
			stepTimeout = 5 * time.Minute
		}

		stepCtx := workflow.WithActivityOptions(ctx, workflow.ActivityOptions{
			StartToCloseTimeout: stepTimeout,
			RetryPolicy: &temporal.RetryPolicy{
				InitialInterval:    time.Second,
				BackoffCoefficient: 2.0,
				MaximumInterval:    time.Minute,
				MaximumAttempts:    int32(step.RetryCount),
			},
		})

		switch step.StepType {
		case journeys.StepTypeAPICall, journeys.StepTypeDataIngestion, journeys.StepTypeReportGeneration, journeys.StepTypeVisualization, journeys.StepTypeBlockchainRecord:
			// Call API endpoint
			payload := resolveInputMapping(step.InputMapping, stepOutputs, input.Inputs)
			err := workflow.ExecuteActivity(stepCtx, "CallAPIEndpoint", step.Endpoint, step.Method, payload).Get(ctx, &result.Output)
			if err != nil {
				result.Error = err.Error()
				result.Status = "failed"
			} else {
				result.Status = "completed"
			}

		case journeys.StepTypeMLInference, journeys.StepTypeSensorFusion:
			// Run ML/sensor module
			inputs := resolveInputMapping(step.InputMapping, stepOutputs, input.Inputs)
			err := workflow.ExecuteActivity(stepCtx, "RunMLInference", step.Module, "run", inputs).Get(ctx, &result.Output)
			if err != nil {
				result.Error = err.Error()
				result.Status = "failed"
			} else {
				result.Status = "completed"
			}

		case journeys.StepTypeEventPublish:
			// Publish event
			payload := resolveInputMapping(step.InputMapping, stepOutputs, input.Inputs)
			if step.KafkaTopic != "" {
				err := workflow.ExecuteActivity(stepCtx, "PublishKafkaEvent", step.KafkaTopic, payload).Get(ctx, nil)
				if err != nil {
					result.Error = err.Error()
					result.Status = "failed"
				} else {
					result.Status = "completed"
				}
			}
			if step.FluvioTopic != "" {
				err := workflow.ExecuteActivity(stepCtx, "PublishFluvioEvent", step.FluvioTopic, payload).Get(ctx, nil)
				if err != nil {
					result.Error = err.Error()
					result.Status = "failed"
				} else {
					result.Status = "completed"
				}
			}

		case journeys.StepTypeLedgerWrite:
			// Write to ledger
			data := resolveInputMapping(step.InputMapping, stepOutputs, input.Inputs)
			var entryID string
			err := workflow.ExecuteActivity(stepCtx, "WriteLedgerEntry", step.LedgerEntryType, data).Get(ctx, &entryID)
			if err != nil {
				result.Error = err.Error()
				result.Status = "failed"
			} else {
				result.Output = map[string]interface{}{"entry_id": entryID}
				result.Status = "completed"
			}

		default:
			result.Status = "skipped"
			result.Output = map[string]interface{}{"reason": "unknown step type"}
		}

		result.StepID = step.ID
		result.CompletedAt = time.Now().UTC().Format(time.RFC3339)

		// Store step output
		stepOutputs[step.ID] = result.Output

		// Publish step completion event
		if step.KafkaTopic != "" && result.Status == "completed" {
			_ = workflow.ExecuteActivity(ctx, "PublishKafkaEvent", step.KafkaTopic, map[string]interface{}{
				"step_id":   step.ID,
				"status":    result.Status,
				"output":    result.Output,
				"timestamp": result.CompletedAt,
			}).Get(ctx, nil)
		}

		// Write ledger entry if specified
		if step.LedgerEntryType != "" && result.Status == "completed" {
			_ = workflow.ExecuteActivity(ctx, "WriteLedgerEntry", step.LedgerEntryType, map[string]interface{}{
				"step_id":    step.ID,
				"journey_id": input.JourneyID,
				"project_id": input.ProjectID,
				"user_id":    input.UserID,
				"output":     result.Output,
				"timestamp":  result.CompletedAt,
			}).Get(ctx, nil)
		}

		// Check for failure
		if result.Status == "failed" {
			lastError = fmt.Errorf("step %s failed: %s", step.ID, result.Error)
			break
		}

		logger.Info("Step completed", "step_id", step.ID, "status", result.Status)
	}

	// Build output
	output := &JourneyOutput{
		JourneyID:   input.JourneyID,
		CompletedAt: time.Now().UTC().Format(time.RFC3339),
		Outputs:     stepOutputs,
	}

	if lastError != nil {
		output.Status = "failed"
		output.Error = lastError.Error()
		logger.Error("Journey failed", "error", lastError)
	} else {
		output.Status = "completed"
		logger.Info("Journey completed successfully")
	}

	return output, lastError
}

// resolveInputMapping resolves input mappings from previous step outputs
func resolveInputMapping(mapping map[string]string, stepOutputs map[string]interface{}, inputs map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})

	// Copy direct inputs
	for k, v := range inputs {
		result[k] = v
	}

	// Resolve mappings (e.g., "$.step-001-1.id" -> value from step output)
	for key, path := range mapping {
		// Simple path resolution - in production would use jsonpath
		if len(path) > 2 && path[:2] == "$." {
			parts := path[2:]
			// For now, just use the path as a key
			if val, ok := stepOutputs[parts]; ok {
				result[key] = val
			}
		} else {
			result[key] = path
		}
	}

	return result
}

// DataIngestionWorkflow is a specialized workflow for data ingestion journeys
func DataIngestionWorkflow(ctx workflow.Context, input JourneyInput) (*JourneyOutput, error) {
	// Reuse the generic journey workflow
	return JourneyWorkflow(ctx, input)
}

// ModelTrainingWorkflow is a specialized workflow for ML training journeys
func ModelTrainingWorkflow(ctx workflow.Context, input JourneyInput) (*JourneyOutput, error) {
	// Reuse the generic journey workflow with longer timeouts
	return JourneyWorkflow(ctx, input)
}

// DigitalTwinWorkflow is a specialized workflow for digital twin sessions
func DigitalTwinWorkflow(ctx workflow.Context, input JourneyInput) (*JourneyOutput, error) {
	// Reuse the generic journey workflow
	return JourneyWorkflow(ctx, input)
}
