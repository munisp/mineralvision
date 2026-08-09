// Package activities provides Temporal activity implementations
package activities

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/mineralvision/orchestrator/internal/middleware"
)

// Activities holds the activity implementations
type Activities struct {
	middleware *middleware.Client
	apiBaseURL string
	httpClient *http.Client
}

// NewActivities creates a new Activities instance
func NewActivities(mw *middleware.Client, apiBaseURL string) *Activities {
	return &Activities{
		middleware: mw,
		apiBaseURL: apiBaseURL,
		httpClient: &http.Client{
			Timeout: 5 * time.Minute,
		},
	}
}

// CallAPIEndpoint calls a FastAPI endpoint
func (a *Activities) CallAPIEndpoint(ctx context.Context, endpoint, method string, payload map[string]interface{}) (map[string]interface{}, error) {
	url := a.apiBaseURL + endpoint

	var body io.Reader
	if payload != nil && (method == "POST" || method == "PUT" || method == "PATCH") {
		jsonData, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal payload: %w", err)
		}
		body = bytes.NewReader(jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := a.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("API error %d: %s", resp.StatusCode, string(respBody))
	}

	var result map[string]interface{}
	if len(respBody) > 0 {
		if err := json.Unmarshal(respBody, &result); err != nil {
			// Try to return as string if not JSON
			result = map[string]interface{}{"response": string(respBody)}
		}
	}

	return result, nil
}

// RunMLInference runs an ML inference module
func (a *Activities) RunMLInference(ctx context.Context, modulePath, functionName string, inputs map[string]interface{}) (map[string]interface{}, error) {
	// In production, this would invoke the Python module via RPC or subprocess
	// For now, we call the API endpoint that wraps the module

	// Map module paths to API endpoints
	endpointMap := map[string]string{
		"src.api.ml.gold_exploration":            "/api/predictive-modeling/gold",
		"src.api.ml.lithium_exploration":         "/api/predictive-modeling/lithium",
		"src.api.ml.soil_suitability":            "/api/predictive-modeling/soil",
		"src.api.ml.advanced_soil_assessment":    "/api/predictive-modeling/soil/advanced",
		"src.api.ml.uncertainty_quantification":  "/api/predictive-modeling/uncertainty",
		"src.api.ml.spatial_cv":                  "/api/predictive-modeling/spatial-cv",
		"src.api.ml.prospectivity_workflow":      "/api/predictive-modeling/prospectivity",
		"src.api.sensor_fusion.kalman_fusion":    "/api/sensor-fusion/kalman",
		"src.api.sensor_fusion.deep_learning_fusion": "/api/sensor-fusion/deep-learning",
		"src.api.sensor_fusion.magnetometry_pipeline": "/api/sensor-fusion/magnetometry",
		"src.api.sensor_fusion.radiometrics_pipeline": "/api/sensor-fusion/radiometrics",
		"src.api.sensor_fusion.lidar_adapter":    "/api/sensor-fusion/lidar",
		"src.api.sensor_fusion.gpr_pipeline":     "/api/sensor-fusion/gpr",
		"src.api.sensor_fusion.segy_ingestion":   "/api/sensor-fusion/segy",
		"src.api.sensor_fusion.tiledb_segy":      "/api/sensor-fusion/segy/tiledb",
		"src.api.sensor_fusion.segy_visualization": "/api/sensor-fusion/segy/visualize",
		"src.api.sensor_fusion.streaming_fusion": "/api/sensor-fusion/stream",
		"src.api.sensor_fusion.drone_telemetry":  "/api/sensor-fusion/drone/telemetry",
		"src.api.sensor_fusion.drone_gpr":        "/api/sensor-fusion/drone/gpr",
		"src.api.molmo.ensemble_pipeline":        "/api/molmo/ensemble",
		"src.api.molmo.drone_video_analysis":     "/api/molmo/drone-video",
		"src.api.vision.sam3.sam3_segmenter":     "/api/sam3/segment",
		"src.api.jepa.vjepa_integration":         "/api/jepa/extract",
		"src.api.jepa.lakehouse_integration":     "/api/jepa/store",
		"src.api.waldo.ensemble_detector":        "/api/waldo/detect",
		"src.api.digital_twin.visualization_3d":  "/api/digital-twin/visualize",
		"src.api.ingestion.lims_ingestion":       "/api/upload/lims",
		"src.api.ingestion.gnss_ingestion":       "/api/upload/gnss",
		"src.api.ingestion.lidar_ingestion":      "/api/upload/lidar",
	}

	endpoint, ok := endpointMap[modulePath]
	if !ok {
		// Default to a generic inference endpoint
		endpoint = "/api/ml/inference"
		inputs["module"] = modulePath
		inputs["function"] = functionName
	}

	return a.CallAPIEndpoint(ctx, endpoint, "POST", inputs)
}

// PublishKafkaEvent publishes an event to Kafka
func (a *Activities) PublishKafkaEvent(ctx context.Context, topic string, event map[string]interface{}) error {
	if a.middleware == nil {
		return nil // Mock mode
	}
	return a.middleware.PublishKafka(topic, event)
}

// PublishFluvioEvent publishes an event to Fluvio
func (a *Activities) PublishFluvioEvent(ctx context.Context, topic string, event map[string]interface{}) error {
	if a.middleware == nil {
		return nil // Mock mode
	}
	return a.middleware.PublishFluvio(topic, event)
}

// CheckPermission checks if a user has a permission
func (a *Activities) CheckPermission(ctx context.Context, userID, permission, resourceID string) (bool, error) {
	if a.middleware == nil {
		return true, nil // Allow in mock mode
	}
	return a.middleware.CheckPermission(userID, permission, resourceID)
}

// WriteLedgerEntry writes an entry to TigerBeetle
func (a *Activities) WriteLedgerEntry(ctx context.Context, entryType string, data map[string]interface{}) (string, error) {
	if a.middleware == nil {
		return fmt.Sprintf("mock-entry-%d", time.Now().UnixNano()), nil
	}
	return a.middleware.WriteLedgerEntry(entryType, data)
}

// StoreToLakehouse stores data to the lakehouse
func (a *Activities) StoreToLakehouse(ctx context.Context, table string, data map[string]interface{}) (string, error) {
	if a.middleware == nil {
		return fmt.Sprintf("mock-record-%d", time.Now().UnixNano()), nil
	}
	return a.middleware.StoreToLakehouse(table, data)
}

// CacheToRedis caches data in Redis
func (a *Activities) CacheToRedis(ctx context.Context, key string, value interface{}, ttlSeconds int) error {
	if a.middleware == nil {
		return nil // Mock mode
	}
	return a.middleware.CacheToRedis(key, value, ttlSeconds)
}
