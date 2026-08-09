// MineralVision Orchestrator - Go Service
//
// This service provides the Temporal workflow orchestration layer for user journeys,
// integrating with all middleware components (Kafka, Dapr, Fluvio, Keycloak, Permify,
// Redis, APISIX, TigerBeetle, Lakehouse).
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/mineralvision/orchestrator/internal/activities"
	"github.com/mineralvision/orchestrator/internal/journeys"
	"github.com/mineralvision/orchestrator/internal/middleware"
	"github.com/mineralvision/orchestrator/internal/workflows"
	"go.temporal.io/sdk/client"
	"go.temporal.io/sdk/worker"
	"go.uber.org/zap"
)

func main() {
	// Initialize logger
	logger, _ := zap.NewProduction()
	defer logger.Sync()
	sugar := logger.Sugar()

	// Load configuration
	config := loadConfig()

	// Initialize middleware connections
	mw, err := middleware.NewMiddlewareClient(config.Middleware)
	if err != nil {
		sugar.Warnw("Failed to connect to some middleware", "error", err)
	}

	// Connect to Temporal
	temporalClient, err := client.Dial(client.Options{
		HostPort:  config.Temporal.Address,
		Namespace: config.Temporal.Namespace,
	})
	if err != nil {
		sugar.Warnw("Failed to connect to Temporal, running in mock mode", "error", err)
		temporalClient = nil
	}

	// Load journey registry
	registry := journeys.NewRegistry()
	sugar.Infow("Loaded journeys", "count", registry.Count())

	// Create worker if Temporal is available
	var w worker.Worker
	if temporalClient != nil {
		w = worker.New(temporalClient, config.Temporal.TaskQueue, worker.Options{})

		// Register workflows
		w.RegisterWorkflow(workflows.JourneyWorkflow)
		w.RegisterWorkflow(workflows.DataIngestionWorkflow)
		w.RegisterWorkflow(workflows.ModelTrainingWorkflow)
		w.RegisterWorkflow(workflows.DigitalTwinWorkflow)

		// Register activities
		acts := activities.NewActivities(mw, config.API.BaseURL)
		w.RegisterActivity(acts.CallAPIEndpoint)
		w.RegisterActivity(acts.RunMLInference)
		w.RegisterActivity(acts.PublishKafkaEvent)
		w.RegisterActivity(acts.PublishFluvioEvent)
		w.RegisterActivity(acts.CheckPermission)
		w.RegisterActivity(acts.WriteLedgerEntry)
		w.RegisterActivity(acts.StoreToLakehouse)
		w.RegisterActivity(acts.CacheToRedis)

		// Start worker in background
		go func() {
			if err := w.Run(worker.InterruptCh()); err != nil {
				sugar.Errorw("Worker failed", "error", err)
			}
		}()
	}

	// Create HTTP server
	router := setupRouter(temporalClient, registry, mw, sugar)

	srv := &http.Server{
		Addr:    ":" + config.Server.Port,
		Handler: router,
	}

	// Start server in goroutine
	go func() {
		sugar.Infow("Starting orchestrator server", "port", config.Server.Port)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			sugar.Fatalw("Server failed", "error", err)
		}
	}()

	// Wait for interrupt signal
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	sugar.Info("Shutting down server...")

	// Graceful shutdown
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		sugar.Fatalw("Server forced to shutdown", "error", err)
	}

	if w != nil {
		w.Stop()
	}

	if temporalClient != nil {
		temporalClient.Close()
	}

	sugar.Info("Server exited")
}

func setupRouter(
	temporalClient client.Client,
	registry *journeys.Registry,
	mw *middleware.Client,
	logger *zap.SugaredLogger,
) *gin.Engine {
	router := gin.Default()

	// Health check
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":    "healthy",
			"timestamp": time.Now().UTC().Format(time.RFC3339),
			"version":   "1.0.0",
		})
	})

	// API routes
	api := router.Group("/api/orchestrator")
	{
		// Journey management
		api.GET("/journeys", func(c *gin.Context) {
			category := c.Query("category")
			journeys := registry.List(category)
			c.JSON(http.StatusOK, gin.H{
				"journeys": journeys,
				"total":    len(journeys),
			})
		})

		api.GET("/journeys/:id", func(c *gin.Context) {
			id := c.Param("id")
			journey := registry.Get(id)
			if journey == nil {
				c.JSON(http.StatusNotFound, gin.H{"error": "Journey not found"})
				return
			}
			c.JSON(http.StatusOK, journey)
		})

		// Workflow execution
		api.POST("/journeys/:id/start", func(c *gin.Context) {
			id := c.Param("id")
			journey := registry.Get(id)
			if journey == nil {
				c.JSON(http.StatusNotFound, gin.H{"error": "Journey not found"})
				return
			}

			var req struct {
				ProjectID string                 `json:"project_id"`
				UserID    string                 `json:"user_id"`
				Inputs    map[string]interface{} `json:"inputs"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			// Generate workflow ID
			workflowID := journeys.GenerateWorkflowID(id)

			if temporalClient == nil {
				// Mock mode
				c.JSON(http.StatusOK, gin.H{
					"workflow_id": workflowID,
					"run_id":      "mock-run-" + workflowID,
					"journey_id":  id,
					"status":      "running",
					"started_at":  time.Now().UTC().Format(time.RFC3339),
				})
				return
			}

			// Start workflow
			options := client.StartWorkflowOptions{
				ID:        workflowID,
				TaskQueue: "mineralvision-journeys",
			}

			we, err := temporalClient.ExecuteWorkflow(
				context.Background(),
				options,
				workflows.JourneyWorkflow,
				workflows.JourneyInput{
					JourneyID: id,
					ProjectID: req.ProjectID,
					UserID:    req.UserID,
					Inputs:    req.Inputs,
					Steps:     journey.Steps,
				},
			)
			if err != nil {
				logger.Errorw("Failed to start workflow", "error", err)
				c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to start workflow"})
				return
			}

			// Publish event to Kafka
			go mw.PublishKafka("mineralvision.journeys.started", map[string]interface{}{
				"workflow_id": we.GetID(),
				"run_id":      we.GetRunID(),
				"journey_id":  id,
				"project_id":  req.ProjectID,
				"user_id":     req.UserID,
			})

			c.JSON(http.StatusOK, gin.H{
				"workflow_id": we.GetID(),
				"run_id":      we.GetRunID(),
				"journey_id":  id,
				"status":      "running",
				"started_at":  time.Now().UTC().Format(time.RFC3339),
			})
		})

		api.GET("/runs/:workflow_id", func(c *gin.Context) {
			workflowID := c.Param("workflow_id")

			if temporalClient == nil {
				c.JSON(http.StatusOK, gin.H{
					"workflow_id": workflowID,
					"status":      "completed",
					"mock":        true,
				})
				return
			}

			handle := temporalClient.GetWorkflow(context.Background(), workflowID, "")
			desc, err := handle.Describe(context.Background())
			if err != nil {
				c.JSON(http.StatusNotFound, gin.H{"error": "Workflow not found"})
				return
			}

			c.JSON(http.StatusOK, gin.H{
				"workflow_id":  workflowID,
				"run_id":       desc.WorkflowExecution.RunID,
				"status":       desc.WorkflowExecutionInfo.Status.String(),
				"started_at":   desc.WorkflowExecutionInfo.StartTime.Format(time.RFC3339),
				"completed_at": desc.WorkflowExecutionInfo.CloseTime.Format(time.RFC3339),
			})
		})

		api.POST("/runs/:workflow_id/signal", func(c *gin.Context) {
			workflowID := c.Param("workflow_id")

			var req struct {
				SignalName string                 `json:"signal_name"`
				Args       map[string]interface{} `json:"args"`
			}
			if err := c.ShouldBindJSON(&req); err != nil {
				c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
				return
			}

			if temporalClient == nil {
				c.JSON(http.StatusOK, gin.H{"status": "mock signal sent"})
				return
			}

			err := temporalClient.SignalWorkflow(
				context.Background(),
				workflowID,
				"",
				req.SignalName,
				req.Args,
			)
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"status": "signal sent"})
		})

		api.POST("/runs/:workflow_id/cancel", func(c *gin.Context) {
			workflowID := c.Param("workflow_id")

			if temporalClient == nil {
				c.JSON(http.StatusOK, gin.H{"status": "mock cancelled"})
				return
			}

			err := temporalClient.CancelWorkflow(context.Background(), workflowID, "")
			if err != nil {
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			c.JSON(http.StatusOK, gin.H{"status": "cancelled"})
		})

		// Middleware status
		api.GET("/middleware/status", func(c *gin.Context) {
			status := mw.GetStatus()
			c.JSON(http.StatusOK, status)
		})
	}

	return router
}

type Config struct {
	Server struct {
		Port string
	}
	Temporal struct {
		Address   string
		Namespace string
		TaskQueue string
	}
	API struct {
		BaseURL string
	}
	Middleware middleware.Config
}

func loadConfig() *Config {
	config := &Config{}

	// Server
	config.Server.Port = getEnv("ORCHESTRATOR_PORT", "8090")

	// Temporal
	config.Temporal.Address = getEnv("TEMPORAL_ADDRESS", "localhost:7233")
	config.Temporal.Namespace = getEnv("TEMPORAL_NAMESPACE", "mineralvision")
	config.Temporal.TaskQueue = getEnv("TEMPORAL_TASK_QUEUE", "mineralvision-journeys")

	// API
	config.API.BaseURL = getEnv("API_BASE_URL", "http://localhost:8000")

	// Middleware
	config.Middleware = middleware.Config{
		KafkaBootstrapServers: getEnv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
		FluvioEndpoint:        getEnv("FLUVIO_ENDPOINT", "localhost:9003"),
		RedisURL:              getEnv("REDIS_URL", "redis://localhost:6379"),
		KeycloakURL:           getEnv("KEYCLOAK_URL", "http://localhost:8080"),
		KeycloakRealm:         getEnv("KEYCLOAK_REALM", "mineralvision"),
		PermifyURL:            getEnv("PERMIFY_URL", "http://localhost:3476"),
		DaprHTTPPort:          getEnv("DAPR_HTTP_PORT", "3500"),
		TigerBeetleAddresses:  getEnv("TIGERBEETLE_ADDRESSES", "127.0.0.1:3000"),
		LakehouseWarehouse:    getEnv("LAKEHOUSE_WAREHOUSE", "s3://mineralvision-lakehouse"),
	}

	return config
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
