# MineralVision WALDO Integration - API Documentation

## Overview

The MineralVision WALDO Integration API provides programmatic access to object detection, tracking, and measurement capabilities. This RESTful API allows developers to integrate WALDO functionality into their applications and workflows.

## Base URL

- Cloud deployment: `https://waldo.mineralvision.com/api`
- On-premise deployment: `http://your-server-ip:8080/api`
- Edge deployment: `http://edge-device-ip:5000/api`

## Authentication

All API requests require authentication using an API key or JWT token.

### API Key Authentication

Include your API key in the request header:

```
Authorization: ApiKey your_api_key
```

### JWT Authentication

Include your JWT token in the request header:

```
Authorization: Bearer your_jwt_token
```

To obtain a JWT token, use the authentication endpoint:

```
POST /auth/login
{
  "username": "your_username",
  "password": "your_password"
}
```

Response:

```json
{
  "token": "your_jwt_token",
  "expires_at": "2025-04-18T10:38:47Z"
}
```

## API Endpoints

### Detection

#### Process Image

```
POST /detection/image
```

Process a single image for object detection.

**Request Body:**
- `image`: Base64 encoded image data or multipart form data
- `confidence_threshold` (optional): Detection confidence threshold (0.0-1.0)
- `classes` (optional): Array of class names to detect

**Response:**
```json
{
  "request_id": "req_123456",
  "detections": [
    {
      "id": "det_1",
      "bbox": [100, 200, 300, 400],
      "class_id": 0,
      "class_name": "vehicle",
      "confidence": 0.95
    },
    {
      "id": "det_2",
      "bbox": [500, 600, 700, 800],
      "class_id": 1,
      "class_name": "person",
      "confidence": 0.87
    }
  ],
  "processing_time": 0.235
}
```

#### Process Video

```
POST /detection/video
```

Process a video file for object detection and tracking.

**Request Body:**
- `video`: Video file as multipart form data
- `confidence_threshold` (optional): Detection confidence threshold (0.0-1.0)
- `classes` (optional): Array of class names to detect
- `track_objects` (optional): Boolean to enable object tracking
- `sample_rate` (optional): Process every N frames

**Response:**
```json
{
  "job_id": "job_123456",
  "status": "processing",
  "estimated_completion_time": "2025-04-17T11:38:47Z"
}
```

#### Get Video Processing Status

```
GET /detection/video/{job_id}
```

Check the status of a video processing job.

**Response:**
```json
{
  "job_id": "job_123456",
  "status": "completed",
  "progress": 100,
  "frames_processed": 1200,
  "total_frames": 1200,
  "processing_time": 45.6,
  "result_url": "/api/detection/video/job_123456/results"
}
```

#### Get Video Processing Results

```
GET /detection/video/{job_id}/results
```

Get the results of a completed video processing job.

**Response:**
```json
{
  "job_id": "job_123456",
  "frames": [
    {
      "frame_id": 0,
      "timestamp": 0.0,
      "detections": [
        {
          "id": "det_1_0",
          "track_id": 1,
          "bbox": [100, 200, 300, 400],
          "class_id": 0,
          "class_name": "vehicle",
          "confidence": 0.95
        }
      ]
    },
    {
      "frame_id": 1,
      "timestamp": 0.033,
      "detections": [
        {
          "id": "det_1_1",
          "track_id": 1,
          "bbox": [102, 202, 302, 402],
          "class_id": 0,
          "class_name": "vehicle",
          "confidence": 0.94
        }
      ]
    }
  ],
  "tracks": [
    {
      "track_id": 1,
      "class_name": "vehicle",
      "first_frame": 0,
      "last_frame": 1200,
      "average_confidence": 0.93
    }
  ]
}
```

### Measurements

#### Calculate Measurements

```
POST /measurements/calculate
```

Calculate measurements for detected objects.

**Request Body:**
- `detections`: Array of detection objects
- `camera_params`: Camera parameters object
- `altitude`: Altitude in meters
- `reference_objects` (optional): Known object dimensions for reference

**Response:**
```json
{
  "measured_objects": [
    {
      "id": "det_1",
      "bbox": [100, 200, 300, 400],
      "class_name": "vehicle",
      "measurements": {
        "width_m": 2.5,
        "height_m": 1.8,
        "area_m2": 4.5,
        "distance_estimate_m": 15.3
      }
    }
  ]
}
```

### Tracking

#### Update Tracks

```
POST /tracking/update
```

Update object tracks with new detections.

**Request Body:**
- `tracks`: Array of existing track objects
- `detections`: Array of new detection objects
- `frame_id`: Current frame ID
- `timestamp`: Current timestamp

**Response:**
```json
{
  "tracks": [
    {
      "track_id": 1,
      "bbox": [102, 202, 302, 402],
      "class_name": "vehicle",
      "confidence": 0.94,
      "age": 0,
      "hits": 2,
      "history": [
        [100, 200, 300, 400],
        [102, 202, 302, 402]
      ]
    }
  ],
  "new_tracks": [
    {
      "track_id": 2,
      "bbox": [500, 600, 700, 800],
      "class_name": "person",
      "confidence": 0.87,
      "age": 0,
      "hits": 1,
      "history": [
        [500, 600, 700, 800]
      ]
    }
  ]
}
```

### Data Management

#### Get Detections

```
GET /data/detections
```

Get stored detections with optional filtering.

**Query Parameters:**
- `source` (optional): Filter by source
- `class_name` (optional): Filter by class name
- `confidence` (optional): Minimum confidence threshold
- `start_time` (optional): Start time for filtering
- `end_time` (optional): End time for filtering
- `limit` (optional): Maximum number of results
- `offset` (optional): Pagination offset

**Response:**
```json
{
  "detections": [
    {
      "id": "det_1",
      "track_id": 1,
      "class_name": "vehicle",
      "confidence": 0.95,
      "bbox": [100, 200, 300, 400],
      "timestamp": "2025-04-17T10:30:00Z",
      "source": "camera_1",
      "measurements": {
        "width_m": 2.5,
        "height_m": 1.8,
        "area_m2": 4.5
      }
    }
  ],
  "total_count": 1256,
  "limit": 1,
  "offset": 0
}
```

#### Get Detection by ID

```
GET /data/detections/{detection_id}
```

Get a specific detection by ID.

**Response:**
```json
{
  "id": "det_1",
  "track_id": 1,
  "class_name": "vehicle",
  "confidence": 0.95,
  "bbox": [100, 200, 300, 400],
  "timestamp": "2025-04-17T10:30:00Z",
  "source": "camera_1",
  "measurements": {
    "width_m": 2.5,
    "height_m": 1.8,
    "area_m2": 4.5
  },
  "metadata": {
    "frame_id": 120,
    "camera_id": "cam_1",
    "location": {
      "latitude": 37.7749,
      "longitude": -122.4194
    }
  }
}
```

### ArcGIS Integration

#### Sync to ArcGIS

```
POST /arcgis/sync
```

Synchronize detection data to ArcGIS.

**Request Body:**
- `detection_ids` (optional): Array of detection IDs to sync
- `time_range` (optional): Time range to sync
- `layer_name` (optional): Name of the ArcGIS layer
- `overwrite` (optional): Boolean to overwrite existing features

**Response:**
```json
{
  "sync_id": "sync_123456",
  "status": "completed",
  "features_synced": 256,
  "layer_url": "https://arcgis.example.com/layers/waldo_detections"
}
```

#### Get ArcGIS Layers

```
GET /arcgis/layers
```

Get available ArcGIS layers.

**Response:**
```json
{
  "layers": [
    {
      "id": "layer_1",
      "name": "WALDO Detections",
      "feature_count": 1256,
      "last_updated": "2025-04-17T10:00:00Z",
      "url": "https://arcgis.example.com/layers/waldo_detections"
    }
  ]
}
```

### System

#### Get System Status

```
GET /system/status
```

Get the current system status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.2.3",
  "uptime": 86400,
  "gpu_available": true,
  "gpu_utilization": 45.2,
  "memory_utilization": 62.8,
  "storage_utilization": 38.5,
  "active_jobs": 2,
  "queued_jobs": 1
}
```

## Error Handling

The API uses standard HTTP status codes to indicate the success or failure of requests.

### Error Response Format

```json
{
  "error": {
    "code": "invalid_request",
    "message": "Invalid request parameters",
    "details": "Parameter 'confidence_threshold' must be between 0.0 and 1.0"
  }
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request parameters
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## Rate Limiting

API requests are subject to rate limiting to ensure fair usage. Rate limits are applied per API key or user.

Rate limit headers are included in all API responses:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1650000000
```

## Webhooks

The API supports webhooks for asynchronous notifications about events.

### Register Webhook

```
POST /webhooks/register
```

Register a new webhook.

**Request Body:**
- `url`: Webhook URL
- `events`: Array of event types to subscribe to
- `secret` (optional): Secret for webhook signature verification

**Response:**
```json
{
  "webhook_id": "webhook_123456",
  "url": "https://your-server.com/webhook",
  "events": ["detection.completed", "job.completed"],
  "created_at": "2025-04-17T10:38:47Z"
}
```

### Webhook Event Types

- `detection.completed`: Detection processing completed
- `job.completed`: Video processing job completed
- `sync.completed`: ArcGIS synchronization completed
- `system.alert`: System alert or notification

### Webhook Payload Format

```json
{
  "event_type": "job.completed",
  "timestamp": "2025-04-17T10:38:47Z",
  "webhook_id": "webhook_123456",
  "data": {
    "job_id": "job_123456",
    "status": "completed",
    "frames_processed": 1200,
    "processing_time": 45.6,
    "result_url": "/api/detection/video/job_123456/results"
  }
}
```

## SDK and Client Libraries

Official client libraries are available for:

- Python: `pip install mineralvision-waldo-client`
- JavaScript: `npm install mineralvision-waldo-client`
- Java: Available on Maven Central
- C#: Available on NuGet

## API Versioning

The API is versioned to ensure backward compatibility. The current version is v1.

To specify a version, include it in the URL path:

```
https://waldo.mineralvision.com/api/v1/detection/image
```

If no version is specified, the latest version is used.

## Support

For API support, contact api-support@mineralvision.com or visit the developer portal at https://developers.mineralvision.com.
