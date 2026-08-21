"""
API Server Module for MineralVision WALDO Integration
====================================================

This module provides the RESTful API endpoints for the WALDO integration.
"""

import base64
import ipaddress
import json
import logging
import os
import secrets
import tempfile
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

from waldo_integration import WALDOIntegrationModule

# WALDO is a private service-to-service API. It deliberately does not enable
# browser CORS. Public browser requests must enter through the authenticated
# MineralVision API and its edge gateway.
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("WALDO_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.environ.get("WALDO_MAX_IMAGE_PIXELS", str(40_000_000)))
WALDO_ENV = os.environ.get("ENV", os.environ.get("ENVIRONMENT", "development")).lower()
WALDO_API_TOKEN = os.environ.get("WALDO_API_TOKEN", "")
if WALDO_ENV == "production" and not WALDO_API_TOKEN:
    raise RuntimeError("WALDO_API_TOKEN is required when ENV=production")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize WALDO integration module
config_path = os.environ.get('CONFIG_PATH', '/etc/mineralvision/waldo/config.yaml')
waldo_module = None

def load_config(config_path):
    """Load configuration from file."""
    try:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            with open(config_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error("Failed to load WALDO configuration")
        if WALDO_ENV == "production":
            raise RuntimeError("WALDO configuration is required in production") from e
        # Development-only fallback; production must use the configured Postgres-backed service.
        return {
            'models_dir': os.environ.get('MODELS_DIR', './models'),
            'model_name': os.environ.get('MODEL_NAME', 'waldo_v3.pt'),
            'confidence_threshold': float(os.environ.get('CONFIDENCE_THRESHOLD', '0.25')),
            'device': os.environ.get('DEVICE', 'cuda'),
            'precision': os.environ.get('PRECISION', 'fp16'),
            'database': {
                'type': 'postgresql',
                'uri': os.environ.get('DATABASE_URI', '')
            },
            'tracker_max_age': int(os.environ.get('TRACKER_MAX_AGE', '10')),
            'tracker_min_hits': int(os.environ.get('TRACKER_MIN_HITS', '3')),
            'tracker_iou_threshold': float(os.environ.get('TRACKER_IOU_THRESHOLD', '0.3')),
            'camera_params': {
                'focal_length': float(os.environ.get('CAMERA_FOCAL_LENGTH', '35.0')),
                'sensor_width': float(os.environ.get('CAMERA_SENSOR_WIDTH', '23.5')),
                'image_width': int(os.environ.get('CAMERA_IMAGE_WIDTH', '1920'))
            }
        }

# Flask 3 removed @app.before_first_request — use lazy one-shot init instead.
def get_waldo_module():
    """Return the WALDO integration module, initializing it on first use."""
    global waldo_module
    if waldo_module is None:
        config = load_config(config_path)
        waldo_module = WALDOIntegrationModule(config)
        logger.info("WALDO integration module initialized")
    return waldo_module


# Record start time at import so /api/status works under any WSGI server.
app.start_time = time.time()


def _detections_to_proxy_shape(detections, max_detections=None):
    """Map internal detection dicts to the MineralVision waldo_proxy
    DetectionResult shape (BoundingBox: x_min/y_min/x_max/y_max/...)."""
    out = []
    for det in detections[:max_detections] if max_detections else detections:
        x1, y1, x2, y2 = det['bbox']
        out.append({
            'x_min': float(x1),
            'y_min': float(y1),
            'x_max': float(x2),
            'y_max': float(y2),
            'confidence': float(det['confidence']),
            'class_name': det.get('class_name', f"class_{det.get('class_id', 0)}"),
            'class_id': int(det.get('class_id', 0)),
        })
    return out

def _service_authorized() -> bool:
    """Validate the private service token without logging caller credentials."""
    return bool(WALDO_API_TOKEN) and secrets.compare_digest(
        request.headers.get("X-Waldo-Service-Token", ""), WALDO_API_TOKEN
    )


@app.before_request
def require_service_authentication():
    """Protect every operational route; only the minimal health probe is public."""
    if request.path == "/health":
        return None
    if not _service_authorized():
        return jsonify({"error": "service authentication required"}), 401
    return None


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "upload exceeds configured size limit"}), 413


@app.errorhandler(Exception)
def unexpected_error(error):
    logger.exception("Unhandled WALDO API error")
    return jsonify({"error": "internal server error"}), 500


def _validated_confidence(value):
    if value is None or value == "":
        return None
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    return confidence


def _decode_image_bytes(data: bytes):
    import cv2
    if not data:
        raise ValueError("image payload is empty")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode image")
    if image.shape[0] * image.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("image exceeds configured pixel limit")
    return image


# API Routes

@app.route('/health', methods=['GET'])
def health():
    """Container healthcheck endpoint (consumed by MineralVision waldo_proxy
    health_check() and docker healthchecks; mirrors /api/status)."""
    return jsonify({'status': 'healthy'})


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get the current system status."""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'uptime': time.time() - app.start_time,
        'gpu_available': os.environ.get('DEVICE', 'cuda') == 'cuda'
    })

@app.route('/api/detection/image', methods=['POST'])
def process_image():
    """Process one authenticated image request entirely in memory."""
    try:
        file = request.files.get("image")
        if file is None or not file.filename:
            return jsonify({"error": "image is required"}), 400
        image = _decode_image_bytes(file.read())
        confidence_threshold = _validated_confidence(request.form.get("confidence_threshold"))
        classes = _parse_classes(request.form.get("classes"))
        response = _run_detection(
            image, secure_filename(file.filename), confidence_threshold, classes, 1000
        )
        return jsonify({
            "request_id": f"req_{uuid.uuid4().hex}",
            "detections": response["detections"],
            "processing_time": response["processing_time_ms"] / 1000.0,
        })
    except (ValueError, TypeError, base64.binascii.Error):
        return jsonify({"error": "invalid image request"}), 400


def _parse_classes(value):
    """Parse classes from JSON string, comma-separated string, or list."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    value = str(value).strip()
    if value.startswith('['):
        return json.loads(value)
    return [c.strip() for c in value.split(',') if c.strip()]


_DETECTOR_LOCK = threading.Lock()


def _decode_json_image(payload):
    """Decode the exact bounded ndarray contract used by the internal API client."""
    encoded = payload.get("image")
    shape = payload.get("shape")
    if not isinstance(encoded, str) or not isinstance(shape, list) or len(shape) not in {2, 3}:
        raise ValueError("JSON detection requires base64 image bytes and a 2D or 3D shape")
    if payload.get("dtype") != "uint8" or not all(isinstance(dimension, int) and dimension > 0 for dimension in shape):
        raise ValueError("only a positive uint8 image array is accepted")
    expected_size = int(np.prod(shape))
    if expected_size > MAX_IMAGE_PIXELS * 3:
        raise ValueError("image exceeds configured pixel limit")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != expected_size:
        raise ValueError("image byte length does not match declared shape")
    return np.frombuffer(raw, dtype=np.uint8).reshape(tuple(shape))


def _run_detection(image, source, confidence_threshold, classes, max_detections):
    """Run one detector request without leaking threshold changes between callers."""
    module = get_waldo_module()
    with _DETECTOR_LOCK:
        original_threshold = module.detector.confidence_threshold
        try:
            if confidence_threshold is not None:
                module.detector.confidence_threshold = confidence_threshold
            result = module.process_frame(image, {"source": source, "timestamp": time.time(), "classes": classes})
        finally:
            module.detector.confidence_threshold = original_threshold
    detections = result["detections"]
    if classes:
        detections = [detection for detection in detections if detection.get("class_name") in classes]
    return {
        "image_id": f"img_{uuid.uuid4().hex}",
        "detections": _detections_to_proxy_shape(detections, max_detections),
        "processing_time_ms": result["processing_time"] * 1000.0,
        "model_version": getattr(module, "model_name", None) or module.config.get("model_name", "waldo"),
    }


@app.route('/detect', methods=['POST'])
def detect():
    """Run bounded private detection for the internal JSON or multipart contract."""
    try:
        if request.is_json:
            payload = request.get_json(force=False, silent=False) or {}
            image = _decode_json_image(payload)
            source = "internal-ndarray"
            confidence_threshold = _validated_confidence(payload.get("confidence_threshold"))
            classes = _parse_classes(payload.get("classes"))
            max_detections = int(payload.get("max_detections", 100))
        else:
            file = request.files.get("image")
            if file is None or not file.filename:
                return jsonify({"error": "image is required"}), 400
            image = _decode_image_bytes(file.read())
            source = secure_filename(file.filename)
            confidence_threshold = _validated_confidence(request.form.get("confidence_threshold"))
            classes = _parse_classes(request.form.get("classes"))
            max_detections = int(request.form.get("max_detections", 100))
        if not 1 <= max_detections <= 1000:
            return jsonify({"error": "max_detections must be between 1 and 1000"}), 400
        return jsonify(_run_detection(image, source, confidence_threshold, classes, max_detections))
    except (ValueError, TypeError, base64.binascii.Error):
        return jsonify({"error": "invalid detection request"}), 400


@app.route('/detect/url', methods=['POST'])
def detect_from_url():
    """Deliberately disabled: server-side URL retrieval is an SSRF risk."""
    return jsonify({"error": "remote URL ingestion is disabled; upload image bytes through /detect"}), 404

WALDO_ENABLE_ASYNC_VIDEO = os.environ.get("WALDO_ENABLE_ASYNC_VIDEO", "").lower() == "true"
VIDEO_JOB_ROOT = Path(os.environ.get("WALDO_JOB_ROOT", "/var/lib/waldo/jobs"))
_VIDEO_JOBS: dict[str, dict] = {}
_VIDEO_JOB_LOCK = threading.Lock()


def _video_job(job_id: str):
    with _VIDEO_JOB_LOCK:
        job = _VIDEO_JOBS.get(job_id)
    if job is None:
        return None
    return job


@app.route('/api/detection/video', methods=['POST'])
def process_video():
    """Start a private, opt-in asynchronous video job with isolated storage."""
    if not WALDO_ENABLE_ASYNC_VIDEO:
        return jsonify({"error": "asynchronous video processing is disabled"}), 404
    file = request.files.get("video")
    if file is None or not file.filename:
        return jsonify({"error": "video is required"}), 400
    try:
        confidence_threshold = _validated_confidence(request.form.get("confidence_threshold"))
        sample_rate = int(request.form.get("sample_rate", "1"))
        if not 1 <= sample_rate <= 60:
            raise ValueError("sample_rate must be between 1 and 60")
        filename = secure_filename(file.filename)
        VIDEO_JOB_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        job_id = f"job_{uuid.uuid4().hex}"
        job_dir = Path(tempfile.mkdtemp(prefix=f"{job_id}-", dir=str(VIDEO_JOB_ROOT)))
        input_path = job_dir / "input"
        file.save(input_path)
        job = {"state": "processing", "directory": job_dir, "input": input_path, "created_at": time.time()}
        with _VIDEO_JOB_LOCK:
            _VIDEO_JOBS[job_id] = job

        def process_video_task():
            try:
                module = get_waldo_module()
                with _DETECTOR_LOCK:
                    original_threshold = module.detector.confidence_threshold
                    try:
                        if confidence_threshold is not None:
                            module.detector.confidence_threshold = confidence_threshold
                        result = module.process_video(str(input_path), {"source": filename, "job_id": job_id, "sample_rate": sample_rate})
                    finally:
                        module.detector.confidence_threshold = original_threshold
                results_path = job_dir / "results.json"
                results_path.write_text(json.dumps(result), encoding="utf-8")
                job.update({"state": "completed", "result": results_path, "summary": result, "completed_at": time.time()})
            except Exception as exc:
                logger.exception("WALDO video job failed")
                job.update({"state": "failed", "error": "video processing failed", "completed_at": time.time()})
            finally:
                input_path.unlink(missing_ok=True)

        threading.Thread(target=process_video_task, daemon=True, name=job_id).start()
        return jsonify({"job_id": job_id, "status": "processing"}), 202
    except (OSError, ValueError):
        return jsonify({"error": "invalid video request"}), 400


@app.route('/api/detection/video/<job_id>', methods=['GET'])
def get_video_status(job_id):
    """Read state only from a server-created random job identifier."""
    job = _video_job(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    if job["state"] == "completed":
        summary = job.get("summary", {})
        return jsonify({"job_id": job_id, "status": "completed", "progress": 100,
                        "frames_processed": summary.get("frames_processed", 0),
                        "processing_time": summary.get("processing_time", 0),
                        "result_url": f"/api/detection/video/{job_id}/results"})
    if job["state"] == "failed":
        return jsonify({"job_id": job_id, "status": "failed", "error": job["error"]}), 500
    return jsonify({"job_id": job_id, "status": "processing", "progress": 0})


@app.route('/api/detection/video/<job_id>/results', methods=['GET'])
def get_video_results(job_id):
    """Return a result only from a completed, in-memory-tracked job."""
    job = _video_job(job_id)
    if job is None or job.get("state") != "completed" or not job.get("result"):
        return jsonify({"error": "completed result not found"}), 404
    return send_file(job["result"], mimetype="application/json", conditional=True)

@app.route('/api/data/detections', methods=['GET'])
def get_detections():
    """Get stored detections with optional filtering."""
    try:
        # Get query parameters
        source = request.args.get('source', None)
        class_name = request.args.get('class_name', None)
        confidence = request.args.get('confidence', None)
        start_time = request.args.get('start_time', None)
        end_time = request.args.get('end_time', None)
        limit = int(request.args.get('limit', '100'))
        offset = int(request.args.get('offset', '0'))
        
        # Build filters
        filters = {}
        if source:
            filters['source'] = source
        if class_name:
            filters['class_name'] = class_name
        if confidence:
            filters['confidence'] = float(confidence)
        if start_time:
            filters['start_time'] = float(start_time)
        if end_time:
            filters['end_time'] = float(end_time)
        
        # Get detections
        detections = get_waldo_module().get_detections(filters, limit)
        
        # Return results
        return jsonify({
            'detections': detections,
            'total_count': len(detections),  # This should be improved to get actual count
            'limit': limit,
            'offset': offset
        })
    
    except Exception as e:
        logger.error(f"Error getting detections: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/arcgis/sync', methods=['POST'])
def sync_to_arcgis():
    """Synchronize detection data to ArcGIS."""
    try:
        # Get request data
        data = request.json
        detection_ids = data.get('detection_ids', None)
        time_range = data.get('time_range', None)
        layer_name = data.get('layer_name', 'WALDO Detections')
        overwrite = data.get('overwrite', False)
        
        # Import ArcGIS connector
        from arcgis_integration import ArcGISConnector
        
        # Get ArcGIS configuration
        arcgis_config = {
            'arcgis_url': os.environ.get('ARCGIS_URL', 'https://arcgis.example.com'),
            'arcgis_username': os.environ.get('ARCGIS_USERNAME', 'mineralvision'),
            'arcgis_password': os.environ.get('ARCGIS_PASSWORD', 'password'),
            'waldo': get_waldo_module().config
        }
        
        # Initialize ArcGIS connector
        arcgis_connector = ArcGISConnector(arcgis_config)
        
        # Sync data
        if detection_ids:
            # Sync specific detections
            result = arcgis_connector.sync_detections_by_id(detection_ids, layer_name, overwrite)
        elif time_range:
            # Sync by time range
            result = arcgis_connector.sync_detections(time_range, layer_name, overwrite)
        else:
            # Sync recent detections
            result = arcgis_connector.sync_recent_detections(layer_name, overwrite)
        
        # Return result
        return jsonify({
            'sync_id': f"sync_{int(time.time())}",
            'status': 'completed',
            'features_synced': result.get('features_synced', 0),
            'layer_url': result.get('layer_url', '')
        })
    
    except Exception as e:
        logger.error(f"Error syncing to ArcGIS: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Main entry point
if __name__ == '__main__':
    # Record start time
    app.start_time = time.time()
    
    # Get port from environment
    port = int(os.environ.get('PORT', 8080))
    
    # Run app
    app.run(host='0.0.0.0', port=port)
