"""
API Server Module for MineralVision WALDO Integration
====================================================

This module provides the RESTful API endpoints for the WALDO integration.
"""

import os
import time
import json
import logging
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from waldo_integration import WALDOIntegrationModule

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
        logger.error(f"Failed to load configuration: {str(e)}")
        # Return default configuration
        return {
            'models_dir': os.environ.get('MODELS_DIR', './models'),
            'model_name': os.environ.get('MODEL_NAME', 'waldo_v3.pt'),
            'confidence_threshold': float(os.environ.get('CONFIDENCE_THRESHOLD', '0.25')),
            'device': os.environ.get('DEVICE', 'cuda'),
            'precision': os.environ.get('PRECISION', 'fp16'),
            'database': {
                'type': 'sqlite',
                'path': os.environ.get('DATABASE_PATH', ':memory:')
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
    """Process a single image for object detection."""
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get parameters
        confidence_threshold = request.form.get('confidence_threshold', None)
        if confidence_threshold:
            confidence_threshold = float(confidence_threshold)
        
        classes = request.form.get('classes', None)
        if classes:
            classes = json.loads(classes)
        
        # Save image temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join('/tmp', filename)
        file.save(temp_path)
        
        # Read image
        import cv2
        image = cv2.imread(temp_path)
        if image is None:
            return jsonify({'error': 'Failed to read image'}), 400
        
        # Process image
        metadata = {
            'source': filename,
            'timestamp': time.time()
        }
        
        # Override confidence threshold if provided
        module = get_waldo_module()
        if confidence_threshold:
            original_threshold = module.detector.confidence_threshold
            module.detector.confidence_threshold = confidence_threshold

        result = module.process_frame(image, metadata)

        # Restore original threshold
        if confidence_threshold:
            module.detector.confidence_threshold = original_threshold
        
        # Clean up
        os.remove(temp_path)
        
        # Return results
        return jsonify({
            'request_id': f"req_{int(time.time())}",
            'detections': result['detections'],
            'processing_time': result['processing_time']
        })
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': str(e)}), 500


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


@app.route('/detect', methods=['POST'])
def detect_multipart():
    """MineralVision waldo_proxy-compatible detection endpoint (multipart).

    Accepts the proxy's payload: files={'image': (filename, bytes)},
    form fields confidence_threshold, max_detections, classes (comma-joined).
    Responds with the proxy's DetectionResult shape.
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400

        confidence_threshold = request.form.get('confidence_threshold', None)
        if confidence_threshold:
            confidence_threshold = float(confidence_threshold)
        max_detections = int(request.form.get('max_detections', 100))
        classes = _parse_classes(request.form.get('classes', None))

        import cv2
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        module = get_waldo_module()
        if confidence_threshold:
            original_threshold = module.detector.confidence_threshold
            module.detector.confidence_threshold = confidence_threshold
        result = module.process_frame(
            image, {'source': file.filename, 'timestamp': time.time(),
                    'classes': classes})
        if confidence_threshold:
            module.detector.confidence_threshold = original_threshold

        detections = result['detections']
        if classes:
            detections = [d for d in detections
                          if d.get('class_name') in classes]

        return jsonify({
            'image_id': f"img_{int(time.time() * 1000)}",
            'detections': _detections_to_proxy_shape(detections, max_detections),
            'processing_time_ms': result['processing_time'] * 1000.0,
            'model_version': getattr(module, 'model_name', None)
                             or module.config.get('model_name', 'waldo'),
        })
    except Exception as e:
        logger.error(f"Error in /detect: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/detect/url', methods=['POST'])
def detect_from_url():
    """MineralVision waldo_proxy-compatible detection endpoint (JSON body).

    Accepts {'image_url', 'confidence_threshold', 'max_detections',
    'classes' (list)} and responds with the proxy's DetectionResult shape.
    """
    try:
        payload = request.get_json(force=True)
        image_url = payload.get('image_url')
        if not image_url:
            return jsonify({'error': 'image_url is required'}), 400
        confidence_threshold = payload.get('confidence_threshold')
        max_detections = int(payload.get('max_detections', 100))
        classes = _parse_classes(payload.get('classes'))

        import cv2
        import urllib.request
        with urllib.request.urlopen(image_url, timeout=30) as resp:
            data = resp.read()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Failed to decode image from URL'}), 400

        module = get_waldo_module()
        if confidence_threshold:
            original_threshold = module.detector.confidence_threshold
            module.detector.confidence_threshold = float(confidence_threshold)
        result = module.process_frame(
            image, {'source': image_url, 'timestamp': time.time(),
                    'classes': classes})
        if confidence_threshold:
            module.detector.confidence_threshold = original_threshold

        detections = result['detections']
        if classes:
            detections = [d for d in detections
                          if d.get('class_name') in classes]

        return jsonify({
            'image_id': f"img_{int(time.time() * 1000)}",
            'detections': _detections_to_proxy_shape(detections, max_detections),
            'processing_time_ms': result['processing_time'] * 1000.0,
            'model_version': getattr(module, 'model_name', None)
                             or module.config.get('model_name', 'waldo'),
        })
    except Exception as e:
        logger.error(f"Error in /detect/url: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection/video', methods=['POST'])
def process_video():
    """Process a video file for object detection and tracking."""
    try:
        # Check if video is provided
        if 'video' not in request.files:
            return jsonify({'error': 'No video provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No video selected'}), 400
        
        # Get parameters
        confidence_threshold = request.form.get('confidence_threshold', None)
        if confidence_threshold:
            confidence_threshold = float(confidence_threshold)
        
        classes = request.form.get('classes', None)
        if classes:
            classes = json.loads(classes)
        
        track_objects = request.form.get('track_objects', 'true').lower() == 'true'
        sample_rate = int(request.form.get('sample_rate', '1'))
        
        # Save video temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join('/tmp', filename)
        file.save(temp_path)
        
        # Generate job ID
        job_id = f"job_{int(time.time())}"
        
        # Start processing in a background thread
        import threading
        def process_video_task():
            try:
                # Process video
                metadata = {
                    'source': filename,
                    'job_id': job_id
                }
                
                # Override confidence threshold if provided
                module = get_waldo_module()
                if confidence_threshold:
                    original_threshold = module.detector.confidence_threshold
                    module.detector.confidence_threshold = confidence_threshold
                
                # Process video
                result = module.process_video(temp_path, metadata)
                
                # Restore original threshold
                if confidence_threshold:
                    module.detector.confidence_threshold = original_threshold
                
                # Store results
                with open(f"/tmp/{job_id}_results.json", 'w') as f:
                    json.dump(result, f)
                
                # Clean up
                os.remove(temp_path)
                
                logger.info(f"Video processing completed for job {job_id}")
            
            except Exception as e:
                logger.error(f"Error processing video: {str(e)}")
                with open(f"/tmp/{job_id}_error.txt", 'w') as f:
                    f.write(str(e))
        
        # Start processing thread
        thread = threading.Thread(target=process_video_task)
        thread.daemon = True
        thread.start()
        
        # Return job ID
        return jsonify({
            'job_id': job_id,
            'status': 'processing',
            'estimated_completion_time': time.time() + 300  # Estimate 5 minutes
        })
    
    except Exception as e:
        logger.error(f"Error starting video processing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection/video/<job_id>', methods=['GET'])
def get_video_status(job_id):
    """Check the status of a video processing job."""
    try:
        # Check if results file exists
        results_path = f"/tmp/{job_id}_results.json"
        error_path = f"/tmp/{job_id}_error.txt"
        
        if os.path.exists(results_path):
            # Job completed successfully
            with open(results_path, 'r') as f:
                result_summary = json.load(f)
            
            return jsonify({
                'job_id': job_id,
                'status': 'completed',
                'progress': 100,
                'frames_processed': result_summary.get('frames_processed', 0),
                'total_frames': result_summary.get('frames_processed', 0),
                'processing_time': result_summary.get('processing_time', 0),
                'result_url': f"/api/detection/video/{job_id}/results"
            })
        
        elif os.path.exists(error_path):
            # Job failed
            with open(error_path, 'r') as f:
                error_message = f.read()
            
            return jsonify({
                'job_id': job_id,
                'status': 'failed',
                'error': error_message
            })
        
        else:
            # Job still processing or not found
            import psutil
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if 'python' in proc.info['name'] and job_id in ' '.join(proc.info.get('cmdline', [])):
                    # Job is still processing
                    return jsonify({
                        'job_id': job_id,
                        'status': 'processing',
                        'progress': 50  # Estimate 50% complete
                    })
            
            # Job not found
            return jsonify({'error': f"Job {job_id} not found"}), 404
    
    except Exception as e:
        logger.error(f"Error checking video status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/detection/video/<job_id>/results', methods=['GET'])
def get_video_results(job_id):
    """Get the results of a completed video processing job."""
    try:
        # Check if results file exists
        results_path = f"/tmp/{job_id}_results.json"
        
        if os.path.exists(results_path):
            # Return results file
            return send_file(results_path, mimetype='application/json')
        else:
            # Job not completed or not found
            return jsonify({'error': f"Results for job {job_id} not found"}), 404
    
    except Exception as e:
        logger.error(f"Error getting video results: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
