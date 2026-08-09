"""
WALDO Integration Module
=======================

This module provides the main integration point between MineralVision and WALDO,
coordinating detection, tracking, and measurement functionality.
"""

import os
import time
import logging
import sqlite3
import json
import numpy as np

from .detection import WALDODetector
from .tracking import ObjectTracker
from .measurement import MeasurementEngine

class WALDOIntegrationModule:
    """
    Main integration module for WALDO in MineralVision.
    
    This class coordinates the detection, tracking, and measurement of objects
    in video streams or image sequences, and manages the storage and retrieval
    of detection results.
    """
    
    def __init__(self, config):
        """
        Initialize the WALDO integration module.
        
        Args:
            config (dict): Configuration dictionary with the following keys:
                - models_dir (str): Directory containing WALDO models
                - model_name (str, optional): Specific model to use
                - camera_params (dict): Camera parameters
                - database (dict): Database configuration
                - tracker_config (dict, optional): Tracker configuration
                - measurement_config (dict, optional): Measurement engine configuration
        """
        self.config = config
        self.logger = logging.getLogger("waldo_integration")
        
        # Set up model path
        models_dir = config.get('models_dir', './models')
        model_name = config.get('model_name', 'waldo_v3.pt')
        model_path = os.path.join(models_dir, model_name)
        
        # Initialize detector
        detector_config = {
            'model_path': model_path,
            'confidence_threshold': config.get('confidence_threshold', 0.25),
            'device': config.get('device', 'cuda'),
            'precision': config.get('precision', 'fp16')
        }
        self.detector = WALDODetector(detector_config)
        
        # Initialize tracker
        tracker_config = config.get('tracker_config', {
            'max_age': config.get('tracker_max_age', 10),
            'min_hits': config.get('tracker_min_hits', 3),
            'iou_threshold': config.get('tracker_iou_threshold', 0.3)
        })
        self.tracker = ObjectTracker(tracker_config)
        
        # Initialize measurement engine
        measurement_config = config.get('measurement_config', {
            'camera_params': config.get('camera_params', {}),
            'reference_objects': config.get('reference_objects', {}),
            'altitude_source': config.get('altitude_source', 'gps'),
            'default_altitude': config.get('default_altitude', 100.0)
        })
        self.measurement_engine = MeasurementEngine(measurement_config)
        
        # Initialize database
        self._initialize_database(config.get('database', {}))
    
    def process_frame(self, frame, metadata=None):
        """
        Process a single frame with WALDO detection.
        
        Args:
            frame (numpy.ndarray): Input image frame
            metadata (dict, optional): Frame metadata
            
        Returns:
            dict: Processing results including detections, tracks, and measurements
        """
        start_time = time.time()
        
        # Ensure metadata is a dictionary
        if metadata is None:
            metadata = {}
        
        # Add timestamp if not present
        if 'timestamp' not in metadata:
            metadata['timestamp'] = time.time()
        
        # Run detection
        detections = self.detector.detect(frame, metadata)
        
        # Update tracker
        tracked_objects = self.tracker.update(detections, metadata)
        
        # Calculate measurements
        measured_objects = self.measurement_engine.calculate(tracked_objects, metadata)
        
        # Store results in database
        self._store_results(measured_objects, metadata)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return results
        return {
            'detections': detections,
            'tracked_objects': tracked_objects,
            'measured_objects': measured_objects,
            'processing_time': processing_time,
            'metadata': metadata
        }
    
    def process_video(self, video_path, metadata=None, callback=None):
        """
        Process a video file with WALDO detection.
        
        Args:
            video_path (str): Path to video file
            metadata (dict, optional): Video metadata
            callback (callable, optional): Callback function for each processed frame
            
        Returns:
            dict: Processing results summary
        """
        import cv2
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize metadata if not provided
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'source': video_path,
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height
        })
        
        # Process each frame
        results = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update frame-specific metadata
            frame_metadata = metadata.copy()
            frame_metadata.update({
                'frame_id': frame_idx,
                'timestamp': time.time()
            })
            
            # Process frame
            result = self.process_frame(frame, frame_metadata)
            results.append(result)
            
            # Call callback if provided
            if callback is not None:
                callback(frame_idx, frame, result)
            
            frame_idx += 1
        
        # Release video capture
        cap.release()
        
        # Return summary
        return {
            'video_path': video_path,
            'frames_processed': frame_idx,
            'total_detections': sum(len(r['detections']) for r in results),
            'total_tracks': len(set(obj['track_id'] for r in results for obj in r['tracked_objects'])),
            'processing_time': sum(r['processing_time'] for r in results),
            'metadata': metadata
        }
    
    def get_detections(self, filters=None, limit=100):
        """
        Get detections from database with optional filtering.
        
        Args:
            filters (dict, optional): Filtering criteria
            limit (int, optional): Maximum number of results to return
            
        Returns:
            list: List of detection dictionaries
        """
        # Build query
        query = "SELECT * FROM detections"
        params = []
        
        if filters:
            conditions = []
            for key, value in filters.items():
                if key in ['start_time', 'end_time']:
                    continue
                conditions.append(f"{key} = ?")
                params.append(value)
            
            if 'start_time' in filters:
                conditions.append("timestamp >= ?")
                params.append(filters['start_time'])
            
            if 'end_time' in filters:
                conditions.append("timestamp <= ?")
                params.append(filters['end_time'])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        # Execute query
        cursor = self.db_conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert rows to dictionaries
        detections = []
        for row in rows:
            detection = dict(zip([column[0] for column in cursor.description], row))
            
            # Parse JSON fields
            for field in ['bbox', 'measurements', 'metadata']:
                if field in detection and detection[field]:
                    try:
                        detection[field] = json.loads(detection[field])
                    except:
                        pass
            
            detections.append(detection)
        
        return detections
    
    def get_detections_in_area(self, area, limit=100):
        """
        Get detections within a specified geographic area.
        
        Args:
            area (dict): Geographic area specification with keys:
                - min_lat (float): Minimum latitude
                - max_lat (float): Maximum latitude
                - min_lon (float): Minimum longitude
                - max_lon (float): Maximum longitude
            limit (int, optional): Maximum number of results to return
            
        Returns:
            list: List of detection dictionaries
        """
        # Build query
        query = """
        SELECT * FROM detections
        WHERE metadata LIKE ? AND metadata LIKE ? AND metadata LIKE ? AND metadata LIKE ?
        ORDER BY timestamp DESC LIMIT ?
        """
        
        # Prepare parameters
        min_lat = area.get('min_lat', -90)
        max_lat = area.get('max_lat', 90)
        min_lon = area.get('min_lon', -180)
        max_lon = area.get('max_lon', 180)
        
        params = [
            f'%"latitude":%',
            f'%"longitude":%',
            f'%"latitude":[{min_lat} TO {max_lat}]%',
            f'%"longitude":[{min_lon} TO {max_lon}]%',
            limit
        ]
        
        # Execute query
        cursor = self.db_conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convert rows to dictionaries
        detections = []
        for row in rows:
            detection = dict(zip([column[0] for column in cursor.description], row))
            
            # Parse JSON fields
            for field in ['bbox', 'measurements', 'metadata']:
                if field in detection and detection[field]:
                    try:
                        detection[field] = json.loads(detection[field])
                    except:
                        pass
            
            # Filter results more precisely (since LIKE is not exact)
            if 'metadata' in detection and detection['metadata']:
                metadata = detection['metadata']
                if 'latitude' in metadata and 'longitude' in metadata:
                    lat = metadata['latitude']
                    lon = metadata['longitude']
                    if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
                        detections.append(detection)
            
        return detections
    
    def get_detections_in_range(self, time_range, limit=100):
        """
        Get detections within a specified time range.
        
        Args:
            time_range (tuple): Tuple of (start_time, end_time) as Unix timestamps
            limit (int, optional): Maximum number of results to return
            
        Returns:
            list: List of detection dictionaries
        """
        start_time, end_time = time_range
        
        return self.get_detections({
            'start_time': start_time,
            'end_time': end_time
        }, limit)
    
    def get_recent_detections(self, limit=100):
        """
        Get most recent detections.
        
        Args:
            limit (int, optional): Maximum number of results to return
            
        Returns:
            list: List of detection dictionaries
        """
        return self.get_detections(limit=limit)
    
    def _initialize_database(self, db_config):
        """
        Initialize the database connection and tables.
        
        Args:
            db_config (dict): Database configuration
        """
        db_type = db_config.get('type', 'sqlite')
        
        if db_type == 'sqlite':
            db_path = db_config.get('path', ':memory:')
            self.db_conn = sqlite3.connect(db_path)
            
            # Create tables if they don't exist
            cursor = self.db_conn.cursor()
            
            # Detections table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id TEXT PRIMARY KEY,
                track_id INTEGER,
                class_id INTEGER,
                class_name TEXT,
                confidence REAL,
                bbox TEXT,
                frame_id INTEGER,
                timestamp REAL,
                source TEXT,
                measurements TEXT,
                metadata TEXT
            )
            ''')
            
            # Create indices
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_track_id ON detections (track_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_class_id ON detections (class_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections (timestamp)')
            
            self.db_conn.commit()
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    
    def _store_results(self, objects, metadata):
        """
        Store detection results in database.
        
        Args:
            objects (list): List of object dictionaries
            metadata (dict): Frame metadata
        """
        if not objects:
            return
        
        cursor = self.db_conn.cursor()
        
        for obj in objects:
            # Generate unique ID
            unique_id = f"{metadata.get('source', 'unknown')}_{metadata.get('frame_id', 0)}_{obj['track_id']}"
            
            # Prepare data
            data = {
                'id': unique_id,
                'track_id': obj['track_id'],
                'class_id': obj['class_id'],
                'class_name': obj['class_name'],
                'confidence': obj['confidence'],
                'bbox': json.dumps(obj['bbox']),
                'frame_id': metadata.get('frame_id', 0),
                'timestamp': metadata.get('timestamp', time.time()),
                'source': metadata.get('source', 'unknown'),
                'measurements': json.dumps(obj.get('measurements', {})),
                'metadata': json.dumps(metadata)
            }
            
            # Insert or replace
            cursor.execute('''
            INSERT OR REPLACE INTO detections
            (id, track_id, class_id, class_name, confidence, bbox, frame_id, timestamp, source, measurements, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['id'], data['track_id'], data['class_id'], data['class_name'], data['confidence'],
                data['bbox'], data['frame_id'], data['timestamp'], data['source'],
                data['measurements'], data['metadata']
            ))
        
        self.db_conn.commit()
