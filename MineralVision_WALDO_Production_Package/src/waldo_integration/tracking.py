"""
Object Tracking Module
=====================

This module provides object tracking functionality to maintain object identity
across multiple frames.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

class ObjectTracker:
    """
    Multi-object tracker using IoU (Intersection over Union) for association.
    
    This class tracks objects across multiple frames, maintaining their identity
    and calculating trajectories.
    """
    
    def __init__(self, config):
        """
        Initialize the object tracker.
        
        Args:
            config (dict): Configuration dictionary with the following keys:
                - max_age (int): Maximum number of frames to keep a track alive without matching
                - min_hits (int): Minimum number of matches needed to establish a track
                - iou_threshold (float): IoU threshold for matching detections to tracks
        """
        self.max_age = config.get('max_age', 10)
        self.min_hits = config.get('min_hits', 3)
        self.iou_threshold = config.get('iou_threshold', 0.3)
        
        # Initialize tracks
        self.tracks = []
        self.frame_count = 0
        self.next_track_id = 0
        
    def update(self, detections, metadata=None):
        """
        Update tracks with new detections.
        
        Args:
            detections (list): List of detection dictionaries
            metadata (dict, optional): Frame metadata
            
        Returns:
            list: List of active tracks
        """
        self.frame_count += 1
        
        # If no tracks exist, initialize with current detections
        if len(self.tracks) == 0:
            for detection in detections:
                self._initialize_track(detection)
            return self.tracks
        
        # Get predictions for existing tracks
        track_predictions = [track['bbox'] for track in self.tracks]
        
        # Get current detections
        detection_bboxes = [detection['bbox'] for detection in detections]
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(track_predictions), len(detection_bboxes)))
        for i, track_bbox in enumerate(track_predictions):
            for j, det_bbox in enumerate(detection_bboxes):
                iou_matrix[i, j] = self._calculate_iou(track_bbox, det_bbox)
        
        # Apply Hungarian algorithm for optimal assignment
        matched_indices = []
        if min(iou_matrix.shape) > 0:
            # Find matches using Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(-iou_matrix)
            for row, col in zip(row_indices, col_indices):
                # Only consider matches with IoU above threshold
                if iou_matrix[row, col] >= self.iou_threshold:
                    matched_indices.append((row, col))
        
        # Process matches, unmatched tracks, and unmatched detections
        unmatched_tracks = set(range(len(self.tracks))) - set([i for i, _ in matched_indices])
        unmatched_detections = set(range(len(detections))) - set([j for _, j in matched_indices])
        
        # Update matched tracks
        for track_idx, detection_idx in matched_indices:
            self._update_track(self.tracks[track_idx], detections[detection_idx])
        
        # Handle unmatched tracks
        for track_idx in unmatched_tracks:
            self._mark_missing(self.tracks[track_idx])
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self._initialize_track(detections[detection_idx])
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks if track['age'] <= self.max_age]
        
        # Return active tracks (those that have been seen enough times)
        active_tracks = [track for track in self.tracks 
                         if track['hits'] >= self.min_hits and track['age'] <= 1]
        
        return active_tracks
    
    def _initialize_track(self, detection):
        """
        Initialize a new track from a detection.
        
        Args:
            detection (dict): Detection dictionary
        """
        track = {
            'track_id': self.next_track_id,
            'bbox': detection['bbox'],
            'class_id': detection['class_id'],
            'class_name': detection['class_name'],
            'confidence': detection['confidence'],
            'hits': 1,
            'age': 0,
            'history': [detection['bbox']],
            'last_detection': detection
        }
        
        self.tracks.append(track)
        self.next_track_id += 1
    
    def _update_track(self, track, detection):
        """
        Update an existing track with a new detection.
        
        Args:
            track (dict): Track dictionary
            detection (dict): Detection dictionary
        """
        track['bbox'] = detection['bbox']
        track['confidence'] = detection['confidence']
        track['hits'] += 1
        track['age'] = 0
        track['history'].append(detection['bbox'])
        track['last_detection'] = detection
    
    def _mark_missing(self, track):
        """
        Mark a track as missing (not detected in current frame).
        
        Args:
            track (dict): Track dictionary
        """
        track['age'] += 1
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1 (list): First bounding box [x1, y1, x2, y2]
            bbox2 (list): Second bounding box [x1, y1, x2, y2]
            
        Returns:
            float: IoU value
        """
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
