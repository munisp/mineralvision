"""
Advanced Object Tracking Module
===============================

Production-grade multi-object tracking with:
- ByteTrack algorithm for high-performance tracking
- Kalman filter for motion prediction
- Appearance embeddings for re-identification
- Track lifecycle management with recovery
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
from scipy.optimize import linear_sum_assignment
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TrackState(Enum):
    """Track lifecycle states."""
    TENTATIVE = 1  # New track, not yet confirmed
    CONFIRMED = 2  # Confirmed track with enough hits
    LOST = 3       # Lost track, waiting for recovery
    DELETED = 4    # Track to be removed


@dataclass
class KalmanState:
    """Kalman filter state for motion prediction."""
    # State vector: [x, y, w, h, vx, vy, vw, vh]
    mean: np.ndarray
    covariance: np.ndarray
    
    @classmethod
    def from_bbox(cls, bbox: List[float]) -> 'KalmanState':
        """Initialize state from bounding box [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        mean = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)
        covariance = np.eye(8, dtype=np.float64)
        # Higher uncertainty for velocity components
        covariance[4:, 4:] *= 10
        
        return cls(mean=mean, covariance=covariance)
    
    def to_bbox(self) -> List[float]:
        """Convert state to bounding box [x1, y1, x2, y2]."""
        cx, cy, w, h = self.mean[:4]
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


class KalmanFilter:
    """
    Kalman filter for bounding box tracking.
    
    State: [cx, cy, w, h, vx, vy, vw, vh]
    Measurement: [cx, cy, w, h]
    """
    
    def __init__(self, dt: float = 1.0):
        self.dt = dt
        
        # State transition matrix (constant velocity model)
        self.F = np.eye(8, dtype=np.float64)
        self.F[:4, 4:] = np.eye(4) * dt
        
        # Measurement matrix
        self.H = np.eye(4, 8, dtype=np.float64)
        
        # Process noise
        self.Q = np.eye(8, dtype=np.float64)
        self.Q[:4, :4] *= 1.0
        self.Q[4:, 4:] *= 0.01
        
        # Measurement noise
        self.R = np.eye(4, dtype=np.float64) * 1.0
    
    def predict(self, state: KalmanState) -> KalmanState:
        """Predict next state."""
        mean = self.F @ state.mean
        covariance = self.F @ state.covariance @ self.F.T + self.Q
        return KalmanState(mean=mean, covariance=covariance)
    
    def update(self, state: KalmanState, measurement: np.ndarray) -> KalmanState:
        """Update state with measurement."""
        # Innovation
        y = measurement - self.H @ state.mean
        
        # Innovation covariance
        S = self.H @ state.covariance @ self.H.T + self.R
        
        # Kalman gain
        K = state.covariance @ self.H.T @ np.linalg.inv(S)
        
        # Updated state
        mean = state.mean + K @ y
        covariance = (np.eye(8) - K @ self.H) @ state.covariance
        
        return KalmanState(mean=mean, covariance=covariance)
    
    def gating_distance(self, state: KalmanState, measurement: np.ndarray) -> float:
        """Calculate Mahalanobis distance for gating."""
        y = measurement - self.H @ state.mean
        S = self.H @ state.covariance @ self.H.T + self.R
        return float(y.T @ np.linalg.inv(S) @ y)


@dataclass
class Track:
    """Single object track."""
    track_id: int
    state: KalmanState
    class_id: int
    class_name: str
    confidence: float
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    track_state: TrackState = TrackState.TENTATIVE
    history: List[List[float]] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    
    def __post_init__(self):
        if not self.history:
            self.history = [self.state.to_bbox()]
    
    @property
    def bbox(self) -> List[float]:
        return self.state.to_bbox()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'track_id': self.track_id,
            'bbox': self.bbox,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'hits': self.hits,
            'age': self.age,
            'state': self.track_state.name,
            'velocity': self.velocity
        }


class AppearanceExtractor(ABC):
    """Abstract base class for appearance feature extraction."""
    
    @abstractmethod
    def extract(self, image: np.ndarray, bboxes: List[List[float]]) -> np.ndarray:
        """Extract appearance features for bounding boxes."""
        pass


class SimpleAppearanceExtractor(AppearanceExtractor):
    """
    Simple appearance extractor using color histograms.
    
    For production, replace with deep learning embeddings (ResNet, OSNet).
    """
    
    def __init__(self, feature_dim: int = 128):
        self.feature_dim = feature_dim
    
    def extract(self, image: np.ndarray, bboxes: List[List[float]]) -> np.ndarray:
        """Extract color histogram features."""
        if image is None or len(bboxes) == 0:
            return np.zeros((len(bboxes), self.feature_dim))
        
        features = []
        h, w = image.shape[:2]
        
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                features.append(np.zeros(self.feature_dim))
                continue
            
            crop = image[y1:y2, x1:x2]
            
            # Compute color histogram
            if len(crop.shape) == 3:
                hist_features = []
                for c in range(min(3, crop.shape[2])):
                    hist, _ = np.histogram(crop[:, :, c].flatten(), bins=self.feature_dim // 3, range=(0, 256))
                    hist_features.extend(hist / (hist.sum() + 1e-6))
                
                # Pad if needed
                while len(hist_features) < self.feature_dim:
                    hist_features.append(0)
                features.append(np.array(hist_features[:self.feature_dim]))
            else:
                hist, _ = np.histogram(crop.flatten(), bins=self.feature_dim, range=(0, 256))
                features.append(hist / (hist.sum() + 1e-6))
        
        return np.array(features)


class DeepAppearanceExtractor(AppearanceExtractor):
    """
    Deep learning appearance extractor using CNN embeddings.
    
    Uses a lightweight CNN for real-time performance.
    """
    
    def __init__(self, model_path: Optional[str] = None, feature_dim: int = 512):
        self.feature_dim = feature_dim
        self.model = None
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            import torch
            import torch.nn as nn
            
            # Simple CNN for embeddings
            class EmbeddingNet(nn.Module):
                def __init__(self, feature_dim):
                    super().__init__()
                    self.conv = nn.Sequential(
                        nn.Conv2d(3, 32, 3, stride=2, padding=1),
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, stride=2, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(64, 128, 3, stride=2, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d((1, 1))
                    )
                    self.fc = nn.Linear(128, feature_dim)
                
                def forward(self, x):
                    x = self.conv(x)
                    x = x.view(x.size(0), -1)
                    x = self.fc(x)
                    return nn.functional.normalize(x, p=2, dim=1)
            
            self.model = EmbeddingNet(self.feature_dim)
            self.model.eval()
            
            if self.model_path:
                self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
        except ImportError:
            logger.warning("PyTorch not available, using simple appearance extractor")
            self.model = None
    
    def extract(self, image: np.ndarray, bboxes: List[List[float]]) -> np.ndarray:
        """Extract deep features."""
        if self.model is None or image is None or len(bboxes) == 0:
            # Fallback to simple extractor
            simple = SimpleAppearanceExtractor(self.feature_dim)
            return simple.extract(image, bboxes)
        
        import torch
        
        h, w = image.shape[:2]
        crops = []
        
        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                crop = np.zeros((64, 64, 3), dtype=np.uint8)
            else:
                crop = image[y1:y2, x1:x2]
                # Resize to fixed size
                import cv2
                crop = cv2.resize(crop, (64, 64))
            
            crops.append(crop)
        
        # Convert to tensor
        crops = np.array(crops).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
        crops_tensor = torch.from_numpy(crops).to(self.device)
        
        with torch.no_grad():
            features = self.model(crops_tensor).cpu().numpy()
        
        return features


class ByteTracker:
    """
    ByteTrack algorithm implementation.
    
    High-performance multi-object tracker that associates detections
    with tracks using both high and low confidence detections.
    
    Reference: https://arxiv.org/abs/2110.06864
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # Thresholds
        self.high_thresh = config.get('high_thresh', 0.6)
        self.low_thresh = config.get('low_thresh', 0.1)
        self.match_thresh = config.get('match_thresh', 0.8)
        
        # Track management
        self.max_age = config.get('max_age', 30)
        self.min_hits = config.get('min_hits', 3)
        self.n_init = config.get('n_init', 3)
        
        # Appearance
        self.use_appearance = config.get('use_appearance', True)
        self.appearance_weight = config.get('appearance_weight', 0.5)
        
        # State
        self.tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []
        self.frame_count = 0
        self.next_track_id = 1
        
        # Components
        self.kalman = KalmanFilter()
        if self.use_appearance:
            self.appearance_extractor = SimpleAppearanceExtractor()
        else:
            self.appearance_extractor = None
    
    def update(self, detections: List[Dict], frame: np.ndarray = None) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection dicts with 'bbox', 'confidence', 'class_id', 'class_name'
            frame: Optional image for appearance extraction
            
        Returns:
            List of active track dictionaries
        """
        self.frame_count += 1
        
        # Predict new locations for all tracks
        for track in self.tracks:
            track.state = self.kalman.predict(track.state)
            track.age += 1
            track.time_since_update += 1
        
        # Separate high and low confidence detections
        high_dets = [d for d in detections if d['confidence'] >= self.high_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d['confidence'] < self.high_thresh]
        
        # Extract appearance features
        if self.use_appearance and frame is not None and len(detections) > 0:
            all_bboxes = [d['bbox'] for d in detections]
            all_features = self.appearance_extractor.extract(frame, all_bboxes)
            
            for i, det in enumerate(detections):
                det['embedding'] = all_features[i]
        
        # First association: high confidence detections with confirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.track_state == TrackState.CONFIRMED]
        matched_track_indices, matched_det_indices, unmatched_tracks, unmatched_dets = \
            self._associate(confirmed_tracks, high_dets)
        
        # Update matched tracks
        for track_idx, det_idx in zip(matched_track_indices, matched_det_indices):
            self._update_track(confirmed_tracks[track_idx], high_dets[det_idx])
        
        # Second association: remaining tracks with low confidence detections
        remaining_tracks = [confirmed_tracks[i] for i in unmatched_tracks]
        matched_track_indices2, matched_det_indices2, unmatched_tracks2, _ = \
            self._associate(remaining_tracks, low_dets)
        
        for track_idx, det_idx in zip(matched_track_indices2, matched_det_indices2):
            self._update_track(remaining_tracks[track_idx], low_dets[det_idx])
        
        # Third association: unconfirmed tracks with remaining high confidence detections
        unconfirmed_tracks = [t for t in self.tracks if t.track_state == TrackState.TENTATIVE]
        remaining_high_dets = [high_dets[i] for i in unmatched_dets]
        matched_track_indices3, matched_det_indices3, unmatched_unconfirmed, unmatched_remaining = \
            self._associate(unconfirmed_tracks, remaining_high_dets)
        
        for track_idx, det_idx in zip(matched_track_indices3, matched_det_indices3):
            self._update_track(unconfirmed_tracks[track_idx], remaining_high_dets[det_idx])
        
        # Mark unmatched tracks
        for track_idx in unmatched_tracks2:
            remaining_tracks[track_idx].track_state = TrackState.LOST
        
        for track_idx in unmatched_unconfirmed:
            unconfirmed_tracks[track_idx].track_state = TrackState.DELETED
        
        # Initialize new tracks for unmatched high confidence detections
        for det_idx in unmatched_remaining:
            self._init_track(remaining_high_dets[det_idx])
        
        # Remove deleted tracks and move lost tracks
        self.tracks = [t for t in self.tracks if t.track_state != TrackState.DELETED]
        
        # Handle lost tracks
        for track in self.tracks:
            if track.track_state == TrackState.LOST:
                if track.time_since_update > self.max_age:
                    track.track_state = TrackState.DELETED
                    self.removed_tracks.append(track)
        
        self.tracks = [t for t in self.tracks if t.track_state != TrackState.DELETED]
        
        # Return confirmed tracks
        return [t.to_dict() for t in self.tracks if t.track_state == TrackState.CONFIRMED]
    
    def _associate(self, tracks: List[Track], detections: List[Dict]) -> Tuple[List, List, List, List]:
        """Associate tracks with detections using IoU and appearance."""
        if len(tracks) == 0 or len(detections) == 0:
            return [], [], list(range(len(tracks))), list(range(len(detections)))
        
        # Compute cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # IoU cost
                iou = self._compute_iou(track.bbox, det['bbox'])
                iou_cost = 1 - iou
                
                # Appearance cost
                if self.use_appearance and track.embedding is not None and 'embedding' in det:
                    appearance_dist = 1 - np.dot(track.embedding, det['embedding'])
                    cost = (1 - self.appearance_weight) * iou_cost + self.appearance_weight * appearance_dist
                else:
                    cost = iou_cost
                
                cost_matrix[i, j] = cost
        
        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matched_tracks = []
        matched_dets = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < self.match_thresh:
                matched_tracks.append(row)
                matched_dets.append(col)
                unmatched_tracks.discard(row)
                unmatched_dets.discard(col)
        
        return matched_tracks, matched_dets, list(unmatched_tracks), list(unmatched_dets)
    
    def _update_track(self, track: Track, detection: Dict):
        """Update track with detection."""
        bbox = detection['bbox']
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        measurement = np.array([cx, cy, w, h])
        
        # Calculate velocity before update
        old_cx, old_cy = track.state.mean[0], track.state.mean[1]
        
        track.state = self.kalman.update(track.state, measurement)
        track.confidence = detection['confidence']
        track.hits += 1
        track.time_since_update = 0
        track.history.append(track.bbox)
        
        # Update velocity
        new_cx, new_cy = track.state.mean[0], track.state.mean[1]
        track.velocity = (new_cx - old_cx, new_cy - old_cy)
        
        # Update embedding
        if 'embedding' in detection:
            if track.embedding is None:
                track.embedding = detection['embedding']
            else:
                # Exponential moving average
                track.embedding = 0.9 * track.embedding + 0.1 * detection['embedding']
                track.embedding /= np.linalg.norm(track.embedding) + 1e-6
        
        # Update state
        if track.track_state == TrackState.TENTATIVE and track.hits >= self.n_init:
            track.track_state = TrackState.CONFIRMED
        elif track.track_state == TrackState.LOST:
            track.track_state = TrackState.CONFIRMED
    
    def _init_track(self, detection: Dict):
        """Initialize new track from detection."""
        track = Track(
            track_id=self.next_track_id,
            state=KalmanState.from_bbox(detection['bbox']),
            class_id=detection['class_id'],
            class_name=detection['class_name'],
            confidence=detection['confidence'],
            embedding=detection.get('embedding')
        )
        self.next_track_id += 1
        self.tracks.append(track)
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracker statistics."""
        return {
            'frame_count': self.frame_count,
            'active_tracks': len([t for t in self.tracks if t.track_state == TrackState.CONFIRMED]),
            'tentative_tracks': len([t for t in self.tracks if t.track_state == TrackState.TENTATIVE]),
            'lost_tracks': len([t for t in self.tracks if t.track_state == TrackState.LOST]),
            'total_tracks_created': self.next_track_id - 1,
            'removed_tracks': len(self.removed_tracks)
        }


class DeepSORTTracker:
    """
    DeepSORT tracker implementation.
    
    Combines Kalman filtering with deep appearance features
    for robust multi-object tracking.
    
    Reference: https://arxiv.org/abs/1703.07402
    """
    
    def __init__(self, config: Dict = None):
        config = config or {}
        
        # Thresholds
        self.max_iou_distance = config.get('max_iou_distance', 0.7)
        self.max_cosine_distance = config.get('max_cosine_distance', 0.3)
        self.nn_budget = config.get('nn_budget', 100)
        
        # Track management
        self.max_age = config.get('max_age', 70)
        self.n_init = config.get('n_init', 3)
        
        # State
        self.tracks: List[Track] = []
        self.frame_count = 0
        self.next_track_id = 1
        
        # Components
        self.kalman = KalmanFilter()
        self.appearance_extractor = DeepAppearanceExtractor()
        
        # Feature gallery for each track
        self.feature_gallery: Dict[int, deque] = {}
    
    def update(self, detections: List[Dict], frame: np.ndarray = None) -> List[Dict]:
        """Update tracks with new detections."""
        self.frame_count += 1
        
        # Predict
        for track in self.tracks:
            track.state = self.kalman.predict(track.state)
            track.age += 1
            track.time_since_update += 1
        
        # Extract features
        if frame is not None and len(detections) > 0:
            bboxes = [d['bbox'] for d in detections]
            features = self.appearance_extractor.extract(frame, bboxes)
            for i, det in enumerate(detections):
                det['embedding'] = features[i]
        
        # Split tracks
        confirmed_tracks = [t for t in self.tracks if t.track_state == TrackState.CONFIRMED]
        unconfirmed_tracks = [t for t in self.tracks if t.track_state == TrackState.TENTATIVE]
        
        # Associate confirmed tracks using appearance + IoU
        matches_a, unmatched_tracks_a, unmatched_dets = \
            self._matching_cascade(confirmed_tracks, detections)
        
        # Associate remaining tracks and detections using IoU
        iou_track_candidates = unconfirmed_tracks + \
            [confirmed_tracks[i] for i in unmatched_tracks_a if confirmed_tracks[i].time_since_update == 1]
        unmatched_tracks_a = [i for i in unmatched_tracks_a if confirmed_tracks[i].time_since_update != 1]
        
        matches_b, unmatched_tracks_b, unmatched_dets = \
            self._iou_matching(iou_track_candidates, [detections[i] for i in unmatched_dets], unmatched_dets)
        
        # Update matched tracks
        for track_idx, det_idx in matches_a + matches_b:
            track = self.tracks[track_idx] if track_idx < len(self.tracks) else iou_track_candidates[track_idx - len(confirmed_tracks)]
            self._update_track(track, detections[det_idx])
        
        # Mark unmatched tracks
        for track_idx in unmatched_tracks_a:
            confirmed_tracks[track_idx].track_state = TrackState.LOST
        
        for track_idx in unmatched_tracks_b:
            iou_track_candidates[track_idx].track_state = TrackState.DELETED
        
        # Initialize new tracks
        for det_idx in unmatched_dets:
            self._init_track(detections[det_idx])
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks 
                      if t.track_state != TrackState.DELETED and 
                      t.time_since_update <= self.max_age]
        
        return [t.to_dict() for t in self.tracks if t.track_state == TrackState.CONFIRMED]
    
    def _matching_cascade(self, tracks: List[Track], detections: List[Dict]) -> Tuple:
        """Matching cascade for confirmed tracks."""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Compute appearance cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            if track.track_id in self.feature_gallery:
                gallery = np.array(list(self.feature_gallery[track.track_id]))
                for j, det in enumerate(detections):
                    if 'embedding' in det:
                        distances = 1 - np.dot(gallery, det['embedding'])
                        cost_matrix[i, j] = distances.min()
                    else:
                        cost_matrix[i, j] = 1.0
            else:
                cost_matrix[i, :] = 1.0
        
        # Gate by cosine distance
        cost_matrix[cost_matrix > self.max_cosine_distance] = 1e5
        
        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 1e5:
                matches.append((row, col))
                unmatched_tracks.discard(row)
                unmatched_dets.discard(col)
        
        return matches, list(unmatched_tracks), list(unmatched_dets)
    
    def _iou_matching(self, tracks: List[Track], detections: List[Dict], 
                     det_indices: List[int]) -> Tuple:
        """IoU-based matching for remaining tracks."""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), det_indices
        
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = self._compute_iou(track.bbox, det['bbox'])
                cost_matrix[i, j] = 1 - iou
        
        cost_matrix[cost_matrix > self.max_iou_distance] = 1e5
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = list(det_indices)
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 1e5:
                matches.append((row, det_indices[col]))
                unmatched_tracks.discard(row)
                if det_indices[col] in unmatched_dets:
                    unmatched_dets.remove(det_indices[col])
        
        return matches, list(unmatched_tracks), unmatched_dets
    
    def _update_track(self, track: Track, detection: Dict):
        """Update track with detection."""
        bbox = detection['bbox']
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        measurement = np.array([cx, cy, w, h])
        
        track.state = self.kalman.update(track.state, measurement)
        track.confidence = detection['confidence']
        track.hits += 1
        track.time_since_update = 0
        track.history.append(track.bbox)
        
        # Update feature gallery
        if 'embedding' in detection:
            if track.track_id not in self.feature_gallery:
                self.feature_gallery[track.track_id] = deque(maxlen=self.nn_budget)
            self.feature_gallery[track.track_id].append(detection['embedding'])
        
        if track.track_state == TrackState.TENTATIVE and track.hits >= self.n_init:
            track.track_state = TrackState.CONFIRMED
        elif track.track_state == TrackState.LOST:
            track.track_state = TrackState.CONFIRMED
    
    def _init_track(self, detection: Dict):
        """Initialize new track."""
        track = Track(
            track_id=self.next_track_id,
            state=KalmanState.from_bbox(detection['bbox']),
            class_id=detection['class_id'],
            class_name=detection['class_name'],
            confidence=detection['confidence']
        )
        
        if 'embedding' in detection:
            self.feature_gallery[track.track_id] = deque(maxlen=self.nn_budget)
            self.feature_gallery[track.track_id].append(detection['embedding'])
        
        self.next_track_id += 1
        self.tracks.append(track)
    
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


def create_tracker(tracker_type: str = 'bytetrack', config: Dict = None) -> Any:
    """Factory function to create tracker."""
    if tracker_type.lower() == 'bytetrack':
        return ByteTracker(config)
    elif tracker_type.lower() == 'deepsort':
        return DeepSORTTracker(config)
    else:
        raise ValueError(f"Unknown tracker type: {tracker_type}")
