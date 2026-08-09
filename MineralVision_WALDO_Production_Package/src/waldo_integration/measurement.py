"""
Measurement Engine Module
========================

This module provides functionality for calculating measurements of detected objects
based on camera parameters and object dimensions.
"""

import math
import numpy as np

class MeasurementEngine:
    """
    Engine for calculating real-world measurements of detected objects.
    
    This class uses camera parameters and object dimensions to calculate
    real-world measurements such as size, distance, and position.
    """
    
    def __init__(self, config):
        """
        Initialize the measurement engine.
        
        Args:
            config (dict): Configuration dictionary with the following keys:
                - camera_params (dict): Camera parameters including:
                    - focal_length (float): Focal length in mm
                    - sensor_width (float): Sensor width in mm
                    - image_width (int): Image width in pixels
                - reference_objects (dict, optional): Known object dimensions for reference
                - altitude_source (str, optional): Source for altitude data ('gps', 'barometer', 'manual')
                - default_altitude (float, optional): Default altitude in meters if not provided in metadata
        """
        self.config = config
        self.camera_params = config.get('camera_params', {})
        self.reference_objects = config.get('reference_objects', {})
        self.altitude_source = config.get('altitude_source', 'gps')
        self.default_altitude = config.get('default_altitude', 100.0)  # meters
        
        # Calculate pixels per meter at 1 meter distance
        self._calculate_scale_factor()
    
    def _calculate_scale_factor(self):
        """
        Calculate the scale factor for converting pixels to real-world units.
        """
        focal_length = self.camera_params.get('focal_length', 35.0)  # mm
        sensor_width = self.camera_params.get('sensor_width', 23.5)  # mm
        image_width = self.camera_params.get('image_width', 1920)  # pixels
        
        # Calculate field of view in radians
        fov_rad = 2 * math.atan(sensor_width / (2 * focal_length))
        
        # Calculate pixels per meter at 1 meter distance
        self.pixels_per_meter_at_1m = image_width / (2 * math.tan(fov_rad / 2))
    
    def calculate(self, tracked_objects, metadata=None):
        """
        Calculate measurements for tracked objects.
        
        Args:
            tracked_objects (list): List of tracked object dictionaries
            metadata (dict, optional): Frame metadata including altitude, camera angle, etc.
            
        Returns:
            list: List of tracked objects with added measurement information
        """
        # Get altitude from metadata or use default
        altitude = self._get_altitude(metadata)
        
        # Calculate measurements for each object
        measured_objects = []
        for obj in tracked_objects:
            measured_obj = obj.copy()
            
            # Calculate dimensions
            bbox = obj['bbox']
            width_px = bbox[2] - bbox[0]
            height_px = bbox[3] - bbox[1]
            
            # Calculate real-world dimensions based on altitude
            width_m = self._pixels_to_meters(width_px, altitude)
            height_m = self._pixels_to_meters(height_px, altitude)
            
            # Calculate area
            area_m2 = width_m * height_m
            
            # Calculate position (center of bounding box)
            center_x_px = (bbox[0] + bbox[2]) / 2
            center_y_px = (bbox[1] + bbox[3]) / 2
            
            # Add measurements to object
            measured_obj['measurements'] = {
                'width_m': width_m,
                'height_m': height_m,
                'area_m2': area_m2,
                'center_px': [center_x_px, center_y_px],
                'altitude_m': altitude
            }
            
            # Add additional measurements based on object class if available
            if obj['class_name'] in self.reference_objects:
                self._add_reference_measurements(measured_obj)
            
            measured_objects.append(measured_obj)
        
        return measured_objects
    
    def _pixels_to_meters(self, pixels, distance_m):
        """
        Convert pixels to meters at a given distance.
        
        Args:
            pixels (float): Size in pixels
            distance_m (float): Distance in meters
            
        Returns:
            float: Size in meters
        """
        return pixels * distance_m / self.pixels_per_meter_at_1m
    
    def _get_altitude(self, metadata):
        """
        Get altitude from metadata or use default.
        
        Args:
            metadata (dict, optional): Frame metadata
            
        Returns:
            float: Altitude in meters
        """
        if metadata is None:
            return self.default_altitude
        
        if self.altitude_source == 'gps' and 'gps_altitude' in metadata:
            return metadata['gps_altitude']
        elif self.altitude_source == 'barometer' and 'barometer_altitude' in metadata:
            return metadata['barometer_altitude']
        elif self.altitude_source == 'manual' and 'manual_altitude' in metadata:
            return metadata['manual_altitude']
        
        return self.default_altitude
    
    def _add_reference_measurements(self, obj):
        """
        Add additional measurements based on reference object dimensions.
        
        Args:
            obj (dict): Object dictionary with measurements
        """
        class_name = obj['class_name']
        ref_obj = self.reference_objects.get(class_name, {})
        
        if 'typical_width_m' in ref_obj and 'typical_height_m' in ref_obj:
            # Calculate distance based on apparent size vs. typical size
            typical_width_m = ref_obj['typical_width_m']
            measured_width_m = obj['measurements']['width_m']
            
            # Distance estimate based on width ratio
            distance_estimate = typical_width_m * self.pixels_per_meter_at_1m / (obj['bbox'][2] - obj['bbox'][0])
            
            # Add to measurements
            obj['measurements']['distance_estimate_m'] = distance_estimate
            obj['measurements']['typical_width_m'] = typical_width_m
            obj['measurements']['typical_height_m'] = ref_obj['typical_height_m']
            
            # Calculate volume if applicable
            if 'typical_depth_m' in ref_obj:
                typical_depth_m = ref_obj['typical_depth_m']
                obj['measurements']['typical_depth_m'] = typical_depth_m
                obj['measurements']['volume_estimate_m3'] = measured_width_m * obj['measurements']['height_m'] * typical_depth_m
