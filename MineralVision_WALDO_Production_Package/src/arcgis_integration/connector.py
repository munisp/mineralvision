"""
ArcGIS Integration Module for MineralVision WALDO Integration
===========================================================

This module provides integration with ArcGIS for spatial visualization and analysis of WALDO detection data.
"""

import os
import time
import json
import logging
import requests
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArcGISConnector:
    """
    Connector class for ArcGIS integration with MineralVision WALDO.
    
    This class handles authentication, data synchronization, and layer management
    for ArcGIS integration.
    """
    
    def __init__(self, config):
        """
        Initialize the ArcGIS connector with configuration.
        
        Args:
            config (dict): Configuration dictionary containing ArcGIS connection parameters
        """
        self.config = config
        self.arcgis_url = config.get('arcgis_url', 'https://arcgis.example.com')
        self.username = config.get('arcgis_username', 'mineralvision')
        self.password = config.get('arcgis_password', 'password')
        self.token = None
        self.token_expires = None
        self.waldo_config = config.get('waldo', {})
        
        # Initialize connection
        self._authenticate()
    
    def _authenticate(self):
        """
        Authenticate with ArcGIS and obtain a token.
        
        Returns:
            bool: True if authentication was successful, False otherwise
        """
        try:
            # Build authentication request
            auth_url = f"{self.arcgis_url}/sharing/rest/generateToken"
            auth_data = {
                'username': self.username,
                'password': self.password,
                'client': 'referer',
                'referer': 'https://mineralvision.com',
                'expiration': 60,  # Token valid for 60 minutes
                'f': 'json'
            }
            
            # Send authentication request
            response = requests.post(auth_url, data=auth_data)
            response_json = response.json()
            
            if 'token' in response_json:
                self.token = response_json['token']
                self.token_expires = datetime.now() + timedelta(minutes=55)  # Refresh before expiration
                logger.info("Successfully authenticated with ArcGIS")
                return True
            else:
                logger.error(f"Authentication failed: {response_json.get('error', 'Unknown error')}")
                return False
        
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return False
    
    def _check_token(self):
        """
        Check if the token is valid and refresh if necessary.
        
        Returns:
            bool: True if token is valid, False otherwise
        """
        if not self.token or not self.token_expires or datetime.now() >= self.token_expires:
            return self._authenticate()
        return True
    
    def get_layers(self):
        """
        Get available layers from ArcGIS.
        
        Returns:
            dict: Dictionary containing layer information
        """
        if not self._check_token():
            return {'error': 'Authentication failed'}
        
        try:
            # Build request URL
            layers_url = f"{self.arcgis_url}/sharing/rest/content/users/{self.username}/items"
            params = {
                'token': self.token,
                'f': 'json',
                'q': 'typekeywords:WALDO'
            }
            
            # Send request
            response = requests.get(layers_url, params=params)
            response_json = response.json()
            
            if 'items' in response_json:
                layers = []
                for item in response_json['items']:
                    layers.append({
                        'id': item.get('id'),
                        'name': item.get('title'),
                        'feature_count': item.get('numFeatures', 0),
                        'last_updated': item.get('modified'),
                        'url': f"{self.arcgis_url}/home/item.html?id={item.get('id')}"
                    })
                
                return {'layers': layers}
            else:
                logger.error(f"Failed to get layers: {response_json.get('error', 'Unknown error')}")
                return {'error': 'Failed to get layers', 'details': response_json.get('error', 'Unknown error')}
        
        except Exception as e:
            logger.error(f"Error getting layers: {str(e)}")
            return {'error': str(e)}
    
    def sync_detections_by_id(self, detection_ids, layer_name='WALDO Detections', overwrite=False):
        """
        Synchronize specific detections to ArcGIS by ID.
        
        Args:
            detection_ids (list): List of detection IDs to synchronize
            layer_name (str): Name of the ArcGIS layer
            overwrite (bool): Whether to overwrite existing features
        
        Returns:
            dict: Synchronization result
        """
        if not self._check_token():
            return {'error': 'Authentication failed'}
        
        try:
            # Get detections from database
            # This would typically call a database function to retrieve detections
            # For this example, we'll simulate the data
            detections = self._get_detections_by_id(detection_ids)
            
            # Convert detections to GeoJSON
            geojson = self._convert_to_geojson(detections)
            
            # Upload to ArcGIS
            result = self._upload_to_arcgis(geojson, layer_name, overwrite)
            
            return result
        
        except Exception as e:
            logger.error(f"Error syncing detections by ID: {str(e)}")
            return {'error': str(e)}
    
    def sync_detections(self, time_range, layer_name='WALDO Detections', overwrite=False):
        """
        Synchronize detections to ArcGIS by time range.
        
        Args:
            time_range (dict): Dictionary containing start_time and end_time
            layer_name (str): Name of the ArcGIS layer
            overwrite (bool): Whether to overwrite existing features
        
        Returns:
            dict: Synchronization result
        """
        if not self._check_token():
            return {'error': 'Authentication failed'}
        
        try:
            # Get detections from database
            # This would typically call a database function to retrieve detections
            # For this example, we'll simulate the data
            start_time = time_range.get('start_time', time.time() - 86400)  # Default to last 24 hours
            end_time = time_range.get('end_time', time.time())
            
            detections = self._get_detections_by_time(start_time, end_time)
            
            # Convert detections to GeoJSON
            geojson = self._convert_to_geojson(detections)
            
            # Upload to ArcGIS
            result = self._upload_to_arcgis(geojson, layer_name, overwrite)
            
            return result
        
        except Exception as e:
            logger.error(f"Error syncing detections by time range: {str(e)}")
            return {'error': str(e)}
    
    def sync_recent_detections(self, layer_name='WALDO Detections', overwrite=False):
        """
        Synchronize recent detections to ArcGIS.
        
        Args:
            layer_name (str): Name of the ArcGIS layer
            overwrite (bool): Whether to overwrite existing features
        
        Returns:
            dict: Synchronization result
        """
        # Default to last hour
        time_range = {
            'start_time': time.time() - 3600,
            'end_time': time.time()
        }
        
        return self.sync_detections(time_range, layer_name, overwrite)
    
    def _get_detections_by_id(self, detection_ids):
        """
        Get detections by ID from the database.
        
        This is a placeholder method that would typically query a database.
        For this example, we'll return simulated data.
        
        Args:
            detection_ids (list): List of detection IDs
        
        Returns:
            list: List of detection objects
        """
        # Simulate database query
        detections = []
        
        for i, detection_id in enumerate(detection_ids):
            # Generate random coordinates near San Francisco for demonstration
            lat = 37.7749 + (i * 0.01)
            lon = -122.4194 + (i * 0.01)
            
            detection = {
                'id': detection_id,
                'class_name': 'vehicle' if i % 3 == 0 else ('person' if i % 3 == 1 else 'equipment'),
                'confidence': 0.85 + (i * 0.01),
                'timestamp': time.time() - (i * 600),  # 10 minutes apart
                'location': {
                    'latitude': lat,
                    'longitude': lon
                },
                'measurements': {
                    'width_m': 2.5,
                    'height_m': 1.8,
                    'area_m2': 4.5
                },
                'metadata': {
                    'source': f"camera_{i % 3 + 1}",
                    'altitude': 100 + (i * 10)
                }
            }
            
            detections.append(detection)
        
        return detections
    
    def _get_detections_by_time(self, start_time, end_time):
        """
        Get detections by time range from the database.
        
        This is a placeholder method that would typically query a database.
        For this example, we'll return simulated data.
        
        Args:
            start_time (float): Start timestamp
            end_time (float): End timestamp
        
        Returns:
            list: List of detection objects
        """
        # Simulate database query
        detections = []
        
        # Generate 10 sample detections
        for i in range(10):
            # Generate random coordinates near San Francisco for demonstration
            lat = 37.7749 + (i * 0.01)
            lon = -122.4194 + (i * 0.01)
            
            # Generate timestamp within range
            timestamp = start_time + ((end_time - start_time) * (i / 10))
            
            detection = {
                'id': f"det_{int(timestamp)}_{i}",
                'class_name': 'vehicle' if i % 3 == 0 else ('person' if i % 3 == 1 else 'equipment'),
                'confidence': 0.85 + (i * 0.01),
                'timestamp': timestamp,
                'location': {
                    'latitude': lat,
                    'longitude': lon
                },
                'measurements': {
                    'width_m': 2.5,
                    'height_m': 1.8,
                    'area_m2': 4.5
                },
                'metadata': {
                    'source': f"camera_{i % 3 + 1}",
                    'altitude': 100 + (i * 10)
                }
            }
            
            detections.append(detection)
        
        return detections
    
    def _convert_to_geojson(self, detections):
        """
        Convert detections to GeoJSON format for ArcGIS.
        
        Args:
            detections (list): List of detection objects
        
        Returns:
            dict: GeoJSON object
        """
        features = []
        
        for detection in detections:
            # Extract location
            lat = detection.get('location', {}).get('latitude')
            lon = detection.get('location', {}).get('longitude')
            
            if not lat or not lon:
                logger.warning(f"Skipping detection {detection.get('id')} without valid location")
                continue
            
            # Create GeoJSON feature
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [lon, lat]  # GeoJSON uses [longitude, latitude]
                },
                'properties': {
                    'id': detection.get('id'),
                    'class_name': detection.get('class_name'),
                    'confidence': detection.get('confidence'),
                    'timestamp': detection.get('timestamp'),
                    'width_m': detection.get('measurements', {}).get('width_m'),
                    'height_m': detection.get('measurements', {}).get('height_m'),
                    'area_m2': detection.get('measurements', {}).get('area_m2'),
                    'source': detection.get('metadata', {}).get('source'),
                    'altitude': detection.get('metadata', {}).get('altitude')
                }
            }
            
            features.append(feature)
        
        # Create GeoJSON object
        geojson = {
            'type': 'FeatureCollection',
            'features': features
        }
        
        return geojson
    
    def _upload_to_arcgis(self, geojson, layer_name, overwrite=False):
        """
        Upload GeoJSON to ArcGIS as a feature layer.
        
        Args:
            geojson (dict): GeoJSON object
            layer_name (str): Name of the ArcGIS layer
            overwrite (bool): Whether to overwrite existing layer
        
        Returns:
            dict: Upload result
        """
        if not self._check_token():
            return {'error': 'Authentication failed'}
        
        try:
            # Check if layer exists
            existing_layer = None
            layers = self.get_layers()
            
            if 'layers' in layers:
                for layer in layers['layers']:
                    if layer['name'] == layer_name:
                        existing_layer = layer
                        break
            
            # If layer exists and overwrite is False, append to existing layer
            if existing_layer and not overwrite:
                return self._append_to_layer(geojson, existing_layer['id'])
            
            # If layer exists and overwrite is True, delete existing layer
            if existing_layer and overwrite:
                self._delete_layer(existing_layer['id'])
            
            # Create new layer
            # Save GeoJSON to temporary file
            temp_file = f"/tmp/waldo_arcgis_{int(time.time())}.geojson"
            with open(temp_file, 'w') as f:
                json.dump(geojson, f)
            
            # Build request
            create_url = f"{self.arcgis_url}/sharing/rest/content/users/{self.username}/addItem"
            
            # Prepare form data
            form_data = {
                'token': self.token,
                'f': 'json',
                'title': layer_name,
                'type': 'GeoJSON',
                'tags': 'WALDO,MineralVision,detection',
                'typeKeywords': 'WALDO,Data,GeoJSON,Feature Collection',
                'description': f"WALDO detection data synchronized on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                'overwrite': 'true' if overwrite else 'false'
            }
            
            # Add file to form
            files = {
                'file': (os.path.basename(temp_file), open(temp_file, 'rb'), 'application/json')
            }
            
            # Send request
            response = requests.post(create_url, data=form_data, files=files)
            response_json = response.json()
            
            # Clean up temporary file
            files['file'][1].close()
            os.remove(temp_file)
            
            if 'success' in response_json and response_json['success']:
                # Publish as feature service
                item_id = response_json['id']
                publish_result = self._publish_feature_service(item_id, layer_name)
                
                if 'error' in publish_result:
                    return publish_result
                
                return {
                    'success': True,
                    'features_synced': len(geojson['features']),
                    'layer_id': publish_result.get('id'),
                    'layer_url': publish_result.get('url')
                }
            else:
                logger.error(f"Failed to create layer: {response_json.get('error', 'Unknown error')}")
                return {'error': 'Failed to create layer', 'details': response_json.get('error', 'Unknown error')}
        
        except Exception as e:
            logger.error(f"Error uploading to ArcGIS: {str(e)}")
            return {'error': str(e)}
    
    def _append_to_layer(self, geojson, layer_id):
        """
        Append features to an existing layer.
        
        Args:
            geojson (dict): GeoJSON object
            layer_id (str): ID of the existing layer
        
        Returns:
            dict: Append result
        """
        if not self._check_token():
            return {'error': 'Authentication failed'}
        
        try:
            # Get layer details
            layer_url = f"{self.arcgis_url}/sharing/rest/content/items/{layer_id}"
            params = {
                'token': self.token,
                'f': 'json'
            }
            
            response = requests.get(layer_url, params=params)
            layer_info = response.json()
            
            if 'error' in layer_info:
                logger.error(f"Failed to get layer info: {layer_info['error']}")
                return {'error': 'Failed to get layer info', 'details': layer_info['error']}
            
            # Get feature service URL
            if 'url' not in layer_info:
                logger.error("Layer is not a feature service")
                return {'error': 'Layer is not a feature service'}
            
            feature_service_url = layer_info['url']
            
            # Append features
            append_url = f"{feature_service_url}/0/addFeatures"
            
            # Convert GeoJSON features to ESRI features
            esri_features = []
            for feature in geojson['features']:
                esri_feature = {
                    'geometry': {
                        'x': feature['geometry']['coordinates'][0],
                        'y': feature['geometry']['coordinates'][1],
                        'spatialReference': {'wkid': 4326}
                    },
                    'attributes': feature['properties']
                }
                esri_features.append(esri_feature)
            
            # Prepare request
            append_data = {
                'token': self.token,
                'f': 'json',
                'features': json.dumps(esri_features)
            }
            
            # Send request
            response = requests.post(append_url, data=append_data)
            response_json = response.json()
            
            if 'addResults' in response_json:
                success_count = sum(1 for result in response_json['addResults'] if result.get('success', False))
                
                return {
                    'success': True,
                    'features_synced': success_count,
                    'layer_id': layer_id,
                    'layer_url': f"{self.arcgis_url}/home/item.html?id={layer_id}"
                }
            else:
                logger.error(f"Failed to append features: {response_json.get('error', 'Unknown error')}")
                return {'error': 'Failed to append features', 'details': response_json.get('error', 'Unknown error')}
        
        except Exception as e:
            logger.error(f"Error appending to layer: {str(e)}")
            return {'error': str(e)}
    
    def _delete_layer(self, layer_id):
        """
        Delete an existing layer.
        
        Args:
            layer_id (str): ID of the layer to delete
        
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if not self._check_token():
            return False
        
        try:
            # Build request
            delete_url = f"{self.arcgis_url}/sharing/rest/content/users/{self.username}/items/{layer_id}/delete"
            
            # Prepare request
            delete_data = {
                'token': self.token,
                'f': 'json'
            }
            
            # Send request
            response = requests.post(delete_url, data=delete_data)
            response_json = response.json()
            
            if 'success' in response_json and response_json['success']:
                logger.info(f"Successfully deleted layer {layer_id}")
                return True
            else:
                logger.error(f"Failed to delete layer: {response_json.get('error', 'Unknown error')}")
                return False
        
        except Exception as e:
            logger.error(f"Error deleting layer: {str(e)}")
            return False
    
    def _publish_feature_service(self, item_id, service_name):
        """
        Publish a GeoJSON item as a feature service.
        
        Args:
            item_id (str): ID of the GeoJSON item
            service_name (str): Name of the feature service
        
        Returns:
            dict: Publishing result
        """
        if not self._check_token():
            return {'error': 'Authentication failed'}
        
        try:
            # Build request
            publish_url = f"{self.arcgis_url}/sharing/rest/content/users/{self.username}/publish"
            
            # Prepare request
            publish_data = {
                'token': self.token,
                'f': 'json',
                'itemId': item_id,
                'filetype': 'geojson',
                'publishParameters': json.dumps({
                    'name': service_name,
                    'hasStaticData': False,
                    'maxRecordCount': 2000,
                    'layerInfo': {
                        'capabilities': 'Query,Create,Delete,Update,Editing'
                    }
                })
            }
            
            # Send request
            response = requests.post(publish_url, data=publish_data)
            response_json = response.json()
            
            if 'services' in response_json and len(response_json['services']) > 0:
                service = response_json['services'][0]
                
                return {
                    'success': True,
                    'id': service.get('serviceItemId'),
                    'url': f"{self.arcgis_url}/home/item.html?id={service.get('serviceItemId')}"
                }
            else:
                logger.error(f"Failed to publish feature service: {response_json.get('error', 'Unknown error')}")
                return {'error': 'Failed to publish feature service', 'details': response_json.get('error', 'Unknown error')}
        
        except Exception as e:
            logger.error(f"Error publishing feature service: {str(e)}")
            return {'error': str(e)}
