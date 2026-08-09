"""
MineralVision Load Testing with Locust

This module provides comprehensive load testing for the MineralVision platform
using Locust for distributed load generation.

Usage:
    locust -f locustfile.py --host=http://localhost:8000
    
For distributed testing:
    locust -f locustfile.py --master
    locust -f locustfile.py --worker --master-host=<master-ip>
"""

import json
import random
import string
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List

from locust import HttpUser, task, between, events, tag
from locust.runners import MasterRunner, WorkerRunner


def random_string(length: int = 8) -> str:
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def generate_coordinates() -> Dict[str, float]:
    return {
        "lat": -23.5 + random.random() * 10,
        "lon": 119.5 + random.random() * 10,
    }


def generate_region() -> Dict[str, float]:
    base_lat = -24.0 + random.random() * 2
    base_lon = 119.0 + random.random() * 2
    return {
        "min_lat": base_lat,
        "max_lat": base_lat + 1.0,
        "min_lon": base_lon,
        "max_lon": base_lon + 1.0,
    }


class MineralVisionUser(HttpUser):
    """Simulates a typical MineralVision platform user."""
    
    wait_time = between(1, 5)
    
    def on_start(self):
        """Called when a simulated user starts."""
        self.api_key = "test-api-key"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.created_entities = []
        self.uploaded_data_ids = []
    
    @task(10)
    @tag('health')
    def health_check(self):
        """Check API health endpoint."""
        with self.client.get("/health", headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(5)
    @tag('sensor-fusion')
    def upload_sensor_data(self):
        """Upload sensor data for fusion."""
        sensor_types = ["hyperspectral", "lidar", "magnetometry"]
        data = {
            "sensor_type": random.choice(sensor_types),
            "data": {
                "values": [random.random() * 1000 for _ in range(100)],
                "coordinates": generate_coordinates(),
                "timestamp": datetime.utcnow().isoformat(),
            },
            "metadata": {
                "source": f"sensor-{random_string()}",
                "quality": random.random(),
            },
        }
        
        with self.client.post(
            "/api/sensor-fusion/upload",
            json=data,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 201]:
                try:
                    result = response.json()
                    data_id = result.get("data_id") or result.get("id")
                    if data_id:
                        self.uploaded_data_ids.append(data_id)
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Upload failed: {response.status_code}")
    
    @task(3)
    @tag('sensor-fusion')
    def trigger_fusion(self):
        """Trigger sensor data fusion."""
        if not self.uploaded_data_ids:
            return
        
        data_ids = random.sample(
            self.uploaded_data_ids,
            min(3, len(self.uploaded_data_ids))
        )
        
        fusion_request = {
            "data_ids": data_ids,
            "algorithm": random.choice(["bayesian", "weighted_average", "kalman"]),
        }
        
        with self.client.post(
            "/api/sensor-fusion/fuse",
            json=fusion_request,
            headers=self.headers,
            catch_response=True,
            name="/api/sensor-fusion/fuse"
        ) as response:
            if response.status_code in [200, 202]:
                response.success()
            else:
                response.failure(f"Fusion failed: {response.status_code}")
    
    @task(4)
    @tag('prediction')
    def predict_mineral_deposits(self):
        """Request mineral deposit prediction."""
        prediction_request = {
            "region": generate_region(),
            "mineral_types": random.choice(["gold", "iron", "copper", "lithium"]),
            "confidence_threshold": 0.5 + random.random() * 0.4,
        }
        
        with self.client.post(
            "/api/predictive-modeling/predict",
            json=prediction_request,
            headers=self.headers,
            catch_response=True,
            name="/api/predictive-modeling/predict"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Prediction failed: {response.status_code}")
    
    @task(3)
    @tag('digital-twin')
    def list_digital_twin_entities(self):
        """List digital twin entities."""
        with self.client.get(
            "/api/digital-twin/entities",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"List entities failed: {response.status_code}")
    
    @task(2)
    @tag('digital-twin')
    def create_digital_twin_entity(self):
        """Create a digital twin entity."""
        entity_types = ["exploration_area", "mineral_deposit", "equipment"]
        entity = {
            "name": f"test-entity-{random_string()}",
            "type": random.choice(entity_types),
            "properties": {
                "area_km2": random.randint(10, 1000),
                "status": random.choice(["active", "planned", "completed"]),
            },
        }
        
        with self.client.post(
            "/api/digital-twin/entities",
            json=entity,
            headers=self.headers,
            catch_response=True,
            name="/api/digital-twin/entities [POST]"
        ) as response:
            if response.status_code in [200, 201]:
                try:
                    result = response.json()
                    entity_id = result.get("entity_id") or result.get("id")
                    if entity_id:
                        self.created_entities.append(entity_id)
                    response.success()
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Create entity failed: {response.status_code}")
    
    @task(2)
    @tag('climate')
    def analyze_climate_resilience(self):
        """Request climate resilience analysis."""
        analysis_request = {
            "region": generate_region(),
            "analysis_type": random.choice(["extreme_weather", "water_resources", "operational_windows"]),
            "time_range": [
                (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d"),
                datetime.utcnow().strftime("%Y-%m-%d"),
            ],
        }
        
        with self.client.post(
            "/api/climate-resilience/analyze",
            json=analysis_request,
            headers=self.headers,
            catch_response=True,
            name="/api/climate-resilience/analyze"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Climate analysis failed: {response.status_code}")
    
    @task(2)
    @tag('blockchain')
    def register_data_provenance(self):
        """Register data on blockchain for provenance."""
        registration = {
            "data_type": random.choice(["sensor_reading", "analysis_result", "prediction"]),
            "metadata": {
                "source": "load-test",
                "timestamp": datetime.utcnow().isoformat(),
            },
            "offline_mode": True,
        }
        
        with self.client.post(
            "/api/blockchain/register",
            json=registration,
            headers=self.headers,
            catch_response=True,
            name="/api/blockchain/register"
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Blockchain register failed: {response.status_code}")
    
    @task(1)
    @tag('autonomous')
    def plan_drone_mission(self):
        """Plan an autonomous drone mission."""
        mission_request = {
            "exploration_area": {
                "name": f"area-{random_string()}",
                "bounds": generate_region(),
            },
            "drone_specs": {
                "model": "DJI Matrice 300",
                "battery_capacity_mah": 5935,
                "max_flight_time_minutes": 55,
            },
            "sampling_points": [
                {
                    "position": generate_coordinates(),
                    "priority": random.randint(1, 5),
                    "sample_type": random.choice(["soil", "rock", "water"]),
                }
                for _ in range(random.randint(3, 10))
            ],
        }
        
        with self.client.post(
            "/api/autonomous-exploration/plan-mission",
            json=mission_request,
            headers=self.headers,
            catch_response=True,
            name="/api/autonomous-exploration/plan-mission"
        ) as response:
            if response.status_code in [200, 201]:
                response.success()
            else:
                response.failure(f"Mission planning failed: {response.status_code}")


class AdminUser(HttpUser):
    """Simulates an admin user performing management tasks."""
    
    wait_time = between(5, 15)
    weight = 1  # Lower weight than regular users
    
    def on_start(self):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer admin-api-key",
        }
    
    @task(5)
    @tag('admin')
    def get_system_metrics(self):
        """Retrieve system metrics."""
        with self.client.get(
            "/api/admin/metrics",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:  # 404 if endpoint doesn't exist
                response.success()
            else:
                response.failure(f"Metrics failed: {response.status_code}")
    
    @task(2)
    @tag('admin')
    def list_all_jobs(self):
        """List all processing jobs."""
        with self.client.get(
            "/api/admin/jobs",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"List jobs failed: {response.status_code}")


# Event handlers for custom reporting
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Load test starting...")
    if isinstance(environment.runner, MasterRunner):
        print("Running as master node")
    elif isinstance(environment.runner, WorkerRunner):
        print("Running as worker node")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print("Load test completed!")
    stats = environment.stats
    print(f"\nTotal requests: {stats.total.num_requests}")
    print(f"Total failures: {stats.total.num_failures}")
    print(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    print(f"Requests per second: {stats.total.current_rps:.2f}")


@events.request.add_listener
def on_request(request_type, name, response_time, response_length, response, context, exception, **kwargs):
    if exception:
        print(f"Request failed: {name} - {exception}")
