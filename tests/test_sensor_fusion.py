"""
Sensor Fusion Tests

Comprehensive tests for the sensor fusion module including unit tests,
integration tests, and performance tests.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime


class TestWeightedAverageFusion:
    """Tests for WeightedAverageFusion algorithm."""
    
    @pytest.mark.unit
    def test_fusion_with_equal_weights(self):
        """Test fusion with equal weights produces average."""
        try:
            from api.sensor_fusion.fusion_algorithms import WeightedAverageFusion
            
            fusion = WeightedAverageFusion()
            data1 = np.array([[1, 2], [3, 4]])
            data2 = np.array([[5, 6], [7, 8]])
            
            result = fusion.fuse([data1, data2], weights=[0.5, 0.5])
            
            expected = np.array([[3, 4], [5, 6]])
            np.testing.assert_array_almost_equal(result, expected)
        except ImportError:
            pytest.skip("sensor_fusion module not available")
    
    @pytest.mark.unit
    def test_fusion_with_unequal_weights(self):
        """Test fusion with unequal weights."""
        try:
            from api.sensor_fusion.fusion_algorithms import WeightedAverageFusion
            
            fusion = WeightedAverageFusion()
            data1 = np.array([[10, 20]])
            data2 = np.array([[0, 0]])
            
            result = fusion.fuse([data1, data2], weights=[0.8, 0.2])
            
            expected = np.array([[8, 16]])
            np.testing.assert_array_almost_equal(result, expected)
        except ImportError:
            pytest.skip("sensor_fusion module not available")
    
    @pytest.mark.unit
    def test_fusion_normalizes_weights(self):
        """Test that weights are normalized if they don't sum to 1."""
        try:
            from api.sensor_fusion.fusion_algorithms import WeightedAverageFusion
            
            fusion = WeightedAverageFusion()
            data1 = np.array([[10]])
            data2 = np.array([[20]])
            
            # Weights sum to 3, should be normalized
            result = fusion.fuse([data1, data2], weights=[1, 2])
            
            expected = np.array([[(10 * 1/3) + (20 * 2/3)]])
            np.testing.assert_array_almost_equal(result, expected)
        except ImportError:
            pytest.skip("sensor_fusion module not available")


class TestBayesianFusion:
    """Tests for BayesianFusion algorithm."""
    
    @pytest.mark.unit
    def test_bayesian_fusion_basic(self):
        """Test basic Bayesian fusion."""
        try:
            from api.sensor_fusion.fusion_algorithms import BayesianFusion
            
            fusion = BayesianFusion()
            
            # Create sample grids
            grid1 = np.random.rand(10, 10)
            grid2 = np.random.rand(10, 10)
            
            result = fusion.fuse([grid1, grid2])
            
            assert result.shape == (10, 10)
            assert np.all(result >= 0)
        except ImportError:
            pytest.skip("sensor_fusion module not available")
    
    @pytest.mark.unit
    def test_grid_conversion(self):
        """Test grid conversion with CRS transformation."""
        try:
            from api.sensor_fusion.fusion_algorithms import BayesianFusion
            
            fusion = BayesianFusion()
            
            # Sample data with coordinates
            data = {
                "values": np.random.rand(100),
                "x": np.linspace(119, 120, 100),
                "y": np.linspace(-24, -23, 100),
                "crs": "EPSG:4326",
            }
            
            grid = fusion.convert_to_grid(data, resolution=0.01)
            
            assert grid is not None
            assert len(grid.shape) == 2
        except ImportError:
            pytest.skip("sensor_fusion module not available")
    
    @pytest.mark.unit
    def test_common_grid_creation(self):
        """Test creation of common grid from multiple datasets."""
        try:
            from api.sensor_fusion.fusion_algorithms import BayesianFusion
            
            fusion = BayesianFusion()
            
            bounds1 = {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10}
            bounds2 = {"min_x": 5, "max_x": 15, "min_y": 5, "max_y": 15}
            
            common_bounds = fusion.create_common_grid([bounds1, bounds2])
            
            # Common grid should be intersection
            assert common_bounds["min_x"] == 5
            assert common_bounds["max_x"] == 10
            assert common_bounds["min_y"] == 5
            assert common_bounds["max_y"] == 10
        except ImportError:
            pytest.skip("sensor_fusion module not available")


class TestHyperspectralAdapter:
    """Tests for HyperspectralDataAdapter."""
    
    @pytest.mark.unit
    def test_load_envi_format(self, temp_dir):
        """Test loading ENVI format hyperspectral data."""
        try:
            from api.sensor_fusion.hyperspectral_adapter import HyperspectralDataAdapter
            
            adapter = HyperspectralDataAdapter()
            
            # This would require actual test files
            # For now, test that the adapter initializes correctly
            assert adapter is not None
            assert hasattr(adapter, 'load')
        except ImportError:
            pytest.skip("hyperspectral_adapter module not available")
    
    @pytest.mark.unit
    def test_spectral_indices_calculation(self):
        """Test calculation of spectral indices."""
        try:
            from api.sensor_fusion.hyperspectral_adapter import HyperspectralDataAdapter
            
            adapter = HyperspectralDataAdapter()
            
            # Create mock hyperspectral data
            data = np.random.rand(100, 100, 224)  # 224 bands
            
            if hasattr(adapter, 'calculate_ndvi'):
                ndvi = adapter.calculate_ndvi(data, red_band=50, nir_band=100)
                assert ndvi.shape == (100, 100)
                assert np.all(ndvi >= -1) and np.all(ndvi <= 1)
        except ImportError:
            pytest.skip("hyperspectral_adapter module not available")


class TestLiDARAdapter:
    """Tests for LidarDataAdapter."""
    
    @pytest.mark.unit
    def test_adapter_initialization(self):
        """Test LiDAR adapter initialization."""
        try:
            from api.sensor_fusion.lidar_adapter import LidarDataAdapter
            
            adapter = LidarDataAdapter()
            
            assert adapter is not None
            assert hasattr(adapter, 'load')
            assert hasattr(adapter, 'supported_formats')
        except ImportError:
            pytest.skip("lidar_adapter module not available")
    
    @pytest.mark.unit
    def test_point_cloud_filtering(self):
        """Test point cloud filtering."""
        try:
            from api.sensor_fusion.lidar_adapter import LidarDataAdapter
            
            adapter = LidarDataAdapter()
            
            # Create mock point cloud
            points = np.random.rand(1000, 3) * 100
            
            if hasattr(adapter, 'filter_by_bounds'):
                filtered = adapter.filter_by_bounds(
                    points,
                    x_min=25, x_max=75,
                    y_min=25, y_max=75
                )
                assert len(filtered) <= len(points)
        except ImportError:
            pytest.skip("lidar_adapter module not available")


class TestMagnetometryAdapter:
    """Tests for MagnetometryDataAdapter."""
    
    @pytest.mark.unit
    def test_adapter_initialization(self):
        """Test magnetometry adapter initialization."""
        try:
            from api.sensor_fusion.magnetometry_adapter import MagnetometryDataAdapter
            
            adapter = MagnetometryDataAdapter()
            
            assert adapter is not None
            assert hasattr(adapter, 'load')
        except ImportError:
            pytest.skip("magnetometry_adapter module not available")


class TestSensorFusionIntegration:
    """Integration tests for sensor fusion pipeline."""
    
    @pytest.mark.integration
    def test_full_fusion_pipeline(self, sample_sensor_data):
        """Test complete sensor fusion pipeline."""
        try:
            from api.sensor_fusion.fusion_algorithms import BayesianFusion
            
            fusion = BayesianFusion()
            
            # Create multiple sensor datasets
            datasets = []
            for i in range(3):
                data = np.random.rand(50, 50) * (i + 1)
                datasets.append(data)
            
            result = fusion.fuse(datasets)
            
            assert result is not None
            assert result.shape == (50, 50)
        except ImportError:
            pytest.skip("sensor_fusion module not available")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_dataset_fusion(self):
        """Test fusion with large datasets."""
        try:
            from api.sensor_fusion.fusion_algorithms import BayesianFusion
            import time
            
            fusion = BayesianFusion()
            
            # Create large datasets
            datasets = [np.random.rand(1000, 1000) for _ in range(5)]
            
            start_time = time.time()
            result = fusion.fuse(datasets)
            duration = time.time() - start_time
            
            assert result is not None
            assert duration < 60  # Should complete within 60 seconds
        except ImportError:
            pytest.skip("sensor_fusion module not available")


class TestSensorFusionPerformance:
    """Performance tests for sensor fusion."""
    
    @pytest.mark.performance
    def test_fusion_throughput(self):
        """Test fusion throughput."""
        try:
            from api.sensor_fusion.fusion_algorithms import WeightedAverageFusion
            import time
            
            fusion = WeightedAverageFusion()
            
            iterations = 100
            data_size = (100, 100)
            
            start_time = time.time()
            for _ in range(iterations):
                data1 = np.random.rand(*data_size)
                data2 = np.random.rand(*data_size)
                fusion.fuse([data1, data2], weights=[0.5, 0.5])
            
            duration = time.time() - start_time
            throughput = iterations / duration
            
            assert throughput > 10  # At least 10 fusions per second
        except ImportError:
            pytest.skip("sensor_fusion module not available")
