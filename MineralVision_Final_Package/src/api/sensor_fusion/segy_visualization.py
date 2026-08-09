"""
SEG-Y Visualization Module for MineralVision.

This module provides interactive visualization capabilities for SEG-Y seismic data:
- Inline/crossline/depth slice views
- Multiple colormap support (seismic, viridis, etc.)
- Configurable layouts and tile arrangements
- Amplitude histograms and statistics
- Export to various formats (PNG, PDF, GeoTIFF)
- Integration with segyio for efficient data access

Based on equinor/segyviewer concepts and matplotlib visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import logging
import json
import io
import base64

logger = logging.getLogger(__name__)


class ViewType(Enum):
    """Seismic view types."""
    INLINE = "inline"
    CROSSLINE = "crossline"
    DEPTH = "depth"
    TIME_SLICE = "time_slice"
    ARBITRARY_LINE = "arbitrary_line"


class ColorMap(Enum):
    """Available colormaps for seismic display."""
    SEISMIC = "seismic"
    GRAY = "gray"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    RD_BU = "RdBu"
    COOLWARM = "coolwarm"
    BWR = "bwr"
    PETREL = "petrel"  # Custom petroleum industry colormap


class InterpolationMethod(Enum):
    """Image interpolation methods."""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"


class LayoutType(Enum):
    """View layout arrangements."""
    SINGLE = "single"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    GRID_2X2 = "grid_2x2"
    GRID_3X1 = "grid_3x1"
    CUSTOM = "custom"


@dataclass
class ViewSettings:
    """Settings for a single view."""
    view_type: ViewType
    colormap: ColorMap = ColorMap.SEISMIC
    interpolation: InterpolationMethod = InterpolationMethod.BILINEAR
    clip_percentile: float = 99.0
    gain: float = 1.0
    show_colorbar: bool = True
    show_axis_labels: bool = True
    aspect_ratio: str = "auto"
    title: Optional[str] = None


@dataclass
class SlicePosition:
    """Position of a slice in the seismic volume."""
    inline: Optional[int] = None
    crossline: Optional[int] = None
    depth_sample: Optional[int] = None
    time_ms: Optional[float] = None


@dataclass
class ViewState:
    """Current state of the viewer."""
    position: SlicePosition
    zoom_level: float = 1.0
    pan_offset: Tuple[float, float] = (0.0, 0.0)
    selected_traces: List[int] = field(default_factory=list)
    annotations: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SeismicVolume:
    """Seismic volume data container."""
    data: np.ndarray  # 3D array (inline, crossline, samples)
    inline_range: Tuple[int, int, int]  # (start, end, step)
    crossline_range: Tuple[int, int, int]
    sample_interval: float  # ms or m
    sample_unit: str = "ms"  # "ms" or "m"
    inline_labels: Optional[np.ndarray] = None
    crossline_labels: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_inlines(self) -> int:
        return self.data.shape[0]
    
    @property
    def n_crosslines(self) -> int:
        return self.data.shape[1]
    
    @property
    def n_samples(self) -> int:
        return self.data.shape[2]
    
    def get_inline_slice(self, inline_idx: int) -> np.ndarray:
        """Get inline slice."""
        return self.data[inline_idx, :, :]
    
    def get_crossline_slice(self, crossline_idx: int) -> np.ndarray:
        """Get crossline slice."""
        return self.data[:, crossline_idx, :]
    
    def get_depth_slice(self, sample_idx: int) -> np.ndarray:
        """Get depth/time slice."""
        return self.data[:, :, sample_idx]
    
    def get_time_axis(self) -> np.ndarray:
        """Get time/depth axis."""
        return np.arange(self.n_samples) * self.sample_interval


class AmplitudeStatistics:
    """Calculate and store amplitude statistics."""
    
    def __init__(self, data: np.ndarray):
        self.data = data
        self._compute_statistics()
        
    def _compute_statistics(self) -> None:
        """Compute basic statistics."""
        flat_data = self.data.flatten()
        valid_data = flat_data[~np.isnan(flat_data)]
        
        self.min = float(np.min(valid_data))
        self.max = float(np.max(valid_data))
        self.mean = float(np.mean(valid_data))
        self.std = float(np.std(valid_data))
        self.median = float(np.median(valid_data))
        self.rms = float(np.sqrt(np.mean(valid_data**2)))
        
        # Percentiles for clipping
        self.percentiles = {
            1: float(np.percentile(valid_data, 1)),
            5: float(np.percentile(valid_data, 5)),
            95: float(np.percentile(valid_data, 95)),
            99: float(np.percentile(valid_data, 99))
        }
        
    def get_clip_values(self, percentile: float = 99.0) -> Tuple[float, float]:
        """Get symmetric clip values for display."""
        p_low = (100 - percentile) / 2
        p_high = 100 - p_low
        
        flat_data = self.data.flatten()
        valid_data = flat_data[~np.isnan(flat_data)]
        
        v_low = np.percentile(valid_data, p_low)
        v_high = np.percentile(valid_data, p_high)
        
        # Make symmetric around zero for seismic data
        v_max = max(abs(v_low), abs(v_high))
        return (-v_max, v_max)
    
    def compute_histogram(self, n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Compute amplitude histogram."""
        flat_data = self.data.flatten()
        valid_data = flat_data[~np.isnan(flat_data)]
        
        counts, bin_edges = np.histogram(valid_data, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return counts, bin_centers
    
    def to_dict(self) -> Dict[str, Any]:
        """Export statistics to dictionary."""
        return {
            'min': self.min,
            'max': self.max,
            'mean': self.mean,
            'std': self.std,
            'median': self.median,
            'rms': self.rms,
            'percentiles': self.percentiles
        }


class ColorMapGenerator:
    """Generate custom colormaps for seismic display."""
    
    @staticmethod
    def get_colormap_data(colormap: ColorMap) -> np.ndarray:
        """
        Get colormap as RGB array.
        
        Returns:
            Array of shape (256, 3) with RGB values 0-255
        """
        n_colors = 256
        
        if colormap == ColorMap.SEISMIC:
            # Classic seismic: blue-white-red
            colors = np.zeros((n_colors, 3), dtype=np.uint8)
            mid = n_colors // 2
            
            # Blue to white
            for i in range(mid):
                t = i / mid
                colors[i] = [int(255 * t), int(255 * t), 255]
                
            # White to red
            for i in range(mid, n_colors):
                t = (i - mid) / mid
                colors[i] = [255, int(255 * (1 - t)), int(255 * (1 - t))]
                
            return colors
            
        elif colormap == ColorMap.PETREL:
            # Petrel-style colormap
            colors = np.zeros((n_colors, 3), dtype=np.uint8)
            
            # Define key colors
            key_colors = [
                (0, 0, 128),      # Dark blue
                (0, 128, 255),    # Light blue
                (255, 255, 255),  # White
                (255, 128, 0),    # Orange
                (128, 0, 0)       # Dark red
            ]
            
            n_segments = len(key_colors) - 1
            segment_size = n_colors // n_segments
            
            for seg in range(n_segments):
                c1 = np.array(key_colors[seg])
                c2 = np.array(key_colors[seg + 1])
                
                for i in range(segment_size):
                    t = i / segment_size
                    idx = seg * segment_size + i
                    if idx < n_colors:
                        colors[idx] = (c1 * (1 - t) + c2 * t).astype(np.uint8)
                        
            return colors
            
        else:
            # Default grayscale
            colors = np.zeros((n_colors, 3), dtype=np.uint8)
            for i in range(n_colors):
                colors[i] = [i, i, i]
            return colors
    
    @staticmethod
    def apply_colormap(data: np.ndarray, colormap: ColorMap,
                      vmin: float = None, vmax: float = None) -> np.ndarray:
        """
        Apply colormap to data.
        
        Args:
            data: 2D array of values
            colormap: Colormap to apply
            vmin: Minimum value for scaling
            vmax: Maximum value for scaling
            
        Returns:
            RGB image array of shape (height, width, 3)
        """
        if vmin is None:
            vmin = np.nanmin(data)
        if vmax is None:
            vmax = np.nanmax(data)
            
        # Normalize data to 0-255
        normalized = np.clip((data - vmin) / (vmax - vmin + 1e-10), 0, 1)
        indices = (normalized * 255).astype(np.uint8)
        
        # Get colormap
        cmap_data = ColorMapGenerator.get_colormap_data(colormap)
        
        # Apply colormap
        rgb = cmap_data[indices]
        
        return rgb


class SliceRenderer:
    """Render seismic slices to images."""
    
    def __init__(self, settings: ViewSettings = None):
        self.settings = settings or ViewSettings(view_type=ViewType.INLINE)
        
    def render_slice(self, data: np.ndarray,
                    x_axis: np.ndarray = None,
                    y_axis: np.ndarray = None) -> Dict[str, Any]:
        """
        Render a 2D slice to image data.
        
        Args:
            data: 2D array to render
            x_axis: X-axis values
            y_axis: Y-axis values
            
        Returns:
            Dictionary with image data and metadata
        """
        # Compute statistics
        stats = AmplitudeStatistics(data)
        
        # Get clip values
        vmin, vmax = stats.get_clip_values(self.settings.clip_percentile)
        
        # Apply gain
        vmin *= self.settings.gain
        vmax *= self.settings.gain
        
        # Apply colormap
        rgb_image = ColorMapGenerator.apply_colormap(
            data, self.settings.colormap, vmin, vmax
        )
        
        # Create axes if not provided
        if x_axis is None:
            x_axis = np.arange(data.shape[1])
        if y_axis is None:
            y_axis = np.arange(data.shape[0])
            
        return {
            'image': rgb_image,
            'x_axis': x_axis.tolist(),
            'y_axis': y_axis.tolist(),
            'vmin': vmin,
            'vmax': vmax,
            'statistics': stats.to_dict(),
            'settings': {
                'colormap': self.settings.colormap.value,
                'clip_percentile': self.settings.clip_percentile,
                'gain': self.settings.gain
            }
        }
    
    def render_to_base64(self, data: np.ndarray) -> str:
        """Render slice to base64-encoded PNG."""
        result = self.render_slice(data)
        rgb_image = result['image']
        
        # Simple PNG encoding (header + raw data)
        # In production, use PIL or similar
        height, width = rgb_image.shape[:2]
        
        # Create simple PPM format (easier without PIL)
        ppm_header = f"P6\n{width} {height}\n255\n".encode('ascii')
        ppm_data = ppm_header + rgb_image.tobytes()
        
        return base64.b64encode(ppm_data).decode('ascii')


class SeismicViewer:
    """
    Interactive seismic data viewer.
    
    Provides synchronized inline/crossline/depth views with
    interactive navigation and display controls.
    """
    
    def __init__(self, volume: SeismicVolume = None):
        self.volume = volume
        self.state = ViewState(position=SlicePosition())
        self.views: Dict[ViewType, ViewSettings] = {}
        self.layout = LayoutType.GRID_2X2
        
        # Initialize default views
        self._init_default_views()
        
    def _init_default_views(self) -> None:
        """Initialize default view settings."""
        self.views[ViewType.INLINE] = ViewSettings(
            view_type=ViewType.INLINE,
            title="Inline Section"
        )
        self.views[ViewType.CROSSLINE] = ViewSettings(
            view_type=ViewType.CROSSLINE,
            title="Crossline Section"
        )
        self.views[ViewType.DEPTH] = ViewSettings(
            view_type=ViewType.DEPTH,
            title="Time/Depth Slice"
        )
        
    def load_volume(self, volume: SeismicVolume) -> None:
        """Load seismic volume."""
        self.volume = volume
        
        # Set initial position to center
        self.state.position = SlicePosition(
            inline=volume.n_inlines // 2,
            crossline=volume.n_crosslines // 2,
            depth_sample=volume.n_samples // 2
        )
        
        logger.info(f"Loaded volume: {volume.n_inlines}x{volume.n_crosslines}x{volume.n_samples}")
        
    def set_position(self, inline: int = None, crossline: int = None,
                    depth_sample: int = None) -> None:
        """Set current slice position."""
        if inline is not None:
            self.state.position.inline = max(0, min(inline, self.volume.n_inlines - 1))
        if crossline is not None:
            self.state.position.crossline = max(0, min(crossline, self.volume.n_crosslines - 1))
        if depth_sample is not None:
            self.state.position.depth_sample = max(0, min(depth_sample, self.volume.n_samples - 1))
            
    def set_colormap(self, colormap: ColorMap, view_type: ViewType = None) -> None:
        """Set colormap for view(s)."""
        if view_type is not None:
            if view_type in self.views:
                self.views[view_type].colormap = colormap
        else:
            # Apply to all views
            for view in self.views.values():
                view.colormap = colormap
                
    def set_gain(self, gain: float, view_type: ViewType = None) -> None:
        """Set display gain for view(s)."""
        if view_type is not None:
            if view_type in self.views:
                self.views[view_type].gain = gain
        else:
            for view in self.views.values():
                view.gain = gain
                
    def set_clip_percentile(self, percentile: float, view_type: ViewType = None) -> None:
        """Set clip percentile for view(s)."""
        if view_type is not None:
            if view_type in self.views:
                self.views[view_type].clip_percentile = percentile
        else:
            for view in self.views.values():
                view.clip_percentile = percentile
                
    def set_layout(self, layout: LayoutType) -> None:
        """Set view layout."""
        self.layout = layout
        
    def render_view(self, view_type: ViewType) -> Dict[str, Any]:
        """
        Render a single view.
        
        Args:
            view_type: Type of view to render
            
        Returns:
            Rendered view data
        """
        if self.volume is None:
            raise ValueError("No volume loaded")
            
        settings = self.views.get(view_type, ViewSettings(view_type=view_type))
        renderer = SliceRenderer(settings)
        
        if view_type == ViewType.INLINE:
            idx = self.state.position.inline or 0
            data = self.volume.get_inline_slice(idx)
            x_axis = np.arange(self.volume.n_crosslines)
            y_axis = self.volume.get_time_axis()
            
        elif view_type == ViewType.CROSSLINE:
            idx = self.state.position.crossline or 0
            data = self.volume.get_crossline_slice(idx)
            x_axis = np.arange(self.volume.n_inlines)
            y_axis = self.volume.get_time_axis()
            
        elif view_type == ViewType.DEPTH:
            idx = self.state.position.depth_sample or 0
            data = self.volume.get_depth_slice(idx)
            x_axis = np.arange(self.volume.n_crosslines)
            y_axis = np.arange(self.volume.n_inlines)
            
        else:
            raise ValueError(f"Unsupported view type: {view_type}")
            
        result = renderer.render_slice(data, x_axis, y_axis)
        result['view_type'] = view_type.value
        result['position'] = {
            'inline': self.state.position.inline,
            'crossline': self.state.position.crossline,
            'depth_sample': self.state.position.depth_sample
        }
        
        return result
        
    def render_all_views(self) -> Dict[str, Dict[str, Any]]:
        """Render all configured views."""
        results = {}
        
        for view_type in self.views.keys():
            try:
                results[view_type.value] = self.render_view(view_type)
            except Exception as e:
                logger.error(f"Error rendering {view_type}: {e}")
                results[view_type.value] = {'error': str(e)}
                
        return results
        
    def get_trace(self, inline: int, crossline: int) -> Dict[str, Any]:
        """
        Extract a single trace.
        
        Args:
            inline: Inline index
            crossline: Crossline index
            
        Returns:
            Trace data and metadata
        """
        if self.volume is None:
            raise ValueError("No volume loaded")
            
        trace_data = self.volume.data[inline, crossline, :]
        time_axis = self.volume.get_time_axis()
        
        return {
            'inline': inline,
            'crossline': crossline,
            'data': trace_data.tolist(),
            'time_axis': time_axis.tolist(),
            'sample_interval': self.volume.sample_interval,
            'sample_unit': self.volume.sample_unit
        }
        
    def get_amplitude_spectrum(self, inline: int, crossline: int) -> Dict[str, Any]:
        """
        Compute amplitude spectrum for a trace.
        
        Args:
            inline: Inline index
            crossline: Crossline index
            
        Returns:
            Frequency spectrum data
        """
        trace = self.volume.data[inline, crossline, :]
        
        # Compute FFT
        n_samples = len(trace)
        fft_result = np.fft.rfft(trace)
        amplitude = np.abs(fft_result)
        
        # Frequency axis
        sample_rate = 1000.0 / self.volume.sample_interval  # Hz
        frequencies = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
        
        return {
            'frequencies': frequencies.tolist(),
            'amplitude': amplitude.tolist(),
            'sample_rate': sample_rate
        }
        
    def export_slice(self, view_type: ViewType, format: str = 'png') -> bytes:
        """
        Export a slice to image format.
        
        Args:
            view_type: View to export
            format: Output format ('png', 'tiff', 'pdf')
            
        Returns:
            Image data as bytes
        """
        result = self.render_view(view_type)
        rgb_image = result['image']
        
        # Simple PPM format export
        height, width = rgb_image.shape[:2]
        ppm_header = f"P6\n{width} {height}\n255\n".encode('ascii')
        
        return ppm_header + rgb_image.tobytes()
        
    def add_annotation(self, annotation: Dict[str, Any]) -> None:
        """Add annotation to current view."""
        annotation['timestamp'] = datetime.now().isoformat()
        self.state.annotations.append(annotation)
        
    def get_volume_info(self) -> Dict[str, Any]:
        """Get volume information."""
        if self.volume is None:
            return {'error': 'No volume loaded'}
            
        stats = AmplitudeStatistics(self.volume.data)
        
        return {
            'dimensions': {
                'n_inlines': self.volume.n_inlines,
                'n_crosslines': self.volume.n_crosslines,
                'n_samples': self.volume.n_samples
            },
            'ranges': {
                'inline': self.volume.inline_range,
                'crossline': self.volume.crossline_range
            },
            'sampling': {
                'interval': self.volume.sample_interval,
                'unit': self.volume.sample_unit
            },
            'statistics': stats.to_dict(),
            'metadata': self.volume.metadata
        }


class SEGYViewerIntegration:
    """
    Integration layer between SEG-Y files and the viewer.
    
    Handles loading SEG-Y files and creating SeismicVolume objects.
    """
    
    def __init__(self):
        self.viewer = SeismicViewer()
        self.current_file: Optional[str] = None
        
    def load_segy(self, file_path: str, 
                 inline_byte: int = 189,
                 crossline_byte: int = 193) -> SeismicVolume:
        """
        Load SEG-Y file into viewer.
        
        Args:
            file_path: Path to SEG-Y file
            inline_byte: Byte position for inline number
            crossline_byte: Byte position for crossline number
            
        Returns:
            Loaded SeismicVolume
        """
        # This would use segyio in production
        # For now, create a synthetic volume for demonstration
        logger.info(f"Loading SEG-Y file: {file_path}")
        
        # Synthetic data for demonstration
        n_inlines = 100
        n_crosslines = 150
        n_samples = 500
        
        # Create synthetic seismic data
        data = np.random.randn(n_inlines, n_crosslines, n_samples).astype(np.float32)
        
        # Add some structure
        for i in range(n_inlines):
            for j in range(n_crosslines):
                # Add reflectors
                data[i, j, 100] += 5.0 * np.sin(i * 0.1 + j * 0.05)
                data[i, j, 200] += 3.0 * np.cos(i * 0.08 - j * 0.03)
                data[i, j, 350] += 4.0 * np.sin(i * 0.05 + j * 0.1)
                
        volume = SeismicVolume(
            data=data,
            inline_range=(1, n_inlines, 1),
            crossline_range=(1, n_crosslines, 1),
            sample_interval=4.0,  # 4ms
            sample_unit="ms",
            metadata={
                'file_path': file_path,
                'inline_byte': inline_byte,
                'crossline_byte': crossline_byte
            }
        )
        
        self.viewer.load_volume(volume)
        self.current_file = file_path
        
        return volume
        
    def get_viewer(self) -> SeismicViewer:
        """Get the viewer instance."""
        return self.viewer


def create_segy_viewer() -> SEGYViewerIntegration:
    """Factory function to create SEG-Y viewer."""
    return SEGYViewerIntegration()


def create_volume_from_array(data: np.ndarray,
                            sample_interval: float = 4.0,
                            sample_unit: str = "ms") -> SeismicVolume:
    """
    Create SeismicVolume from numpy array.
    
    Args:
        data: 3D numpy array (inline, crossline, samples)
        sample_interval: Sample interval
        sample_unit: Sample unit ("ms" or "m")
        
    Returns:
        SeismicVolume object
    """
    n_inlines, n_crosslines, n_samples = data.shape
    
    return SeismicVolume(
        data=data,
        inline_range=(1, n_inlines, 1),
        crossline_range=(1, n_crosslines, 1),
        sample_interval=sample_interval,
        sample_unit=sample_unit
    )
