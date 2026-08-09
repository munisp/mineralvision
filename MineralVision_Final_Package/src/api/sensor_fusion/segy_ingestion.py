"""
SEG-Y Seismic Data Ingestion Pipeline for MineralVision.

This module provides comprehensive SEG-Y file handling including:
- Robust SEG-Y I/O with cloud storage support (fsspec)
- Schema-driven header parsing for different contractor formats
- Support for SEG-Y Rev 0, 1, 2, 2.1 standards
- Trace indexing by inline/xline or CDP
- Integration with lakehouse for chunked storage
- Custom schema support for non-standard formats

Based on TGSAI/segy library patterns and SEG technical standards.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, List, Tuple, Any, Optional, Union, Iterator, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from abc import ABC, abstractmethod
import struct
import io
import logging
import json
import os

logger = logging.getLogger(__name__)


class SEGYRevision(Enum):
    """SEG-Y revision standards."""
    REV0 = "0"
    REV1 = "1"
    REV2 = "2"
    REV2_1 = "2.1"


class DataSampleFormat(Enum):
    """SEG-Y data sample formats."""
    IBM_FLOAT_4 = 1
    INT_4 = 2
    INT_2 = 3
    FIXED_POINT = 4  # Obsolete
    IEEE_FLOAT_4 = 5
    IEEE_FLOAT_8 = 8
    INT_1 = 8
    INT_8 = 9
    UINT_4 = 10
    UINT_2 = 11
    UINT_8 = 12
    UINT_3 = 15  # 24-bit unsigned
    INT_3 = 16   # 24-bit signed


class SortingCode(Enum):
    """Trace sorting codes."""
    UNKNOWN = -1
    AS_RECORDED = 0
    ENSEMBLE = 1
    CDP = 2
    SINGLE_FOLD = 3
    HORIZONTALLY_STACKED = 4
    COMMON_SOURCE = 5
    COMMON_RECEIVER = 6
    COMMON_OFFSET = 7
    COMMON_MID_POINT = 8
    COMMON_CONVERSION_POINT = 9


@dataclass
class TextHeader:
    """SEG-Y textual file header (3200 bytes)."""
    text: str
    encoding: str = 'ebcdic'  # or 'ascii'
    
    @classmethod
    def from_bytes(cls, data: bytes, encoding: str = 'ebcdic') -> 'TextHeader':
        """Parse text header from bytes."""
        if encoding == 'ebcdic':
            try:
                text = data.decode('cp500')  # EBCDIC
            except:
                text = data.decode('ascii', errors='replace')
        else:
            text = data.decode('ascii', errors='replace')
            
        # Format into 40 lines of 80 characters
        lines = [text[i:i+80] for i in range(0, 3200, 80)]
        formatted_text = '\n'.join(lines)
        
        return cls(text=formatted_text, encoding=encoding)
        
    def to_bytes(self) -> bytes:
        """Convert text header to bytes."""
        # Remove newlines and pad to 3200 bytes
        text = self.text.replace('\n', '')
        text = text[:3200].ljust(3200)
        
        if self.encoding == 'ebcdic':
            return text.encode('cp500')
        else:
            return text.encode('ascii')


@dataclass
class BinaryHeader:
    """SEG-Y binary file header (400 bytes)."""
    job_id: int = 0
    line_number: int = 0
    reel_number: int = 0
    traces_per_ensemble: int = 1
    aux_traces_per_ensemble: int = 0
    sample_interval: int = 4000  # microseconds
    sample_interval_original: int = 4000
    samples_per_trace: int = 1000
    samples_per_trace_original: int = 1000
    data_sample_format: DataSampleFormat = DataSampleFormat.IEEE_FLOAT_4
    ensemble_fold: int = 1
    trace_sorting: SortingCode = SortingCode.CDP
    vertical_sum: int = 1
    sweep_freq_start: int = 0
    sweep_freq_end: int = 0
    sweep_length: int = 0
    sweep_type: int = 0
    trace_number_sweep: int = 0
    sweep_taper_start: int = 0
    sweep_taper_end: int = 0
    taper_type: int = 0
    correlated_traces: int = 0
    binary_gain: int = 0
    amp_recovery_method: int = 0
    measurement_system: int = 1  # 1=meters, 2=feet
    impulse_polarity: int = 0
    vibratory_polarity: int = 0
    segy_revision: SEGYRevision = SEGYRevision.REV1
    fixed_length_trace: int = 1
    num_extended_headers: int = 0
    
    @classmethod
    def from_bytes(cls, data: bytes, endian: str = '>') -> 'BinaryHeader':
        """Parse binary header from bytes."""
        header = cls()
        
        # Unpack key fields (big-endian by default)
        header.job_id = struct.unpack(f'{endian}i', data[0:4])[0]
        header.line_number = struct.unpack(f'{endian}i', data[4:8])[0]
        header.reel_number = struct.unpack(f'{endian}i', data[8:12])[0]
        header.traces_per_ensemble = struct.unpack(f'{endian}h', data[12:14])[0]
        header.aux_traces_per_ensemble = struct.unpack(f'{endian}h', data[14:16])[0]
        header.sample_interval = struct.unpack(f'{endian}h', data[16:18])[0]
        header.sample_interval_original = struct.unpack(f'{endian}h', data[18:20])[0]
        header.samples_per_trace = struct.unpack(f'{endian}h', data[20:22])[0]
        header.samples_per_trace_original = struct.unpack(f'{endian}h', data[22:24])[0]
        
        format_code = struct.unpack(f'{endian}h', data[24:26])[0]
        try:
            header.data_sample_format = DataSampleFormat(format_code)
        except ValueError:
            header.data_sample_format = DataSampleFormat.IEEE_FLOAT_4
            
        header.ensemble_fold = struct.unpack(f'{endian}h', data[26:28])[0]
        
        sorting_code = struct.unpack(f'{endian}h', data[28:30])[0]
        try:
            header.trace_sorting = SortingCode(sorting_code)
        except ValueError:
            header.trace_sorting = SortingCode.UNKNOWN
            
        header.measurement_system = struct.unpack(f'{endian}h', data[54:56])[0]
        
        # SEG-Y revision (bytes 300-301)
        if len(data) >= 302:
            rev_major = struct.unpack(f'{endian}B', data[300:301])[0]
            rev_minor = struct.unpack(f'{endian}B', data[301:302])[0]
            
            if rev_major == 0:
                header.segy_revision = SEGYRevision.REV0
            elif rev_major == 1:
                header.segy_revision = SEGYRevision.REV1
            elif rev_major == 2:
                if rev_minor == 0:
                    header.segy_revision = SEGYRevision.REV2
                else:
                    header.segy_revision = SEGYRevision.REV2_1
                    
        # Fixed length trace flag (bytes 302-303)
        if len(data) >= 304:
            header.fixed_length_trace = struct.unpack(f'{endian}h', data[302:304])[0]
            
        # Number of extended headers (bytes 304-305)
        if len(data) >= 306:
            header.num_extended_headers = struct.unpack(f'{endian}h', data[304:306])[0]
            
        return header
        
    def to_bytes(self, endian: str = '>') -> bytes:
        """Convert binary header to bytes."""
        data = bytearray(400)
        
        struct.pack_into(f'{endian}i', data, 0, self.job_id)
        struct.pack_into(f'{endian}i', data, 4, self.line_number)
        struct.pack_into(f'{endian}i', data, 8, self.reel_number)
        struct.pack_into(f'{endian}h', data, 12, self.traces_per_ensemble)
        struct.pack_into(f'{endian}h', data, 14, self.aux_traces_per_ensemble)
        struct.pack_into(f'{endian}h', data, 16, self.sample_interval)
        struct.pack_into(f'{endian}h', data, 18, self.sample_interval_original)
        struct.pack_into(f'{endian}h', data, 20, self.samples_per_trace)
        struct.pack_into(f'{endian}h', data, 22, self.samples_per_trace_original)
        struct.pack_into(f'{endian}h', data, 24, self.data_sample_format.value)
        struct.pack_into(f'{endian}h', data, 26, self.ensemble_fold)
        struct.pack_into(f'{endian}h', data, 28, self.trace_sorting.value)
        struct.pack_into(f'{endian}h', data, 54, self.measurement_system)
        
        # SEG-Y revision
        if self.segy_revision == SEGYRevision.REV0:
            struct.pack_into(f'{endian}BB', data, 300, 0, 0)
        elif self.segy_revision == SEGYRevision.REV1:
            struct.pack_into(f'{endian}BB', data, 300, 1, 0)
        elif self.segy_revision == SEGYRevision.REV2:
            struct.pack_into(f'{endian}BB', data, 300, 2, 0)
        else:
            struct.pack_into(f'{endian}BB', data, 300, 2, 1)
            
        struct.pack_into(f'{endian}h', data, 302, self.fixed_length_trace)
        struct.pack_into(f'{endian}h', data, 304, self.num_extended_headers)
        
        return bytes(data)


@dataclass
class TraceHeader:
    """SEG-Y trace header (240 bytes)."""
    trace_sequence_line: int = 0
    trace_sequence_file: int = 0
    field_record: int = 0
    trace_number_field: int = 0
    energy_source_point: int = 0
    cdp: int = 0
    trace_number_ensemble: int = 0
    trace_id: int = 1
    num_vertically_summed: int = 1
    num_horizontally_stacked: int = 1
    data_use: int = 1
    source_receiver_offset: int = 0
    receiver_elevation: int = 0
    source_elevation: int = 0
    source_depth: int = 0
    receiver_datum_elevation: int = 0
    source_datum_elevation: int = 0
    water_depth_source: int = 0
    water_depth_receiver: int = 0
    scalar_elevation: int = 1
    scalar_coordinates: int = 1
    source_x: int = 0
    source_y: int = 0
    receiver_x: int = 0
    receiver_y: int = 0
    coordinate_units: int = 1
    weathering_velocity: int = 0
    subweathering_velocity: int = 0
    uphole_time_source: int = 0
    uphole_time_receiver: int = 0
    source_static: int = 0
    receiver_static: int = 0
    total_static: int = 0
    lag_time_a: int = 0
    lag_time_b: int = 0
    delay_recording_time: int = 0
    mute_time_start: int = 0
    mute_time_end: int = 0
    num_samples: int = 0
    sample_interval: int = 0
    gain_type: int = 0
    instrument_gain: int = 0
    instrument_early_gain: int = 0
    correlated: int = 0
    sweep_freq_start: int = 0
    sweep_freq_end: int = 0
    sweep_length: int = 0
    sweep_type: int = 0
    sweep_taper_start: int = 0
    sweep_taper_end: int = 0
    taper_type: int = 0
    alias_filter_freq: int = 0
    alias_filter_slope: int = 0
    notch_filter_freq: int = 0
    notch_filter_slope: int = 0
    low_cut_freq: int = 0
    high_cut_freq: int = 0
    low_cut_slope: int = 0
    high_cut_slope: int = 0
    year: int = 0
    day_of_year: int = 0
    hour: int = 0
    minute: int = 0
    second: int = 0
    time_basis: int = 1
    trace_weighting_factor: int = 0
    geophone_group_roll: int = 0
    geophone_group_first: int = 0
    geophone_group_last: int = 0
    gap_size: int = 0
    over_travel: int = 0
    cdp_x: int = 0
    cdp_y: int = 0
    inline: int = 0
    crossline: int = 0
    shotpoint: int = 0
    shotpoint_scalar: int = 1
    trace_value_unit: int = 0
    transduction_constant: int = 0
    transduction_units: int = 0
    device_id: int = 0
    scalar_times: int = 1
    source_type: int = 0
    source_energy_direction: int = 0
    source_measurement: int = 0
    source_measurement_unit: int = 0
    
    # Custom fields for extended headers
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_bytes(cls, data: bytes, endian: str = '>') -> 'TraceHeader':
        """Parse trace header from bytes."""
        header = cls()
        
        header.trace_sequence_line = struct.unpack(f'{endian}i', data[0:4])[0]
        header.trace_sequence_file = struct.unpack(f'{endian}i', data[4:8])[0]
        header.field_record = struct.unpack(f'{endian}i', data[8:12])[0]
        header.trace_number_field = struct.unpack(f'{endian}i', data[12:16])[0]
        header.energy_source_point = struct.unpack(f'{endian}i', data[16:20])[0]
        header.cdp = struct.unpack(f'{endian}i', data[20:24])[0]
        header.trace_number_ensemble = struct.unpack(f'{endian}i', data[24:28])[0]
        header.trace_id = struct.unpack(f'{endian}h', data[28:30])[0]
        header.num_vertically_summed = struct.unpack(f'{endian}h', data[30:32])[0]
        header.num_horizontally_stacked = struct.unpack(f'{endian}h', data[32:34])[0]
        header.data_use = struct.unpack(f'{endian}h', data[34:36])[0]
        header.source_receiver_offset = struct.unpack(f'{endian}i', data[36:40])[0]
        header.receiver_elevation = struct.unpack(f'{endian}i', data[40:44])[0]
        header.source_elevation = struct.unpack(f'{endian}i', data[44:48])[0]
        header.source_depth = struct.unpack(f'{endian}i', data[48:52])[0]
        header.scalar_elevation = struct.unpack(f'{endian}h', data[68:70])[0]
        header.scalar_coordinates = struct.unpack(f'{endian}h', data[70:72])[0]
        header.source_x = struct.unpack(f'{endian}i', data[72:76])[0]
        header.source_y = struct.unpack(f'{endian}i', data[76:80])[0]
        header.receiver_x = struct.unpack(f'{endian}i', data[80:84])[0]
        header.receiver_y = struct.unpack(f'{endian}i', data[84:88])[0]
        header.coordinate_units = struct.unpack(f'{endian}h', data[88:90])[0]
        header.num_samples = struct.unpack(f'{endian}h', data[114:116])[0]
        header.sample_interval = struct.unpack(f'{endian}h', data[116:118])[0]
        header.year = struct.unpack(f'{endian}h', data[156:158])[0]
        header.day_of_year = struct.unpack(f'{endian}h', data[158:160])[0]
        header.hour = struct.unpack(f'{endian}h', data[160:162])[0]
        header.minute = struct.unpack(f'{endian}h', data[162:164])[0]
        header.second = struct.unpack(f'{endian}h', data[164:166])[0]
        header.cdp_x = struct.unpack(f'{endian}i', data[180:184])[0]
        header.cdp_y = struct.unpack(f'{endian}i', data[184:188])[0]
        header.inline = struct.unpack(f'{endian}i', data[188:192])[0]
        header.crossline = struct.unpack(f'{endian}i', data[192:196])[0]
        header.shotpoint = struct.unpack(f'{endian}i', data[196:200])[0]
        
        return header
        
    def to_bytes(self, endian: str = '>') -> bytes:
        """Convert trace header to bytes."""
        data = bytearray(240)
        
        struct.pack_into(f'{endian}i', data, 0, self.trace_sequence_line)
        struct.pack_into(f'{endian}i', data, 4, self.trace_sequence_file)
        struct.pack_into(f'{endian}i', data, 8, self.field_record)
        struct.pack_into(f'{endian}i', data, 12, self.trace_number_field)
        struct.pack_into(f'{endian}i', data, 16, self.energy_source_point)
        struct.pack_into(f'{endian}i', data, 20, self.cdp)
        struct.pack_into(f'{endian}i', data, 24, self.trace_number_ensemble)
        struct.pack_into(f'{endian}h', data, 28, self.trace_id)
        struct.pack_into(f'{endian}h', data, 30, self.num_vertically_summed)
        struct.pack_into(f'{endian}h', data, 32, self.num_horizontally_stacked)
        struct.pack_into(f'{endian}h', data, 34, self.data_use)
        struct.pack_into(f'{endian}i', data, 36, self.source_receiver_offset)
        struct.pack_into(f'{endian}i', data, 40, self.receiver_elevation)
        struct.pack_into(f'{endian}i', data, 44, self.source_elevation)
        struct.pack_into(f'{endian}i', data, 48, self.source_depth)
        struct.pack_into(f'{endian}h', data, 68, self.scalar_elevation)
        struct.pack_into(f'{endian}h', data, 70, self.scalar_coordinates)
        struct.pack_into(f'{endian}i', data, 72, self.source_x)
        struct.pack_into(f'{endian}i', data, 76, self.source_y)
        struct.pack_into(f'{endian}i', data, 80, self.receiver_x)
        struct.pack_into(f'{endian}i', data, 84, self.receiver_y)
        struct.pack_into(f'{endian}h', data, 88, self.coordinate_units)
        struct.pack_into(f'{endian}h', data, 114, self.num_samples)
        struct.pack_into(f'{endian}h', data, 116, self.sample_interval)
        struct.pack_into(f'{endian}h', data, 156, self.year)
        struct.pack_into(f'{endian}h', data, 158, self.day_of_year)
        struct.pack_into(f'{endian}h', data, 160, self.hour)
        struct.pack_into(f'{endian}h', data, 162, self.minute)
        struct.pack_into(f'{endian}h', data, 164, self.second)
        struct.pack_into(f'{endian}i', data, 180, self.cdp_x)
        struct.pack_into(f'{endian}i', data, 184, self.cdp_y)
        struct.pack_into(f'{endian}i', data, 188, self.inline)
        struct.pack_into(f'{endian}i', data, 192, self.crossline)
        struct.pack_into(f'{endian}i', data, 196, self.shotpoint)
        
        return bytes(data)
        
    def get_coordinates(self) -> Tuple[float, float]:
        """Get scaled coordinates."""
        scalar = self.scalar_coordinates
        if scalar < 0:
            scalar = -1.0 / scalar
        elif scalar == 0:
            scalar = 1.0
            
        x = self.cdp_x * scalar if self.cdp_x != 0 else self.source_x * scalar
        y = self.cdp_y * scalar if self.cdp_y != 0 else self.source_y * scalar
        
        return x, y


@dataclass
class Trace:
    """SEG-Y trace with header and data."""
    header: TraceHeader
    data: np.ndarray
    
    def get_time_axis(self, start_time: float = 0.0) -> np.ndarray:
        """Get time axis for trace in seconds."""
        dt = self.header.sample_interval / 1e6  # Convert microseconds to seconds
        return start_time + np.arange(len(self.data)) * dt


@dataclass
class HeaderSchema:
    """Schema for custom header field mapping."""
    name: str
    byte_offset: int
    byte_length: int
    data_type: str  # 'int16', 'int32', 'float32', 'float64'
    endian: str = '>'
    scalar_field: Optional[str] = None
    description: str = ""
    
    def read_value(self, data: bytes) -> Any:
        """Read value from header bytes."""
        format_map = {
            'int16': 'h',
            'int32': 'i',
            'float32': 'f',
            'float64': 'd',
            'uint16': 'H',
            'uint32': 'I'
        }
        
        fmt = format_map.get(self.data_type, 'i')
        end = self.byte_offset + self.byte_length
        
        return struct.unpack(f'{self.endian}{fmt}', data[self.byte_offset:end])[0]


class HeaderSchemaRegistry:
    """
    Registry for custom header schemas.
    
    Allows different contractors' SEG-Y variants to be parsed consistently.
    """
    
    # Standard SEG-Y Rev 1 trace header schema
    STANDARD_SCHEMA = {
        'trace_sequence_line': HeaderSchema('trace_sequence_line', 0, 4, 'int32'),
        'trace_sequence_file': HeaderSchema('trace_sequence_file', 4, 4, 'int32'),
        'field_record': HeaderSchema('field_record', 8, 4, 'int32'),
        'trace_number_field': HeaderSchema('trace_number_field', 12, 4, 'int32'),
        'cdp': HeaderSchema('cdp', 20, 4, 'int32'),
        'trace_number_ensemble': HeaderSchema('trace_number_ensemble', 24, 4, 'int32'),
        'source_receiver_offset': HeaderSchema('source_receiver_offset', 36, 4, 'int32'),
        'scalar_coordinates': HeaderSchema('scalar_coordinates', 70, 2, 'int16'),
        'source_x': HeaderSchema('source_x', 72, 4, 'int32'),
        'source_y': HeaderSchema('source_y', 76, 4, 'int32'),
        'receiver_x': HeaderSchema('receiver_x', 80, 4, 'int32'),
        'receiver_y': HeaderSchema('receiver_y', 84, 4, 'int32'),
        'num_samples': HeaderSchema('num_samples', 114, 2, 'int16'),
        'sample_interval': HeaderSchema('sample_interval', 116, 2, 'int16'),
        'cdp_x': HeaderSchema('cdp_x', 180, 4, 'int32'),
        'cdp_y': HeaderSchema('cdp_y', 184, 4, 'int32'),
        'inline': HeaderSchema('inline', 188, 4, 'int32'),
        'crossline': HeaderSchema('crossline', 192, 4, 'int32'),
    }
    
    def __init__(self):
        self.schemas: Dict[str, Dict[str, HeaderSchema]] = {
            'standard': dict(self.STANDARD_SCHEMA)
        }
        
    def register_schema(self, name: str, schema: Dict[str, HeaderSchema]) -> None:
        """Register a custom schema."""
        self.schemas[name] = schema
        
    def get_schema(self, name: str) -> Dict[str, HeaderSchema]:
        """Get a schema by name."""
        return self.schemas.get(name, self.schemas['standard'])
        
    def parse_header(self, data: bytes, schema_name: str = 'standard') -> Dict[str, Any]:
        """Parse header using specified schema."""
        schema = self.get_schema(schema_name)
        result = {}
        
        for field_name, field_schema in schema.items():
            try:
                result[field_name] = field_schema.read_value(data)
            except Exception as e:
                logger.warning(f"Failed to read field {field_name}: {e}")
                result[field_name] = 0
                
        return result
        
    def export_schema(self, name: str) -> str:
        """Export schema to JSON."""
        schema = self.get_schema(name)
        export = {}
        
        for field_name, field_schema in schema.items():
            export[field_name] = {
                'byte_offset': field_schema.byte_offset,
                'byte_length': field_schema.byte_length,
                'data_type': field_schema.data_type,
                'endian': field_schema.endian,
                'description': field_schema.description
            }
            
        return json.dumps(export, indent=2)
        
    def import_schema(self, name: str, json_str: str) -> None:
        """Import schema from JSON."""
        data = json.loads(json_str)
        schema = {}
        
        for field_name, field_data in data.items():
            schema[field_name] = HeaderSchema(
                name=field_name,
                byte_offset=field_data['byte_offset'],
                byte_length=field_data['byte_length'],
                data_type=field_data['data_type'],
                endian=field_data.get('endian', '>'),
                description=field_data.get('description', '')
            )
            
        self.register_schema(name, schema)


class DataFormatConverter:
    """Convert between SEG-Y data formats."""
    
    @staticmethod
    def ibm_to_ieee(ibm_float: bytes, endian: str = '>') -> float:
        """Convert IBM floating point to IEEE."""
        # Unpack as unsigned int
        uint = struct.unpack(f'{endian}I', ibm_float)[0]
        
        if uint == 0:
            return 0.0
            
        # Extract components
        sign = (uint >> 31) & 1
        exponent = (uint >> 24) & 0x7F
        fraction = uint & 0x00FFFFFF
        
        # Convert
        value = fraction / (2**24) * (16 ** (exponent - 64))
        
        if sign:
            value = -value
            
        return value
        
    @staticmethod
    def ieee_to_ibm(value: float, endian: str = '>') -> bytes:
        """Convert IEEE floating point to IBM."""
        if value == 0:
            return struct.pack(f'{endian}I', 0)
            
        sign = 0 if value >= 0 else 1
        value = abs(value)
        
        # Find exponent
        exponent = 64
        while value >= 1:
            value /= 16
            exponent += 1
        while value < 1/16:
            value *= 16
            exponent -= 1
            
        # Compute fraction
        fraction = int(value * (2**24))
        
        # Pack
        uint = (sign << 31) | (exponent << 24) | (fraction & 0x00FFFFFF)
        
        return struct.pack(f'{endian}I', uint)
        
    @staticmethod
    def read_samples(data: bytes, format_code: DataSampleFormat, 
                    num_samples: int, endian: str = '>') -> np.ndarray:
        """Read trace samples in specified format."""
        if format_code == DataSampleFormat.IBM_FLOAT_4:
            samples = np.zeros(num_samples)
            for i in range(num_samples):
                samples[i] = DataFormatConverter.ibm_to_ieee(
                    data[i*4:(i+1)*4], endian
                )
            return samples
            
        elif format_code == DataSampleFormat.IEEE_FLOAT_4:
            return np.frombuffer(data[:num_samples*4], 
                               dtype=f'{endian}f4').copy()
            
        elif format_code == DataSampleFormat.IEEE_FLOAT_8:
            return np.frombuffer(data[:num_samples*8], 
                               dtype=f'{endian}f8').copy()
            
        elif format_code == DataSampleFormat.INT_4:
            return np.frombuffer(data[:num_samples*4], 
                               dtype=f'{endian}i4').astype(float).copy()
            
        elif format_code == DataSampleFormat.INT_2:
            return np.frombuffer(data[:num_samples*2], 
                               dtype=f'{endian}i2').astype(float).copy()
            
        elif format_code == DataSampleFormat.INT_1:
            return np.frombuffer(data[:num_samples], 
                               dtype='i1').astype(float).copy()
            
        else:
            # Default to IEEE float
            return np.frombuffer(data[:num_samples*4], 
                               dtype=f'{endian}f4').copy()


class SEGYReader:
    """
    SEG-Y file reader with streaming support.
    """
    
    def __init__(self, file_path: str = None, file_obj: BinaryIO = None,
                 endian: str = '>'):
        self.file_path = file_path
        self.file_obj = file_obj
        self.endian = endian
        self.text_header: Optional[TextHeader] = None
        self.binary_header: Optional[BinaryHeader] = None
        self.extended_headers: List[TextHeader] = []
        self._trace_offsets: List[int] = []
        self._is_open = False
        
    def open(self) -> None:
        """Open the SEG-Y file."""
        if self.file_obj is None:
            # Support for fsspec-style paths
            if self.file_path.startswith('s3://'):
                try:
                    import fsspec
                    fs = fsspec.filesystem('s3')
                    self.file_obj = fs.open(self.file_path, 'rb')
                except ImportError:
                    raise ImportError("fsspec required for cloud storage support")
            elif self.file_path.startswith('gs://'):
                try:
                    import fsspec
                    fs = fsspec.filesystem('gcs')
                    self.file_obj = fs.open(self.file_path, 'rb')
                except ImportError:
                    raise ImportError("fsspec required for cloud storage support")
            elif self.file_path.startswith('az://'):
                try:
                    import fsspec
                    fs = fsspec.filesystem('abfs')
                    self.file_obj = fs.open(self.file_path, 'rb')
                except ImportError:
                    raise ImportError("fsspec required for cloud storage support")
            else:
                self.file_obj = open(self.file_path, 'rb')
                
        self._is_open = True
        self._read_headers()
        
    def close(self) -> None:
        """Close the SEG-Y file."""
        if self.file_obj is not None and self._is_open:
            self.file_obj.close()
            self._is_open = False
            
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def _read_headers(self) -> None:
        """Read file headers."""
        # Text header (3200 bytes)
        text_data = self.file_obj.read(3200)
        self.text_header = TextHeader.from_bytes(text_data)
        
        # Binary header (400 bytes)
        binary_data = self.file_obj.read(400)
        self.binary_header = BinaryHeader.from_bytes(binary_data, self.endian)
        
        # Extended text headers
        for _ in range(self.binary_header.num_extended_headers):
            ext_data = self.file_obj.read(3200)
            self.extended_headers.append(TextHeader.from_bytes(ext_data))
            
    def _get_trace_size(self) -> int:
        """Get size of a single trace in bytes."""
        samples = self.binary_header.samples_per_trace
        format_code = self.binary_header.data_sample_format
        
        bytes_per_sample = {
            DataSampleFormat.IBM_FLOAT_4: 4,
            DataSampleFormat.IEEE_FLOAT_4: 4,
            DataSampleFormat.IEEE_FLOAT_8: 8,
            DataSampleFormat.INT_4: 4,
            DataSampleFormat.INT_2: 2,
            DataSampleFormat.INT_1: 1,
        }.get(format_code, 4)
        
        return 240 + samples * bytes_per_sample
        
    def build_index(self) -> pd.DataFrame:
        """
        Build trace index for fast access.
        
        Returns:
            DataFrame with trace header information
        """
        self._trace_offsets = []
        index_data = []
        
        # Start after headers
        header_size = 3600 + self.binary_header.num_extended_headers * 3200
        self.file_obj.seek(header_size)
        
        trace_size = self._get_trace_size()
        trace_num = 0
        
        while True:
            offset = self.file_obj.tell()
            header_data = self.file_obj.read(240)
            
            if len(header_data) < 240:
                break
                
            header = TraceHeader.from_bytes(header_data, self.endian)
            self._trace_offsets.append(offset)
            
            x, y = header.get_coordinates()
            
            index_data.append({
                'trace_num': trace_num,
                'offset': offset,
                'inline': header.inline,
                'crossline': header.crossline,
                'cdp': header.cdp,
                'x': x,
                'y': y,
                'field_record': header.field_record,
                'trace_number_field': header.trace_number_field
            })
            
            # Skip trace data
            data_size = trace_size - 240
            self.file_obj.seek(data_size, 1)
            trace_num += 1
            
        return pd.DataFrame(index_data)
        
    def read_trace(self, trace_num: int) -> Trace:
        """Read a single trace by number."""
        if not self._trace_offsets:
            self.build_index()
            
        if trace_num >= len(self._trace_offsets):
            raise IndexError(f"Trace {trace_num} out of range")
            
        offset = self._trace_offsets[trace_num]
        self.file_obj.seek(offset)
        
        # Read header
        header_data = self.file_obj.read(240)
        header = TraceHeader.from_bytes(header_data, self.endian)
        
        # Read data
        samples = header.num_samples or self.binary_header.samples_per_trace
        format_code = self.binary_header.data_sample_format
        
        bytes_per_sample = {
            DataSampleFormat.IBM_FLOAT_4: 4,
            DataSampleFormat.IEEE_FLOAT_4: 4,
            DataSampleFormat.IEEE_FLOAT_8: 8,
            DataSampleFormat.INT_4: 4,
            DataSampleFormat.INT_2: 2,
            DataSampleFormat.INT_1: 1,
        }.get(format_code, 4)
        
        data_bytes = self.file_obj.read(samples * bytes_per_sample)
        data = DataFormatConverter.read_samples(
            data_bytes, format_code, samples, self.endian
        )
        
        return Trace(header=header, data=data)
        
    def read_traces(self, trace_nums: List[int]) -> List[Trace]:
        """Read multiple traces."""
        return [self.read_trace(n) for n in trace_nums]
        
    def iter_traces(self) -> Iterator[Trace]:
        """Iterate over all traces."""
        header_size = 3600 + self.binary_header.num_extended_headers * 3200
        self.file_obj.seek(header_size)
        
        trace_size = self._get_trace_size()
        samples = self.binary_header.samples_per_trace
        format_code = self.binary_header.data_sample_format
        
        while True:
            header_data = self.file_obj.read(240)
            
            if len(header_data) < 240:
                break
                
            header = TraceHeader.from_bytes(header_data, self.endian)
            
            bytes_per_sample = {
                DataSampleFormat.IBM_FLOAT_4: 4,
                DataSampleFormat.IEEE_FLOAT_4: 4,
                DataSampleFormat.IEEE_FLOAT_8: 8,
                DataSampleFormat.INT_4: 4,
                DataSampleFormat.INT_2: 2,
                DataSampleFormat.INT_1: 1,
            }.get(format_code, 4)
            
            data_bytes = self.file_obj.read(samples * bytes_per_sample)
            data = DataFormatConverter.read_samples(
                data_bytes, format_code, samples, self.endian
            )
            
            yield Trace(header=header, data=data)
            
    def read_inline(self, inline_num: int, index: pd.DataFrame = None) -> List[Trace]:
        """Read all traces for an inline."""
        if index is None:
            index = self.build_index()
            
        trace_nums = index[index['inline'] == inline_num]['trace_num'].tolist()
        return self.read_traces(trace_nums)
        
    def read_crossline(self, crossline_num: int, 
                      index: pd.DataFrame = None) -> List[Trace]:
        """Read all traces for a crossline."""
        if index is None:
            index = self.build_index()
            
        trace_nums = index[index['crossline'] == crossline_num]['trace_num'].tolist()
        return self.read_traces(trace_nums)


class SEGYWriter:
    """
    SEG-Y file writer.
    """
    
    def __init__(self, file_path: str, endian: str = '>'):
        self.file_path = file_path
        self.endian = endian
        self.file_obj: Optional[BinaryIO] = None
        self.text_header: Optional[TextHeader] = None
        self.binary_header: Optional[BinaryHeader] = None
        self._is_open = False
        
    def open(self, text_header: TextHeader = None, 
            binary_header: BinaryHeader = None) -> None:
        """Open file for writing."""
        self.file_obj = open(self.file_path, 'wb')
        self._is_open = True
        
        # Write headers
        self.text_header = text_header or TextHeader(text="SEG-Y file created by MineralVision")
        self.binary_header = binary_header or BinaryHeader()
        
        self.file_obj.write(self.text_header.to_bytes())
        self.file_obj.write(self.binary_header.to_bytes(self.endian))
        
    def close(self) -> None:
        """Close the file."""
        if self.file_obj is not None and self._is_open:
            self.file_obj.close()
            self._is_open = False
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def write_trace(self, trace: Trace) -> None:
        """Write a single trace."""
        if not self._is_open:
            raise RuntimeError("File not open for writing")
            
        # Write header
        self.file_obj.write(trace.header.to_bytes(self.endian))
        
        # Write data as IEEE float
        self.file_obj.write(trace.data.astype(f'{self.endian}f4').tobytes())
        
    def write_traces(self, traces: List[Trace]) -> None:
        """Write multiple traces."""
        for trace in traces:
            self.write_trace(trace)


class SEGYIngestionPipeline:
    """
    Complete SEG-Y ingestion pipeline for MineralVision.
    """
    
    def __init__(self):
        self.schema_registry = HeaderSchemaRegistry()
        self.processing_history: List[Dict] = []
        
    def ingest_file(self, file_path: str, 
                   schema_name: str = 'standard') -> Dict[str, Any]:
        """
        Ingest a SEG-Y file.
        
        Args:
            file_path: Path to SEG-Y file (local or cloud)
            schema_name: Header schema to use
            
        Returns:
            Ingestion results
        """
        with SEGYReader(file_path) as reader:
            # Build index
            index = reader.build_index()
            
            # Get metadata
            metadata = {
                'file_path': file_path,
                'segy_revision': reader.binary_header.segy_revision.value,
                'data_format': reader.binary_header.data_sample_format.name,
                'samples_per_trace': reader.binary_header.samples_per_trace,
                'sample_interval_us': reader.binary_header.sample_interval,
                'num_traces': len(index),
                'sorting': reader.binary_header.trace_sorting.name,
                'measurement_system': 'meters' if reader.binary_header.measurement_system == 1 else 'feet'
            }
            
            # Compute geometry
            if len(index) > 0:
                metadata['inline_range'] = (index['inline'].min(), index['inline'].max())
                metadata['crossline_range'] = (index['crossline'].min(), index['crossline'].max())
                metadata['x_range'] = (index['x'].min(), index['x'].max())
                metadata['y_range'] = (index['y'].min(), index['y'].max())
                
            self.processing_history.append({
                'timestamp': datetime.now().isoformat(),
                'file_path': file_path,
                'num_traces': len(index),
                'schema': schema_name
            })
            
            return {
                'metadata': metadata,
                'index': index,
                'text_header': reader.text_header.text,
                'binary_header': reader.binary_header
            }
            
    def extract_volume(self, file_path: str,
                      inline_range: Tuple[int, int] = None,
                      crossline_range: Tuple[int, int] = None) -> xr.DataArray:
        """
        Extract 3D seismic volume.
        
        Args:
            file_path: Path to SEG-Y file
            inline_range: (min, max) inline range
            crossline_range: (min, max) crossline range
            
        Returns:
            3D volume as xarray DataArray
        """
        with SEGYReader(file_path) as reader:
            index = reader.build_index()
            
            # Filter by range
            if inline_range:
                index = index[(index['inline'] >= inline_range[0]) & 
                             (index['inline'] <= inline_range[1])]
            if crossline_range:
                index = index[(index['crossline'] >= crossline_range[0]) & 
                             (index['crossline'] <= crossline_range[1])]
                             
            if len(index) == 0:
                return None
                
            # Get dimensions
            inlines = sorted(index['inline'].unique())
            crosslines = sorted(index['crossline'].unique())
            num_samples = reader.binary_header.samples_per_trace
            
            # Create time axis
            dt = reader.binary_header.sample_interval / 1e6  # seconds
            time_axis = np.arange(num_samples) * dt
            
            # Initialize volume
            volume = np.zeros((len(inlines), len(crosslines), num_samples))
            
            # Read traces
            for _, row in index.iterrows():
                trace = reader.read_trace(row['trace_num'])
                
                il_idx = inlines.index(row['inline'])
                xl_idx = crosslines.index(row['crossline'])
                
                volume[il_idx, xl_idx, :] = trace.data[:num_samples]
                
            return xr.DataArray(
                data=volume,
                dims=['inline', 'crossline', 'time'],
                coords={
                    'inline': inlines,
                    'crossline': crosslines,
                    'time': time_axis
                },
                attrs={
                    'file_path': file_path,
                    'sample_interval_s': dt
                }
            )
            
    def extract_horizon_slice(self, file_path: str,
                             time_ms: float) -> xr.DataArray:
        """
        Extract horizontal time slice.
        
        Args:
            file_path: Path to SEG-Y file
            time_ms: Time in milliseconds
            
        Returns:
            2D slice as xarray DataArray
        """
        with SEGYReader(file_path) as reader:
            index = reader.build_index()
            
            # Get dimensions
            inlines = sorted(index['inline'].unique())
            crosslines = sorted(index['crossline'].unique())
            
            # Find sample index
            dt = reader.binary_header.sample_interval / 1000  # ms
            sample_idx = int(time_ms / dt)
            
            # Initialize slice
            slice_data = np.zeros((len(inlines), len(crosslines)))
            
            # Read traces
            for _, row in index.iterrows():
                trace = reader.read_trace(row['trace_num'])
                
                il_idx = inlines.index(row['inline'])
                xl_idx = crosslines.index(row['crossline'])
                
                if sample_idx < len(trace.data):
                    slice_data[il_idx, xl_idx] = trace.data[sample_idx]
                    
            return xr.DataArray(
                data=slice_data,
                dims=['inline', 'crossline'],
                coords={
                    'inline': inlines,
                    'crossline': crosslines
                },
                attrs={
                    'time_ms': time_ms,
                    'sample_index': sample_idx
                }
            )
            
    def export_to_lakehouse(self, file_path: str,
                           output_dir: str,
                           chunk_size: int = 1000) -> Dict[str, str]:
        """
        Export SEG-Y to lakehouse-friendly format.
        
        Converts to chunked Parquet files with metadata tables.
        
        Args:
            file_path: Path to SEG-Y file
            output_dir: Output directory
            chunk_size: Number of traces per chunk
            
        Returns:
            Dictionary of output file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        with SEGYReader(file_path) as reader:
            index = reader.build_index()
            
            # Save index
            index_path = os.path.join(output_dir, 'trace_index.parquet')
            index.to_parquet(index_path)
            
            # Save metadata
            metadata = {
                'source_file': file_path,
                'segy_revision': reader.binary_header.segy_revision.value,
                'data_format': reader.binary_header.data_sample_format.name,
                'samples_per_trace': reader.binary_header.samples_per_trace,
                'sample_interval_us': reader.binary_header.sample_interval,
                'num_traces': len(index),
                'chunk_size': chunk_size,
                'export_time': datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(output_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            # Save text header
            text_path = os.path.join(output_dir, 'text_header.txt')
            with open(text_path, 'w') as f:
                f.write(reader.text_header.text)
                
            # Export trace data in chunks
            traces_dir = os.path.join(output_dir, 'traces')
            os.makedirs(traces_dir, exist_ok=True)
            
            chunk_paths = []
            chunk_num = 0
            chunk_data = []
            
            for i, trace in enumerate(reader.iter_traces()):
                chunk_data.append({
                    'trace_num': i,
                    'data': trace.data.tolist()
                })
                
                if len(chunk_data) >= chunk_size:
                    chunk_path = os.path.join(traces_dir, f'chunk_{chunk_num:06d}.parquet')
                    pd.DataFrame(chunk_data).to_parquet(chunk_path)
                    chunk_paths.append(chunk_path)
                    chunk_data = []
                    chunk_num += 1
                    
            # Save remaining
            if chunk_data:
                chunk_path = os.path.join(traces_dir, f'chunk_{chunk_num:06d}.parquet')
                pd.DataFrame(chunk_data).to_parquet(chunk_path)
                chunk_paths.append(chunk_path)
                
            return {
                'index': index_path,
                'metadata': metadata_path,
                'text_header': text_path,
                'trace_chunks': chunk_paths
            }


def create_segy_pipeline() -> SEGYIngestionPipeline:
    """Factory function to create SEG-Y pipeline."""
    return SEGYIngestionPipeline()


def validate_segy_file(file_path: str) -> Dict[str, Any]:
    """
    Validate a SEG-Y file.
    
    Args:
        file_path: Path to SEG-Y file
        
    Returns:
        Validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'info': {}
    }
    
    try:
        with SEGYReader(file_path) as reader:
            # Check binary header
            if reader.binary_header.samples_per_trace <= 0:
                results['errors'].append("Invalid samples per trace")
                results['valid'] = False
                
            if reader.binary_header.sample_interval <= 0:
                results['errors'].append("Invalid sample interval")
                results['valid'] = False
                
            # Build index and check consistency
            index = reader.build_index()
            
            if len(index) == 0:
                results['errors'].append("No traces found")
                results['valid'] = False
            else:
                results['info']['num_traces'] = len(index)
                
                # Check for duplicate positions
                duplicates = index.duplicated(subset=['inline', 'crossline'])
                if duplicates.any():
                    results['warnings'].append(
                        f"Found {duplicates.sum()} duplicate inline/crossline positions"
                    )
                    
                # Check coordinate consistency
                if index['x'].std() == 0 and index['y'].std() == 0:
                    results['warnings'].append("All coordinates are identical")
                    
            results['info']['segy_revision'] = reader.binary_header.segy_revision.value
            results['info']['data_format'] = reader.binary_header.data_sample_format.name
            
    except Exception as e:
        results['valid'] = False
        results['errors'].append(str(e))
        
    return results
