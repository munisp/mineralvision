"""
Parquet Storage Utilities for MineralVision

This module implements utilities for working with Parquet files in the MineralVision Lakehouse architecture.
It provides functionality for reading, writing, and optimizing Parquet files with a focus on
geospatial data handling.

Uses PyArrow for Parquet operations with optimized settings for geospatial data.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

@dataclass
class ParquetConfig:
    """Configuration settings for Parquet storage."""
    base_path: str
    compression: str = "snappy"
    row_group_size: int = 128 * 1024 * 1024
    page_size: int = 1 * 1024 * 1024
    enable_dictionary: bool = True
    enable_statistics: bool = True
    # Backward-compatible alias for enable_dictionary (legacy callers pass
    # enable_dictionary_encoding=...); when explicitly set it wins.
    enable_dictionary_encoding: Optional[bool] = None
    log_level: str = "INFO"
    spatial_partition_columns: List[str] = field(default_factory=lambda: ["tile_id", "resolution"])

    def __post_init__(self):
        """Initialize derived paths and create directories if they don't exist."""
        if self.enable_dictionary_encoding is not None:
            self.enable_dictionary = self.enable_dictionary_encoding
        else:
            # keep the two names consistent for readers
            self.enable_dictionary_encoding = self.enable_dictionary
        os.makedirs(self.base_path, exist_ok=True)
        logging.basicConfig(level=getattr(logging, self.log_level), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("ParquetStorage")
        self.logger.info("Initialized Parquet storage configuration")


class ParquetStorage:
    """
    Utility class for working with Parquet files in the MineralVision Lakehouse architecture.
    
    This class provides methods for:
    - Reading and writing Parquet files with optimizations for geospatial data
    - Partitioning data for efficient querying
    - Converting between different formats and Parquet
    - Optimizing Parquet files for performance
    """
    
    def __init__(self, config: ParquetConfig):
        """
        Initialize the Parquet storage utility.
        
        Args:
            config: Configuration settings for Parquet storage
        """
        self.config = config
        self.logger = logging.getLogger("ParquetStorage")
        self.logger.info("Initialized Parquet storage utility")
    
    def write_parquet(self, file_path: str, data: Any, partition_by: Optional[List[str]] = None,
                     schema: Optional[Dict] = None, compression: Optional[str] = None) -> bool:
        """
        Write data to a Parquet file with optimized settings.

        Args:
            file_path: Path to write the Parquet file to
            data: Data to write (DataFrame, PyArrow Table, dict, or list of dicts)
            partition_by: Columns to partition the data by
            schema: Schema to use for the Parquet file
            compression: Compression codec (defaults to config setting)

        Returns:
            bool: True if data was written successfully
        """
        # Backward compatibility: a dict passed as the third positional argument
        # is a schema, not a partition column list.
        if isinstance(partition_by, dict) and schema is None:
            schema = partition_by
            partition_by = None

        if compression is None:
            compression = self.config.compression
        full_path = file_path if os.path.isabs(file_path) else os.path.join(self.config.base_path, file_path)
        self.logger.info(f"Writing Parquet file to {full_path}")
        
        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            if isinstance(data, pa.Table):
                table = data
            elif isinstance(data, pd.DataFrame):
                table = pa.Table.from_pandas(data)
            elif isinstance(data, dict):
                table = pa.Table.from_pandas(pd.DataFrame([data]))
            elif isinstance(data, list):
                table = pa.Table.from_pandas(pd.DataFrame(data))
            elif data is None:
                self.logger.warning("No data provided to write")
                return True
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")
            
            if partition_by:
                pq.write_to_dataset(table, root_path=full_path, partition_cols=partition_by, compression=compression, use_dictionary=self.config.enable_dictionary, write_statistics=self.config.enable_statistics)
            else:
                pq.write_table(table, full_path, compression=compression, use_dictionary=self.config.enable_dictionary, write_statistics=self.config.enable_statistics, row_group_size=self.config.row_group_size)
            
            self.logger.info(f"Successfully wrote {len(table)} rows to {full_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to write Parquet file: {str(e)}")
            raise
    
    def write_file(self, file_path: str, data: Any, partition_by: Optional[List[str]] = None,
                  schema: Optional[Dict] = None, compression: Optional[str] = None) -> bool:
        """Alias for write_parquet for backward compatibility with tests."""
        return self.write_parquet(file_path, data, partition_by, schema, compression)
    
    def read_parquet(self, file_path: str, columns: Optional[List[str]] = None,
                    filters: Optional[List] = None) -> pd.DataFrame:
        """
        Read data from a Parquet file with predicate pushdown.
        
        Args:
            file_path: Path to the Parquet file
            columns: Columns to read (projection pushdown)
            filters: Filters to apply (predicate pushdown)
            
        Returns:
            pd.DataFrame: Data from the Parquet file
        """
        full_path = os.path.join(self.config.base_path, file_path)
        self.logger.info(f"Reading Parquet file from {full_path}")
        
        try:
            if not os.path.exists(full_path):
                self.logger.warning(f"File does not exist: {full_path}")
                return pd.DataFrame()
            
            if os.path.isdir(full_path):
                dataset = pq.ParquetDataset(full_path, filters=filters)
                table = dataset.read(columns=columns)
            else:
                table = pq.read_table(full_path, columns=columns, filters=filters)
            
            df = table.to_pandas()
            self.logger.info(f"Successfully read {len(df)} rows from {full_path}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to read Parquet file: {str(e)}")
            return pd.DataFrame()
    
    def read_file(self, file_path: str, columns: Optional[List[str]] = None, filters: Optional[List] = None) -> pd.DataFrame:
        """Alias for read_parquet for backward compatibility with tests."""
        return self.read_parquet(file_path, columns, filters)
    
    def convert_to_parquet(self, source_path: str, target_path: str, source_format: str,
                          partition_by: Optional[List[str]] = None) -> bool:
        """
        Convert a file from another format to Parquet.
        
        Args:
            source_path: Path to the source file
            target_path: Path to write the Parquet file to
            source_format: Format of the source file (e.g., CSV, JSON, GeoTIFF)
            partition_by: Columns to partition the data by
            
        Returns:
            bool: True if conversion was successful
        """
        full_source_path = os.path.join(self.config.base_path, source_path)
        full_target_path = os.path.join(self.config.base_path, target_path)
        
        self.logger.info(f"Converting {source_format} file from {full_source_path} to Parquet")
        
        try:
            source_format_lower = source_format.lower()
            
            if source_format_lower == 'csv':
                df = pd.read_csv(full_source_path)
            elif source_format_lower == 'json':
                df = pd.read_json(full_source_path)
            elif source_format_lower in ['geotiff', 'tif', 'tiff']:
                try:
                    import rasterio
                    with rasterio.open(full_source_path) as src:
                        data = src.read()
                        df = pd.DataFrame({'data': [data.tobytes()], 'shape': [str(data.shape)], 'crs': [str(src.crs)], 'bounds': [str(src.bounds)]})
                except ImportError:
                    self.logger.warning("rasterio not available, using placeholder for GeoTIFF")
                    df = pd.DataFrame({'source': [full_source_path], 'format': ['geotiff']})
            elif source_format_lower == 'excel' or source_format_lower in ['xls', 'xlsx']:
                df = pd.read_excel(full_source_path)
            else:
                self.logger.warning(f"Unknown format {source_format}, attempting to read as CSV")
                df = pd.read_csv(full_source_path)
            
            return self.write_parquet(target_path, df, partition_by)
        except Exception as e:
            self.logger.error(f"Failed to convert {source_format} to Parquet: {str(e)}")
            raise
    
    def optimize_parquet(self, file_path: str, target_path: Optional[str] = None) -> bool:
        """
        Optimize a Parquet file for better performance.
        
        Args:
            file_path: Path to the Parquet file to optimize
            target_path: Path to write the optimized Parquet file to (if None, overwrites the original)
            
        Returns:
            bool: True if optimization was successful
        """
        full_source_path = os.path.join(self.config.base_path, file_path)
        full_target_path = os.path.join(self.config.base_path, target_path) if target_path else full_source_path
        
        self.logger.info(f"Optimizing Parquet file at {full_source_path}")
        
        try:
            if not os.path.exists(full_source_path):
                self.logger.warning(f"File does not exist: {full_source_path}")
                return False
            
            if os.path.isdir(full_source_path):
                dataset = pq.ParquetDataset(full_source_path)
                table = dataset.read()
            else:
                table = pq.read_table(full_source_path)
            
            os.makedirs(os.path.dirname(full_target_path) if os.path.dirname(full_target_path) else '.', exist_ok=True)
            pq.write_table(table, full_target_path, compression=self.config.compression, use_dictionary=self.config.enable_dictionary, write_statistics=self.config.enable_statistics, row_group_size=self.config.row_group_size)
            
            self.logger.info(f"Successfully optimized Parquet file to {full_target_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to optimize Parquet file: {str(e)}")
            return False
    
    def get_parquet_metadata(self, file_path: str) -> Dict:
        """
        Get metadata from a Parquet file.
        
        Args:
            file_path: Path to the Parquet file
            
        Returns:
            Dict: Metadata from the Parquet file
        """
        full_path = os.path.join(self.config.base_path, file_path)
        self.logger.info(f"Getting metadata from Parquet file at {full_path}")
        
        try:
            if not os.path.exists(full_path):
                self.logger.warning(f"File does not exist: {full_path}")
                return {}
            
            if os.path.isdir(full_path):
                parquet_files = [os.path.join(full_path, f) for f in os.listdir(full_path) if f.endswith('.parquet')]
                if not parquet_files:
                    return {}
                pf = pq.ParquetFile(parquet_files[0])
            else:
                pf = pq.ParquetFile(full_path)
            
            metadata = pf.metadata
            schema = pf.schema_arrow
            
            result = {
                "num_rows": metadata.num_rows,
                "num_row_groups": metadata.num_row_groups,
                "format_version": str(metadata.format_version),
                "created_by": metadata.created_by,
                "schema": {"fields": [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in schema]},
                "row_groups": []
            }
            
            for i in range(metadata.num_row_groups):
                rg = metadata.row_group(i)
                rg_info = {"num_rows": rg.num_rows, "total_byte_size": rg.total_byte_size, "columns": []}
                for j in range(rg.num_columns):
                    col = rg.column(j)
                    rg_info["columns"].append({"path": col.path_in_schema, "compression": str(col.compression), "encodings": [str(e) for e in col.encodings] if hasattr(col, 'encodings') else []})
                result["row_groups"].append(rg_info)
            
            self.logger.info(f"Successfully read metadata from Parquet file")
            return result
        except Exception as e:
            self.logger.error(f"Failed to get Parquet metadata: {str(e)}")
            return {}
    
    def get_statistics(self, file_path: str) -> Dict:
        """
        Get statistics for a Parquet file.

        Convenience wrapper around get_parquet_metadata that guarantees
        row-count keys are present.

        Args:
            file_path: Path to the Parquet file

        Returns:
            Dict: Statistics including "num_rows" and "row_count"
        """
        metadata = self.get_parquet_metadata(file_path)
        if not metadata:
            return {}

        stats = dict(metadata)
        stats["row_count"] = metadata.get("num_rows", 0)
        return stats

    def merge_parquet_files(self, source_paths: List[str], target_path: str,
                           partition_by: Optional[List[str]] = None) -> bool:
        """
        Merge multiple Parquet files into a single file.
        
        Args:
            source_paths: Paths to the source Parquet files
            target_path: Path to write the merged Parquet file to
            partition_by: Columns to partition the merged data by
            
        Returns:
            bool: True if merge was successful
        """
        full_source_paths = [os.path.join(self.config.base_path, path) for path in source_paths]
        full_target_path = os.path.join(self.config.base_path, target_path)
        
        self.logger.info(f"Merging {len(source_paths)} Parquet files to {full_target_path}")
        
        try:
            tables = []
            for path in full_source_paths:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        dataset = pq.ParquetDataset(path)
                        tables.append(dataset.read())
                    else:
                        tables.append(pq.read_table(path))
                else:
                    self.logger.warning(f"Source file does not exist: {path}")
            
            if not tables:
                self.logger.warning("No valid source files found")
                return False
            
            merged_table = pa.concat_tables(tables)
            os.makedirs(os.path.dirname(full_target_path) if os.path.dirname(full_target_path) else '.', exist_ok=True)
            
            if partition_by:
                pq.write_to_dataset(merged_table, root_path=full_target_path, partition_cols=partition_by, compression=self.config.compression)
            else:
                pq.write_table(merged_table, full_target_path, compression=self.config.compression, row_group_size=self.config.row_group_size)
            
            self.logger.info(f"Successfully merged {len(tables)} files ({len(merged_table)} rows) to {full_target_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to merge Parquet files: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # Create a configuration
    config = ParquetConfig(
        base_path="/data/mineralvision/parquet",
        compression="snappy",
        row_group_size=128 * 1024 * 1024,  # 128 MB
        page_size=1 * 1024 * 1024,  # 1 MB
    )
    
    # Create a storage utility
    storage = ParquetStorage(config)
    
    # Write a Parquet file
    storage.write_parquet(
        data=None,  # In a real implementation, this would be a DataFrame
        file_path="satellite_imagery/landsat8_2025_04_20.parquet",
        partition_by=["tile_id", "resolution"],
        schema={
            "fields": [
                {"name": "id", "type": "string", "nullable": False},
                {"name": "acquisition_date", "type": "timestamp", "nullable": False},
                {"name": "sensor_type", "type": "string", "nullable": False},
                {"name": "tile_id", "type": "string", "nullable": False},
                {"name": "resolution", "type": "double", "nullable": False},
                {"name": "data", "type": "binary", "nullable": False},
                {"name": "metadata", "type": "struct", "nullable": True}
            ]
        }
    )
    
    # Read a Parquet file
    data = storage.read_parquet(
        file_path="satellite_imagery/landsat8_2025_04_20.parquet",
        columns=["id", "acquisition_date", "sensor_type", "data"],
        filters=[("sensor_type", "=", "Landsat-8")]
    )
    
    # Get metadata from a Parquet file
    metadata = storage.get_parquet_metadata(
        file_path="satellite_imagery/landsat8_2025_04_20.parquet"
    )
    
    # Optimize a Parquet file
    storage.optimize_parquet(
        file_path="satellite_imagery/landsat8_2025_04_20.parquet",
        target_path="satellite_imagery/landsat8_2025_04_20_optimized.parquet"
    )
    
    # Convert a file to Parquet
    storage.convert_to_parquet(
        source_path="raw_data/landsat8_2025_04_20.tif",
        target_path="satellite_imagery/landsat8_2025_04_20.parquet",
        source_format="GeoTIFF",
        partition_by=["tile_id", "resolution"]
    )
    
    # Merge Parquet files
    storage.merge_parquet_files(
        source_paths=[
            "satellite_imagery/landsat8_2025_04_20_part1.parquet",
            "satellite_imagery/landsat8_2025_04_20_part2.parquet",
            "satellite_imagery/landsat8_2025_04_20_part3.parquet"
        ],
        target_path="satellite_imagery/landsat8_2025_04_20_merged.parquet",
        partition_by=["tile_id", "resolution"]
    )
