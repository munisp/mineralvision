"""
Delta Lake Storage Configuration for MineralVision

This module implements the Delta Lake storage layer for the MineralVision Lakehouse architecture.
It provides functionality for creating, reading, writing, and managing Delta tables with
optimizations for geospatial data.

Uses PyArrow and Parquet for data storage with Delta Lake-like metadata tracking.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import numpy as np

try:
    from deltalake import DeltaTable, write_deltalake
    from deltalake.exceptions import TableNotFoundError
    DELTA_AVAILABLE = True
except ImportError:
    DELTA_AVAILABLE = False
    DeltaTable = None
    write_deltalake = None
    TableNotFoundError = Exception

# Configuration for Delta Lake storage
@dataclass
class DeltaLakeConfig:
    """Configuration settings for Delta Lake storage."""
    base_path: str
    raw_zone_path: Optional[str] = None
    processed_zone_path: Optional[str] = None
    curated_zone_path: Optional[str] = None
    feature_zone_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    log_level: str = "INFO"
    
    # Delta-specific configurations
    enable_versioning: bool = True
    auto_compact: bool = True
    optimize_write: bool = True
    auto_optimize: bool = True
    vacuum_retention_hours: int = 168  # 7 days
    
    # Geospatial-specific configurations
    spatial_partition_columns: List[str] = field(default_factory=lambda: ["tile_id", "resolution"])
    z_order_columns: List[str] = field(default_factory=lambda: ["acquisition_date", "sensor_type"])
    
    # Additional config fields for compatibility with tests
    partitioning_columns: List[str] = field(default_factory=list)
    enable_schema_evolution: bool = True
    enable_time_travel: bool = True
    
    def __post_init__(self):
        """Initialize derived paths and create directories if they don't exist."""

        # Derive zone paths from base_path when not explicitly provided
        if self.raw_zone_path is None:
            self.raw_zone_path = os.path.join(self.base_path, "raw")
        if self.processed_zone_path is None:
            self.processed_zone_path = os.path.join(self.base_path, "processed")
        if self.curated_zone_path is None:
            self.curated_zone_path = os.path.join(self.base_path, "curated")
        if self.feature_zone_path is None:
            self.feature_zone_path = os.path.join(self.base_path, "feature")
        if self.checkpoint_path is None:
            self.checkpoint_path = os.path.join(self.base_path, "checkpoints")

        # Create directories if they don't exist
        for path in [self.base_path, self.raw_zone_path, self.processed_zone_path, 
                    self.curated_zone_path, self.feature_zone_path, self.checkpoint_path]:
            os.makedirs(path, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("DeltaLakeStorage")
        self.logger.info("Initialized Delta Lake storage configuration")


class DeltaLakeStorage:
    """
    Main class for managing Delta Lake storage in the MineralVision Lakehouse architecture.
    
    This class provides methods for:
    - Creating and managing Delta tables
    - Reading and writing data with optimizations for geospatial datasets
    - Implementing data versioning and time travel
    - Optimizing storage through compaction and Z-ordering
    - Managing schema evolution
    """
    
    def __init__(self, config: DeltaLakeConfig):
        """
        Initialize the Delta Lake storage manager.
        
        Args:
            config: Configuration settings for Delta Lake storage
        """
        self.config = config
        self.logger = logging.getLogger("DeltaLakeStorage")
        self._tables: Dict[str, Any] = {}
        self._table_metadata: Dict[str, Dict] = {}
        
        if DELTA_AVAILABLE:
            self.logger.info("Initialized Delta Lake storage manager with delta-rs")
        else:
            self.logger.info("Initialized Delta Lake storage manager with Parquet fallback")
    
    def _get_zone_path(self, zone: str) -> str:
        """Get the path for a specific zone."""
        zone_paths = {
            "raw": self.config.raw_zone_path,
            "processed": self.config.processed_zone_path,
            "curated": self.config.curated_zone_path,
            "feature": self.config.feature_zone_path
        }
        if zone not in zone_paths:
            raise ValueError(f"Invalid zone: {zone}. Must be one of {list(zone_paths.keys())}")
        return zone_paths[zone]
    
    def _get_table_path(self, table_name: str, zone: str) -> str:
        """Get the full path for a table."""
        return os.path.join(self._get_zone_path(zone), table_name)
    
    @staticmethod
    def _normalize_schema(schema: Dict) -> Dict:
        """Normalize a schema dict to the {"fields": [...]} form.

        Accepts either {"fields": [{"name": ..., "type": ...}, ...]} or a flat
        {"column_name": "type"} mapping.
        """
        if "fields" in schema:
            return schema
        return {"fields": [{"name": name, "type": str(type_), "nullable": True}
                           for name, type_ in schema.items()]}

    def _schema_dict_to_pyarrow(self, schema: Dict) -> pa.Schema:
        """Convert a schema dictionary to PyArrow schema."""
        schema = self._normalize_schema(schema)
        type_mapping = {
            "string": pa.string(),
            "int": pa.int64(),
            "int32": pa.int32(),
            "int64": pa.int64(),
            "float": pa.float64(),
            "float32": pa.float32(),
            "float64": pa.float64(),
            "double": pa.float64(),
            "boolean": pa.bool_(),
            "bool": pa.bool_(),
            "timestamp": pa.timestamp('us'),
            "date": pa.date32(),
            "binary": pa.binary(),
            "struct": pa.struct([]),
        }
        
        fields = []
        for field_def in schema.get("fields", []):
            field_type = field_def.get("type", "string").lower()
            pa_type = type_mapping.get(field_type, pa.string())
            nullable = field_def.get("nullable", True)
            fields.append(pa.field(field_def["name"], pa_type, nullable=nullable))
        
        return pa.schema(fields)
    
    def _get_pandas_dtype(self, type_str: str) -> str:
        """Convert type string to pandas dtype."""
        type_mapping = {
            "string": "object",
            "int": "int64",
            "int32": "int32",
            "int64": "int64",
            "float": "float64",
            "float32": "float32",
            "float64": "float64",
            "double": "float64",
            "boolean": "bool",
            "bool": "bool",
            "timestamp": "datetime64[ns]",
            "date": "datetime64[ns]",
            "binary": "object",
            "struct": "object",
        }
        return type_mapping.get(type_str.lower(), "object")
    
    def _get_next_version(self, table_path: str) -> int:
        """Get the next version number for a table."""
        if not os.path.exists(table_path):
            return 0
        parquet_files = [f for f in os.listdir(table_path) if f.endswith('.parquet')]
        if not parquet_files:
            return 0
        versions = []
        for f in parquet_files:
            try:
                version = int(f.split('-')[1].split('.')[0])
                versions.append(version)
            except (IndexError, ValueError):
                continue
        return max(versions) + 1 if versions else 0
    
    def _update_table_metadata(self, table_name: str, zone: str, operation: str, params: Dict):
        """Update table metadata with a new operation."""
        key = f"{zone}/{table_name}"
        if key not in self._table_metadata:
            self._table_metadata[key] = {"table_name": table_name, "zone": zone, "version": 0, "history": []}
        
        metadata = self._table_metadata[key]
        metadata["version"] += 1
        metadata["history"].append({
            "version": metadata["version"],
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "operationParameters": params
        })
        
        table_path = self._get_table_path(table_name, zone)
        metadata_path = os.path.join(table_path, "_delta_log")
        os.makedirs(metadata_path, exist_ok=True)
        version_file = f"{metadata['version']:020d}.json"
        with open(os.path.join(metadata_path, version_file), 'w') as f:
            json.dump(metadata, f, indent=2)
        
    def create_table(self, table_name: str, schema: Dict, zone: str = "processed", 
                    partition_by: Optional[List[str]] = None) -> bool:
        """
        Create a new Delta table with the specified schema.
        
        Args:
            table_name: Name of the table to create
            schema: Schema definition for the table
            zone: Storage zone (raw, processed, curated, feature)
            partition_by: Columns to partition the table by
            
        Returns:
            bool: True if table was created successfully
        """
        table_path = self._get_table_path(table_name, zone)
        self.logger.info(f"Creating Delta table {table_name} at {table_path}")

        schema = self._normalize_schema(schema)
        try:
            os.makedirs(table_path, exist_ok=True)
            pa_schema = self._schema_dict_to_pyarrow(schema)
            
            metadata = {
                "table_name": table_name,
                "zone": zone,
                "schema": schema,
                "partition_by": partition_by or [],
                "created_at": datetime.now().isoformat(),
                "version": 0,
                "history": [{"version": 0, "timestamp": datetime.now().isoformat(), "operation": "CREATE TABLE"}]
            }
            
            metadata_path = os.path.join(table_path, "_delta_log")
            os.makedirs(metadata_path, exist_ok=True)
            with open(os.path.join(metadata_path, "00000000000000000000.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self._table_metadata[f"{zone}/{table_name}"] = metadata
            
            if DELTA_AVAILABLE:
                empty_df = pd.DataFrame({fd["name"]: pd.Series(dtype=self._get_pandas_dtype(fd["type"])) 
                                        for fd in schema.get("fields", [])})
                empty_table = pa.Table.from_pandas(empty_df, schema=pa_schema)
                write_deltalake(table_path, empty_table, mode="overwrite", partition_by=partition_by, schema_mode="overwrite")
            
            self.logger.info(f"Created Delta table at {table_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create table {table_name}: {str(e)}")
            raise
    
    def write_data(self, table_name: str, data: Any, zone: str = "processed", 
                  mode: str = "append", partition_by: Optional[List[str]] = None) -> bool:
        """
        Write data to a Delta table.
        
        Args:
            table_name: Name of the table to write to
            data: Data to write (DataFrame, PyArrow Table, dict, or list of dicts)
            zone: Storage zone (raw, processed, curated, feature)
            mode: Write mode (append, overwrite, error, ignore)
            partition_by: Columns to partition the data by
            
        Returns:
            bool: True if data was written successfully
        """
        table_path = self._get_table_path(table_name, zone)
        self.logger.info(f"Writing data to {table_name} in {zone} zone with mode {mode}")
        
        try:
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
            
            if DELTA_AVAILABLE:
                write_deltalake(table_path, table, mode=mode, partition_by=partition_by)
            else:
                os.makedirs(table_path, exist_ok=True)
                version = self._get_next_version(table_path)
                data_path = os.path.join(table_path, f"part-{version:05d}.parquet")
                pq.write_table(table, data_path)
                self._update_table_metadata(table_name, zone, "WRITE", {"mode": mode, "rows": len(table)})
            
            self.logger.info(f"Wrote {len(table)} rows to {table_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to write data to {table_name}: {str(e)}")
            raise
    
    def read_table(self, table_name: str, zone: str = "processed", 
                  version: Optional[int] = None, timestamp: Optional[str] = None,
                  filter_expr: Optional[str] = None) -> pd.DataFrame:
        """
        Read data from a Delta table with optional time travel.
        
        Args:
            table_name: Name of the table to read
            zone: Storage zone (raw, processed, curated, feature)
            version: Specific version to read (for time travel)
            timestamp: Specific timestamp to read (for time travel)
            filter_expr: Filter expression to apply
            
        Returns:
            pd.DataFrame: Data from the table
        """
        table_path = self._get_table_path(table_name, zone)
        self.logger.info(f"Reading data from {table_name} in {zone} zone")
        
        try:
            if DELTA_AVAILABLE:
                if version is not None:
                    dt = DeltaTable(table_path, version=version)
                elif timestamp is not None:
                    dt = DeltaTable(table_path)
                    dt.load_as_version(timestamp)
                else:
                    dt = DeltaTable(table_path)
                df = dt.to_pandas()
            else:
                if not os.path.exists(table_path):
                    self.logger.warning(f"Table {table_name} does not exist at {table_path}")
                    return pd.DataFrame()
                
                parquet_files = [os.path.join(table_path, f) for f in os.listdir(table_path) 
                                if f.endswith('.parquet')]
                if not parquet_files:
                    return pd.DataFrame()
                
                dfs = [pq.read_table(f).to_pandas() for f in parquet_files]
                df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            
            if filter_expr and not df.empty:
                try:
                    df = df.query(filter_expr)
                except Exception as e:
                    self.logger.warning(f"Could not apply filter '{filter_expr}': {e}")
            
            self.logger.info(f"Read {len(df)} rows from {table_path}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to read table {table_name}: {str(e)}")
            return pd.DataFrame()
    
    def read_data(self, table_name: str, zone: str = "processed", 
                  version: Optional[int] = None, timestamp: Optional[str] = None,
                  filter_expr: Optional[str] = None) -> pd.DataFrame:
        """Alias for read_table for backward compatibility with tests."""
        return self.read_table(table_name, zone, version, timestamp, filter_expr)
    
    def optimize_table(self, table_name: str, zone: str = "processed", 
                      z_order_by: Optional[List[str]] = None) -> bool:
        """
        Optimize a Delta table through compaction and Z-ordering.
        
        Args:
            table_name: Name of the table to optimize
            zone: Storage zone (raw, processed, curated, feature)
            z_order_by: Columns to Z-order by
            
        Returns:
            bool: True if optimization was successful
        """
        table_path = self._get_table_path(table_name, zone)
        self.logger.info(f"Optimizing table {table_name} at {table_path}")
        
        try:
            if DELTA_AVAILABLE:
                dt = DeltaTable(table_path)
                if z_order_by:
                    dt.optimize.z_order(z_order_by)
                else:
                    dt.optimize.compact()
                self.logger.info(f"Optimized Delta table at {table_path}")
            else:
                if not os.path.exists(table_path):
                    self.logger.warning(f"Table {table_name} does not exist")
                    return False
                
                parquet_files = [os.path.join(table_path, f) for f in os.listdir(table_path) 
                                if f.endswith('.parquet')]
                if len(parquet_files) <= 1:
                    self.logger.info("No compaction needed - single or no files")
                    return True
                
                dfs = [pq.read_table(f).to_pandas() for f in parquet_files]
                combined_df = pd.concat(dfs, ignore_index=True)
                
                if z_order_by:
                    valid_cols = [c for c in z_order_by if c in combined_df.columns]
                    if valid_cols:
                        combined_df = combined_df.sort_values(by=valid_cols)
                
                for f in parquet_files:
                    os.remove(f)
                
                compacted_path = os.path.join(table_path, "part-00000.parquet")
                pq.write_table(pa.Table.from_pandas(combined_df), compacted_path)
                self._update_table_metadata(table_name, zone, "OPTIMIZE", {"z_order_by": z_order_by})
                self.logger.info(f"Compacted {len(parquet_files)} files into 1")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to optimize table {table_name}: {str(e)}")
            return False
    
    def vacuum_table(self, table_name: str, zone: str = "processed", 
                    retention_hours: Optional[int] = None) -> bool:
        """
        Vacuum a Delta table to remove old files.
        
        Args:
            table_name: Name of the table to vacuum
            zone: Storage zone (raw, processed, curated, feature)
            retention_hours: Retention period in hours
            
        Returns:
            bool: True if vacuum was successful
        """
        if retention_hours is None:
            retention_hours = self.config.vacuum_retention_hours
        
        table_path = self._get_table_path(table_name, zone)
        self.logger.info(f"Vacuuming table {table_name} with retention {retention_hours} hours")
        
        try:
            if DELTA_AVAILABLE:
                dt = DeltaTable(table_path)
                dt.vacuum(retention_hours=retention_hours, enforce_retention_duration=False)
                self.logger.info(f"Vacuumed Delta table at {table_path}")
            else:
                if not os.path.exists(table_path):
                    self.logger.warning(f"Table {table_name} does not exist")
                    return False
                
                cutoff_time = datetime.now().timestamp() - (retention_hours * 3600)
                removed_count = 0
                
                for f in os.listdir(table_path):
                    if f.startswith('.') or f == '_delta_log':
                        continue
                    file_path = os.path.join(table_path, f)
                    if os.path.isfile(file_path):
                        file_mtime = os.path.getmtime(file_path)
                        if file_mtime < cutoff_time:
                            os.remove(file_path)
                            removed_count += 1
                
                self._update_table_metadata(table_name, zone, "VACUUM", {"retention_hours": retention_hours, "removed_files": removed_count})
                self.logger.info(f"Removed {removed_count} old files from {table_path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to vacuum table {table_name}: {str(e)}")
            return False
    
    def get_table_history(self, table_name: str, zone: str = "processed") -> List[Dict]:
        """
        Get the history of a Delta table.
        
        Args:
            table_name: Name of the table
            zone: Storage zone (raw, processed, curated, feature)
            
        Returns:
            List[Dict]: History of the table
        """
        table_path = self._get_table_path(table_name, zone)
        self.logger.info(f"Getting history for table {table_name}")
        
        try:
            if DELTA_AVAILABLE:
                dt = DeltaTable(table_path)
                history = dt.history()
                return history.to_pydict() if hasattr(history, 'to_pydict') else [{"version": i, "timestamp": str(datetime.now()), "operation": "UNKNOWN"} for i in range(len(history))]
            else:
                key = f"{zone}/{table_name}"
                if key in self._table_metadata:
                    return self._table_metadata[key].get("history", [])
                
                metadata_path = os.path.join(table_path, "_delta_log")
                if not os.path.exists(metadata_path):
                    return []
                
                history = []
                for f in sorted(os.listdir(metadata_path)):
                    if f.endswith('.json'):
                        with open(os.path.join(metadata_path, f), 'r') as fp:
                            data = json.load(fp)
                            if "history" in data:
                                history.extend(data["history"])
                            else:
                                history.append(data)
                
                return history
        except Exception as e:
            self.logger.error(f"Failed to get history for {table_name}: {str(e)}")
            return []
    
    def describe_table(self, table_name: str, zone: str = "processed") -> Dict:
        """
        Get the schema and metadata of a Delta table.
        
        Args:
            table_name: Name of the table
            zone: Storage zone (raw, processed, curated, feature)
            
        Returns:
            Dict: Schema and metadata of the table
        """
        table_path = self._get_table_path(table_name, zone)
        self.logger.info(f"Describing table {table_name}")
        
        try:
            if DELTA_AVAILABLE:
                dt = DeltaTable(table_path)
                schema = dt.schema()
                metadata = dt.metadata()
                return {
                    "schema": {"fields": [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in schema]},
                    "metadata": {"id": metadata.id, "name": metadata.name, "description": metadata.description, "partitionColumns": metadata.partition_columns, "format": "delta", "createdAt": str(metadata.created_time)}
                }
            else:
                key = f"{zone}/{table_name}"
                if key in self._table_metadata:
                    meta = self._table_metadata[key]
                    return {"schema": meta.get("schema", {}), "metadata": {"partitionColumns": meta.get("partition_by", []), "format": "parquet", "createdAt": meta.get("created_at", "")}}
                
                metadata_path = os.path.join(table_path, "_delta_log", "00000000000000000000.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        meta = json.load(f)
                        return {"schema": meta.get("schema", {}), "metadata": {"partitionColumns": meta.get("partition_by", []), "format": "parquet", "createdAt": meta.get("created_at", "")}}
                
                parquet_files = [os.path.join(table_path, f) for f in os.listdir(table_path) if f.endswith('.parquet')] if os.path.exists(table_path) else []
                if parquet_files:
                    table = pq.read_table(parquet_files[0])
                    schema = table.schema
                    return {"schema": {"fields": [{"name": f.name, "type": str(f.type), "nullable": f.nullable} for f in schema]}, "metadata": {"format": "parquet"}}
                
                return {"schema": {}, "metadata": {}}
        except Exception as e:
            self.logger.error(f"Failed to describe table {table_name}: {str(e)}")
            return {"schema": {}, "metadata": {}}


# Example usage
if __name__ == "__main__":
    # Create a configuration
    config = DeltaLakeConfig(
        base_path="/data/mineralvision/lakehouse",
        raw_zone_path="/data/mineralvision/lakehouse/raw",
        processed_zone_path="/data/mineralvision/lakehouse/processed",
        curated_zone_path="/data/mineralvision/lakehouse/curated",
        feature_zone_path="/data/mineralvision/lakehouse/feature",
        checkpoint_path="/data/mineralvision/lakehouse/checkpoints"
    )
    
    # Create a storage manager
    storage = DeltaLakeStorage(config)
    
    # Create a table
    schema = {
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
    
    storage.create_table(
        table_name="satellite_imagery",
        schema=schema,
        zone="raw",
        partition_by=["tile_id", "resolution"]
    )
    
    # Write data to the table
    storage.write_data(
        table_name="satellite_imagery",
        data=None,  # In a real implementation, this would be a DataFrame
        zone="raw",
        mode="append",
        partition_by=["tile_id", "resolution"]
    )
    
    # Read data from the table
    data = storage.read_table(
        table_name="satellite_imagery",
        zone="raw",
        filter_expr="sensor_type = 'Landsat-8'"
    )
    
    # Optimize the table
    storage.optimize_table(
        table_name="satellite_imagery",
        zone="raw",
        z_order_by=["acquisition_date", "sensor_type"]
    )
    
    # Get table history
    history = storage.get_table_history(
        table_name="satellite_imagery",
        zone="raw"
    )
    
    # Describe the table
    schema = storage.describe_table(
        table_name="satellite_imagery",
        zone="raw"
    )
