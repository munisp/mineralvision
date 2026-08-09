"""
Apache Iceberg Table Format Module
==================================

Production-grade Iceberg table format with:
- Table creation and management
- Schema evolution
- Partition evolution
- Time travel queries
- Snapshot management
- Compaction and optimization
"""

import os
import json
import logging
import uuid
import hashlib
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import threading

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class IcebergType(Enum):
    """Iceberg data types."""
    BOOLEAN = "boolean"
    INTEGER = "int"
    LONG = "long"
    FLOAT = "float"
    DOUBLE = "double"
    DECIMAL = "decimal"
    DATE = "date"
    TIME = "time"
    TIMESTAMP = "timestamp"
    TIMESTAMPTZ = "timestamptz"
    STRING = "string"
    UUID = "uuid"
    FIXED = "fixed"
    BINARY = "binary"
    STRUCT = "struct"
    LIST = "list"
    MAP = "map"


class PartitionTransform(Enum):
    """Partition transforms."""
    IDENTITY = "identity"
    BUCKET = "bucket"
    TRUNCATE = "truncate"
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    HOUR = "hour"
    VOID = "void"


@dataclass
class IcebergField:
    """Represents an Iceberg schema field."""
    field_id: int
    name: str
    field_type: IcebergType
    required: bool = False
    doc: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'id': self.field_id,
            'name': self.name,
            'type': self.field_type.value,
            'required': self.required
        }
        if self.doc:
            result['doc'] = self.doc
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'IcebergField':
        """Create from dictionary."""
        return cls(
            field_id=d['id'],
            name=d['name'],
            field_type=IcebergType(d['type']) if isinstance(d['type'], str) else IcebergType.STRING,
            required=d.get('required', False),
            doc=d.get('doc')
        )


@dataclass
class IcebergSchema:
    """Represents an Iceberg table schema."""
    schema_id: int
    fields: List[IcebergField] = field(default_factory=list)
    identifier_field_ids: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'schema-id': self.schema_id,
            'type': 'struct',
            'fields': [f.to_dict() for f in self.fields],
            'identifier-field-ids': self.identifier_field_ids
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'IcebergSchema':
        """Create from dictionary."""
        return cls(
            schema_id=d.get('schema-id', 0),
            fields=[IcebergField.from_dict(f) for f in d.get('fields', [])],
            identifier_field_ids=d.get('identifier-field-ids', [])
        )
    
    def add_field(self, name: str, field_type: IcebergType,
                 required: bool = False, doc: str = None) -> 'IcebergSchema':
        """Add a field to the schema."""
        field_id = max([f.field_id for f in self.fields], default=0) + 1
        self.fields.append(IcebergField(
            field_id=field_id,
            name=name,
            field_type=field_type,
            required=required,
            doc=doc
        ))
        return self
    
    def get_field(self, name: str) -> Optional[IcebergField]:
        """Get a field by name."""
        for f in self.fields:
            if f.name == name:
                return f
        return None


@dataclass
class PartitionField:
    """Represents a partition field."""
    source_id: int
    field_id: int
    name: str
    transform: PartitionTransform
    transform_param: Optional[int] = None  # For bucket/truncate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        transform_str = self.transform.value
        if self.transform_param is not None:
            transform_str = f"{self.transform.value}[{self.transform_param}]"
        
        return {
            'source-id': self.source_id,
            'field-id': self.field_id,
            'name': self.name,
            'transform': transform_str
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PartitionField':
        """Create from dictionary."""
        transform_str = d.get('transform', 'identity')
        transform_param = None
        
        if '[' in transform_str:
            transform_name = transform_str.split('[')[0]
            transform_param = int(transform_str.split('[')[1].rstrip(']'))
        else:
            transform_name = transform_str
        
        return cls(
            source_id=d['source-id'],
            field_id=d['field-id'],
            name=d['name'],
            transform=PartitionTransform(transform_name),
            transform_param=transform_param
        )


@dataclass
class PartitionSpec:
    """Represents a partition specification."""
    spec_id: int
    fields: List[PartitionField] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'spec-id': self.spec_id,
            'fields': [f.to_dict() for f in self.fields]
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PartitionSpec':
        """Create from dictionary."""
        return cls(
            spec_id=d.get('spec-id', 0),
            fields=[PartitionField.from_dict(f) for f in d.get('fields', [])]
        )


@dataclass
class Snapshot:
    """Represents a table snapshot."""
    snapshot_id: int
    parent_snapshot_id: Optional[int]
    sequence_number: int
    timestamp_ms: int
    manifest_list: str
    summary: Dict[str, str] = field(default_factory=dict)
    schema_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'snapshot-id': self.snapshot_id,
            'sequence-number': self.sequence_number,
            'timestamp-ms': self.timestamp_ms,
            'manifest-list': self.manifest_list,
            'summary': self.summary,
            'schema-id': self.schema_id
        }
        if self.parent_snapshot_id is not None:
            result['parent-snapshot-id'] = self.parent_snapshot_id
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Snapshot':
        """Create from dictionary."""
        return cls(
            snapshot_id=d['snapshot-id'],
            parent_snapshot_id=d.get('parent-snapshot-id'),
            sequence_number=d.get('sequence-number', 0),
            timestamp_ms=d['timestamp-ms'],
            manifest_list=d['manifest-list'],
            summary=d.get('summary', {}),
            schema_id=d.get('schema-id', 0)
        )


@dataclass
class TableMetadata:
    """Represents Iceberg table metadata."""
    format_version: int = 2
    table_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    location: str = ""
    last_sequence_number: int = 0
    last_updated_ms: int = 0
    last_column_id: int = 0
    current_schema_id: int = 0
    schemas: List[IcebergSchema] = field(default_factory=list)
    default_spec_id: int = 0
    partition_specs: List[PartitionSpec] = field(default_factory=list)
    last_partition_id: int = 0
    default_sort_order_id: int = 0
    sort_orders: List[Dict] = field(default_factory=list)
    properties: Dict[str, str] = field(default_factory=dict)
    current_snapshot_id: Optional[int] = None
    snapshots: List[Snapshot] = field(default_factory=list)
    snapshot_log: List[Dict] = field(default_factory=list)
    metadata_log: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'format-version': self.format_version,
            'table-uuid': self.table_uuid,
            'location': self.location,
            'last-sequence-number': self.last_sequence_number,
            'last-updated-ms': self.last_updated_ms,
            'last-column-id': self.last_column_id,
            'current-schema-id': self.current_schema_id,
            'schemas': [s.to_dict() for s in self.schemas],
            'default-spec-id': self.default_spec_id,
            'partition-specs': [p.to_dict() for p in self.partition_specs],
            'last-partition-id': self.last_partition_id,
            'default-sort-order-id': self.default_sort_order_id,
            'sort-orders': self.sort_orders,
            'properties': self.properties,
            'snapshots': [s.to_dict() for s in self.snapshots],
            'snapshot-log': self.snapshot_log,
            'metadata-log': self.metadata_log
        }
        if self.current_snapshot_id is not None:
            result['current-snapshot-id'] = self.current_snapshot_id
        return result
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TableMetadata':
        """Create from dictionary."""
        return cls(
            format_version=d.get('format-version', 2),
            table_uuid=d.get('table-uuid', str(uuid.uuid4())),
            location=d.get('location', ''),
            last_sequence_number=d.get('last-sequence-number', 0),
            last_updated_ms=d.get('last-updated-ms', 0),
            last_column_id=d.get('last-column-id', 0),
            current_schema_id=d.get('current-schema-id', 0),
            schemas=[IcebergSchema.from_dict(s) for s in d.get('schemas', [])],
            default_spec_id=d.get('default-spec-id', 0),
            partition_specs=[PartitionSpec.from_dict(p) for p in d.get('partition-specs', [])],
            last_partition_id=d.get('last-partition-id', 0),
            default_sort_order_id=d.get('default-sort-order-id', 0),
            sort_orders=d.get('sort-orders', []),
            properties=d.get('properties', {}),
            current_snapshot_id=d.get('current-snapshot-id'),
            snapshots=[Snapshot.from_dict(s) for s in d.get('snapshots', [])],
            snapshot_log=d.get('snapshot-log', []),
            metadata_log=d.get('metadata-log', [])
        )


class IcebergTable:
    """
    Represents an Iceberg table.
    """
    
    def __init__(self, name: str, location: str, metadata: TableMetadata = None):
        self.name = name
        self.location = location
        self.metadata = metadata or TableMetadata(location=location)
        self._lock = threading.Lock()
    
    @property
    def schema(self) -> Optional[IcebergSchema]:
        """Get current schema."""
        for s in self.metadata.schemas:
            if s.schema_id == self.metadata.current_schema_id:
                return s
        return self.metadata.schemas[0] if self.metadata.schemas else None
    
    @property
    def partition_spec(self) -> Optional[PartitionSpec]:
        """Get current partition spec."""
        for p in self.metadata.partition_specs:
            if p.spec_id == self.metadata.default_spec_id:
                return p
        return self.metadata.partition_specs[0] if self.metadata.partition_specs else None
    
    @property
    def current_snapshot(self) -> Optional[Snapshot]:
        """Get current snapshot."""
        if self.metadata.current_snapshot_id is None:
            return None
        for s in self.metadata.snapshots:
            if s.snapshot_id == self.metadata.current_snapshot_id:
                return s
        return None
    
    def update_schema(self, new_schema: IcebergSchema):
        """Update table schema."""
        with self._lock:
            new_schema.schema_id = max([s.schema_id for s in self.metadata.schemas], default=-1) + 1
            self.metadata.schemas.append(new_schema)
            self.metadata.current_schema_id = new_schema.schema_id
            self.metadata.last_updated_ms = int(datetime.utcnow().timestamp() * 1000)
    
    def update_partition_spec(self, new_spec: PartitionSpec):
        """Update partition specification."""
        with self._lock:
            new_spec.spec_id = max([p.spec_id for p in self.metadata.partition_specs], default=-1) + 1
            self.metadata.partition_specs.append(new_spec)
            self.metadata.default_spec_id = new_spec.spec_id
            self.metadata.last_updated_ms = int(datetime.utcnow().timestamp() * 1000)
    
    def add_snapshot(self, manifest_list: str, summary: Dict[str, str] = None) -> Snapshot:
        """Add a new snapshot."""
        with self._lock:
            snapshot_id = int(datetime.utcnow().timestamp() * 1000000)
            
            snapshot = Snapshot(
                snapshot_id=snapshot_id,
                parent_snapshot_id=self.metadata.current_snapshot_id,
                sequence_number=self.metadata.last_sequence_number + 1,
                timestamp_ms=int(datetime.utcnow().timestamp() * 1000),
                manifest_list=manifest_list,
                summary=summary or {},
                schema_id=self.metadata.current_schema_id
            )
            
            self.metadata.snapshots.append(snapshot)
            self.metadata.current_snapshot_id = snapshot_id
            self.metadata.last_sequence_number += 1
            self.metadata.last_updated_ms = int(datetime.utcnow().timestamp() * 1000)
            
            self.metadata.snapshot_log.append({
                'timestamp-ms': snapshot.timestamp_ms,
                'snapshot-id': snapshot_id
            })
            
            return snapshot
    
    def rollback_to_snapshot(self, snapshot_id: int) -> bool:
        """Rollback to a previous snapshot."""
        with self._lock:
            for snapshot in self.metadata.snapshots:
                if snapshot.snapshot_id == snapshot_id:
                    self.metadata.current_snapshot_id = snapshot_id
                    self.metadata.last_updated_ms = int(datetime.utcnow().timestamp() * 1000)
                    return True
            return False
    
    def get_snapshot_at_timestamp(self, timestamp_ms: int) -> Optional[Snapshot]:
        """Get snapshot at or before a timestamp."""
        valid_snapshots = [s for s in self.metadata.snapshots if s.timestamp_ms <= timestamp_ms]
        if not valid_snapshots:
            return None
        return max(valid_snapshots, key=lambda s: s.timestamp_ms)
    
    def expire_snapshots(self, older_than_ms: int) -> int:
        """Expire old snapshots."""
        with self._lock:
            cutoff = int(datetime.utcnow().timestamp() * 1000) - older_than_ms
            
            # Keep current snapshot and recent ones
            to_keep = [s for s in self.metadata.snapshots 
                      if s.timestamp_ms >= cutoff or s.snapshot_id == self.metadata.current_snapshot_id]
            
            expired_count = len(self.metadata.snapshots) - len(to_keep)
            self.metadata.snapshots = to_keep
            
            return expired_count


class IcebergCatalog(ABC):
    """Abstract base class for Iceberg catalogs."""
    
    @abstractmethod
    def create_table(self, name: str, schema: IcebergSchema,
                    partition_spec: PartitionSpec = None,
                    location: str = None) -> IcebergTable:
        """Create a new table."""
        pass
    
    @abstractmethod
    def load_table(self, name: str) -> Optional[IcebergTable]:
        """Load an existing table."""
        pass
    
    @abstractmethod
    def drop_table(self, name: str) -> bool:
        """Drop a table."""
        pass
    
    @abstractmethod
    def list_tables(self, namespace: str = None) -> List[str]:
        """List all tables."""
        pass
    
    @abstractmethod
    def table_exists(self, name: str) -> bool:
        """Check if table exists."""
        pass


class LocalIcebergCatalog(IcebergCatalog):
    """
    Local file-based Iceberg catalog.
    """
    
    def __init__(self, warehouse_path: str):
        self.warehouse_path = warehouse_path
        os.makedirs(warehouse_path, exist_ok=True)
        
        self._tables: Dict[str, IcebergTable] = {}
        self._lock = threading.Lock()
        
        self._load_catalog()
    
    def _load_catalog(self):
        """Load catalog from storage."""
        catalog_path = os.path.join(self.warehouse_path, 'catalog.json')
        
        if os.path.exists(catalog_path):
            try:
                with open(catalog_path, 'r') as f:
                    data = json.load(f)
                
                for table_name, table_location in data.get('tables', {}).items():
                    metadata_path = os.path.join(table_location, 'metadata', 'current.json')
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            metadata = TableMetadata.from_dict(json.load(f))
                        self._tables[table_name] = IcebergTable(table_name, table_location, metadata)
                
                logger.info(f"Loaded {len(self._tables)} tables from catalog")
                
            except Exception as e:
                logger.error(f"Error loading catalog: {e}")
    
    def _save_catalog(self):
        """Save catalog to storage."""
        catalog_path = os.path.join(self.warehouse_path, 'catalog.json')
        
        data = {
            'tables': {name: table.location for name, table in self._tables.items()}
        }
        
        with open(catalog_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_table_metadata(self, table: IcebergTable):
        """Save table metadata."""
        metadata_dir = os.path.join(table.location, 'metadata')
        os.makedirs(metadata_dir, exist_ok=True)
        
        # Save versioned metadata
        version = len([f for f in os.listdir(metadata_dir) if f.startswith('v')]) + 1
        versioned_path = os.path.join(metadata_dir, f'v{version}.metadata.json')
        
        with open(versioned_path, 'w') as f:
            json.dump(table.metadata.to_dict(), f, indent=2)
        
        # Update current pointer
        current_path = os.path.join(metadata_dir, 'current.json')
        with open(current_path, 'w') as f:
            json.dump(table.metadata.to_dict(), f, indent=2)
    
    def create_table(self, name: str, schema: IcebergSchema,
                    partition_spec: PartitionSpec = None,
                    location: str = None) -> IcebergTable:
        """Create a new table."""
        with self._lock:
            if name in self._tables:
                raise ValueError(f"Table {name} already exists")
            
            table_location = location or os.path.join(self.warehouse_path, name)
            os.makedirs(table_location, exist_ok=True)
            
            metadata = TableMetadata(
                location=table_location,
                schemas=[schema],
                current_schema_id=schema.schema_id,
                partition_specs=[partition_spec] if partition_spec else [PartitionSpec(spec_id=0)],
                default_spec_id=partition_spec.spec_id if partition_spec else 0,
                last_updated_ms=int(datetime.utcnow().timestamp() * 1000)
            )
            
            table = IcebergTable(name, table_location, metadata)
            self._tables[name] = table
            
            self._save_table_metadata(table)
            self._save_catalog()
            
            logger.info(f"Created Iceberg table: {name}")
            
            return table
    
    def load_table(self, name: str) -> Optional[IcebergTable]:
        """Load an existing table."""
        return self._tables.get(name)
    
    def drop_table(self, name: str) -> bool:
        """Drop a table."""
        with self._lock:
            if name not in self._tables:
                return False
            
            del self._tables[name]
            self._save_catalog()
            
            logger.info(f"Dropped Iceberg table: {name}")
            return True
    
    def list_tables(self, namespace: str = None) -> List[str]:
        """List all tables."""
        return list(self._tables.keys())
    
    def table_exists(self, name: str) -> bool:
        """Check if table exists."""
        return name in self._tables


class IcebergStorage:
    """
    High-level Iceberg storage interface.
    """
    
    def __init__(self, catalog: IcebergCatalog = None, warehouse_path: str = None):
        self.catalog = catalog or LocalIcebergCatalog(warehouse_path or '/tmp/iceberg_warehouse')
        self._data_cache: Dict[str, pd.DataFrame] = {}
    
    def create_table(self, name: str, schema: Dict[str, str],
                    partition_by: List[str] = None,
                    location: str = None) -> IcebergTable:
        """
        Create a new Iceberg table.
        
        Args:
            name: Table name
            schema: Schema as {column_name: type_string}
            partition_by: Columns to partition by
            location: Optional table location
            
        Returns:
            Created table
        """
        # Convert schema dict to IcebergSchema
        iceberg_schema = IcebergSchema(schema_id=0)
        
        type_mapping = {
            'string': IcebergType.STRING,
            'str': IcebergType.STRING,
            'int': IcebergType.INTEGER,
            'integer': IcebergType.INTEGER,
            'long': IcebergType.LONG,
            'bigint': IcebergType.LONG,
            'float': IcebergType.FLOAT,
            'double': IcebergType.DOUBLE,
            'boolean': IcebergType.BOOLEAN,
            'bool': IcebergType.BOOLEAN,
            'date': IcebergType.DATE,
            'timestamp': IcebergType.TIMESTAMP,
            'binary': IcebergType.BINARY
        }
        
        for col_name, col_type in schema.items():
            iceberg_type = type_mapping.get(col_type.lower(), IcebergType.STRING)
            iceberg_schema.add_field(col_name, iceberg_type)
        
        # Create partition spec if specified
        partition_spec = None
        if partition_by:
            partition_spec = PartitionSpec(spec_id=0)
            for i, col in enumerate(partition_by):
                field = iceberg_schema.get_field(col)
                if field:
                    partition_spec.fields.append(PartitionField(
                        source_id=field.field_id,
                        field_id=1000 + i,
                        name=col,
                        transform=PartitionTransform.IDENTITY
                    ))
        
        return self.catalog.create_table(name, iceberg_schema, partition_spec, location)
    
    def write_data(self, table_name: str, data: pd.DataFrame,
                  mode: str = "append") -> bool:
        """
        Write data to an Iceberg table.
        
        Args:
            table_name: Table name
            data: DataFrame to write
            mode: Write mode ('append', 'overwrite')
            
        Returns:
            True if successful
        """
        table = self.catalog.load_table(table_name)
        if not table:
            raise ValueError(f"Table {table_name} not found")
        
        try:
            # Write data files
            data_dir = os.path.join(table.location, 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            # Generate unique file name
            file_id = str(uuid.uuid4())[:8]
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            data_file = os.path.join(data_dir, f'data_{timestamp}_{file_id}.parquet')
            
            # Write parquet file
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            arrow_table = pa.Table.from_pandas(data)
            pq.write_table(arrow_table, data_file)
            
            # Create manifest
            manifest_dir = os.path.join(table.location, 'metadata')
            manifest_file = os.path.join(manifest_dir, f'manifest_{timestamp}_{file_id}.avro')
            
            # Write manifest (simplified - just JSON for now)
            manifest_data = {
                'data_files': [{
                    'file_path': data_file,
                    'file_format': 'PARQUET',
                    'record_count': len(data),
                    'file_size_in_bytes': os.path.getsize(data_file)
                }]
            }
            
            with open(manifest_file + '.json', 'w') as f:
                json.dump(manifest_data, f)
            
            # Add snapshot
            summary = {
                'operation': mode,
                'added-data-files': '1',
                'added-records': str(len(data)),
                'total-records': str(len(data))
            }
            
            table.add_snapshot(manifest_file, summary)
            
            # Save updated metadata
            if isinstance(self.catalog, LocalIcebergCatalog):
                self.catalog._save_table_metadata(table)
            
            # Update cache
            if mode == 'overwrite':
                self._data_cache[table_name] = data.copy()
            else:
                if table_name in self._data_cache:
                    self._data_cache[table_name] = pd.concat([
                        self._data_cache[table_name], data
                    ], ignore_index=True)
                else:
                    self._data_cache[table_name] = data.copy()
            
            logger.info(f"Wrote {len(data)} rows to {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing data: {e}")
            return False
    
    def read_data(self, table_name: str, snapshot_id: int = None,
                 as_of_timestamp: datetime = None,
                 columns: List[str] = None,
                 filter_expr: str = None) -> pd.DataFrame:
        """
        Read data from an Iceberg table.
        
        Args:
            table_name: Table name
            snapshot_id: Specific snapshot to read
            as_of_timestamp: Read as of timestamp (time travel)
            columns: Columns to read
            filter_expr: Filter expression
            
        Returns:
            DataFrame with data
        """
        table = self.catalog.load_table(table_name)
        if not table:
            raise ValueError(f"Table {table_name} not found")
        
        try:
            # Determine which snapshot to read
            if snapshot_id:
                snapshot = None
                for s in table.metadata.snapshots:
                    if s.snapshot_id == snapshot_id:
                        snapshot = s
                        break
            elif as_of_timestamp:
                timestamp_ms = int(as_of_timestamp.timestamp() * 1000)
                snapshot = table.get_snapshot_at_timestamp(timestamp_ms)
            else:
                snapshot = table.current_snapshot
            
            if not snapshot:
                logger.warning(f"No snapshot found for {table_name}")
                return pd.DataFrame()
            
            # Read data files from manifest
            manifest_json = snapshot.manifest_list + '.json'
            
            if os.path.exists(manifest_json):
                with open(manifest_json, 'r') as f:
                    manifest = json.load(f)
                
                dfs = []
                for data_file in manifest.get('data_files', []):
                    file_path = data_file['file_path']
                    if os.path.exists(file_path):
                        import pyarrow.parquet as pq
                        df = pq.read_table(file_path, columns=columns).to_pandas()
                        dfs.append(df)
                
                if dfs:
                    result = pd.concat(dfs, ignore_index=True)
                else:
                    result = pd.DataFrame()
            else:
                # Fallback to cache
                result = self._data_cache.get(table_name, pd.DataFrame())
                if columns and not result.empty:
                    valid_cols = [c for c in columns if c in result.columns]
                    result = result[valid_cols]
            
            # Apply filter
            if filter_expr and not result.empty:
                try:
                    result = result.query(filter_expr)
                except Exception as e:
                    logger.warning(f"Could not apply filter: {e}")
            
            logger.info(f"Read {len(result)} rows from {table_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error reading data: {e}")
            return pd.DataFrame()
    
    def time_travel(self, table_name: str, as_of: Union[datetime, int]) -> pd.DataFrame:
        """
        Read table as of a specific time or snapshot.
        
        Args:
            table_name: Table name
            as_of: Timestamp or snapshot ID
            
        Returns:
            DataFrame with historical data
        """
        if isinstance(as_of, datetime):
            return self.read_data(table_name, as_of_timestamp=as_of)
        else:
            return self.read_data(table_name, snapshot_id=as_of)
    
    def get_history(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get table history (snapshots).
        
        Args:
            table_name: Table name
            
        Returns:
            List of snapshot information
        """
        table = self.catalog.load_table(table_name)
        if not table:
            return []
        
        history = []
        for snapshot in table.metadata.snapshots:
            history.append({
                'snapshot_id': snapshot.snapshot_id,
                'timestamp': datetime.fromtimestamp(snapshot.timestamp_ms / 1000).isoformat(),
                'parent_snapshot_id': snapshot.parent_snapshot_id,
                'operation': snapshot.summary.get('operation', 'unknown'),
                'records': snapshot.summary.get('total-records', '0')
            })
        
        return history
    
    def rollback(self, table_name: str, snapshot_id: int) -> bool:
        """
        Rollback table to a previous snapshot.
        
        Args:
            table_name: Table name
            snapshot_id: Snapshot to rollback to
            
        Returns:
            True if successful
        """
        table = self.catalog.load_table(table_name)
        if not table:
            return False
        
        success = table.rollback_to_snapshot(snapshot_id)
        
        if success and isinstance(self.catalog, LocalIcebergCatalog):
            self.catalog._save_table_metadata(table)
        
        return success
    
    def expire_snapshots(self, table_name: str, older_than_days: int = 7) -> int:
        """
        Expire old snapshots.
        
        Args:
            table_name: Table name
            older_than_days: Expire snapshots older than this
            
        Returns:
            Number of expired snapshots
        """
        table = self.catalog.load_table(table_name)
        if not table:
            return 0
        
        older_than_ms = older_than_days * 24 * 60 * 60 * 1000
        expired = table.expire_snapshots(older_than_ms)
        
        if expired > 0 and isinstance(self.catalog, LocalIcebergCatalog):
            self.catalog._save_table_metadata(table)
        
        return expired
    
    def compact_table(self, table_name: str) -> bool:
        """
        Compact table data files.
        
        Args:
            table_name: Table name
            
        Returns:
            True if successful
        """
        table = self.catalog.load_table(table_name)
        if not table:
            return False
        
        try:
            # Read all current data
            data = self.read_data(table_name)
            
            if data.empty:
                return True
            
            # Write compacted data
            data_dir = os.path.join(table.location, 'data')
            
            # Remove old data files
            for f in os.listdir(data_dir):
                if f.endswith('.parquet'):
                    os.remove(os.path.join(data_dir, f))
            
            # Write single compacted file
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            file_id = str(uuid.uuid4())[:8]
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            compacted_file = os.path.join(data_dir, f'compacted_{timestamp}_{file_id}.parquet')
            
            arrow_table = pa.Table.from_pandas(data)
            pq.write_table(arrow_table, compacted_file)
            
            # Create new manifest
            manifest_dir = os.path.join(table.location, 'metadata')
            manifest_file = os.path.join(manifest_dir, f'manifest_{timestamp}_{file_id}.avro')
            
            manifest_data = {
                'data_files': [{
                    'file_path': compacted_file,
                    'file_format': 'PARQUET',
                    'record_count': len(data),
                    'file_size_in_bytes': os.path.getsize(compacted_file)
                }]
            }
            
            with open(manifest_file + '.json', 'w') as f:
                json.dump(manifest_data, f)
            
            # Add compaction snapshot
            summary = {
                'operation': 'compact',
                'total-data-files': '1',
                'total-records': str(len(data))
            }
            
            table.add_snapshot(manifest_file, summary)
            
            if isinstance(self.catalog, LocalIcebergCatalog):
                self.catalog._save_table_metadata(table)
            
            logger.info(f"Compacted table {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error compacting table: {e}")
            return False


def create_iceberg_storage(warehouse_path: str = None) -> IcebergStorage:
    """Factory function to create Iceberg storage."""
    return IcebergStorage(warehouse_path=warehouse_path)


def create_iceberg_catalog(warehouse_path: str = None) -> IcebergCatalog:
    """Factory function to create Iceberg catalog."""
    return LocalIcebergCatalog(warehouse_path or '/tmp/iceberg_warehouse')
