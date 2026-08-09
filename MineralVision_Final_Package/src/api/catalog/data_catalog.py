"""
Data Catalog & Discovery for MineralVision.

Provides:
- Unified data catalog for all geospatial assets
- Metadata management and search
- Data lineage visualization
- Asset discovery across projects
- Schema registry
- Data quality profiling
- Tag-based organization
"""

import json
import hashlib
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import re

logger = logging.getLogger(__name__)


class AssetType(Enum):
    """Types of data assets."""
    DATASET = "dataset"
    TABLE = "table"
    FILE = "file"
    MODEL = "model"
    PIPELINE = "pipeline"
    DASHBOARD = "dashboard"
    FEATURE = "feature"
    SCHEMA = "schema"
    GLOSSARY_TERM = "glossary_term"


class DataFormat(Enum):
    """Data formats."""
    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"
    GEOTIFF = "geotiff"
    LAS = "las"
    SEGY = "segy"
    SHAPEFILE = "shapefile"
    GEOPACKAGE = "geopackage"
    TILEDB = "tiledb"
    ZARR = "zarr"
    DELTA = "delta"
    ICEBERG = "iceberg"


class DataDomain(Enum):
    """Data domains for organization."""
    GEOLOGY = "geology"
    GEOPHYSICS = "geophysics"
    GEOCHEMISTRY = "geochemistry"
    REMOTE_SENSING = "remote_sensing"
    DRILLING = "drilling"
    SURVEY = "survey"
    ENVIRONMENTAL = "environmental"
    SOIL = "soil"
    CLIMATE = "climate"
    INFRASTRUCTURE = "infrastructure"


class QualityLevel(Enum):
    """Data quality levels."""
    GOLD = "gold"      # Production-ready, validated
    SILVER = "silver"  # Processed, needs validation
    BRONZE = "bronze"  # Raw, unprocessed


class LineageType(Enum):
    """Types of lineage relationships."""
    DERIVED_FROM = "derived_from"
    TRANSFORMED_BY = "transformed_by"
    CONSUMED_BY = "consumed_by"
    PRODUCED_BY = "produced_by"
    DEPENDS_ON = "depends_on"


@dataclass
class SchemaField:
    """Schema field definition."""
    name: str
    data_type: str
    description: str = ""
    nullable: bool = True
    is_primary_key: bool = False
    is_partition_key: bool = False
    default_value: Any = None
    constraints: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'data_type': self.data_type,
            'description': self.description,
            'nullable': self.nullable,
            'is_primary_key': self.is_primary_key,
            'is_partition_key': self.is_partition_key,
            'default_value': self.default_value,
            'constraints': self.constraints,
            'tags': self.tags
        }


@dataclass
class Schema:
    """Data schema definition."""
    schema_id: str
    name: str
    version: str
    fields: List[SchemaField]
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema_id': self.schema_id,
            'name': self.name,
            'version': self.version,
            'fields': [f.to_dict() for f in self.fields],
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class DataQualityProfile:
    """Data quality profile."""
    profile_id: str
    asset_id: str
    profiled_at: datetime
    row_count: int
    column_count: int
    null_counts: Dict[str, int]
    unique_counts: Dict[str, int]
    min_values: Dict[str, Any]
    max_values: Dict[str, Any]
    mean_values: Dict[str, float]
    completeness_score: float  # 0-1
    validity_score: float  # 0-1
    uniqueness_score: float  # 0-1
    consistency_score: float  # 0-1
    overall_score: float  # 0-1
    issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'profile_id': self.profile_id,
            'asset_id': self.asset_id,
            'profiled_at': self.profiled_at.isoformat(),
            'row_count': self.row_count,
            'column_count': self.column_count,
            'null_counts': self.null_counts,
            'unique_counts': self.unique_counts,
            'min_values': self.min_values,
            'max_values': self.max_values,
            'mean_values': self.mean_values,
            'completeness_score': self.completeness_score,
            'validity_score': self.validity_score,
            'uniqueness_score': self.uniqueness_score,
            'consistency_score': self.consistency_score,
            'overall_score': self.overall_score,
            'issues': self.issues
        }


@dataclass
class SpatialExtent:
    """Spatial extent of a dataset."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    crs: str = "EPSG:4326"
    min_z: Optional[float] = None
    max_z: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'min_x': self.min_x,
            'min_y': self.min_y,
            'max_x': self.max_x,
            'max_y': self.max_y,
            'crs': self.crs,
            'min_z': self.min_z,
            'max_z': self.max_z
        }
        
    def intersects(self, other: 'SpatialExtent') -> bool:
        """Check if extents intersect."""
        return not (self.max_x < other.min_x or self.min_x > other.max_x or
                   self.max_y < other.min_y or self.min_y > other.max_y)
                   
    def contains_point(self, x: float, y: float) -> bool:
        """Check if extent contains a point."""
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y


@dataclass
class TemporalExtent:
    """Temporal extent of a dataset."""
    start_time: datetime
    end_time: datetime
    granularity: str = "day"  # second, minute, hour, day, month, year
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'granularity': self.granularity
        }


@dataclass
class LineageEdge:
    """Lineage relationship between assets."""
    edge_id: str
    source_id: str
    target_id: str
    relationship: LineageType
    transformation: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'edge_id': self.edge_id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relationship': self.relationship.value,
            'transformation': self.transformation,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class Tag:
    """Tag for organizing assets."""
    tag_id: str
    name: str
    category: str = "general"
    description: str = ""
    color: str = "#808080"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tag_id': self.tag_id,
            'name': self.name,
            'category': self.category,
            'description': self.description,
            'color': self.color
        }


@dataclass
class GlossaryTerm:
    """Business glossary term."""
    term_id: str
    name: str
    definition: str
    domain: DataDomain
    synonyms: List[str] = field(default_factory=list)
    related_terms: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    owner: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'term_id': self.term_id,
            'name': self.name,
            'definition': self.definition,
            'domain': self.domain.value,
            'synonyms': self.synonyms,
            'related_terms': self.related_terms,
            'examples': self.examples,
            'owner': self.owner,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class DataAsset:
    """Data asset in the catalog."""
    asset_id: str
    name: str
    asset_type: AssetType
    description: str
    domain: DataDomain
    quality_level: QualityLevel
    
    # Location
    location: str  # URI or path
    format: Optional[DataFormat] = None
    
    # Schema
    schema: Optional[Schema] = None
    
    # Extents
    spatial_extent: Optional[SpatialExtent] = None
    temporal_extent: Optional[TemporalExtent] = None
    
    # Ownership
    owner: str = ""
    steward: str = ""
    tenant_id: str = ""
    project_id: str = ""
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    glossary_terms: List[str] = field(default_factory=list)
    custom_properties: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    size_bytes: int = 0
    row_count: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed_at: Optional[datetime] = None
    
    # Quality
    quality_profile: Optional[DataQualityProfile] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'asset_id': self.asset_id,
            'name': self.name,
            'asset_type': self.asset_type.value,
            'description': self.description,
            'domain': self.domain.value,
            'quality_level': self.quality_level.value,
            'location': self.location,
            'format': self.format.value if self.format else None,
            'schema': self.schema.to_dict() if self.schema else None,
            'spatial_extent': self.spatial_extent.to_dict() if self.spatial_extent else None,
            'temporal_extent': self.temporal_extent.to_dict() if self.temporal_extent else None,
            'owner': self.owner,
            'steward': self.steward,
            'tenant_id': self.tenant_id,
            'project_id': self.project_id,
            'tags': self.tags,
            'glossary_terms': self.glossary_terms,
            'custom_properties': self.custom_properties,
            'size_bytes': self.size_bytes,
            'row_count': self.row_count,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_accessed_at': self.last_accessed_at.isoformat() if self.last_accessed_at else None,
            'quality_profile': self.quality_profile.to_dict() if self.quality_profile else None
        }


@dataclass
class SearchQuery:
    """Search query for catalog."""
    text: str = ""
    asset_types: List[AssetType] = field(default_factory=list)
    domains: List[DataDomain] = field(default_factory=list)
    quality_levels: List[QualityLevel] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    owner: str = ""
    tenant_id: str = ""
    project_id: str = ""
    spatial_extent: Optional[SpatialExtent] = None
    temporal_extent: Optional[TemporalExtent] = None
    min_quality_score: float = 0.0
    limit: int = 100
    offset: int = 0


@dataclass
class SearchResult:
    """Search result."""
    asset: DataAsset
    score: float
    highlights: Dict[str, List[str]] = field(default_factory=dict)


class SearchIndex:
    """Full-text search index for catalog."""
    
    def __init__(self):
        self._index: Dict[str, Set[str]] = {}  # term -> asset_ids
        self._assets: Dict[str, DataAsset] = {}
        self._lock = threading.Lock()
        
    def index_asset(self, asset: DataAsset) -> None:
        """Index an asset for search."""
        with self._lock:
            self._assets[asset.asset_id] = asset
            
            terms = self._tokenize(asset.name)
            terms.update(self._tokenize(asset.description))
            for tag in asset.tags:
                terms.update(self._tokenize(tag))
                
            for term in terms:
                if term not in self._index:
                    self._index[term] = set()
                self._index[term].add(asset.asset_id)
                
    def remove_asset(self, asset_id: str) -> None:
        """Remove asset from index."""
        with self._lock:
            asset = self._assets.pop(asset_id, None)
            if asset:
                terms = self._tokenize(asset.name)
                terms.update(self._tokenize(asset.description))
                for term in terms:
                    if term in self._index:
                        self._index[term].discard(asset_id)
                        
    def search(self, query: str, limit: int = 100) -> List[Tuple[str, float]]:
        """
        Search for assets.
        
        Returns:
            List of (asset_id, score) tuples
        """
        with self._lock:
            terms = self._tokenize(query)
            if not terms:
                return []
                
            scores: Dict[str, float] = {}
            
            for term in terms:
                if term in self._index:
                    for asset_id in self._index[term]:
                        scores[asset_id] = scores.get(asset_id, 0) + 1
                        
            for term in terms:
                for indexed_term in self._index:
                    if term in indexed_term and term != indexed_term:
                        for asset_id in self._index[indexed_term]:
                            scores[asset_id] = scores.get(asset_id, 0) + 0.5
                            
            results = sorted(scores.items(), key=lambda x: -x[1])
            return results[:limit]
            
    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text for indexing."""
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return set(tokens)


class LineageGraph:
    """Graph of data lineage relationships."""
    
    def __init__(self):
        self._edges: Dict[str, LineageEdge] = {}
        self._outgoing: Dict[str, List[str]] = {}  # asset_id -> edge_ids
        self._incoming: Dict[str, List[str]] = {}  # asset_id -> edge_ids
        self._lock = threading.Lock()
        
    def add_edge(self, edge: LineageEdge) -> None:
        """Add lineage edge."""
        with self._lock:
            self._edges[edge.edge_id] = edge
            
            if edge.source_id not in self._outgoing:
                self._outgoing[edge.source_id] = []
            self._outgoing[edge.source_id].append(edge.edge_id)
            
            if edge.target_id not in self._incoming:
                self._incoming[edge.target_id] = []
            self._incoming[edge.target_id].append(edge.edge_id)
            
    def remove_edge(self, edge_id: str) -> None:
        """Remove lineage edge."""
        with self._lock:
            edge = self._edges.pop(edge_id, None)
            if edge:
                if edge.source_id in self._outgoing:
                    self._outgoing[edge.source_id].remove(edge_id)
                if edge.target_id in self._incoming:
                    self._incoming[edge.target_id].remove(edge_id)
                    
    def get_upstream(self, asset_id: str, depth: int = 10) -> List[DataAsset]:
        """Get upstream assets (sources)."""
        visited = set()
        result = []
        self._traverse_upstream(asset_id, depth, visited, result)
        return result
        
    def _traverse_upstream(self, asset_id: str, depth: int,
                          visited: Set[str], result: List[str]) -> None:
        """Traverse upstream in lineage graph."""
        if depth <= 0 or asset_id in visited:
            return
            
        visited.add(asset_id)
        
        with self._lock:
            edge_ids = self._incoming.get(asset_id, [])
            for edge_id in edge_ids:
                edge = self._edges.get(edge_id)
                if edge and edge.source_id not in visited:
                    result.append(edge.source_id)
                    self._traverse_upstream(edge.source_id, depth - 1, visited, result)
                    
    def get_downstream(self, asset_id: str, depth: int = 10) -> List[str]:
        """Get downstream assets (consumers)."""
        visited = set()
        result = []
        self._traverse_downstream(asset_id, depth, visited, result)
        return result
        
    def _traverse_downstream(self, asset_id: str, depth: int,
                            visited: Set[str], result: List[str]) -> None:
        """Traverse downstream in lineage graph."""
        if depth <= 0 or asset_id in visited:
            return
            
        visited.add(asset_id)
        
        with self._lock:
            edge_ids = self._outgoing.get(asset_id, [])
            for edge_id in edge_ids:
                edge = self._edges.get(edge_id)
                if edge and edge.target_id not in visited:
                    result.append(edge.target_id)
                    self._traverse_downstream(edge.target_id, depth - 1, visited, result)
                    
    def get_lineage_subgraph(self, asset_id: str,
                            upstream_depth: int = 3,
                            downstream_depth: int = 3) -> Dict[str, Any]:
        """Get lineage subgraph around an asset."""
        upstream = self.get_upstream(asset_id, upstream_depth)
        downstream = self.get_downstream(asset_id, downstream_depth)
        
        all_assets = set([asset_id] + upstream + downstream)
        
        edges = []
        with self._lock:
            for edge in self._edges.values():
                if edge.source_id in all_assets and edge.target_id in all_assets:
                    edges.append(edge.to_dict())
                    
        return {
            'center': asset_id,
            'nodes': list(all_assets),
            'edges': edges
        }


class DataQualityProfiler:
    """Profile data quality for assets."""
    
    def profile_dataset(self, asset_id: str, data: Any) -> DataQualityProfile:
        """
        Profile a dataset.
        
        Args:
            asset_id: Asset ID
            data: Data to profile (dict, list, or dataframe-like)
            
        Returns:
            Quality profile
        """
        profile_id = hashlib.md5(
            f"{asset_id}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        if isinstance(data, dict):
            return self._profile_dict(profile_id, asset_id, data)
        elif isinstance(data, list):
            return self._profile_list(profile_id, asset_id, data)
        else:
            return self._profile_generic(profile_id, asset_id, data)
            
    def _profile_dict(self, profile_id: str, asset_id: str,
                     data: Dict[str, Any]) -> DataQualityProfile:
        """Profile dictionary data."""
        columns = list(data.keys())
        
        null_counts = {}
        unique_counts = {}
        min_values = {}
        max_values = {}
        mean_values = {}
        
        row_count = 0
        
        for col, values in data.items():
            if isinstance(values, list):
                row_count = max(row_count, len(values))
                null_counts[col] = sum(1 for v in values if v is None)
                unique_counts[col] = len(set(v for v in values if v is not None))
                
                numeric_values = [v for v in values if isinstance(v, (int, float)) and v is not None]
                if numeric_values:
                    min_values[col] = min(numeric_values)
                    max_values[col] = max(numeric_values)
                    mean_values[col] = sum(numeric_values) / len(numeric_values)
                    
        completeness = 1 - (sum(null_counts.values()) / (row_count * len(columns))) if row_count > 0 and columns else 1.0
        uniqueness = sum(unique_counts.values()) / (row_count * len(columns)) if row_count > 0 and columns else 1.0
        
        return DataQualityProfile(
            profile_id=profile_id,
            asset_id=asset_id,
            profiled_at=datetime.utcnow(),
            row_count=row_count,
            column_count=len(columns),
            null_counts=null_counts,
            unique_counts=unique_counts,
            min_values=min_values,
            max_values=max_values,
            mean_values=mean_values,
            completeness_score=completeness,
            validity_score=0.95,
            uniqueness_score=min(uniqueness, 1.0),
            consistency_score=0.9,
            overall_score=(completeness + 0.95 + min(uniqueness, 1.0) + 0.9) / 4
        )
        
    def _profile_list(self, profile_id: str, asset_id: str,
                     data: List[Any]) -> DataQualityProfile:
        """Profile list data."""
        if not data:
            return DataQualityProfile(
                profile_id=profile_id,
                asset_id=asset_id,
                profiled_at=datetime.utcnow(),
                row_count=0,
                column_count=0,
                null_counts={},
                unique_counts={},
                min_values={},
                max_values={},
                mean_values={},
                completeness_score=0.0,
                validity_score=0.0,
                uniqueness_score=0.0,
                consistency_score=0.0,
                overall_score=0.0
            )
            
        if isinstance(data[0], dict):
            columns = set()
            for row in data:
                if isinstance(row, dict):
                    columns.update(row.keys())
            columns = list(columns)
            
            null_counts = {col: 0 for col in columns}
            unique_counts = {col: set() for col in columns}
            
            for row in data:
                if isinstance(row, dict):
                    for col in columns:
                        val = row.get(col)
                        if val is None:
                            null_counts[col] += 1
                        else:
                            unique_counts[col].add(str(val))
                            
            unique_counts = {k: len(v) for k, v in unique_counts.items()}
            completeness = 1 - (sum(null_counts.values()) / (len(data) * len(columns))) if columns else 1.0
            
            return DataQualityProfile(
                profile_id=profile_id,
                asset_id=asset_id,
                profiled_at=datetime.utcnow(),
                row_count=len(data),
                column_count=len(columns),
                null_counts=null_counts,
                unique_counts=unique_counts,
                min_values={},
                max_values={},
                mean_values={},
                completeness_score=completeness,
                validity_score=0.95,
                uniqueness_score=0.8,
                consistency_score=0.9,
                overall_score=(completeness + 0.95 + 0.8 + 0.9) / 4
            )
        else:
            return self._profile_generic(profile_id, asset_id, data)
            
    def _profile_generic(self, profile_id: str, asset_id: str,
                        data: Any) -> DataQualityProfile:
        """Profile generic data."""
        return DataQualityProfile(
            profile_id=profile_id,
            asset_id=asset_id,
            profiled_at=datetime.utcnow(),
            row_count=1,
            column_count=1,
            null_counts={},
            unique_counts={},
            min_values={},
            max_values={},
            mean_values={},
            completeness_score=1.0,
            validity_score=1.0,
            uniqueness_score=1.0,
            consistency_score=1.0,
            overall_score=1.0
        )


class SchemaRegistry:
    """Registry for data schemas."""
    
    def __init__(self):
        self._schemas: Dict[str, Dict[str, Schema]] = {}  # name -> version -> schema
        self._lock = threading.Lock()
        
    def register_schema(self, schema: Schema) -> None:
        """Register a schema."""
        with self._lock:
            if schema.name not in self._schemas:
                self._schemas[schema.name] = {}
            self._schemas[schema.name][schema.version] = schema
            
    def get_schema(self, name: str, version: str = None) -> Optional[Schema]:
        """Get schema by name and version."""
        with self._lock:
            if name not in self._schemas:
                return None
            if version:
                return self._schemas[name].get(version)
            versions = sorted(self._schemas[name].keys())
            return self._schemas[name][versions[-1]] if versions else None
            
    def get_all_versions(self, name: str) -> List[Schema]:
        """Get all versions of a schema."""
        with self._lock:
            if name not in self._schemas:
                return []
            return list(self._schemas[name].values())
            
    def check_compatibility(self, schema1: Schema, schema2: Schema) -> Dict[str, Any]:
        """Check compatibility between two schemas."""
        fields1 = {f.name: f for f in schema1.fields}
        fields2 = {f.name: f for f in schema2.fields}
        
        added = set(fields2.keys()) - set(fields1.keys())
        removed = set(fields1.keys()) - set(fields2.keys())
        
        type_changes = []
        for name in set(fields1.keys()) & set(fields2.keys()):
            if fields1[name].data_type != fields2[name].data_type:
                type_changes.append({
                    'field': name,
                    'from': fields1[name].data_type,
                    'to': fields2[name].data_type
                })
                
        is_compatible = len(removed) == 0 and len(type_changes) == 0
        
        return {
            'compatible': is_compatible,
            'added_fields': list(added),
            'removed_fields': list(removed),
            'type_changes': type_changes
        }


class BusinessGlossary:
    """Business glossary for domain terms."""
    
    def __init__(self):
        self._terms: Dict[str, GlossaryTerm] = {}
        self._by_domain: Dict[DataDomain, List[str]] = {}
        self._lock = threading.Lock()
        
    def add_term(self, term: GlossaryTerm) -> None:
        """Add glossary term."""
        with self._lock:
            self._terms[term.term_id] = term
            if term.domain not in self._by_domain:
                self._by_domain[term.domain] = []
            self._by_domain[term.domain].append(term.term_id)
            
    def get_term(self, term_id: str) -> Optional[GlossaryTerm]:
        """Get term by ID."""
        with self._lock:
            return self._terms.get(term_id)
            
    def search_terms(self, query: str, domain: DataDomain = None) -> List[GlossaryTerm]:
        """Search glossary terms."""
        query_lower = query.lower()
        results = []
        
        with self._lock:
            for term in self._terms.values():
                if domain and term.domain != domain:
                    continue
                    
                if (query_lower in term.name.lower() or
                    query_lower in term.definition.lower() or
                    any(query_lower in s.lower() for s in term.synonyms)):
                    results.append(term)
                    
        return results
        
    def get_terms_by_domain(self, domain: DataDomain) -> List[GlossaryTerm]:
        """Get all terms in a domain."""
        with self._lock:
            term_ids = self._by_domain.get(domain, [])
            return [self._terms[tid] for tid in term_ids if tid in self._terms]


class DataCatalog:
    """Main data catalog service."""
    
    def __init__(self):
        self._assets: Dict[str, DataAsset] = {}
        self._tags: Dict[str, Tag] = {}
        self._lock = threading.Lock()
        
        self.search_index = SearchIndex()
        self.lineage_graph = LineageGraph()
        self.quality_profiler = DataQualityProfiler()
        self.schema_registry = SchemaRegistry()
        self.glossary = BusinessGlossary()
        
        self._setup_default_tags()
        self._setup_default_glossary()
        
    def _setup_default_tags(self) -> None:
        """Setup default tags."""
        default_tags = [
            Tag("pii", "PII", "compliance", "Contains personally identifiable information", "#ff0000"),
            Tag("sensitive", "Sensitive", "compliance", "Contains sensitive data", "#ff6600"),
            Tag("validated", "Validated", "quality", "Data has been validated", "#00ff00"),
            Tag("deprecated", "Deprecated", "lifecycle", "Asset is deprecated", "#999999"),
            Tag("production", "Production", "environment", "Production data", "#0066ff"),
            Tag("staging", "Staging", "environment", "Staging data", "#ffcc00"),
        ]
        for tag in default_tags:
            self._tags[tag.tag_id] = tag
            
    def _setup_default_glossary(self) -> None:
        """Setup default glossary terms."""
        terms = [
            GlossaryTerm(
                "assay", "Assay", 
                "Chemical analysis to determine the composition of a sample",
                DataDomain.GEOCHEMISTRY,
                synonyms=["chemical analysis", "sample analysis"]
            ),
            GlossaryTerm(
                "drillhole", "Drillhole",
                "A hole drilled into the earth for exploration or extraction",
                DataDomain.DRILLING,
                synonyms=["borehole", "drill hole", "well"]
            ),
            GlossaryTerm(
                "anomaly", "Anomaly",
                "A deviation from expected values in geophysical or geochemical data",
                DataDomain.GEOPHYSICS,
                synonyms=["deviation", "outlier"]
            ),
            GlossaryTerm(
                "prospectivity", "Prospectivity",
                "The likelihood of an area containing economic mineral deposits",
                DataDomain.GEOLOGY,
                synonyms=["mineral potential", "favorability"]
            ),
        ]
        for term in terms:
            self.glossary.add_term(term)
            
    def register_asset(self, asset: DataAsset) -> str:
        """Register a new asset."""
        with self._lock:
            if not asset.asset_id:
                asset.asset_id = hashlib.md5(
                    f"{asset.name}:{asset.location}:{datetime.utcnow().isoformat()}".encode()
                ).hexdigest()[:16]
                
            self._assets[asset.asset_id] = asset
            
        self.search_index.index_asset(asset)
        
        if asset.schema:
            self.schema_registry.register_schema(asset.schema)
            
        logger.info(f"Asset registered: {asset.asset_id} - {asset.name}")
        return asset.asset_id
        
    def update_asset(self, asset_id: str, updates: Dict[str, Any]) -> Optional[DataAsset]:
        """Update an asset."""
        with self._lock:
            asset = self._assets.get(asset_id)
            if not asset:
                return None
                
            for key, value in updates.items():
                if hasattr(asset, key):
                    setattr(asset, key, value)
                    
            asset.updated_at = datetime.utcnow()
            
        self.search_index.remove_asset(asset_id)
        self.search_index.index_asset(asset)
        
        return asset
        
    def delete_asset(self, asset_id: str) -> bool:
        """Delete an asset."""
        with self._lock:
            if asset_id not in self._assets:
                return False
            del self._assets[asset_id]
            
        self.search_index.remove_asset(asset_id)
        return True
        
    def get_asset(self, asset_id: str) -> Optional[DataAsset]:
        """Get asset by ID."""
        with self._lock:
            asset = self._assets.get(asset_id)
            if asset:
                asset.last_accessed_at = datetime.utcnow()
            return asset
            
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for assets."""
        text_results = {}
        if query.text:
            text_results = dict(self.search_index.search(query.text, query.limit * 2))
            
        results = []
        
        with self._lock:
            for asset_id, asset in self._assets.items():
                if query.asset_types and asset.asset_type not in query.asset_types:
                    continue
                if query.domains and asset.domain not in query.domains:
                    continue
                if query.quality_levels and asset.quality_level not in query.quality_levels:
                    continue
                if query.tags and not any(t in asset.tags for t in query.tags):
                    continue
                if query.owner and asset.owner != query.owner:
                    continue
                if query.tenant_id and asset.tenant_id != query.tenant_id:
                    continue
                if query.project_id and asset.project_id != query.project_id:
                    continue
                    
                if query.spatial_extent and asset.spatial_extent:
                    if not query.spatial_extent.intersects(asset.spatial_extent):
                        continue
                        
                if query.min_quality_score > 0 and asset.quality_profile:
                    if asset.quality_profile.overall_score < query.min_quality_score:
                        continue
                        
                score = text_results.get(asset_id, 1.0 if not query.text else 0.0)
                if score > 0 or not query.text:
                    results.append(SearchResult(asset=asset, score=score))
                    
        results.sort(key=lambda x: -x.score)
        return results[query.offset:query.offset + query.limit]
        
    def add_lineage(self, source_id: str, target_id: str,
                   relationship: LineageType,
                   transformation: str = "") -> str:
        """Add lineage relationship."""
        edge_id = hashlib.md5(
            f"{source_id}:{target_id}:{relationship.value}".encode()
        ).hexdigest()[:16]
        
        edge = LineageEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            transformation=transformation
        )
        
        self.lineage_graph.add_edge(edge)
        return edge_id
        
    def get_lineage(self, asset_id: str,
                   upstream_depth: int = 3,
                   downstream_depth: int = 3) -> Dict[str, Any]:
        """Get lineage for an asset."""
        subgraph = self.lineage_graph.get_lineage_subgraph(
            asset_id, upstream_depth, downstream_depth
        )
        
        nodes_with_details = []
        for node_id in subgraph['nodes']:
            asset = self.get_asset(node_id)
            if asset:
                nodes_with_details.append({
                    'id': node_id,
                    'name': asset.name,
                    'type': asset.asset_type.value,
                    'domain': asset.domain.value
                })
            else:
                nodes_with_details.append({'id': node_id, 'name': 'Unknown'})
                
        subgraph['nodes'] = nodes_with_details
        return subgraph
        
    def profile_asset(self, asset_id: str, data: Any) -> DataQualityProfile:
        """Profile data quality for an asset."""
        profile = self.quality_profiler.profile_dataset(asset_id, data)
        
        with self._lock:
            asset = self._assets.get(asset_id)
            if asset:
                asset.quality_profile = profile
                
        return profile
        
    def get_catalog_stats(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        with self._lock:
            by_type = {}
            by_domain = {}
            by_quality = {}
            total_size = 0
            
            for asset in self._assets.values():
                by_type[asset.asset_type.value] = by_type.get(asset.asset_type.value, 0) + 1
                by_domain[asset.domain.value] = by_domain.get(asset.domain.value, 0) + 1
                by_quality[asset.quality_level.value] = by_quality.get(asset.quality_level.value, 0) + 1
                total_size += asset.size_bytes
                
            return {
                'total_assets': len(self._assets),
                'by_type': by_type,
                'by_domain': by_domain,
                'by_quality': by_quality,
                'total_size_bytes': total_size,
                'total_schemas': len(self.schema_registry._schemas),
                'total_glossary_terms': len(self.glossary._terms)
            }


def create_data_catalog() -> DataCatalog:
    """Factory function to create data catalog."""
    return DataCatalog()


def create_schema_registry() -> SchemaRegistry:
    """Factory function to create schema registry."""
    return SchemaRegistry()


def create_business_glossary() -> BusinessGlossary:
    """Factory function to create business glossary."""
    return BusinessGlossary()
