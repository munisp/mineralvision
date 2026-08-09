"""
OpenSearch Integration
=======================

Production-grade search and analytics integration for MineralVision:
- Full-text search for geological data
- Log aggregation and analysis
- Real-time analytics dashboards
- Anomaly detection
- Geospatial search capabilities
- Index lifecycle management

OpenSearch provides distributed search and analytics
with powerful visualization capabilities.
"""

import asyncio
import json
import logging
import uuid
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import re

logger = logging.getLogger(__name__)

try:
    from opensearchpy import OpenSearch, AsyncOpenSearch
    from opensearchpy.helpers import bulk, async_bulk
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False

from .._mock_fallback import real_client_unavailable


class IndexType(Enum):
    """Types of indices."""
    LOGS = "logs"
    METRICS = "metrics"
    TRACES = "traces"
    GEOLOGICAL = "geological"
    SENSOR = "sensor"
    ALERTS = "alerts"


class AggregationType(Enum):
    """Types of aggregations."""
    TERMS = "terms"
    DATE_HISTOGRAM = "date_histogram"
    AVG = "avg"
    SUM = "sum"
    MIN = "min"
    MAX = "max"
    PERCENTILES = "percentiles"
    CARDINALITY = "cardinality"
    GEO_BOUNDS = "geo_bounds"
    GEO_CENTROID = "geo_centroid"


@dataclass
class OpenSearchConfig:
    """OpenSearch configuration."""
    hosts: List[str] = field(default_factory=lambda: ["localhost:9200"])
    username: str = "admin"
    password: str = "admin"
    use_ssl: bool = True
    verify_certs: bool = False
    ssl_show_warn: bool = False
    index_prefix: str = "mineralvision"
    default_shards: int = 1
    default_replicas: int = 1


@dataclass
class SearchQuery:
    """Search query configuration."""
    query: Dict[str, Any]
    index: str
    size: int = 10
    from_: int = 0
    sort: List[Dict[str, Any]] = field(default_factory=list)
    aggregations: Dict[str, Any] = field(default_factory=dict)
    highlight: Dict[str, Any] = field(default_factory=dict)
    source: Union[bool, List[str]] = True


@dataclass
class SearchResult:
    """Search result."""
    total: int
    hits: List[Dict[str, Any]]
    aggregations: Dict[str, Any] = field(default_factory=dict)
    took_ms: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total': self.total,
            'hits': self.hits,
            'aggregations': self.aggregations,
            'took_ms': self.took_ms
        }


@dataclass
class IndexMapping:
    """Index mapping configuration."""
    properties: Dict[str, Dict[str, Any]]
    dynamic: str = "strict"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mappings': {
                'dynamic': self.dynamic,
                'properties': self.properties
            }
        }


class MockOpenSearchClient:
    """Mock OpenSearch client."""
    
    def __init__(self, config: OpenSearchConfig):
        self.config = config
        self._indices: Dict[str, Dict[str, Any]] = {}
        self._documents: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._aliases: Dict[str, str] = {}
    
    async def create_index(self, index: str, body: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create an index."""
        self._indices[index] = body or {}
        self._documents[index] = {}
        return {'acknowledged': True, 'index': index}
    
    async def delete_index(self, index: str) -> Dict[str, Any]:
        """Delete an index."""
        if index in self._indices:
            del self._indices[index]
            del self._documents[index]
            return {'acknowledged': True}
        return {'acknowledged': False}
    
    async def index_exists(self, index: str) -> bool:
        """Check if index exists."""
        return index in self._indices
    
    async def index_document(self, index: str, body: Dict[str, Any],
                            doc_id: str = None) -> Dict[str, Any]:
        """Index a document."""
        doc_id = doc_id or str(uuid.uuid4())
        
        if index not in self._documents:
            self._documents[index] = {}
        
        self._documents[index][doc_id] = {
            '_id': doc_id,
            '_source': body,
            '_index': index,
            '_version': 1
        }
        
        return {
            '_id': doc_id,
            '_index': index,
            'result': 'created',
            '_version': 1
        }
    
    async def get_document(self, index: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        if index in self._documents and doc_id in self._documents[index]:
            return self._documents[index][doc_id]
        return None
    
    async def delete_document(self, index: str, doc_id: str) -> Dict[str, Any]:
        """Delete a document."""
        if index in self._documents and doc_id in self._documents[index]:
            del self._documents[index][doc_id]
            return {'result': 'deleted'}
        return {'result': 'not_found'}
    
    async def search(self, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Search documents."""
        if index not in self._documents:
            return {'hits': {'total': {'value': 0}, 'hits': []}, 'took': 1}
        
        docs = list(self._documents[index].values())
        query = body.get('query', {})
        size = body.get('size', 10)
        from_ = body.get('from', 0)
        
        # Simple query matching
        if 'match_all' in query:
            results = docs
        elif 'match' in query:
            field = list(query['match'].keys())[0]
            value = query['match'][field]
            if isinstance(value, dict):
                value = value.get('query', '')
            results = [
                d for d in docs
                if str(value).lower() in str(d['_source'].get(field, '')).lower()
            ]
        elif 'term' in query:
            field = list(query['term'].keys())[0]
            value = query['term'][field]
            if isinstance(value, dict):
                value = value.get('value', '')
            results = [
                d for d in docs
                if d['_source'].get(field) == value
            ]
        elif 'bool' in query:
            results = docs  # Simplified
        else:
            results = docs
        
        total = len(results)
        hits = results[from_:from_ + size]
        
        response = {
            'took': 5,
            'hits': {
                'total': {'value': total, 'relation': 'eq'},
                'hits': [
                    {
                        '_id': h['_id'],
                        '_index': h['_index'],
                        '_source': h['_source'],
                        '_score': 1.0
                    }
                    for h in hits
                ]
            }
        }
        
        # Handle aggregations
        if 'aggs' in body or 'aggregations' in body:
            aggs = body.get('aggs', body.get('aggregations', {}))
            response['aggregations'] = self._compute_aggregations(docs, aggs)
        
        return response
    
    def _compute_aggregations(self, docs: List[Dict[str, Any]],
                             aggs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute aggregations."""
        results = {}
        
        for agg_name, agg_config in aggs.items():
            if 'terms' in agg_config:
                field = agg_config['terms']['field']
                buckets = {}
                for doc in docs:
                    value = doc['_source'].get(field)
                    if value:
                        buckets[value] = buckets.get(value, 0) + 1
                
                results[agg_name] = {
                    'buckets': [
                        {'key': k, 'doc_count': v}
                        for k, v in sorted(buckets.items(), key=lambda x: -x[1])[:10]
                    ]
                }
            
            elif 'avg' in agg_config:
                field = agg_config['avg']['field']
                values = [doc['_source'].get(field, 0) for doc in docs if field in doc['_source']]
                results[agg_name] = {'value': sum(values) / len(values) if values else 0}
            
            elif 'sum' in agg_config:
                field = agg_config['sum']['field']
                values = [doc['_source'].get(field, 0) for doc in docs if field in doc['_source']]
                results[agg_name] = {'value': sum(values)}
            
            elif 'date_histogram' in agg_config:
                results[agg_name] = {'buckets': []}
        
        return results
    
    async def bulk_index(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk index documents."""
        success = 0
        errors = []
        
        for action in actions:
            if '_index' in action and '_source' in action:
                await self.index_document(
                    action['_index'],
                    action['_source'],
                    action.get('_id')
                )
                success += 1
        
        return {'took': 10, 'errors': len(errors) > 0, 'items': success}
    
    async def update_alias(self, index: str, alias: str) -> Dict[str, Any]:
        """Update index alias."""
        self._aliases[alias] = index
        return {'acknowledged': True}
    
    async def get_cluster_health(self) -> Dict[str, Any]:
        """Get cluster health."""
        return {
            'cluster_name': 'mineralvision',
            'status': 'green',
            'number_of_nodes': 1,
            'number_of_data_nodes': 1,
            'active_primary_shards': len(self._indices),
            'active_shards': len(self._indices)
        }
    
    async def close(self) -> None:
        """Close the client."""
        pass


class IndexManager:
    """
    Index management for OpenSearch.
    
    Provides:
    - Index creation with mappings
    - Index lifecycle management
    - Alias management
    - Template management
    """
    
    def __init__(self, client: MockOpenSearchClient, config: OpenSearchConfig):
        self.client = client
        self.config = config
    
    async def create_index(self, name: str, mapping: IndexMapping = None,
                          settings: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create an index with mapping."""
        index_name = f"{self.config.index_prefix}-{name}"
        
        body = {
            'settings': settings or {
                'number_of_shards': self.config.default_shards,
                'number_of_replicas': self.config.default_replicas
            }
        }
        
        if mapping:
            body.update(mapping.to_dict())
        
        return await self.client.create_index(index_name, body)
    
    async def delete_index(self, name: str) -> Dict[str, Any]:
        """Delete an index."""
        index_name = f"{self.config.index_prefix}-{name}"
        return await self.client.delete_index(index_name)
    
    async def index_exists(self, name: str) -> bool:
        """Check if index exists."""
        index_name = f"{self.config.index_prefix}-{name}"
        return await self.client.index_exists(index_name)
    
    async def create_geological_index(self) -> Dict[str, Any]:
        """Create index for geological data."""
        mapping = IndexMapping(
            properties={
                'sample_id': {'type': 'keyword'},
                'project_id': {'type': 'keyword'},
                'location': {'type': 'geo_point'},
                'depth': {'type': 'float'},
                'rock_type': {'type': 'keyword'},
                'mineral_content': {'type': 'object'},
                'assay_results': {'type': 'nested'},
                'collected_at': {'type': 'date'},
                'description': {'type': 'text'},
                'tags': {'type': 'keyword'}
            },
            dynamic='false'
        )
        
        return await self.create_index('geological', mapping)
    
    async def create_sensor_index(self) -> Dict[str, Any]:
        """Create index for sensor data."""
        mapping = IndexMapping(
            properties={
                'sensor_id': {'type': 'keyword'},
                'sensor_type': {'type': 'keyword'},
                'location': {'type': 'geo_point'},
                'timestamp': {'type': 'date'},
                'value': {'type': 'float'},
                'unit': {'type': 'keyword'},
                'quality': {'type': 'float'},
                'metadata': {'type': 'object'}
            },
            dynamic='false'
        )
        
        return await self.create_index('sensor', mapping)
    
    async def create_logs_index(self) -> Dict[str, Any]:
        """Create index for logs."""
        mapping = IndexMapping(
            properties={
                'timestamp': {'type': 'date'},
                'level': {'type': 'keyword'},
                'logger': {'type': 'keyword'},
                'message': {'type': 'text'},
                'service': {'type': 'keyword'},
                'trace_id': {'type': 'keyword'},
                'span_id': {'type': 'keyword'},
                'exception': {'type': 'text'},
                'metadata': {'type': 'object'}
            },
            dynamic='true'
        )
        
        return await self.create_index('logs', mapping)
    
    async def create_alerts_index(self) -> Dict[str, Any]:
        """Create index for alerts."""
        mapping = IndexMapping(
            properties={
                'alert_id': {'type': 'keyword'},
                'timestamp': {'type': 'date'},
                'severity': {'type': 'keyword'},
                'source': {'type': 'keyword'},
                'title': {'type': 'text'},
                'description': {'type': 'text'},
                'status': {'type': 'keyword'},
                'acknowledged_by': {'type': 'keyword'},
                'resolved_at': {'type': 'date'},
                'metadata': {'type': 'object'}
            },
            dynamic='false'
        )
        
        return await self.create_index('alerts', mapping)


class SearchEngine:
    """
    Search engine for OpenSearch.
    
    Provides:
    - Full-text search
    - Geospatial queries
    - Aggregations
    - Highlighting
    """
    
    def __init__(self, client: MockOpenSearchClient, config: OpenSearchConfig):
        self.client = client
        self.config = config
    
    async def search(self, query: SearchQuery) -> SearchResult:
        """Execute a search query."""
        index_name = f"{self.config.index_prefix}-{query.index}"
        
        body = {
            'query': query.query,
            'size': query.size,
            'from': query.from_
        }
        
        if query.sort:
            body['sort'] = query.sort
        if query.aggregations:
            body['aggs'] = query.aggregations
        if query.highlight:
            body['highlight'] = query.highlight
        if query.source is not True:
            body['_source'] = query.source
        
        response = await self.client.search(index_name, body)
        
        return SearchResult(
            total=response['hits']['total']['value'],
            hits=[hit['_source'] for hit in response['hits']['hits']],
            aggregations=response.get('aggregations', {}),
            took_ms=response['took']
        )
    
    async def full_text_search(self, index: str, query_text: str,
                              fields: List[str] = None,
                              size: int = 10) -> SearchResult:
        """Perform full-text search."""
        if fields:
            query = {
                'multi_match': {
                    'query': query_text,
                    'fields': fields
                }
            }
        else:
            query = {
                'query_string': {
                    'query': query_text
                }
            }
        
        return await self.search(SearchQuery(
            query=query,
            index=index,
            size=size,
            highlight={'fields': {'*': {}}}
        ))
    
    async def geo_search(self, index: str, lat: float, lon: float,
                        distance: str = "10km",
                        size: int = 10) -> SearchResult:
        """Search by geographic location."""
        query = {
            'bool': {
                'filter': {
                    'geo_distance': {
                        'distance': distance,
                        'location': {
                            'lat': lat,
                            'lon': lon
                        }
                    }
                }
            }
        }
        
        return await self.search(SearchQuery(
            query=query,
            index=index,
            size=size,
            sort=[{
                '_geo_distance': {
                    'location': {'lat': lat, 'lon': lon},
                    'order': 'asc',
                    'unit': 'km'
                }
            }]
        ))
    
    async def date_range_search(self, index: str, field: str,
                               start: datetime, end: datetime,
                               size: int = 10) -> SearchResult:
        """Search by date range."""
        query = {
            'range': {
                field: {
                    'gte': start.isoformat(),
                    'lte': end.isoformat()
                }
            }
        }
        
        return await self.search(SearchQuery(
            query=query,
            index=index,
            size=size,
            sort=[{field: {'order': 'desc'}}]
        ))
    
    async def aggregate(self, index: str, aggregations: Dict[str, Any],
                       query: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run aggregations."""
        result = await self.search(SearchQuery(
            query=query or {'match_all': {}},
            index=index,
            size=0,
            aggregations=aggregations
        ))
        
        return result.aggregations


class DocumentManager:
    """
    Document management for OpenSearch.
    
    Provides:
    - Document indexing
    - Bulk operations
    - Document updates
    - Document deletion
    """
    
    def __init__(self, client: MockOpenSearchClient, config: OpenSearchConfig):
        self.client = client
        self.config = config
    
    async def index(self, index: str, document: Dict[str, Any],
                   doc_id: str = None) -> Dict[str, Any]:
        """Index a document."""
        index_name = f"{self.config.index_prefix}-{index}"
        return await self.client.index_document(index_name, document, doc_id)
    
    async def get(self, index: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get a document by ID."""
        index_name = f"{self.config.index_prefix}-{index}"
        doc = await self.client.get_document(index_name, doc_id)
        return doc['_source'] if doc else None
    
    async def delete(self, index: str, doc_id: str) -> Dict[str, Any]:
        """Delete a document."""
        index_name = f"{self.config.index_prefix}-{index}"
        return await self.client.delete_document(index_name, doc_id)
    
    async def bulk_index(self, index: str, 
                        documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Bulk index documents."""
        index_name = f"{self.config.index_prefix}-{index}"
        
        actions = [
            {
                '_index': index_name,
                '_source': doc,
                '_id': doc.get('id', str(uuid.uuid4()))
            }
            for doc in documents
        ]
        
        return await self.client.bulk_index(actions)
    
    async def index_geological_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Index a geological sample."""
        return await self.index('geological', sample, sample.get('sample_id'))
    
    async def index_sensor_reading(self, reading: Dict[str, Any]) -> Dict[str, Any]:
        """Index a sensor reading."""
        return await self.index('sensor', reading)
    
    async def index_log(self, log_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Index a log entry."""
        if 'timestamp' not in log_entry:
            log_entry['timestamp'] = datetime.now().isoformat()
        return await self.index('logs', log_entry)
    
    async def index_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Index an alert."""
        if 'alert_id' not in alert:
            alert['alert_id'] = str(uuid.uuid4())
        if 'timestamp' not in alert:
            alert['timestamp'] = datetime.now().isoformat()
        return await self.index('alerts', alert, alert['alert_id'])


class OpenSearchIntegration:
    """
    OpenSearch integration for MineralVision.
    
    Provides comprehensive search and analytics:
    - Index management
    - Full-text search
    - Geospatial search
    - Aggregations
    - Document management
    
    Example:
        opensearch = OpenSearchIntegration()
        await opensearch.connect()
        
        # Create indices
        await opensearch.indices.create_geological_index()
        
        # Index a document
        await opensearch.documents.index_geological_sample({
            'sample_id': 'S001',
            'rock_type': 'granite',
            'location': {'lat': 45.0, 'lon': -122.0}
        })
        
        # Search
        results = await opensearch.search.full_text_search(
            'geological', 'granite'
        )
    """
    
    def __init__(self, config: OpenSearchConfig = None):
        self.config = config or OpenSearchConfig()
        self.client: Optional[MockOpenSearchClient] = None
        self.indices: Optional[IndexManager] = None
        self.search: Optional[SearchEngine] = None
        self.documents: Optional[DocumentManager] = None
        self._connected = False
        self._degraded = False

    @property
    def degraded(self) -> bool:
        """True when running on the explicit in-memory mock fallback."""
        return self._degraded

    async def connect(self) -> 'OpenSearchIntegration':
        """
        Connect to OpenSearch (real client first).

        Falls back to the in-memory mock ONLY when
        MV_ALLOW_MOCK_FALLBACK=true; otherwise raises RuntimeError.
        """
        if OPENSEARCH_AVAILABLE:
            try:
                self.client = AsyncOpenSearch(
                    hosts=self.config.hosts,
                    http_auth=(self.config.username, self.config.password),
                    use_ssl=self.config.use_ssl,
                    verify_certs=self.config.verify_certs,
                    ssl_show_warn=self.config.ssl_show_warn,
                    timeout=5
                )
                # Test connection
                await self.client.cluster.health()
                logger.info(f"Connected to OpenSearch at {self.config.hosts}")
            except Exception as e:
                if real_client_unavailable("OpenSearch", "connection failed", e):
                    self._degraded = True
                    self.client = MockOpenSearchClient(self.config)
        else:
            if real_client_unavailable("OpenSearch", "opensearch-py package not installed"):
                self._degraded = True
                self.client = MockOpenSearchClient(self.config)

        self.indices = IndexManager(self.client, self.config)
        self.search = SearchEngine(self.client, self.config)
        self.documents = DocumentManager(self.client, self.config)
        
        self._connected = True
        return self
    
    async def setup_indices(self) -> Dict[str, Any]:
        """Setup all MineralVision indices."""
        results = {}
        
        results['geological'] = await self.indices.create_geological_index()
        results['sensor'] = await self.indices.create_sensor_index()
        results['logs'] = await self.indices.create_logs_index()
        results['alerts'] = await self.indices.create_alerts_index()
        
        return results
    
    async def health_check(self) -> Dict[str, Any]:
        """Check OpenSearch health."""
        if isinstance(self.client, MockOpenSearchClient):
            health = await self.client.get_cluster_health()
            health['degraded'] = True
            return health
        health = await self.client.cluster.health()
        health['degraded'] = self._degraded
        return health
    
    async def close(self) -> None:
        """Close the connection."""
        if self.client:
            await self.client.close()
        self._connected = False
    
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected


# Factory functions

def create_opensearch(config: OpenSearchConfig = None) -> OpenSearchIntegration:
    """Create an OpenSearch integration instance."""
    return OpenSearchIntegration(config)


async def create_and_connect_opensearch(config: OpenSearchConfig = None) -> OpenSearchIntegration:
    """Create and connect OpenSearch."""
    opensearch = OpenSearchIntegration(config)
    await opensearch.connect()
    return opensearch
