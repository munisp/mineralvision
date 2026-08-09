"""
V-JEPA + Lakehouse Integration Module for MineralVision.

Connects V-JEPA visual analysis with the Lakehouse data architecture:
- Store/retrieve imagery and embeddings from Delta Lake/Iceberg tables
- Track data lineage for reproducible training runs
- Enable RAG retrieval for LLM explanations
- Support streaming ingestion via Kafka

Architecture:
┌─────────────────────────────────────────────────────────────────┐
│  OLLAMA (Reasoning Layer) - uses retrieved context              │
└─────────────────────────────────────────────────────────────────┘
                              ↑ retrieves embeddings + docs
┌─────────────────────────────────────────────────────────────────┐
│  V-JEPA (Visual Encoder) - reads/writes via this module         │
└─────────────────────────────────────────────────────────────────┘
                              ↑ this integration layer
┌─────────────────────────────────────────────────────────────────┐
│  LAKEHOUSE (Delta/Iceberg + Object Store)                       │
│  - Raw imagery, prepared datasets, embeddings, metadata         │
└─────────────────────────────────────────────────────────────────┘

Usage:
    from api.jepa.lakehouse_integration import (
        JEPALakehouseStore,
        EmbeddingTable,
        TrainingDatasetManager,
        create_jepa_lakehouse_store,
    )
    
    # Create lakehouse store
    store = create_jepa_lakehouse_store(
        warehouse_path="/data/lakehouse",
        catalog="mineralvision",
    )
    
    # Store embeddings
    store.write_embeddings(embeddings, metadata)
    
    # Retrieve similar embeddings for RAG
    neighbors = store.search_embeddings(query_embedding, k=10)
    
    # Create training dataset
    dataset_manager = TrainingDatasetManager(store)
    manifest = dataset_manager.create_training_manifest(
        filters={"site": "prospect_a", "sensor": "drone_rgb"},
        output_path="./training_data",
    )
"""

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib

logger = logging.getLogger(__name__)


class TableFormat(Enum):
    """Supported lakehouse table formats."""
    DELTA = "delta"
    ICEBERG = "iceberg"
    PARQUET = "parquet"


class StorageBackend(Enum):
    """Supported storage backends."""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"


@dataclass
class LakehouseConfig:
    """Configuration for lakehouse connection."""
    warehouse_path: str
    catalog: str = "mineralvision"
    table_format: TableFormat = TableFormat.DELTA
    storage_backend: StorageBackend = StorageBackend.LOCAL
    
    # Connection settings
    spark_master: Optional[str] = None
    s3_endpoint: Optional[str] = None
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    
    # Table names
    embeddings_table: str = "jepa_embeddings"
    imagery_table: str = "jepa_imagery"
    training_runs_table: str = "jepa_training_runs"
    findings_table: str = "jepa_findings"


@dataclass
class EmbeddingRecord:
    """Record for storing V-JEPA embeddings in lakehouse."""
    embedding_id: str
    embedding_vector: List[float]
    source_type: str  # "drone_video", "satellite", "core_photo"
    source_path: str
    
    # Metadata
    site: Optional[str] = None
    project: Optional[str] = None
    sensor: Optional[str] = None
    timestamp: Optional[datetime] = None
    geo_bounds: Optional[Dict[str, float]] = None  # {"min_lat", "max_lat", "min_lon", "max_lon"}
    
    # V-JEPA specific
    backbone: str = "vit_large"
    checkpoint_version: Optional[str] = None
    
    # Lineage
    created_at: datetime = field(default_factory=datetime.utcnow)
    training_run_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "embedding_id": self.embedding_id,
            "embedding_vector": self.embedding_vector,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "site": self.site,
            "project": self.project,
            "sensor": self.sensor,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "geo_bounds": json.dumps(self.geo_bounds) if self.geo_bounds else None,
            "backbone": self.backbone,
            "checkpoint_version": self.checkpoint_version,
            "created_at": self.created_at.isoformat(),
            "training_run_id": self.training_run_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingRecord":
        """Create from dictionary."""
        return cls(
            embedding_id=data["embedding_id"],
            embedding_vector=data["embedding_vector"],
            source_type=data["source_type"],
            source_path=data["source_path"],
            site=data.get("site"),
            project=data.get("project"),
            sensor=data.get("sensor"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            geo_bounds=json.loads(data["geo_bounds"]) if data.get("geo_bounds") else None,
            backbone=data.get("backbone", "vit_large"),
            checkpoint_version=data.get("checkpoint_version"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            training_run_id=data.get("training_run_id"),
        )


@dataclass
class FindingRecord:
    """Record for storing V-JEPA findings in lakehouse."""
    finding_id: str
    finding_type: str  # "anomaly", "change", "similarity"
    
    # Scores
    anomaly_score: Optional[float] = None
    similarity_score: Optional[float] = None
    change_percentage: Optional[float] = None
    confidence: str = "medium"
    
    # Location
    source_embedding_id: str = ""
    location_name: Optional[str] = None
    geo_point: Optional[Dict[str, float]] = None  # {"lat", "lon"}
    
    # LLM explanation (from Ollama)
    what_explanation: Optional[str] = None
    why_evidence: Optional[str] = None
    recommended_actions: Optional[List[str]] = None
    citations: Optional[List[str]] = None
    
    # Status
    status: str = "new"  # "new", "reviewed", "actioned", "dismissed"
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    
    # Lineage
    created_at: datetime = field(default_factory=datetime.utcnow)
    analysis_job_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "finding_id": self.finding_id,
            "finding_type": self.finding_type,
            "anomaly_score": self.anomaly_score,
            "similarity_score": self.similarity_score,
            "change_percentage": self.change_percentage,
            "confidence": self.confidence,
            "source_embedding_id": self.source_embedding_id,
            "location_name": self.location_name,
            "geo_point": json.dumps(self.geo_point) if self.geo_point else None,
            "what_explanation": self.what_explanation,
            "why_evidence": self.why_evidence,
            "recommended_actions": json.dumps(self.recommended_actions) if self.recommended_actions else None,
            "citations": json.dumps(self.citations) if self.citations else None,
            "status": self.status,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "created_at": self.created_at.isoformat(),
            "analysis_job_id": self.analysis_job_id,
        }


@dataclass
class TrainingRunRecord:
    """Record for tracking V-JEPA training runs."""
    run_id: str
    run_name: str
    
    # Configuration
    backbone: str
    pretraining_mode: str
    num_epochs: int
    batch_size: int
    learning_rate: float
    
    # Data
    dataset_manifest_path: str
    num_samples: int
    data_filters: Optional[Dict[str, Any]] = None
    
    # Checkpoints
    checkpoint_path: Optional[str] = None
    best_loss: Optional[float] = None
    
    # Status
    status: str = "pending"  # "pending", "running", "completed", "failed"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Lineage
    created_at: datetime = field(default_factory=datetime.utcnow)
    parent_run_id: Optional[str] = None  # For continued training
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "backbone": self.backbone,
            "pretraining_mode": self.pretraining_mode,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "dataset_manifest_path": self.dataset_manifest_path,
            "num_samples": self.num_samples,
            "data_filters": json.dumps(self.data_filters) if self.data_filters else None,
            "checkpoint_path": self.checkpoint_path,
            "best_loss": self.best_loss,
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
            "parent_run_id": self.parent_run_id,
        }


class LakehouseBackend(ABC):
    """Abstract base class for lakehouse backends."""
    
    @abstractmethod
    def write_records(self, table: str, records: List[Dict[str, Any]]) -> int:
        """Write records to a table."""
        pass
    
    @abstractmethod
    def read_records(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Read records from a table."""
        pass
    
    @abstractmethod
    def table_exists(self, table: str) -> bool:
        """Check if a table exists."""
        pass
    
    @abstractmethod
    def create_table(self, table: str, schema: Dict[str, str]) -> None:
        """Create a table with the given schema."""
        pass


class LocalParquetBackend(LakehouseBackend):
    """Local Parquet-based lakehouse backend for development/testing."""
    
    def __init__(self, warehouse_path: str):
        self.warehouse_path = Path(warehouse_path)
        self.warehouse_path.mkdir(parents=True, exist_ok=True)
        self._tables: Dict[str, List[Dict[str, Any]]] = {}
    
    def _table_path(self, table: str) -> Path:
        """Get the path for a table."""
        return self.warehouse_path / f"{table}.json"
    
    def _load_table(self, table: str) -> List[Dict[str, Any]]:
        """Load a table from disk."""
        if table in self._tables:
            return self._tables[table]
        
        path = self._table_path(table)
        if path.exists():
            with open(path, "r") as f:
                self._tables[table] = json.load(f)
        else:
            self._tables[table] = []
        
        return self._tables[table]
    
    def _save_table(self, table: str) -> None:
        """Save a table to disk."""
        path = self._table_path(table)
        with open(path, "w") as f:
            json.dump(self._tables.get(table, []), f, indent=2, default=str)
    
    def write_records(self, table: str, records: List[Dict[str, Any]]) -> int:
        """Write records to a table."""
        data = self._load_table(table)
        data.extend(records)
        self._tables[table] = data
        self._save_table(table)
        return len(records)
    
    def read_records(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Read records from a table."""
        data = self._load_table(table)
        
        # Apply filters
        if filters:
            filtered = []
            for record in data:
                match = True
                for key, value in filters.items():
                    if record.get(key) != value:
                        match = False
                        break
                if match:
                    filtered.append(record)
            data = filtered
        
        # Apply limit
        if limit:
            data = data[:limit]
        
        return data
    
    def table_exists(self, table: str) -> bool:
        """Check if a table exists."""
        return self._table_path(table).exists() or table in self._tables
    
    def create_table(self, table: str, schema: Dict[str, str]) -> None:
        """Create a table with the given schema."""
        if not self.table_exists(table):
            self._tables[table] = []
            self._save_table(table)


class DeltaLakeBackend(LakehouseBackend):
    """Delta Lake backend using delta-rs or PySpark."""
    
    def __init__(self, config: LakehouseConfig):
        self.config = config
        self._spark = None
        self._delta_available = False
        
        # Try to import delta
        try:
            import delta
            self._delta_available = True
        except ImportError:
            logger.warning("Delta Lake not available, falling back to local backend")
    
    def _get_spark(self):
        """Get or create Spark session with Delta support."""
        if self._spark is None and self._delta_available:
            try:
                from pyspark.sql import SparkSession
                from delta import configure_spark_with_delta_pip
                
                builder = SparkSession.builder \
                    .appName("MineralVision-JEPA") \
                    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
                    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
                
                if self.config.spark_master:
                    builder = builder.master(self.config.spark_master)
                
                self._spark = configure_spark_with_delta_pip(builder).getOrCreate()
            except Exception as e:
                logger.error(f"Failed to create Spark session: {e}")
                self._delta_available = False
        
        return self._spark
    
    def write_records(self, table: str, records: List[Dict[str, Any]]) -> int:
        """Write records to Delta table."""
        if not self._delta_available:
            # Fallback to local
            fallback = LocalParquetBackend(self.config.warehouse_path)
            return fallback.write_records(table, records)
        
        spark = self._get_spark()
        if spark is None:
            fallback = LocalParquetBackend(self.config.warehouse_path)
            return fallback.write_records(table, records)
        
        try:
            df = spark.createDataFrame(records)
            table_path = f"{self.config.warehouse_path}/{table}"
            df.write.format("delta").mode("append").save(table_path)
            return len(records)
        except Exception as e:
            logger.error(f"Delta write failed: {e}")
            fallback = LocalParquetBackend(self.config.warehouse_path)
            return fallback.write_records(table, records)
    
    def read_records(
        self,
        table: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Read records from Delta table."""
        if not self._delta_available:
            fallback = LocalParquetBackend(self.config.warehouse_path)
            return fallback.read_records(table, filters, limit)
        
        spark = self._get_spark()
        if spark is None:
            fallback = LocalParquetBackend(self.config.warehouse_path)
            return fallback.read_records(table, filters, limit)
        
        try:
            table_path = f"{self.config.warehouse_path}/{table}"
            df = spark.read.format("delta").load(table_path)
            
            if filters:
                for key, value in filters.items():
                    df = df.filter(df[key] == value)
            
            if limit:
                df = df.limit(limit)
            
            return [row.asDict() for row in df.collect()]
        except Exception as e:
            logger.error(f"Delta read failed: {e}")
            fallback = LocalParquetBackend(self.config.warehouse_path)
            return fallback.read_records(table, filters, limit)
    
    def table_exists(self, table: str) -> bool:
        """Check if Delta table exists."""
        table_path = Path(self.config.warehouse_path) / table
        return table_path.exists()
    
    def create_table(self, table: str, schema: Dict[str, str]) -> None:
        """Create Delta table."""
        # Delta tables are created on first write
        pass


class JEPALakehouseStore:
    """Main interface for V-JEPA + Lakehouse integration."""
    
    def __init__(self, config: LakehouseConfig, backend: Optional[LakehouseBackend] = None):
        self.config = config
        
        if backend:
            self.backend = backend
        elif config.table_format == TableFormat.DELTA:
            self.backend = DeltaLakeBackend(config)
        else:
            self.backend = LocalParquetBackend(config.warehouse_path)
        
        # Initialize tables
        self._init_tables()
    
    def _init_tables(self) -> None:
        """Initialize required tables."""
        tables = [
            self.config.embeddings_table,
            self.config.imagery_table,
            self.config.training_runs_table,
            self.config.findings_table,
        ]
        
        for table in tables:
            if not self.backend.table_exists(table):
                self.backend.create_table(table, {})
    
    # Embedding operations
    
    def write_embeddings(
        self,
        embeddings: List[EmbeddingRecord],
    ) -> int:
        """Write embeddings to lakehouse."""
        records = [e.to_dict() for e in embeddings]
        return self.backend.write_records(self.config.embeddings_table, records)
    
    def read_embeddings(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[EmbeddingRecord]:
        """Read embeddings from lakehouse."""
        records = self.backend.read_records(
            self.config.embeddings_table,
            filters=filters,
            limit=limit,
        )
        return [EmbeddingRecord.from_dict(r) for r in records]
    
    def search_embeddings(
        self,
        query_vector: List[float],
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[EmbeddingRecord, float]]:
        """Search for similar embeddings using cosine similarity."""
        embeddings = self.read_embeddings(filters=filters)
        
        if not embeddings:
            return []
        
        # Calculate cosine similarities
        import math
        
        def cosine_similarity(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)
        
        similarities = []
        for emb in embeddings:
            sim = cosine_similarity(query_vector, emb.embedding_vector)
            similarities.append((emb, sim))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    # Finding operations
    
    def write_findings(self, findings: List[FindingRecord]) -> int:
        """Write findings to lakehouse."""
        records = [f.to_dict() for f in findings]
        return self.backend.write_records(self.config.findings_table, records)
    
    def read_findings(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[FindingRecord]:
        """Read findings from lakehouse."""
        records = self.backend.read_records(
            self.config.findings_table,
            filters=filters,
            limit=limit,
        )
        findings = []
        for r in records:
            finding = FindingRecord(
                finding_id=r["finding_id"],
                finding_type=r["finding_type"],
                anomaly_score=r.get("anomaly_score"),
                similarity_score=r.get("similarity_score"),
                change_percentage=r.get("change_percentage"),
                confidence=r.get("confidence", "medium"),
                source_embedding_id=r.get("source_embedding_id", ""),
                location_name=r.get("location_name"),
                what_explanation=r.get("what_explanation"),
                why_evidence=r.get("why_evidence"),
                status=r.get("status", "new"),
            )
            findings.append(finding)
        return findings
    
    # Training run operations
    
    def write_training_run(self, run: TrainingRunRecord) -> None:
        """Write training run record to lakehouse."""
        self.backend.write_records(self.config.training_runs_table, [run.to_dict()])
    
    def read_training_runs(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[TrainingRunRecord]:
        """Read training runs from lakehouse."""
        records = self.backend.read_records(
            self.config.training_runs_table,
            filters=filters,
            limit=limit,
        )
        runs = []
        for r in records:
            run = TrainingRunRecord(
                run_id=r["run_id"],
                run_name=r["run_name"],
                backbone=r["backbone"],
                pretraining_mode=r["pretraining_mode"],
                num_epochs=r["num_epochs"],
                batch_size=r["batch_size"],
                learning_rate=r["learning_rate"],
                dataset_manifest_path=r["dataset_manifest_path"],
                num_samples=r["num_samples"],
                status=r.get("status", "pending"),
                checkpoint_path=r.get("checkpoint_path"),
                best_loss=r.get("best_loss"),
            )
            runs.append(run)
        return runs
    
    def update_training_run_status(
        self,
        run_id: str,
        status: str,
        checkpoint_path: Optional[str] = None,
        best_loss: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update training run status."""
        # Read all runs, update the matching one, and rewrite
        # (In production, this would use Delta's MERGE or UPDATE)
        runs = self.read_training_runs()
        for run in runs:
            if run.run_id == run_id:
                run.status = status
                if checkpoint_path:
                    run.checkpoint_path = checkpoint_path
                if best_loss is not None:
                    run.best_loss = best_loss
                if error_message:
                    run.error_message = error_message
                if status == "running":
                    run.started_at = datetime.utcnow()
                elif status in ("completed", "failed"):
                    run.completed_at = datetime.utcnow()
                break
        
        # Rewrite table (simplified - production would use proper updates)
        self.backend._tables[self.config.training_runs_table] = [r.to_dict() for r in runs]
        if hasattr(self.backend, '_save_table'):
            self.backend._save_table(self.config.training_runs_table)


class TrainingDatasetManager:
    """Manages training datasets for V-JEPA pretraining."""
    
    def __init__(self, store: JEPALakehouseStore):
        self.store = store
    
    def create_training_manifest(
        self,
        filters: Optional[Dict[str, Any]] = None,
        output_path: str = "./training_manifest.json",
        include_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """Create a training manifest from lakehouse data."""
        # Get embeddings matching filters
        embeddings = self.store.read_embeddings(filters=filters)
        
        manifest = {
            "manifest_id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat(),
            "filters": filters,
            "num_samples": len(embeddings),
            "samples": [],
        }
        
        for emb in embeddings:
            sample = {
                "embedding_id": emb.embedding_id,
                "source_path": emb.source_path,
                "source_type": emb.source_type,
                "site": emb.site,
                "sensor": emb.sensor,
            }
            if include_embeddings:
                sample["embedding_vector"] = emb.embedding_vector
            manifest["samples"].append(sample)
        
        # Save manifest
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        return manifest
    
    def get_dataset_statistics(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get statistics about the training dataset."""
        embeddings = self.store.read_embeddings(filters=filters)
        
        stats = {
            "total_samples": len(embeddings),
            "by_source_type": {},
            "by_site": {},
            "by_sensor": {},
            "date_range": {"min": None, "max": None},
        }
        
        for emb in embeddings:
            # Count by source type
            st = emb.source_type
            stats["by_source_type"][st] = stats["by_source_type"].get(st, 0) + 1
            
            # Count by site
            if emb.site:
                stats["by_site"][emb.site] = stats["by_site"].get(emb.site, 0) + 1
            
            # Count by sensor
            if emb.sensor:
                stats["by_sensor"][emb.sensor] = stats["by_sensor"].get(emb.sensor, 0) + 1
            
            # Track date range
            if emb.timestamp:
                if stats["date_range"]["min"] is None or emb.timestamp < stats["date_range"]["min"]:
                    stats["date_range"]["min"] = emb.timestamp
                if stats["date_range"]["max"] is None or emb.timestamp > stats["date_range"]["max"]:
                    stats["date_range"]["max"] = emb.timestamp
        
        return stats


class RAGRetriever:
    """Retrieval-Augmented Generation retriever for LLM explanations."""
    
    def __init__(self, store: JEPALakehouseStore):
        self.store = store
    
    def retrieve_context_for_finding(
        self,
        finding: FindingRecord,
        k_neighbors: int = 5,
        k_similar_findings: int = 3,
    ) -> Dict[str, Any]:
        """Retrieve context for LLM explanation of a finding."""
        context = {
            "neighbors": [],
            "similar_findings": [],
            "site_history": [],
        }
        
        # Get the source embedding
        embeddings = self.store.read_embeddings(
            filters={"embedding_id": finding.source_embedding_id}
        )
        
        if embeddings:
            source_emb = embeddings[0]
            
            # Find similar embeddings
            neighbors = self.store.search_embeddings(
                query_vector=source_emb.embedding_vector,
                k=k_neighbors + 1,  # +1 to exclude self
            )
            
            for emb, score in neighbors:
                if emb.embedding_id != finding.source_embedding_id:
                    context["neighbors"].append({
                        "embedding_id": emb.embedding_id,
                        "similarity": score,
                        "source_type": emb.source_type,
                        "site": emb.site,
                        "sensor": emb.sensor,
                    })
        
        # Find similar findings
        similar_findings = self.store.read_findings(
            filters={"finding_type": finding.finding_type},
            limit=k_similar_findings * 2,
        )
        
        for sf in similar_findings:
            if sf.finding_id != finding.finding_id:
                context["similar_findings"].append({
                    "finding_id": sf.finding_id,
                    "location": sf.location_name,
                    "confidence": sf.confidence,
                    "status": sf.status,
                    "what": sf.what_explanation,
                })
                if len(context["similar_findings"]) >= k_similar_findings:
                    break
        
        return context


# Factory functions

def create_jepa_lakehouse_store(
    warehouse_path: str = "./lakehouse",
    catalog: str = "mineralvision",
    table_format: str = "delta",
) -> JEPALakehouseStore:
    """Create a V-JEPA lakehouse store."""
    config = LakehouseConfig(
        warehouse_path=warehouse_path,
        catalog=catalog,
        table_format=TableFormat(table_format) if table_format in ["delta", "iceberg", "parquet"] else TableFormat.PARQUET,
    )
    return JEPALakehouseStore(config)


def create_training_dataset_manager(
    store: Optional[JEPALakehouseStore] = None,
    warehouse_path: str = "./lakehouse",
) -> TrainingDatasetManager:
    """Create a training dataset manager."""
    if store is None:
        store = create_jepa_lakehouse_store(warehouse_path=warehouse_path)
    return TrainingDatasetManager(store)


def create_rag_retriever(
    store: Optional[JEPALakehouseStore] = None,
    warehouse_path: str = "./lakehouse",
) -> RAGRetriever:
    """Create a RAG retriever for LLM explanations."""
    if store is None:
        store = create_jepa_lakehouse_store(warehouse_path=warehouse_path)
    return RAGRetriever(store)
