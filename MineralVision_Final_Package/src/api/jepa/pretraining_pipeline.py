"""
V-JEPA Pretraining Pipeline for Mining Domain.

Provides end-to-end pretraining workflow including:
- Dataset preparation and tiling
- Multi-GPU distributed training
- Checkpoint management
- Training monitoring and logging
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import hashlib

from .vjepa_integration import (
    VJEPAConfig,
    VJEPAPretrainer,
    MiningDataLoader,
    ImageryType,
    PretrainingMode,
    BackboneSize,
    create_vjepa_config,
    create_mining_data_loader,
)

logger = logging.getLogger(__name__)


@dataclass
class PretrainingJob:
    """Represents a pretraining job configuration."""
    job_id: str
    job_name: str
    config: VJEPAConfig
    
    data_sources: Dict[str, str] = field(default_factory=dict)
    manifest_paths: Dict[str, str] = field(default_factory=dict)
    
    output_dir: str = "./pretraining_output"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    num_gpus: int = 1
    num_nodes: int = 1
    distributed_backend: str = "nccl"
    
    resume_from: Optional[str] = None
    pretrained_weights: Optional[str] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_name": self.job_name,
            "config": self.config.to_dict(),
            "data_sources": self.data_sources,
            "manifest_paths": self.manifest_paths,
            "output_dir": self.output_dir,
            "checkpoint_dir": self.checkpoint_dir,
            "log_dir": self.log_dir,
            "num_gpus": self.num_gpus,
            "num_nodes": self.num_nodes,
            "distributed_backend": self.distributed_backend,
            "resume_from": self.resume_from,
            "pretrained_weights": self.pretrained_weights,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PretrainingJob":
        config_data = data.pop("config")
        config = VJEPAConfig(
            backbone=BackboneSize(config_data["backbone"]),
            pretraining_mode=PretrainingMode(config_data["pretraining_mode"]),
            **{k: v for k, v in config_data.items() if k not in ["backbone", "pretraining_mode"]}
        )
        
        created_at = datetime.fromisoformat(data.pop("created_at"))
        
        return cls(config=config, created_at=created_at, **data)


@dataclass
class TilingConfig:
    """Configuration for image tiling."""
    tile_size: Tuple[int, int] = (224, 224)
    overlap: int = 32
    min_valid_ratio: float = 0.5
    
    output_format: str = "png"
    preserve_georeference: bool = True
    
    normalize: bool = True
    normalize_percentile: Tuple[float, float] = (2, 98)
    
    bands_to_use: Optional[List[int]] = None
    band_mapping: Optional[Dict[int, str]] = None


class ImageTiler:
    """Tiles large images into V-JEPA compatible patches."""
    
    def __init__(self, config: TilingConfig):
        self.config = config
        logger.info(f"Initialized ImageTiler with tile_size={config.tile_size}")
    
    def tile_image(
        self,
        image_path: str,
        output_dir: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Tile a single image into patches."""
        import numpy as np
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            image = self._load_image(image_path)
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return []
        
        height, width = image.shape[:2]
        tile_h, tile_w = self.config.tile_size
        overlap = self.config.overlap
        
        stride_h = tile_h - overlap
        stride_w = tile_w - overlap
        
        tiles = []
        tile_idx = 0
        
        for y in range(0, height - tile_h + 1, stride_h):
            for x in range(0, width - tile_w + 1, stride_w):
                tile = image[y:y + tile_h, x:x + tile_w]
                
                valid_ratio = np.count_nonzero(tile) / tile.size
                if valid_ratio < self.config.min_valid_ratio:
                    continue
                
                tile_id = f"{Path(image_path).stem}_tile_{tile_idx:04d}"
                tile_filename = f"{tile_id}.{self.config.output_format}"
                tile_path = output_path / tile_filename
                
                self._save_tile(tile, str(tile_path))
                
                tile_meta = {
                    "tile_id": tile_id,
                    "source_image": image_path,
                    "tile_path": str(tile_path),
                    "bbox_in_image": (x, y, x + tile_w, y + tile_h),
                    "tile_size": self.config.tile_size,
                    "valid_ratio": valid_ratio,
                }
                
                if metadata:
                    tile_meta["source_metadata"] = metadata
                    
                    if metadata.get("geo_bbox"):
                        tile_meta["geo_bbox"] = self._compute_tile_geobbox(
                            metadata["geo_bbox"],
                            (width, height),
                            (x, y, x + tile_w, y + tile_h)
                        )
                
                tiles.append(tile_meta)
                tile_idx += 1
        
        logger.info(f"Created {len(tiles)} tiles from {image_path}")
        return tiles
    
    def _load_image(self, image_path: str) -> Any:
        """Load image from file."""
        import numpy as np
        
        path = Path(image_path)
        
        if path.suffix.lower() in [".tif", ".tiff"]:
            try:
                import rasterio
                with rasterio.open(image_path) as src:
                    if self.config.bands_to_use:
                        image = src.read(self.config.bands_to_use)
                    else:
                        image = src.read()
                    
                    if image.shape[0] <= 4:
                        image = np.transpose(image, (1, 2, 0))
                    
                    return image
            except ImportError:
                pass
        
        import cv2
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _save_tile(self, tile: Any, path: str) -> None:
        """Save tile to file."""
        import cv2
        import numpy as np
        
        if tile.dtype != np.uint8:
            tile = ((tile - tile.min()) / (tile.max() - tile.min() + 1e-8) * 255).astype(np.uint8)
        
        if len(tile.shape) == 3 and tile.shape[2] == 3:
            tile = cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(path, tile)
    
    def _compute_tile_geobbox(
        self,
        image_bbox: Tuple[float, float, float, float],
        image_size: Tuple[int, int],
        tile_bbox: Tuple[int, int, int, int]
    ) -> Tuple[float, float, float, float]:
        """Compute geographic bounding box for a tile."""
        min_x, min_y, max_x, max_y = image_bbox
        img_w, img_h = image_size
        tile_x1, tile_y1, tile_x2, tile_y2 = tile_bbox
        
        pixel_w = (max_x - min_x) / img_w
        pixel_h = (max_y - min_y) / img_h
        
        tile_min_x = min_x + tile_x1 * pixel_w
        tile_max_x = min_x + tile_x2 * pixel_w
        tile_min_y = max_y - tile_y2 * pixel_h
        tile_max_y = max_y - tile_y1 * pixel_h
        
        return (tile_min_x, tile_min_y, tile_max_x, tile_max_y)
    
    def tile_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Tile all images in a directory."""
        extensions = extensions or [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
        
        input_path = Path(input_dir)
        all_tiles = []
        
        for ext in extensions:
            for image_path in input_path.rglob(f"*{ext}"):
                tiles = self.tile_image(str(image_path), output_dir)
                all_tiles.extend(tiles)
        
        logger.info(f"Created {len(all_tiles)} total tiles from {input_dir}")
        return all_tiles


class VideoChunker:
    """Chunks videos into V-JEPA compatible clips."""
    
    def __init__(
        self,
        num_frames: int = 16,
        frame_stride: int = 4,
        clip_overlap: int = 8,
        resolution: Tuple[int, int] = (224, 224)
    ):
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.clip_overlap = clip_overlap
        self.resolution = resolution
        
        logger.info(f"Initialized VideoChunker with num_frames={num_frames}")
    
    def chunk_video(
        self,
        video_path: str,
        output_dir: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Chunk a video into clips."""
        import cv2
        import numpy as np
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames_per_clip = self.num_frames * self.frame_stride
        clip_stride = frames_per_clip - self.clip_overlap * self.frame_stride
        
        clips = []
        clip_idx = 0
        
        for start_frame in range(0, total_frames - frames_per_clip + 1, clip_stride):
            clip_frames = []
            
            for i in range(self.num_frames):
                frame_idx = start_frame + i * self.frame_stride
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.resolution)
                clip_frames.append(frame)
            
            if len(clip_frames) < self.num_frames:
                continue
            
            clip_id = f"{Path(video_path).stem}_clip_{clip_idx:04d}"
            clip_filename = f"{clip_id}.npz"
            clip_path = output_path / clip_filename
            
            np.savez_compressed(str(clip_path), frames=np.array(clip_frames))
            
            clip_meta = {
                "clip_id": clip_id,
                "source_video": video_path,
                "clip_path": str(clip_path),
                "start_frame": start_frame,
                "end_frame": start_frame + frames_per_clip,
                "num_frames": self.num_frames,
                "fps": fps,
                "timestamp_start": start_frame / fps if fps > 0 else 0,
                "timestamp_end": (start_frame + frames_per_clip) / fps if fps > 0 else 0,
            }
            
            if metadata:
                clip_meta["source_metadata"] = metadata
            
            clips.append(clip_meta)
            clip_idx += 1
        
        cap.release()
        logger.info(f"Created {len(clips)} clips from {video_path}")
        return clips
    
    def chunk_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Chunk all videos in a directory."""
        extensions = extensions or [".mp4", ".avi", ".mov", ".mkv"]
        
        input_path = Path(input_dir)
        all_clips = []
        
        for ext in extensions:
            for video_path in input_path.rglob(f"*{ext}"):
                clips = self.chunk_video(str(video_path), output_dir)
                all_clips.extend(clips)
        
        logger.info(f"Created {len(all_clips)} total clips from {input_dir}")
        return all_clips


class CorePhotoProcessor:
    """Processes core photography for V-JEPA training."""
    
    def __init__(
        self,
        tile_size: Tuple[int, int] = (224, 224),
        depth_interval: float = 1.0,
        overlap_ratio: float = 0.25
    ):
        self.tile_size = tile_size
        self.depth_interval = depth_interval
        self.overlap_ratio = overlap_ratio
        
        logger.info(f"Initialized CorePhotoProcessor with tile_size={tile_size}")
    
    def process_core_tray(
        self,
        image_path: str,
        output_dir: str,
        hole_id: str,
        depth_from: float,
        depth_to: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Process a core tray image into depth-ordered tiles."""
        import cv2
        import numpy as np
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load core image {image_path}")
            return []
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        is_horizontal = width > height
        
        if is_horizontal:
            core_length = width
            core_width = height
        else:
            core_length = height
            core_width = width
        
        depth_range = depth_to - depth_from
        pixels_per_meter = core_length / depth_range if depth_range > 0 else core_length
        
        tile_h, tile_w = self.tile_size
        overlap = int(tile_w * self.overlap_ratio)
        stride = tile_w - overlap
        
        tiles = []
        tile_idx = 0
        
        if is_horizontal:
            for x in range(0, width - tile_w + 1, stride):
                tile_depth_from = depth_from + (x / pixels_per_meter)
                tile_depth_to = depth_from + ((x + tile_w) / pixels_per_meter)
                
                y_center = height // 2
                y_start = max(0, y_center - tile_h // 2)
                
                tile = image[y_start:y_start + tile_h, x:x + tile_w]
                
                if tile.shape[0] < tile_h or tile.shape[1] < tile_w:
                    tile = cv2.resize(tile, (tile_w, tile_h))
                
                tile_id = f"{hole_id}_{depth_from:.1f}_{depth_to:.1f}_tile_{tile_idx:04d}"
                tile_filename = f"{tile_id}.png"
                tile_path = output_path / tile_filename
                
                cv2.imwrite(str(tile_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
                
                tile_meta = {
                    "tile_id": tile_id,
                    "source_image": image_path,
                    "tile_path": str(tile_path),
                    "hole_id": hole_id,
                    "depth_from": tile_depth_from,
                    "depth_to": tile_depth_to,
                    "depth_center": (tile_depth_from + tile_depth_to) / 2,
                    "bbox_in_image": (x, y_start, x + tile_w, y_start + tile_h),
                }
                
                if metadata:
                    tile_meta["source_metadata"] = metadata
                
                tiles.append(tile_meta)
                tile_idx += 1
        
        tiles.sort(key=lambda t: t["depth_center"])
        
        logger.info(f"Created {len(tiles)} core tiles from {image_path}")
        return tiles
    
    def create_depth_sequences(
        self,
        tiles: List[Dict[str, Any]],
        sequence_length: int = 16,
        overlap: int = 8
    ) -> List[List[Dict[str, Any]]]:
        """Create depth-ordered sequences for temporal modeling."""
        if len(tiles) < sequence_length:
            return [tiles] if tiles else []
        
        sequences = []
        stride = sequence_length - overlap
        
        for i in range(0, len(tiles) - sequence_length + 1, stride):
            sequence = tiles[i:i + sequence_length]
            sequences.append(sequence)
        
        return sequences


class DatasetManifestBuilder:
    """Builds unified manifest for pretraining dataset."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.manifest: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized DatasetManifestBuilder with output_dir={output_dir}")
    
    def add_tiles(self, tiles: List[Dict[str, Any]], imagery_type: str) -> None:
        """Add tiles to manifest."""
        for tile in tiles:
            entry = {
                "id": tile.get("tile_id") or tile.get("clip_id"),
                "path": tile.get("tile_path") or tile.get("clip_path"),
                "type": imagery_type,
                "metadata": tile,
            }
            self.manifest.append(entry)
        
        logger.info(f"Added {len(tiles)} {imagery_type} entries to manifest")
    
    def add_clips(self, clips: List[Dict[str, Any]], imagery_type: str) -> None:
        """Add video clips to manifest."""
        self.add_tiles(clips, imagery_type)
    
    def save_manifest(self, filename: str = "manifest.json") -> str:
        """Save manifest to file."""
        manifest_path = self.output_dir / filename
        
        with open(manifest_path, "w") as f:
            json.dump({
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "num_samples": len(self.manifest),
                "samples": self.manifest,
            }, f, indent=2)
        
        logger.info(f"Saved manifest with {len(self.manifest)} samples to {manifest_path}")
        return str(manifest_path)
    
    def load_manifest(self, path: str) -> None:
        """Load existing manifest."""
        with open(path, "r") as f:
            data = json.load(f)
        
        self.manifest = data.get("samples", [])
        logger.info(f"Loaded manifest with {len(self.manifest)} samples")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        type_counts = {}
        for entry in self.manifest:
            entry_type = entry.get("type", "unknown")
            type_counts[entry_type] = type_counts.get(entry_type, 0) + 1
        
        return {
            "total_samples": len(self.manifest),
            "samples_by_type": type_counts,
        }


class PretrainingRunner:
    """Runs V-JEPA pretraining jobs."""
    
    def __init__(self, jobs_dir: str = "./pretraining_jobs"):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_jobs: Dict[str, PretrainingJob] = {}
        
        logger.info(f"Initialized PretrainingRunner with jobs_dir={jobs_dir}")
    
    def create_job(
        self,
        job_name: str,
        config: VJEPAConfig,
        data_sources: Dict[str, str],
        **kwargs
    ) -> PretrainingJob:
        """Create a new pretraining job."""
        job_id = hashlib.md5(f"{job_name}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        
        job = PretrainingJob(
            job_id=job_id,
            job_name=job_name,
            config=config,
            data_sources=data_sources,
            output_dir=str(self.jobs_dir / job_id / "output"),
            checkpoint_dir=str(self.jobs_dir / job_id / "checkpoints"),
            log_dir=str(self.jobs_dir / job_id / "logs"),
            **kwargs
        )
        
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        with open(job_dir / "job_config.json", "w") as f:
            json.dump(job.to_dict(), f, indent=2)
        
        self.active_jobs[job_id] = job
        
        logger.info(f"Created pretraining job {job_id}: {job_name}")
        return job
    
    def run_job(self, job_id: str) -> Dict[str, Any]:
        """Run a pretraining job."""
        job = self.active_jobs.get(job_id)
        if not job:
            job = self.load_job(job_id)
        
        if not job:
            raise ValueError(f"Job {job_id} not found")
        
        job.status = "running"
        self._save_job(job)
        
        logger.info(f"Starting pretraining job {job_id}")
        
        try:
            data_loaders = []
            for imagery_type, data_root in job.data_sources.items():
                loader = create_mining_data_loader(
                    imagery_type=imagery_type,
                    data_root=data_root,
                    config=job.config,
                    manifest_path=job.manifest_paths.get(imagery_type)
                )
                loader.load_manifest()
                data_loaders.append(loader)
            
            pretrainer = VJEPAPretrainer(
                config=job.config,
                data_loaders=data_loaders
            )
            
            if job.resume_from:
                start_epoch = pretrainer.load_checkpoint(job.resume_from)
            else:
                start_epoch = 0
            
            def checkpoint_callback(epoch, encoder, stats):
                pretrainer.save_checkpoint(job.checkpoint_dir, epoch)
                self._log_progress(job, epoch, stats)
            
            results = pretrainer.train(
                num_epochs=job.config.total_epochs - start_epoch,
                checkpoint_callback=checkpoint_callback
            )
            
            job.status = "completed"
            self._save_job(job)
            
            logger.info(f"Completed pretraining job {job_id}")
            return results
            
        except Exception as e:
            job.status = "failed"
            self._save_job(job)
            logger.error(f"Pretraining job {job_id} failed: {e}")
            raise
    
    def load_job(self, job_id: str) -> Optional[PretrainingJob]:
        """Load a job from disk."""
        job_path = self.jobs_dir / job_id / "job_config.json"
        
        if not job_path.exists():
            return None
        
        with open(job_path, "r") as f:
            data = json.load(f)
        
        job = PretrainingJob.from_dict(data)
        self.active_jobs[job_id] = job
        
        return job
    
    def _save_job(self, job: PretrainingJob) -> None:
        """Save job state to disk."""
        job_path = self.jobs_dir / job.job_id / "job_config.json"
        
        with open(job_path, "w") as f:
            json.dump(job.to_dict(), f, indent=2)
    
    def _log_progress(self, job: PretrainingJob, epoch: int, stats: Dict[str, Any]) -> None:
        """Log training progress."""
        log_path = Path(job.log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "stats": stats,
        }
        
        log_file = log_path / "training_log.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs."""
        jobs = []
        
        for job_dir in self.jobs_dir.iterdir():
            if job_dir.is_dir():
                job = self.load_job(job_dir.name)
                if job:
                    jobs.append({
                        "job_id": job.job_id,
                        "job_name": job.job_name,
                        "status": job.status,
                        "created_at": job.created_at.isoformat(),
                    })
        
        return jobs
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get detailed job status."""
        job = self.active_jobs.get(job_id) or self.load_job(job_id)
        
        if not job:
            return {"error": f"Job {job_id} not found"}
        
        log_path = Path(job.log_dir) / "training_log.jsonl"
        latest_stats = None
        
        if log_path.exists():
            with open(log_path, "r") as f:
                lines = f.readlines()
                if lines:
                    latest_stats = json.loads(lines[-1])
        
        return {
            "job_id": job.job_id,
            "job_name": job.job_name,
            "status": job.status,
            "created_at": job.created_at.isoformat(),
            "config": job.config.to_dict(),
            "latest_stats": latest_stats,
        }


def prepare_mining_dataset(
    data_sources: Dict[str, str],
    output_dir: str,
    tile_size: Tuple[int, int] = (224, 224),
    num_frames: int = 16
) -> str:
    """Prepare mining dataset for V-JEPA pretraining."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    manifest_builder = DatasetManifestBuilder(output_dir)
    
    image_tiler = ImageTiler(TilingConfig(tile_size=tile_size))
    video_chunker = VideoChunker(num_frames=num_frames, resolution=tile_size)
    core_processor = CorePhotoProcessor(tile_size=tile_size)
    
    for imagery_type, data_root in data_sources.items():
        tiles_dir = output_path / "tiles" / imagery_type
        
        if imagery_type in ["drone_video", "site_camera"]:
            clips = video_chunker.chunk_directory(data_root, str(tiles_dir))
            manifest_builder.add_clips(clips, imagery_type)
            
        elif imagery_type == "core_photo":
            for image_path in Path(data_root).rglob("*.jpg"):
                tiles = core_processor.process_core_tray(
                    str(image_path),
                    str(tiles_dir),
                    hole_id=image_path.parent.name,
                    depth_from=0.0,
                    depth_to=10.0,
                )
                manifest_builder.add_tiles(tiles, imagery_type)
        else:
            tiles = image_tiler.tile_directory(data_root, str(tiles_dir))
            manifest_builder.add_tiles(tiles, imagery_type)
    
    manifest_path = manifest_builder.save_manifest()
    
    stats = manifest_builder.get_statistics()
    logger.info(f"Dataset preparation complete: {stats}")
    
    return manifest_path


def run_pretraining(
    job_name: str,
    data_sources: Dict[str, str],
    backbone: str = "vit_large",
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    pretrained_weights: Optional[str] = None,
    output_dir: str = "./pretraining_output"
) -> Dict[str, Any]:
    """Run V-JEPA pretraining on mining data."""
    config = create_vjepa_config(
        backbone=backbone,
        total_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pretrained_checkpoint=pretrained_weights,
        output_dir=output_dir,
    )
    
    runner = PretrainingRunner(output_dir)
    
    job = runner.create_job(
        job_name=job_name,
        config=config,
        data_sources=data_sources,
        pretrained_weights=pretrained_weights,
    )
    
    results = runner.run_job(job.job_id)
    
    return {
        "job_id": job.job_id,
        "results": results,
        "checkpoint_dir": job.checkpoint_dir,
    }
