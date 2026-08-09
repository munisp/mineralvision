"""
Edge Deployment Optimization Module
====================================

Production-grade edge deployment with:
- ONNX export and optimization
- TensorRT acceleration
- Batch inference support
- Frame skipping and backpressure
- Memory-efficient inference
"""

import os
import time
import logging
import threading
import queue
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Model export formats."""
    ONNX = "onnx"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"
    COREML = "coreml"
    TFLITE = "tflite"
    TORCHSCRIPT = "torchscript"


class PrecisionMode(Enum):
    """Inference precision modes."""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    MIXED = "mixed"


@dataclass
class ExportConfig:
    """Configuration for model export."""
    format: ExportFormat = ExportFormat.ONNX
    precision: PrecisionMode = PrecisionMode.FP16
    input_size: Tuple[int, int] = (640, 640)
    batch_size: int = 1
    dynamic_batch: bool = True
    opset_version: int = 17
    simplify: bool = True
    optimize: bool = True
    calibration_data: Optional[str] = None  # For INT8 calibration


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    batch_size: int = 1
    max_batch_wait_ms: float = 10.0
    frame_skip: int = 0
    backpressure_threshold: int = 100
    warmup_iterations: int = 10
    use_cuda_graphs: bool = True
    memory_pool_size_mb: int = 512


@dataclass
class InferenceResult:
    """Inference result."""
    detections: List[Dict]
    latency_ms: float
    batch_size: int
    frame_id: int
    timestamp: float


class ModelExporter:
    """
    Export models to various deployment formats.
    """
    
    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()
    
    def export_onnx(self, model_path: str, output_path: str,
                   class_names: List[str] = None) -> str:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model_path: Path to PyTorch model
            output_path: Output ONNX path
            class_names: List of class names for metadata
            
        Returns:
            Path to exported ONNX model
        """
        try:
            import torch
            from ultralytics import YOLO
            
            # Load model
            model = YOLO(model_path)
            
            # Export to ONNX
            export_path = model.export(
                format='onnx',
                imgsz=self.config.input_size,
                batch=self.config.batch_size,
                dynamic=self.config.dynamic_batch,
                simplify=self.config.simplify,
                opset=self.config.opset_version,
                half=self.config.precision == PrecisionMode.FP16
            )
            
            # Move to output path if different
            if export_path != output_path:
                import shutil
                shutil.move(export_path, output_path)
            
            logger.info(f"Exported ONNX model to {output_path}")
            return output_path
            
        except ImportError as e:
            logger.error(f"Required packages not available: {e}")
            return self._export_onnx_manual(model_path, output_path)
    
    def _export_onnx_manual(self, model_path: str, output_path: str) -> str:
        """Manual ONNX export for custom models."""
        try:
            import torch
            import torch.onnx
            
            # Load model
            model = torch.load(model_path, map_location='cpu')
            if isinstance(model, dict):
                model = model.get('model', model)
            
            model.eval()
            
            # Create dummy input
            batch_size = self.config.batch_size if not self.config.dynamic_batch else 1
            dummy_input = torch.randn(batch_size, 3, *self.config.input_size)
            
            # Export
            dynamic_axes = None
            if self.config.dynamic_batch:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                opset_version=self.config.opset_version,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )
            
            # Simplify if requested
            if self.config.simplify:
                self._simplify_onnx(output_path)
            
            return output_path
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
    
    def _simplify_onnx(self, onnx_path: str):
        """Simplify ONNX model."""
        try:
            import onnx
            from onnxsim import simplify
            
            model = onnx.load(onnx_path)
            model_simplified, check = simplify(model)
            
            if check:
                onnx.save(model_simplified, onnx_path)
                logger.info("ONNX model simplified successfully")
            else:
                logger.warning("ONNX simplification check failed")
                
        except ImportError:
            logger.warning("onnxsim not available, skipping simplification")
    
    def export_tensorrt(self, onnx_path: str, output_path: str,
                       workspace_size_gb: int = 4) -> str:
        """
        Convert ONNX model to TensorRT engine.
        
        Args:
            onnx_path: Path to ONNX model
            output_path: Output TensorRT engine path
            workspace_size_gb: Workspace size in GB
            
        Returns:
            Path to TensorRT engine
        """
        try:
            import tensorrt as trt
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            with trt.Builder(TRT_LOGGER) as builder, \
                 builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
                 trt.OnnxParser(network, TRT_LOGGER) as parser:
                
                # Configure builder
                config = builder.create_builder_config()
                config.max_workspace_size = workspace_size_gb * (1 << 30)
                
                # Set precision
                if self.config.precision == PrecisionMode.FP16:
                    if builder.platform_has_fast_fp16:
                        config.set_flag(trt.BuilderFlag.FP16)
                elif self.config.precision == PrecisionMode.INT8:
                    if builder.platform_has_fast_int8:
                        config.set_flag(trt.BuilderFlag.INT8)
                        # Would need calibrator for INT8
                
                # Parse ONNX
                with open(onnx_path, 'rb') as f:
                    if not parser.parse(f.read()):
                        for error in range(parser.num_errors):
                            logger.error(f"TensorRT parser error: {parser.get_error(error)}")
                        raise RuntimeError("Failed to parse ONNX model")
                
                # Build engine
                engine = builder.build_engine(network, config)
                
                if engine is None:
                    raise RuntimeError("Failed to build TensorRT engine")
                
                # Serialize
                with open(output_path, 'wb') as f:
                    f.write(engine.serialize())
                
                logger.info(f"Exported TensorRT engine to {output_path}")
                return output_path
                
        except ImportError:
            logger.error("TensorRT not available")
            raise
    
    def export_openvino(self, onnx_path: str, output_dir: str) -> str:
        """
        Convert ONNX model to OpenVINO IR format.
        
        Args:
            onnx_path: Path to ONNX model
            output_dir: Output directory
            
        Returns:
            Path to OpenVINO model XML
        """
        try:
            from openvino.tools import mo
            
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            mo.convert_model(
                onnx_path,
                output_model=str(output_path / 'model'),
                compress_to_fp16=self.config.precision == PrecisionMode.FP16
            )
            
            return str(output_path / 'model.xml')
            
        except ImportError:
            logger.error("OpenVINO not available")
            raise


class ONNXInferenceEngine:
    """
    ONNX Runtime inference engine.
    """
    
    def __init__(self, model_path: str, config: InferenceConfig = None):
        self.model_path = model_path
        self.config = config or InferenceConfig()
        self.session = None
        self.input_name = None
        self.output_names = None
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            
            # Configure session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            
            # Select execution provider
            providers = []
            if ort.get_device() == 'GPU':
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': self.config.memory_pool_size_mb * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE'
                }))
            providers.append('CPUExecutionProvider')
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            # Warmup
            self._warmup()
            
            logger.info(f"Loaded ONNX model from {self.model_path}")
            
        except ImportError:
            logger.error("ONNX Runtime not available")
            raise
    
    def _warmup(self):
        """Warmup inference."""
        input_shape = self.session.get_inputs()[0].shape
        batch_size = input_shape[0] if isinstance(input_shape[0], int) else 1
        height = input_shape[2] if isinstance(input_shape[2], int) else 640
        width = input_shape[3] if isinstance(input_shape[3], int) else 640
        
        dummy_input = np.random.randn(batch_size, 3, height, width).astype(np.float32)
        
        for _ in range(self.config.warmup_iterations):
            self.session.run(self.output_names, {self.input_name: dummy_input})
    
    def infer(self, images: np.ndarray) -> List[np.ndarray]:
        """
        Run inference on images.
        
        Args:
            images: Input images [N, C, H, W] or [C, H, W]
            
        Returns:
            List of output arrays
        """
        if len(images.shape) == 3:
            images = images[np.newaxis, ...]
        
        images = images.astype(np.float32)
        
        outputs = self.session.run(self.output_names, {self.input_name: images})
        
        return outputs
    
    def infer_with_timing(self, images: np.ndarray) -> Tuple[List[np.ndarray], float]:
        """Run inference with timing."""
        start = time.perf_counter()
        outputs = self.infer(images)
        latency = (time.perf_counter() - start) * 1000
        return outputs, latency


class TensorRTInferenceEngine:
    """
    TensorRT inference engine.
    """
    
    def __init__(self, engine_path: str, config: InferenceConfig = None):
        self.engine_path = engine_path
        self.config = config or InferenceConfig()
        self.engine = None
        self.context = None
        self.bindings = None
        self.stream = None
        self._load_engine()
    
    def _load_engine(self):
        """Load TensorRT engine."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            with open(self.engine_path, 'rb') as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            
            # Allocate buffers
            self.bindings = []
            self.inputs = []
            self.outputs = []
            
            for binding in self.engine:
                size = trt.volume(self.engine.get_binding_shape(binding))
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self.bindings.append(int(device_mem))
                
                if self.engine.binding_is_input(binding):
                    self.inputs.append({'host': host_mem, 'device': device_mem})
                else:
                    self.outputs.append({'host': host_mem, 'device': device_mem})
            
            # Warmup
            self._warmup()
            
            logger.info(f"Loaded TensorRT engine from {self.engine_path}")
            
        except ImportError as e:
            logger.error(f"TensorRT/PyCUDA not available: {e}")
            raise
    
    def _warmup(self):
        """Warmup inference."""
        import pycuda.driver as cuda
        
        for _ in range(self.config.warmup_iterations):
            # Copy random data to input
            np.copyto(self.inputs[0]['host'], 
                     np.random.randn(*self.inputs[0]['host'].shape).astype(np.float32).ravel())
            
            cuda.memcpy_htod_async(self.inputs[0]['device'], 
                                   self.inputs[0]['host'], self.stream)
            
            self.context.execute_async_v2(bindings=self.bindings, 
                                          stream_handle=self.stream.handle)
            
            cuda.memcpy_dtoh_async(self.outputs[0]['host'],
                                   self.outputs[0]['device'], self.stream)
            
            self.stream.synchronize()
    
    def infer(self, images: np.ndarray) -> List[np.ndarray]:
        """Run inference."""
        import pycuda.driver as cuda
        
        # Copy input
        np.copyto(self.inputs[0]['host'], images.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'],
                               self.inputs[0]['host'], self.stream)
        
        # Execute
        self.context.execute_async_v2(bindings=self.bindings,
                                      stream_handle=self.stream.handle)
        
        # Copy outputs
        outputs = []
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output['host'], output['device'], self.stream)
        
        self.stream.synchronize()
        
        for output in self.outputs:
            outputs.append(output['host'].copy())
        
        return outputs
    
    def infer_with_timing(self, images: np.ndarray) -> Tuple[List[np.ndarray], float]:
        """Run inference with timing."""
        start = time.perf_counter()
        outputs = self.infer(images)
        latency = (time.perf_counter() - start) * 1000
        return outputs, latency


class BatchInferenceManager:
    """
    Manages batched inference with dynamic batching.
    """
    
    def __init__(self, engine: Any, config: InferenceConfig = None):
        self.engine = engine
        self.config = config or InferenceConfig()
        
        self.request_queue = queue.Queue(maxsize=self.config.backpressure_threshold)
        self.result_queues: Dict[int, queue.Queue] = {}
        self.next_request_id = 0
        self._lock = threading.Lock()
        
        self.running = False
        self.worker_thread = None
    
    def start(self):
        """Start batch inference worker."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
    
    def stop(self):
        """Stop batch inference worker."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
    
    def submit(self, image: np.ndarray, frame_id: int = 0) -> int:
        """
        Submit image for inference.
        
        Args:
            image: Input image [C, H, W]
            frame_id: Frame identifier
            
        Returns:
            Request ID
        """
        with self._lock:
            request_id = self.next_request_id
            self.next_request_id += 1
            self.result_queues[request_id] = queue.Queue()
        
        try:
            self.request_queue.put((request_id, image, frame_id), timeout=1.0)
        except queue.Full:
            logger.warning("Request queue full, applying backpressure")
            raise RuntimeError("Inference backpressure - queue full")
        
        return request_id
    
    def get_result(self, request_id: int, timeout: float = 10.0) -> InferenceResult:
        """
        Get inference result.
        
        Args:
            request_id: Request ID from submit()
            timeout: Timeout in seconds
            
        Returns:
            InferenceResult
        """
        result_queue = self.result_queues.get(request_id)
        if result_queue is None:
            raise ValueError(f"Unknown request ID: {request_id}")
        
        try:
            result = result_queue.get(timeout=timeout)
        finally:
            with self._lock:
                del self.result_queues[request_id]
        
        return result
    
    def _worker_loop(self):
        """Worker loop for batch inference."""
        while self.running:
            batch = []
            request_ids = []
            frame_ids = []
            
            # Collect batch
            deadline = time.time() + self.config.max_batch_wait_ms / 1000
            
            while len(batch) < self.config.batch_size:
                timeout = max(0, deadline - time.time())
                
                try:
                    request_id, image, frame_id = self.request_queue.get(timeout=timeout)
                    batch.append(image)
                    request_ids.append(request_id)
                    frame_ids.append(frame_id)
                except queue.Empty:
                    break
            
            if not batch:
                continue
            
            # Run inference
            images = np.stack(batch, axis=0)
            outputs, latency = self.engine.infer_with_timing(images)
            
            # Distribute results
            for i, request_id in enumerate(request_ids):
                # Parse detections from output
                detections = self._parse_detections(outputs, i)
                
                result = InferenceResult(
                    detections=detections,
                    latency_ms=latency / len(batch),
                    batch_size=len(batch),
                    frame_id=frame_ids[i],
                    timestamp=time.time()
                )
                
                result_queue = self.result_queues.get(request_id)
                if result_queue:
                    result_queue.put(result)
    
    def _parse_detections(self, outputs: List[np.ndarray], batch_idx: int) -> List[Dict]:
        """Parse detections from model output."""
        detections = []
        
        # Assume YOLO-style output
        if len(outputs) > 0:
            output = outputs[0]
            
            if len(output.shape) == 3:
                # [batch, num_detections, 6] format
                batch_output = output[batch_idx]
                
                for det in batch_output:
                    if det[4] > 0.25:  # Confidence threshold
                        detections.append({
                            'bbox': det[:4].tolist(),
                            'confidence': float(det[4]),
                            'class_id': int(det[5])
                        })
        
        return detections


class FrameSkipManager:
    """
    Manages frame skipping for real-time performance.
    """
    
    def __init__(self, target_fps: float = 30.0, min_fps: float = 10.0):
        self.target_fps = target_fps
        self.min_fps = min_fps
        self.target_latency = 1000 / target_fps
        self.max_latency = 1000 / min_fps
        
        self.frame_count = 0
        self.skip_count = 0
        self.latency_history: List[float] = []
        self.history_size = 30
    
    def should_process(self, frame_id: int) -> bool:
        """
        Determine if frame should be processed.
        
        Args:
            frame_id: Frame identifier
            
        Returns:
            True if frame should be processed
        """
        self.frame_count += 1
        
        # Always process first few frames
        if len(self.latency_history) < 5:
            return True
        
        # Calculate adaptive skip rate
        avg_latency = np.mean(self.latency_history)
        
        if avg_latency <= self.target_latency:
            # Processing fast enough, no skip
            return True
        elif avg_latency >= self.max_latency:
            # Too slow, skip every other frame
            skip = frame_id % 2 != 0
        else:
            # Adaptive skip rate
            skip_rate = (avg_latency - self.target_latency) / (self.max_latency - self.target_latency)
            skip = np.random.random() < skip_rate
        
        if skip:
            self.skip_count += 1
        
        return not skip
    
    def record_latency(self, latency_ms: float):
        """Record inference latency."""
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > self.history_size:
            self.latency_history.pop(0)
    
    def get_stats(self) -> Dict[str, float]:
        """Get frame skip statistics."""
        return {
            'total_frames': self.frame_count,
            'skipped_frames': self.skip_count,
            'skip_rate': self.skip_count / max(1, self.frame_count),
            'avg_latency_ms': np.mean(self.latency_history) if self.latency_history else 0,
            'effective_fps': 1000 / np.mean(self.latency_history) if self.latency_history else 0
        }


class EdgeDeploymentManager:
    """
    Complete edge deployment manager.
    """
    
    def __init__(self, config: InferenceConfig = None):
        self.config = config or InferenceConfig()
        self.exporter = ModelExporter()
        self.engine = None
        self.batch_manager = None
        self.frame_skip = FrameSkipManager()
    
    def prepare_model(self, model_path: str, output_dir: str,
                     format: ExportFormat = ExportFormat.ONNX,
                     precision: PrecisionMode = PrecisionMode.FP16) -> str:
        """
        Prepare model for edge deployment.
        
        Args:
            model_path: Path to source model
            output_dir: Output directory
            format: Export format
            precision: Precision mode
            
        Returns:
            Path to optimized model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.exporter.config.precision = precision
        
        if format == ExportFormat.ONNX:
            onnx_path = str(output_path / 'model.onnx')
            return self.exporter.export_onnx(model_path, onnx_path)
        
        elif format == ExportFormat.TENSORRT:
            # First export to ONNX
            onnx_path = str(output_path / 'model.onnx')
            self.exporter.export_onnx(model_path, onnx_path)
            
            # Then convert to TensorRT
            trt_path = str(output_path / 'model.engine')
            return self.exporter.export_tensorrt(onnx_path, trt_path)
        
        elif format == ExportFormat.OPENVINO:
            # First export to ONNX
            onnx_path = str(output_path / 'model.onnx')
            self.exporter.export_onnx(model_path, onnx_path)
            
            # Then convert to OpenVINO
            return self.exporter.export_openvino(onnx_path, str(output_path))
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_engine(self, model_path: str, format: ExportFormat = ExportFormat.ONNX):
        """
        Load inference engine.
        
        Args:
            model_path: Path to optimized model
            format: Model format
        """
        if format == ExportFormat.ONNX:
            self.engine = ONNXInferenceEngine(model_path, self.config)
        elif format == ExportFormat.TENSORRT:
            self.engine = TensorRTInferenceEngine(model_path, self.config)
        else:
            raise ValueError(f"Unsupported format for inference: {format}")
        
        # Setup batch manager
        self.batch_manager = BatchInferenceManager(self.engine, self.config)
        self.batch_manager.start()
    
    def infer(self, image: np.ndarray, frame_id: int = 0) -> Optional[InferenceResult]:
        """
        Run inference with frame skipping.
        
        Args:
            image: Input image
            frame_id: Frame identifier
            
        Returns:
            InferenceResult or None if frame was skipped
        """
        if not self.frame_skip.should_process(frame_id):
            return None
        
        if self.batch_manager:
            request_id = self.batch_manager.submit(image, frame_id)
            result = self.batch_manager.get_result(request_id)
        else:
            outputs, latency = self.engine.infer_with_timing(image)
            detections = self._parse_detections(outputs)
            result = InferenceResult(
                detections=detections,
                latency_ms=latency,
                batch_size=1,
                frame_id=frame_id,
                timestamp=time.time()
            )
        
        self.frame_skip.record_latency(result.latency_ms)
        
        return result
    
    def _parse_detections(self, outputs: List[np.ndarray]) -> List[Dict]:
        """Parse detections from output."""
        detections = []
        
        if len(outputs) > 0:
            output = outputs[0]
            
            if len(output.shape) == 2:
                for det in output:
                    if det[4] > 0.25:
                        detections.append({
                            'bbox': det[:4].tolist(),
                            'confidence': float(det[4]),
                            'class_id': int(det[5]) if len(det) > 5 else 0
                        })
        
        return detections
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deployment statistics."""
        return {
            'frame_skip': self.frame_skip.get_stats(),
            'config': {
                'batch_size': self.config.batch_size,
                'backpressure_threshold': self.config.backpressure_threshold
            }
        }
    
    def shutdown(self):
        """Shutdown deployment manager."""
        if self.batch_manager:
            self.batch_manager.stop()


def create_edge_deployment(config: Optional[Dict] = None) -> EdgeDeploymentManager:
    """Factory function to create edge deployment manager."""
    if config:
        inf_config = InferenceConfig(
            batch_size=config.get('batch_size', 1),
            max_batch_wait_ms=config.get('max_batch_wait_ms', 10.0),
            frame_skip=config.get('frame_skip', 0),
            backpressure_threshold=config.get('backpressure_threshold', 100)
        )
    else:
        inf_config = InferenceConfig()
    
    return EdgeDeploymentManager(inf_config)
