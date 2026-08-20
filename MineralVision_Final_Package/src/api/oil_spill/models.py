"""Pluggable local model adapters for oil-spill semantic segmentation.

No checkpoint is bundled, downloaded, or trusted automatically. Operators must configure a
versioned local TorchScript or ONNX artifact before image inference is enabled. A caller can
always use the mask-assessment pathway with independently produced evidence.
"""

from __future__ import annotations

import hashlib
import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


class ModelNotConfiguredError(RuntimeError):
    """Raised when a deployment has not configured an approved model artifact."""


class ModelInferenceError(RuntimeError):
    """Raised when a configured model cannot produce a usable probability map."""


@dataclass(frozen=True)
class ModelDescriptor:
    """Identity of a model approved by the deployment operator."""

    model_id: str
    version: str
    engine: str
    path: Path
    artifact_sha256: str
    input_size: int = 512
    oil_class_index: int = 1


class OilSpillSegmentationModel(ABC):
    """Contract for a local image-to-oil-probability model adapter."""

    def __init__(self, descriptor: ModelDescriptor):
        self.descriptor = descriptor

    @abstractmethod
    def predict_probability(self, image_bytes: bytes) -> np.ndarray:
        """Return a normalized 2-D oil probability map at the image's native size."""

    @staticmethod
    def _load_image(image_bytes: bytes) -> Image.Image:
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                return image.convert("RGB").copy()
        except Exception as exc:
            raise ModelInferenceError("The uploaded image cannot be decoded as RGB imagery") from exc

    @staticmethod
    def _validate_probability_map(probability: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        if probability.ndim != 2:
            raise ModelInferenceError("Model output must resolve to a 2-D probability map")
        if not np.isfinite(probability).all():
            raise ModelInferenceError("Model output contains non-finite values")
        if probability.min() < 0 or probability.max() > 1:
            raise ModelInferenceError("Model output must be normalized to [0, 1]")
        width, height = target_size
        if probability.shape != (height, width):
            probability_image = Image.fromarray(np.round(probability * 255).astype(np.uint8), mode="L")
            probability = np.asarray(probability_image.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        return probability.astype(np.float32)


class TorchScriptOilSpillModel(OilSpillSegmentationModel):
    """Adapter for an operator-approved local TorchScript segmentation artifact."""

    def __init__(self, descriptor: ModelDescriptor):
        super().__init__(descriptor)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - torch is a project dependency
            raise ModelNotConfiguredError("PyTorch is required for a TorchScript oil-spill model") from exc

        self._torch = torch
        try:
            self._model = torch.jit.load(str(descriptor.path), map_location="cpu")
            self._model.eval()
        except Exception as exc:
            raise ModelNotConfiguredError(
                "Unable to load configured TorchScript model. Only deploy artifacts from a trusted model registry."
            ) from exc

    def predict_probability(self, image_bytes: bytes) -> np.ndarray:
        image = self._load_image(image_bytes)
        original_size = image.size
        normalized = np.asarray(
            image.resize((self.descriptor.input_size, self.descriptor.input_size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        tensor = self._torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)

        try:
            with self._torch.no_grad():
                output = self._model(tensor)
        except Exception as exc:
            raise ModelInferenceError("The configured TorchScript model failed during inference") from exc

        if isinstance(output, (tuple, list)):
            output = output[0]
        if not hasattr(output, "detach"):
            raise ModelInferenceError("TorchScript model returned an unsupported output type")
        array = output.detach().cpu().float().numpy()
        if array.ndim == 4:
            array = array[0]
        if array.ndim == 3:
            if array.shape[0] == 1:
                probability = 1 / (1 + np.exp(-array[0]))
            else:
                if not 0 <= self.descriptor.oil_class_index < array.shape[0]:
                    raise ModelInferenceError("oil_class_index is outside the model output channel range")
                stabilized = array - array.max(axis=0, keepdims=True)
                softmax = np.exp(stabilized) / np.exp(stabilized).sum(axis=0, keepdims=True)
                probability = softmax[self.descriptor.oil_class_index]
        elif array.ndim == 2:
            probability = 1 / (1 + np.exp(-array))
        else:
            raise ModelInferenceError("TorchScript model output must have 2, 3, or 4 dimensions")
        return self._validate_probability_map(probability, original_size)


class ONNXOilSpillModel(OilSpillSegmentationModel):
    """Adapter for an operator-approved local ONNX segmentation artifact."""

    def __init__(self, descriptor: ModelDescriptor):
        super().__init__(descriptor)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ModelNotConfiguredError("onnxruntime is required for an ONNX oil-spill model") from exc
        try:
            self._session = ort.InferenceSession(str(descriptor.path), providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
        except Exception as exc:
            raise ModelNotConfiguredError("Unable to load configured ONNX model") from exc

    def predict_probability(self, image_bytes: bytes) -> np.ndarray:
        image = self._load_image(image_bytes)
        original_size = image.size
        normalized = np.asarray(
            image.resize((self.descriptor.input_size, self.descriptor.input_size), Image.Resampling.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        try:
            array = np.asarray(self._session.run(None, {self._input_name: tensor})[0], dtype=np.float32)
        except Exception as exc:
            raise ModelInferenceError("The configured ONNX model failed during inference") from exc

        if array.ndim == 4:
            array = array[0]
        if array.ndim == 3:
            if array.shape[0] == 1:
                probability = 1 / (1 + np.exp(-array[0]))
            else:
                if not 0 <= self.descriptor.oil_class_index < array.shape[0]:
                    raise ModelInferenceError("oil_class_index is outside the model output channel range")
                stabilized = array - array.max(axis=0, keepdims=True)
                softmax = np.exp(stabilized) / np.exp(stabilized).sum(axis=0, keepdims=True)
                probability = softmax[self.descriptor.oil_class_index]
        elif array.ndim == 2:
            probability = 1 / (1 + np.exp(-array))
        else:
            raise ModelInferenceError("ONNX model output must have 2, 3, or 4 dimensions")
        return self._validate_probability_map(probability, original_size)


def model_from_environment() -> OilSpillSegmentationModel:
    """Load the locally configured model or fail closed when none is configured.

    Environment variables:
    - OIL_SPILL_MODEL_PATH: absolute or deployment-controlled relative local model path.
    - OIL_SPILL_MODEL_ENGINE: `torchscript` or `onnx`.
    - OIL_SPILL_MODEL_ID / OIL_SPILL_MODEL_VERSION: mandatory provenance labels.
    - OIL_SPILL_MODEL_SHA256: mandatory SHA-256 of the approved local artifact.
    - OIL_SPILL_MODEL_INPUT_SIZE: square model input size (default 512).
    - OIL_SPILL_OIL_CLASS_INDEX: output channel representing oil (default 1).
    """
    model_path_value = os.getenv("OIL_SPILL_MODEL_PATH")
    if not model_path_value:
        raise ModelNotConfiguredError(
            "Image inference is disabled. Configure a trusted local model with OIL_SPILL_MODEL_PATH, "
            "or use /api/oil-spill/analyze/mask for independently generated mask evidence."
        )

    path = Path(model_path_value).expanduser().resolve()
    if not path.is_file():
        raise ModelNotConfiguredError("OIL_SPILL_MODEL_PATH does not point to a readable local file")
    model_id = os.getenv("OIL_SPILL_MODEL_ID")
    model_version = os.getenv("OIL_SPILL_MODEL_VERSION")
    configured_sha256 = os.getenv("OIL_SPILL_MODEL_SHA256")
    if not model_id or not model_version or not configured_sha256:
        raise ModelNotConfiguredError(
            "OIL_SPILL_MODEL_ID, OIL_SPILL_MODEL_VERSION, and OIL_SPILL_MODEL_SHA256 are required "
            "to enable raw-image inference."
        )
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256.lower() != configured_sha256.lower():
        raise ModelNotConfiguredError(
            "Configured model hash does not match OIL_SPILL_MODEL_PATH. Refusing to load an unverified artifact."
        )
    engine = os.getenv("OIL_SPILL_MODEL_ENGINE", "torchscript").strip().lower()
    descriptor = ModelDescriptor(
        model_id=model_id,
        version=model_version,
        engine=engine,
        path=path,
        artifact_sha256=actual_sha256,
        input_size=int(os.getenv("OIL_SPILL_MODEL_INPUT_SIZE", "512")),
        oil_class_index=int(os.getenv("OIL_SPILL_OIL_CLASS_INDEX", "1")),
    )
    if descriptor.input_size <= 0:
        raise ModelNotConfiguredError("OIL_SPILL_MODEL_INPUT_SIZE must be positive")
    if descriptor.oil_class_index < 0:
        raise ModelNotConfiguredError("OIL_SPILL_OIL_CLASS_INDEX cannot be negative")
    if engine == "torchscript":
        return TorchScriptOilSpillModel(descriptor)
    if engine == "onnx":
        return ONNXOilSpillModel(descriptor)
    raise ModelNotConfiguredError("OIL_SPILL_MODEL_ENGINE must be either 'torchscript' or 'onnx'")
