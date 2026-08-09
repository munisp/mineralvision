"""
Molmo2 Optimization Module for MineralVision.

Provides domain-specific optimization for highest accuracy in
mining/geological exploration applications.

Key Features:
- Structured JSON output schemas (no regex parsing)
- LoRA/QLoRA fine-tuning pipeline
- Domain-specific prompt templates
- Multi-adapter architecture
- Ensemble optimization with YOLO11/RF-DETR/SAM3/V-JEPA
- Mining-specific evaluation metrics
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import threading

logger = logging.getLogger(__name__)


# =============================================================================
# STRUCTURED OUTPUT SCHEMAS
# =============================================================================

class OutputSchema(Enum):
    """Predefined output schemas for structured responses."""
    SCENE_ANALYSIS = "scene_analysis"
    OBJECT_DETECTION = "object_detection"
    GEOLOGICAL_FEATURES = "geological_features"
    ARTISANAL_MINING = "artisanal_mining"
    CHANGE_DETECTION = "change_detection"
    ENVIRONMENTAL = "environmental"
    TRACKING = "tracking"


# JSON Schemas for structured outputs
STRUCTURED_SCHEMAS = {
    OutputSchema.SCENE_ANALYSIS: {
        "type": "object",
        "required": ["scene_type", "confidence", "description"],
        "properties": {
            "scene_type": {
                "type": "string",
                "enum": ["open_pit", "underground", "exploration", "processing", 
                        "drill_site", "tailings", "stockpile", "haul_road", 
                        "camp", "vegetation", "water_body", "unknown"]
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "description": {"type": "string"},
            "activities": {
                "type": "array",
                "items": {"type": "string"}
            },
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "bbox": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4
                        },
                        "confidence": {"type": "number"}
                    }
                }
            },
            "safety_flags": {
                "type": "array",
                "items": {"type": "string"}
            },
            "rationale": {"type": "string"}
        }
    },
    
    OutputSchema.GEOLOGICAL_FEATURES: {
        "type": "object",
        "required": ["features", "confidence"],
        "properties": {
            "features": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["type", "confidence"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["outcrop", "fault", "fold", "vein", 
                                    "alteration_zone", "gossan", "contact",
                                    "intrusion", "sedimentary_layer", "alluvial"]
                        },
                        "confidence": {"type": "number"},
                        "bbox": {"type": "array"},
                        "description": {"type": "string"},
                        "mineral_indicators": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "color_anomalies": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "structural_notes": {"type": "string"}
                    }
                }
            },
            "confidence": {"type": "number"},
            "rock_types": {
                "type": "array",
                "items": {"type": "string"}
            },
            "mineralization_potential": {
                "type": "string",
                "enum": ["low", "medium", "high", "very_high"]
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    },
    
    OutputSchema.ARTISANAL_MINING: {
        "type": "object",
        "required": ["is_artisanal_mining", "confidence"],
        "properties": {
            "is_artisanal_mining": {"type": "boolean"},
            "confidence": {"type": "number"},
            "activity_type": {
                "type": "string",
                "enum": ["excavation", "panning", "sluicing", "processing",
                        "transport", "camp", "none", "unknown"]
            },
            "indicators": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "bbox": {"type": "array"},
                        "confidence": {"type": "number"},
                        "description": {"type": "string"}
                    }
                }
            },
            "people_count": {"type": "integer"},
            "equipment": {
                "type": "array",
                "items": {"type": "string"}
            },
            "environmental_impact": {
                "type": "string",
                "enum": ["none", "low", "medium", "high", "severe"]
            },
            "safety_concerns": {
                "type": "array",
                "items": {"type": "string"}
            },
            "rationale": {"type": "string"}
        }
    },
    
    OutputSchema.CHANGE_DETECTION: {
        "type": "object",
        "required": ["has_changes", "confidence"],
        "properties": {
            "has_changes": {"type": "boolean"},
            "confidence": {"type": "number"},
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["new_excavation", "vegetation_change",
                                    "new_structure", "water_change",
                                    "road_modification", "equipment_change",
                                    "erosion", "other"]
                        },
                        "magnitude": {
                            "type": "string",
                            "enum": ["small", "medium", "large"]
                        },
                        "significance": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "critical"]
                        },
                        "bbox": {"type": "array"},
                        "description": {"type": "string"}
                    }
                }
            },
            "overall_change_magnitude": {"type": "number"},
            "recommendations": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    },
    
    OutputSchema.ENVIRONMENTAL: {
        "type": "object",
        "required": ["assessment", "confidence"],
        "properties": {
            "assessment": {
                "type": "string",
                "enum": ["healthy", "minor_concerns", "moderate_impact",
                        "significant_impact", "severe_degradation"]
            },
            "confidence": {"type": "number"},
            "vegetation": {
                "type": "object",
                "properties": {
                    "health": {"type": "string"},
                    "coverage_percent": {"type": "number"},
                    "changes": {"type": "string"}
                }
            },
            "water": {
                "type": "object",
                "properties": {
                    "present": {"type": "boolean"},
                    "quality_indicators": {"type": "string"},
                    "concerns": {"type": "array", "items": {"type": "string"}}
                }
            },
            "erosion": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string"},
                    "areas": {"type": "array", "items": {"type": "string"}}
                }
            },
            "pollution_indicators": {
                "type": "array",
                "items": {"type": "string"}
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"}
            }
        }
    },
    
    OutputSchema.TRACKING: {
        "type": "object",
        "required": ["tracks", "confidence"],
        "properties": {
            "tracks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "track_id": {"type": "string"},
                        "object_class": {"type": "string"},
                        "positions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "frame": {"type": "integer"},
                                    "x": {"type": "number"},
                                    "y": {"type": "number"},
                                    "bbox": {"type": "array"}
                                }
                            }
                        },
                        "first_frame": {"type": "integer"},
                        "last_frame": {"type": "integer"},
                        "confidence": {"type": "number"}
                    }
                }
            },
            "confidence": {"type": "number"},
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "frame": {"type": "integer"},
                        "description": {"type": "string"}
                    }
                }
            }
        }
    }
}


@dataclass
class StructuredOutput:
    """Structured output from Molmo2 with schema validation."""
    schema_type: OutputSchema
    data: Dict[str, Any]
    raw_response: str
    confidence: float
    is_valid: bool
    validation_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_type": self.schema_type.value,
            "data": self.data,
            "confidence": self.confidence,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
        }


class StructuredOutputParser:
    """Parser for structured JSON outputs from Molmo2."""
    
    def __init__(self):
        self.schemas = STRUCTURED_SCHEMAS
    
    def parse(
        self,
        response: str,
        schema_type: OutputSchema,
    ) -> StructuredOutput:
        """Parse and validate structured output."""
        # Try to extract JSON from response
        json_data = self._extract_json(response)
        
        if json_data is None:
            return StructuredOutput(
                schema_type=schema_type,
                data={},
                raw_response=response,
                confidence=0.0,
                is_valid=False,
                validation_errors=["Failed to extract JSON from response"],
            )
        
        # Validate against schema
        is_valid, errors = self._validate_schema(json_data, schema_type)
        
        # Extract confidence
        confidence = json_data.get("confidence", 0.5)
        
        return StructuredOutput(
            schema_type=schema_type,
            data=json_data,
            raw_response=response,
            confidence=confidence,
            is_valid=is_valid,
            validation_errors=errors,
        )
    
    def _extract_json(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from response text."""
        import re
        
        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON block in markdown
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _validate_schema(
        self,
        data: Dict[str, Any],
        schema_type: OutputSchema,
    ) -> Tuple[bool, List[str]]:
        """Validate data against schema."""
        schema = self.schemas.get(schema_type)
        if not schema:
            return True, []
        
        errors = []
        
        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Basic type validation
        properties = schema.get("properties", {})
        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type:
                    if not self._check_type(value, expected_type):
                        errors.append(f"Invalid type for {field}: expected {expected_type}")
        
        return len(errors) == 0, errors
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)
        return True


# =============================================================================
# DOMAIN-SPECIFIC PROMPT TEMPLATES
# =============================================================================

class PromptTemplate:
    """Domain-specific prompt template."""
    
    def __init__(
        self,
        name: str,
        template: str,
        schema_type: OutputSchema,
        variables: List[str] = None,
    ):
        self.name = name
        self.template = template
        self.schema_type = schema_type
        self.variables = variables or []
    
    def format(self, **kwargs) -> str:
        """Format template with variables."""
        return self.template.format(**kwargs)


# Mining/Geological Domain Prompts
DOMAIN_PROMPTS = {
    "scene_analysis": PromptTemplate(
        name="scene_analysis",
        template="""Analyze this mining/exploration scene image.

You MUST respond with a valid JSON object following this exact structure:
{{
    "scene_type": "<one of: open_pit, underground, exploration, processing, drill_site, tailings, stockpile, haul_road, camp, vegetation, water_body, unknown>",
    "confidence": <0.0-1.0>,
    "description": "<brief description>",
    "activities": ["<activity1>", "<activity2>"],
    "objects": [
        {{"label": "<object>", "bbox": [x1, y1, x2, y2], "confidence": <0.0-1.0>}}
    ],
    "safety_flags": ["<concern1>", "<concern2>"],
    "rationale": "<explanation of classification>"
}}

{context}

Analyze the image and provide your response as valid JSON only.""",
        schema_type=OutputSchema.SCENE_ANALYSIS,
        variables=["context"],
    ),
    
    "geological_features": PromptTemplate(
        name="geological_features",
        template="""Identify geological features in this image for mineral exploration.

You MUST respond with a valid JSON object following this exact structure:
{{
    "features": [
        {{
            "type": "<one of: outcrop, fault, fold, vein, alteration_zone, gossan, contact, intrusion, sedimentary_layer, alluvial>",
            "confidence": <0.0-1.0>,
            "bbox": [x1, y1, x2, y2],
            "description": "<description>",
            "mineral_indicators": ["<indicator1>", "<indicator2>"],
            "color_anomalies": ["<color1>", "<color2>"],
            "structural_notes": "<notes>"
        }}
    ],
    "confidence": <0.0-1.0>,
    "rock_types": ["<rock1>", "<rock2>"],
    "mineralization_potential": "<one of: low, medium, high, very_high>",
    "recommendations": ["<rec1>", "<rec2>"]
}}

Target minerals: {target_minerals}
{context}

Analyze the image and provide your response as valid JSON only.""",
        schema_type=OutputSchema.GEOLOGICAL_FEATURES,
        variables=["target_minerals", "context"],
    ),
    
    "artisanal_mining": PromptTemplate(
        name="artisanal_mining",
        template="""Detect artisanal/small-scale mining activity in this image.

You MUST respond with a valid JSON object following this exact structure:
{{
    "is_artisanal_mining": <true/false>,
    "confidence": <0.0-1.0>,
    "activity_type": "<one of: excavation, panning, sluicing, processing, transport, camp, none, unknown>",
    "indicators": [
        {{
            "type": "<indicator type>",
            "bbox": [x1, y1, x2, y2],
            "confidence": <0.0-1.0>,
            "description": "<description>"
        }}
    ],
    "people_count": <integer>,
    "equipment": ["<equipment1>", "<equipment2>"],
    "environmental_impact": "<one of: none, low, medium, high, severe>",
    "safety_concerns": ["<concern1>", "<concern2>"],
    "rationale": "<explanation>"
}}

Known detector results: {detector_context}
{context}

IMPORTANT: Distinguish artisanal mining from similar activities (construction, farming, logging).
Analyze the image and provide your response as valid JSON only.""",
        schema_type=OutputSchema.ARTISANAL_MINING,
        variables=["detector_context", "context"],
    ),
    
    "change_detection": PromptTemplate(
        name="change_detection",
        template="""Compare these images to detect site changes over time.

You MUST respond with a valid JSON object following this exact structure:
{{
    "has_changes": <true/false>,
    "confidence": <0.0-1.0>,
    "changes": [
        {{
            "type": "<one of: new_excavation, vegetation_change, new_structure, water_change, road_modification, equipment_change, erosion, other>",
            "magnitude": "<one of: small, medium, large>",
            "significance": "<one of: low, medium, high, critical>",
            "bbox": [x1, y1, x2, y2],
            "description": "<description>"
        }}
    ],
    "overall_change_magnitude": <0.0-1.0>,
    "recommendations": ["<rec1>", "<rec2>"]
}}

Time between images: {time_delta}
{context}

IMPORTANT: Distinguish real changes from lighting/seasonal variations.
Analyze the images and provide your response as valid JSON only.""",
        schema_type=OutputSchema.CHANGE_DETECTION,
        variables=["time_delta", "context"],
    ),
    
    "environmental": PromptTemplate(
        name="environmental",
        template="""Assess environmental conditions in this mining/exploration area.

You MUST respond with a valid JSON object following this exact structure:
{{
    "assessment": "<one of: healthy, minor_concerns, moderate_impact, significant_impact, severe_degradation>",
    "confidence": <0.0-1.0>,
    "vegetation": {{
        "health": "<description>",
        "coverage_percent": <0-100>,
        "changes": "<description>"
    }},
    "water": {{
        "present": <true/false>,
        "quality_indicators": "<description>",
        "concerns": ["<concern1>", "<concern2>"]
    }},
    "erosion": {{
        "severity": "<none/minor/moderate/severe>",
        "areas": ["<area1>", "<area2>"]
    }},
    "pollution_indicators": ["<indicator1>", "<indicator2>"],
    "recommendations": ["<rec1>", "<rec2>"]
}}

{context}

Analyze the image and provide your response as valid JSON only.""",
        schema_type=OutputSchema.ENVIRONMENTAL,
        variables=["context"],
    ),
    
    "gold_exploration": PromptTemplate(
        name="gold_exploration",
        template="""Analyze this image for gold exploration indicators.

You MUST respond with a valid JSON object following this exact structure:
{{
    "features": [
        {{
            "type": "<feature type>",
            "confidence": <0.0-1.0>,
            "bbox": [x1, y1, x2, y2],
            "description": "<description>",
            "mineral_indicators": ["<indicator>"],
            "color_anomalies": ["<color>"],
            "structural_notes": "<notes>"
        }}
    ],
    "confidence": <0.0-1.0>,
    "rock_types": ["<rock>"],
    "mineralization_potential": "<one of: low, medium, high, very_high>",
    "recommendations": ["<rec>"]
}}

Focus on gold indicators:
- Quartz veins (white, glassy)
- Gossans (iron-stained caps, red/orange/brown)
- Alteration zones (color changes, bleaching)
- Structural controls (faults, shear zones)
- Alluvial deposits (stream channels, terraces)
- Sulfide minerals (pyrite, arsenopyrite)

Deposit type: {deposit_type}
{context}

Analyze the image and provide your response as valid JSON only.""",
        schema_type=OutputSchema.GEOLOGICAL_FEATURES,
        variables=["deposit_type", "context"],
    ),
    
    "lithium_exploration": PromptTemplate(
        name="lithium_exploration",
        template="""Analyze this image for lithium exploration indicators.

You MUST respond with a valid JSON object following this exact structure:
{{
    "features": [
        {{
            "type": "<feature type>",
            "confidence": <0.0-1.0>,
            "bbox": [x1, y1, x2, y2],
            "description": "<description>",
            "mineral_indicators": ["<indicator>"],
            "color_anomalies": ["<color>"],
            "structural_notes": "<notes>"
        }}
    ],
    "confidence": <0.0-1.0>,
    "rock_types": ["<rock>"],
    "mineralization_potential": "<one of: low, medium, high, very_high>",
    "recommendations": ["<rec>"]
}}

Focus on lithium indicators:
- Pegmatites (coarse-grained, white/pink feldspar)
- Spodumene crystals (elongated, greenish)
- Clay deposits (white/cream colored)
- Brine pools (salt flats, evaporites)
- Alteration halos around pegmatites

Deposit type: {deposit_type}
{context}

Analyze the image and provide your response as valid JSON only.""",
        schema_type=OutputSchema.GEOLOGICAL_FEATURES,
        variables=["deposit_type", "context"],
    ),
}


class PromptLibrary:
    """Library of domain-specific prompts."""
    
    def __init__(self):
        self.prompts = DOMAIN_PROMPTS.copy()
        self.custom_prompts: Dict[str, PromptTemplate] = {}
    
    def get_prompt(self, name: str) -> Optional[PromptTemplate]:
        """Get prompt template by name."""
        return self.prompts.get(name) or self.custom_prompts.get(name)
    
    def add_custom_prompt(self, prompt: PromptTemplate) -> None:
        """Add custom prompt template."""
        self.custom_prompts[prompt.name] = prompt
    
    def list_prompts(self) -> List[str]:
        """List all available prompts."""
        return list(self.prompts.keys()) + list(self.custom_prompts.keys())
    
    def format_prompt(self, name: str, **kwargs) -> str:
        """Format prompt with variables."""
        prompt = self.get_prompt(name)
        if prompt:
            # Set defaults for missing variables
            for var in prompt.variables:
                if var not in kwargs:
                    kwargs[var] = ""
            return prompt.format(**kwargs)
        raise ValueError(f"Unknown prompt: {name}")


# =============================================================================
# LORA/QLORA FINE-TUNING PIPELINE
# =============================================================================

@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    r: int = 16  # LoRA rank
    lora_alpha: int = 32  # LoRA alpha
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class QLoRAConfig(LoRAConfig):
    """Configuration for QLoRA (quantized LoRA) fine-tuning."""
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning training."""
    output_dir: str = "./molmo2_finetuned"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = True
    max_grad_norm: float = 0.3
    optim: str = "paged_adamw_32bit"
    gradient_checkpointing: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "lr_scheduler_type": self.lr_scheduler_type,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "bf16": self.bf16,
            "optim": self.optim,
            "gradient_checkpointing": self.gradient_checkpointing,
        }


@dataclass
class TrainingExample:
    """Single training example for fine-tuning."""
    image_path: str
    prompt: str
    response: str  # Expected JSON response
    schema_type: OutputSchema
    metadata: Dict[str, Any] = field(default_factory=dict)


class FineTuningDataset:
    """Dataset for Molmo2 fine-tuning."""
    
    def __init__(self, examples: List[TrainingExample] = None):
        self.examples = examples or []
    
    def add_example(self, example: TrainingExample) -> None:
        """Add training example."""
        self.examples.append(example)
    
    def add_from_json(self, json_path: str) -> None:
        """Load examples from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        for item in data:
            self.examples.append(TrainingExample(
                image_path=item["image_path"],
                prompt=item["prompt"],
                response=item["response"],
                schema_type=OutputSchema(item.get("schema_type", "scene_analysis")),
                metadata=item.get("metadata", {}),
            ))
    
    def save_to_json(self, json_path: str) -> None:
        """Save examples to JSON file."""
        data = []
        for ex in self.examples:
            data.append({
                "image_path": ex.image_path,
                "prompt": ex.prompt,
                "response": ex.response,
                "schema_type": ex.schema_type.value,
                "metadata": ex.metadata,
            })
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
    ) -> Tuple["FineTuningDataset", "FineTuningDataset", "FineTuningDataset"]:
        """Split dataset into train/val/test."""
        import random
        
        examples = self.examples.copy()
        random.shuffle(examples)
        
        n = len(examples)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return (
            FineTuningDataset(examples[:train_end]),
            FineTuningDataset(examples[train_end:val_end]),
            FineTuningDataset(examples[val_end:]),
        )
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> TrainingExample:
        return self.examples[idx]


class Molmo2FineTuner:
    """Fine-tuning pipeline for Molmo2 with LoRA/QLoRA."""
    
    def __init__(
        self,
        model_name: str = "allenai/Molmo2-8B",
        lora_config: Optional[LoRAConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        use_qlora: bool = True,
    ):
        self.model_name = model_name
        self.lora_config = lora_config or (QLoRAConfig() if use_qlora else LoRAConfig())
        self.training_config = training_config or TrainingConfig()
        self.use_qlora = use_qlora
        
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._peft_model = None
    
    def prepare_model(self) -> None:
        """Prepare model for fine-tuning."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            logger.info(f"Loading model: {self.model_name}")
            
            # Quantization config for QLoRA
            if self.use_qlora:
                from transformers import BitsAndBytesConfig
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                
                self._model = prepare_model_for_kbit_training(self._model)
            else:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            
            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            
            # Apply LoRA
            peft_config = LoraConfig(
                r=self.lora_config.r,
                lora_alpha=self.lora_config.lora_alpha,
                lora_dropout=self.lora_config.lora_dropout,
                target_modules=self.lora_config.target_modules,
                bias=self.lora_config.bias,
                task_type=self.lora_config.task_type,
            )
            
            self._peft_model = get_peft_model(self._model, peft_config)
            
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in self._peft_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self._peft_model.parameters())
            logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
            
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            raise ImportError(
                "Fine-tuning requires: pip install transformers peft bitsandbytes accelerate"
            )
    
    def train(
        self,
        train_dataset: FineTuningDataset,
        eval_dataset: Optional[FineTuningDataset] = None,
    ) -> Dict[str, Any]:
        """Run fine-tuning training."""
        try:
            from transformers import Trainer, TrainingArguments
            
            if self._peft_model is None:
                self.prepare_model()
            
            # Prepare training arguments
            training_args = TrainingArguments(
                **self.training_config.to_dict()
            )
            
            # Create trainer
            trainer = Trainer(
                model=self._peft_model,
                args=training_args,
                train_dataset=self._prepare_hf_dataset(train_dataset),
                eval_dataset=self._prepare_hf_dataset(eval_dataset) if eval_dataset else None,
            )
            
            # Train
            logger.info("Starting fine-tuning...")
            train_result = trainer.train()
            
            # Save model
            self._peft_model.save_pretrained(self.training_config.output_dir)
            self._processor.save_pretrained(self.training_config.output_dir)
            
            logger.info(f"Model saved to {self.training_config.output_dir}")
            
            return {
                "train_loss": train_result.training_loss,
                "train_runtime": train_result.metrics.get("train_runtime"),
                "output_dir": self.training_config.output_dir,
            }
            
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            raise
    
    def _prepare_hf_dataset(self, dataset: FineTuningDataset):
        """Convert to HuggingFace dataset format."""
        from datasets import Dataset
        
        data = {
            "image_path": [],
            "prompt": [],
            "response": [],
        }
        
        for ex in dataset.examples:
            data["image_path"].append(ex.image_path)
            data["prompt"].append(ex.prompt)
            data["response"].append(ex.response)
        
        return Dataset.from_dict(data)
    
    def load_adapter(self, adapter_path: str) -> None:
        """Load a trained LoRA adapter."""
        try:
            from peft import PeftModel
            
            if self._model is None:
                self.prepare_model()
            
            self._peft_model = PeftModel.from_pretrained(
                self._model,
                adapter_path,
            )
            logger.info(f"Loaded adapter from {adapter_path}")
            
        except Exception as e:
            logger.error(f"Failed to load adapter: {e}")
            raise


# =============================================================================
# MULTI-ADAPTER ARCHITECTURE
# =============================================================================

class AdapterType(Enum):
    """Types of domain-specific adapters."""
    MINING_OPS = "mining_ops"
    GEOLOGY = "geology"
    ENVIRONMENT = "environment"
    CHANGE_DETECTION = "change_detection"
    GOLD = "gold"
    LITHIUM = "lithium"


@dataclass
class AdapterInfo:
    """Information about a trained adapter."""
    adapter_type: AdapterType
    path: str
    version: str
    trained_on: str
    metrics: Dict[str, float] = field(default_factory=dict)


class MultiAdapterManager:
    """Manager for multiple domain-specific LoRA adapters."""
    
    def __init__(self, base_model_name: str = "allenai/Molmo2-8B"):
        self.base_model_name = base_model_name
        self.adapters: Dict[AdapterType, AdapterInfo] = {}
        self.active_adapter: Optional[AdapterType] = None
        
        self._model = None
        self._processor = None
    
    def register_adapter(self, adapter_info: AdapterInfo) -> None:
        """Register a trained adapter."""
        self.adapters[adapter_info.adapter_type] = adapter_info
        logger.info(f"Registered adapter: {adapter_info.adapter_type.value}")
    
    def load_adapter(self, adapter_type: AdapterType) -> None:
        """Load and activate a specific adapter."""
        if adapter_type not in self.adapters:
            raise ValueError(f"Adapter not registered: {adapter_type}")
        
        adapter_info = self.adapters[adapter_type]
        
        try:
            from peft import PeftModel
            
            if self._model is None:
                self._load_base_model()
            
            # Load adapter
            self._model = PeftModel.from_pretrained(
                self._model,
                adapter_info.path,
                adapter_name=adapter_type.value,
            )
            
            self.active_adapter = adapter_type
            logger.info(f"Activated adapter: {adapter_type.value}")
            
        except Exception as e:
            logger.error(f"Failed to load adapter: {e}")
            raise
    
    def switch_adapter(self, adapter_type: AdapterType) -> None:
        """Switch to a different adapter."""
        if adapter_type == self.active_adapter:
            return
        
        if adapter_type not in self.adapters:
            raise ValueError(f"Adapter not registered: {adapter_type}")
        
        try:
            self._model.set_adapter(adapter_type.value)
            self.active_adapter = adapter_type
            logger.info(f"Switched to adapter: {adapter_type.value}")
        except Exception as e:
            logger.error(f"Failed to switch adapter: {e}")
            raise
    
    def _load_base_model(self) -> None:
        """Load base model."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            self._processor = AutoProcessor.from_pretrained(
                self.base_model_name,
                trust_remote_code=True,
            )
            
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            raise
    
    def get_adapter_for_task(self, task: str) -> AdapterType:
        """Get recommended adapter for a task."""
        task_adapter_map = {
            "artisanal_mining": AdapterType.MINING_OPS,
            "scene_analysis": AdapterType.MINING_OPS,
            "geological_features": AdapterType.GEOLOGY,
            "gold_exploration": AdapterType.GOLD,
            "lithium_exploration": AdapterType.LITHIUM,
            "environmental": AdapterType.ENVIRONMENT,
            "change_detection": AdapterType.CHANGE_DETECTION,
        }
        
        return task_adapter_map.get(task, AdapterType.MINING_OPS)
    
    def list_adapters(self) -> List[Dict[str, Any]]:
        """List all registered adapters."""
        return [
            {
                "type": info.adapter_type.value,
                "path": info.path,
                "version": info.version,
                "metrics": info.metrics,
            }
            for info in self.adapters.values()
        ]
