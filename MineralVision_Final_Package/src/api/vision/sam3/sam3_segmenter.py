"""
SAM3 Segmenter for MineralVision

Meta's Segment Anything Model 3 integration for geology, mining, 
geospatial, and geophysics applications.

Features:
- Text-based concept segmentation
- Image exemplar prompts
- Video tracking
- Domain-specific fine-tuning support
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import numpy as np

logger = logging.getLogger(__name__)

# Optional SAM3 imports with graceful fallback
try:
    import torch
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. SAM3 features disabled.")

try:
    from sam3 import SAM3Predictor, SAM3VideoPredictor
    from sam3.build_sam3 import build_sam3
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    logger.info("SAM3 not installed. Install with: pip install sam3")


class Modality(str, Enum):
    """Supported imaging modalities for geology/mining."""
    DRILLCORE = "drillcore"
    THIN_SECTION = "thin_section"
    UAV_ORTHO = "uav_ortho"
    SATELLITE = "satellite"
    GEOPHYSICS = "geophysics"
    OUTCROP = "outcrop"
    SOIL = "soil"
    SOIL_PIT = "soil_pit"


@dataclass
class GeologyConcept:
    """Domain-specific concept for segmentation."""
    name: str
    modality: Modality
    text_prompts: List[str]
    description: str = ""
    parent_concept: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)


# Domain-specific concept vocabulary for geology/mining
GEOLOGY_CONCEPTS: Dict[str, GeologyConcept] = {
    # Drillcore concepts
    "vein": GeologyConcept(
        name="vein",
        modality=Modality.DRILLCORE,
        text_prompts=["quartz vein", "vein", "veinlet", "stockwork vein"],
        description="Mineral-filled fracture in rock",
        synonyms=["veinlet", "stringer", "stockwork"]
    ),
    "sulfide_zone": GeologyConcept(
        name="sulfide_zone",
        modality=Modality.DRILLCORE,
        text_prompts=["sulfide minerals", "pyrite zone", "sulfide-rich zone", "massive sulfide"],
        description="Zone rich in sulfide minerals",
        synonyms=["sulfide band", "sulfide interval"]
    ),
    "alteration_halo": GeologyConcept(
        name="alteration_halo",
        modality=Modality.DRILLCORE,
        text_prompts=["alteration zone", "altered rock", "hydrothermal alteration"],
        description="Hydrothermally altered rock surrounding mineralization",
        synonyms=["alteration envelope", "alteration aureole"]
    ),
    "oxidation_boundary": GeologyConcept(
        name="oxidation_boundary",
        modality=Modality.DRILLCORE,
        text_prompts=["oxidation front", "oxide zone boundary", "redox boundary"],
        description="Boundary between oxidized and fresh rock",
    ),
    "lithology_contact": GeologyConcept(
        name="lithology_contact",
        modality=Modality.DRILLCORE,
        text_prompts=["rock contact", "lithology boundary", "geological contact"],
        description="Contact between different rock types",
    ),
    "fracture_zone": GeologyConcept(
        name="fracture_zone",
        modality=Modality.DRILLCORE,
        text_prompts=["fractured rock", "fracture zone", "broken core"],
        description="Intensely fractured interval",
    ),
    "core_loss": GeologyConcept(
        name="core_loss",
        modality=Modality.DRILLCORE,
        text_prompts=["core loss", "missing core", "void in core tray"],
        description="Missing core interval",
    ),
    
    # Thin section concepts
    "quartz": GeologyConcept(
        name="quartz",
        modality=Modality.THIN_SECTION,
        text_prompts=["quartz grain", "quartz crystal", "silica"],
        description="Quartz mineral phase",
    ),
    "feldspar": GeologyConcept(
        name="feldspar",
        modality=Modality.THIN_SECTION,
        text_prompts=["feldspar", "plagioclase", "k-feldspar", "orthoclase"],
        description="Feldspar mineral group",
        synonyms=["plagioclase", "orthoclase", "microcline"]
    ),
    "pyrite": GeologyConcept(
        name="pyrite",
        modality=Modality.THIN_SECTION,
        text_prompts=["pyrite", "iron sulfide", "fool's gold"],
        description="Iron sulfide mineral",
    ),
    "chalcopyrite": GeologyConcept(
        name="chalcopyrite",
        modality=Modality.THIN_SECTION,
        text_prompts=["chalcopyrite", "copper iron sulfide", "yellow copper ore"],
        description="Copper iron sulfide mineral",
    ),
    "arsenopyrite": GeologyConcept(
        name="arsenopyrite",
        modality=Modality.THIN_SECTION,
        text_prompts=["arsenopyrite", "iron arsenic sulfide"],
        description="Iron arsenic sulfide mineral",
    ),
    "mica": GeologyConcept(
        name="mica",
        modality=Modality.THIN_SECTION,
        text_prompts=["mica", "biotite", "muscovite", "sheet silicate"],
        description="Mica mineral group",
        synonyms=["biotite", "muscovite", "sericite"]
    ),
    
    # UAV/Satellite concepts
    "lineament": GeologyConcept(
        name="lineament",
        modality=Modality.UAV_ORTHO,
        text_prompts=["lineament", "linear feature", "geological lineament"],
        description="Linear geological feature visible from above",
    ),
    "fault_trace": GeologyConcept(
        name="fault_trace",
        modality=Modality.UAV_ORTHO,
        text_prompts=["fault", "fault trace", "fault line", "fracture trace"],
        description="Surface expression of a fault",
    ),
    "gossan": GeologyConcept(
        name="gossan",
        modality=Modality.UAV_ORTHO,
        text_prompts=["gossan", "iron cap", "oxidized outcrop", "rusty outcrop"],
        description="Oxidized surface expression of sulfide mineralization",
    ),
    "artisanal_working": GeologyConcept(
        name="artisanal_working",
        modality=Modality.UAV_ORTHO,
        text_prompts=["artisanal mining", "small scale mining", "pit", "excavation"],
        description="Artisanal or small-scale mining activity",
    ),
    "tailings": GeologyConcept(
        name="tailings",
        modality=Modality.UAV_ORTHO,
        text_prompts=["tailings", "mine waste", "tailings dam", "waste pile"],
        description="Mine waste material",
    ),
    
    # Geophysics concepts
    "magnetic_high": GeologyConcept(
        name="magnetic_high",
        modality=Modality.GEOPHYSICS,
        text_prompts=["magnetic anomaly", "magnetic high", "positive magnetic anomaly"],
        description="Positive magnetic anomaly",
    ),
    "magnetic_low": GeologyConcept(
        name="magnetic_low",
        modality=Modality.GEOPHYSICS,
        text_prompts=["magnetic low", "negative magnetic anomaly", "demagnetized zone"],
        description="Negative magnetic anomaly",
    ),
    "gravity_anomaly": GeologyConcept(
        name="gravity_anomaly",
        modality=Modality.GEOPHYSICS,
        text_prompts=["gravity anomaly", "density anomaly", "bouguer anomaly"],
        description="Gravity anomaly indicating density contrast",
    ),
    "conductivity_high": GeologyConcept(
        name="conductivity_high",
        modality=Modality.GEOPHYSICS,
        text_prompts=["conductive zone", "conductivity anomaly", "EM anomaly"],
        description="High conductivity zone from EM surveys",
    ),
    "ip_chargeability": GeologyConcept(
        name="ip_chargeability",
        modality=Modality.GEOPHYSICS,
        text_prompts=["chargeability anomaly", "IP anomaly", "induced polarization high"],
        description="High chargeability from IP surveys",
    ),
    
    # =========================================================================
    # SOIL ASSESSMENT CONCEPTS
    # =========================================================================
    
    # Soil horizon concepts (soil pit photos)
    "soil_horizon_a": GeologyConcept(
        name="soil_horizon_a",
        modality=Modality.SOIL_PIT,
        text_prompts=["topsoil", "A horizon", "organic layer", "humus layer", "dark soil layer"],
        description="Topsoil layer rich in organic matter",
        synonyms=["topsoil", "humus", "organic horizon"]
    ),
    "soil_horizon_b": GeologyConcept(
        name="soil_horizon_b",
        modality=Modality.SOIL_PIT,
        text_prompts=["subsoil", "B horizon", "clay accumulation", "illuvial horizon"],
        description="Subsoil layer with accumulated clay and minerals",
        synonyms=["subsoil", "illuvial zone"]
    ),
    "soil_horizon_c": GeologyConcept(
        name="soil_horizon_c",
        modality=Modality.SOIL_PIT,
        text_prompts=["parent material", "C horizon", "weathered rock", "regolith"],
        description="Weathered parent material layer",
        synonyms=["regolith", "saprolite"]
    ),
    "laterite": GeologyConcept(
        name="laterite",
        modality=Modality.SOIL_PIT,
        text_prompts=["laterite", "iron-rich soil", "hardpan", "ferricrete", "red laterite"],
        description="Iron and aluminum rich tropical soil",
        synonyms=["ferricrete", "duricrust", "ironstone"]
    ),
    "waterlogging": GeologyConcept(
        name="waterlogging",
        modality=Modality.SOIL_PIT,
        text_prompts=["waterlogged soil", "saturated zone", "gleyed soil", "mottled soil"],
        description="Water-saturated soil zone",
        synonyms=["gleying", "saturation zone"]
    ),
    "root_zone": GeologyConcept(
        name="root_zone",
        modality=Modality.SOIL_PIT,
        text_prompts=["root zone", "rooting depth", "root penetration", "rhizosphere"],
        description="Zone of active root growth",
    ),
    "compaction_layer": GeologyConcept(
        name="compaction_layer",
        modality=Modality.SOIL_PIT,
        text_prompts=["compacted soil", "hardpan", "plow pan", "compaction layer"],
        description="Compacted soil restricting root growth",
        synonyms=["plow pan", "traffic pan"]
    ),
    
    # Soil surface features (UAV/field photos)
    "erosion_rill": GeologyConcept(
        name="erosion_rill",
        modality=Modality.SOIL,
        text_prompts=["rill erosion", "small erosion channel", "rill", "erosion groove"],
        description="Small erosion channels from water flow",
    ),
    "erosion_gully": GeologyConcept(
        name="erosion_gully",
        modality=Modality.SOIL,
        text_prompts=["gully erosion", "gully", "erosion gully", "deep erosion channel"],
        description="Large erosion channels",
    ),
    "sheet_erosion": GeologyConcept(
        name="sheet_erosion",
        modality=Modality.SOIL,
        text_prompts=["sheet erosion", "surface erosion", "soil loss", "exposed subsoil"],
        description="Uniform removal of soil surface",
    ),
    "soil_crust": GeologyConcept(
        name="soil_crust",
        modality=Modality.SOIL,
        text_prompts=["soil crust", "surface crust", "sealed soil", "crusted surface"],
        description="Hard surface crust limiting infiltration",
    ),
    "drainage_pattern": GeologyConcept(
        name="drainage_pattern",
        modality=Modality.SOIL,
        text_prompts=["drainage pattern", "water flow", "drainage channel", "wet area"],
        description="Surface water drainage patterns",
    ),
    "vegetation_stress": GeologyConcept(
        name="vegetation_stress",
        modality=Modality.SOIL,
        text_prompts=["stressed vegetation", "yellowing plants", "nutrient deficiency", "crop stress"],
        description="Vegetation showing soil-related stress",
    ),
    
    # Soil texture (close-up photos)
    "clay_soil": GeologyConcept(
        name="clay_soil",
        modality=Modality.SOIL,
        text_prompts=["clay soil", "heavy soil", "clay texture", "fine-grained soil"],
        description="Clay-dominated soil texture",
    ),
    "sandy_soil": GeologyConcept(
        name="sandy_soil",
        modality=Modality.SOIL,
        text_prompts=["sandy soil", "light soil", "sand texture", "coarse-grained soil"],
        description="Sand-dominated soil texture",
    ),
    "loam_soil": GeologyConcept(
        name="loam_soil",
        modality=Modality.SOIL,
        text_prompts=["loam soil", "loamy soil", "balanced texture", "good tilth"],
        description="Balanced soil texture ideal for agriculture",
    ),
    
    # =========================================================================
    # GOLD EXPLORATION CONCEPTS
    # =========================================================================
    
    # Gold in drillcore
    "visible_gold": GeologyConcept(
        name="visible_gold",
        modality=Modality.DRILLCORE,
        text_prompts=["visible gold", "free gold", "gold grain", "native gold", "gold speck"],
        description="Visible gold particles in core",
        synonyms=["free gold", "coarse gold", "VG"]
    ),
    "gold_bearing_vein": GeologyConcept(
        name="gold_bearing_vein",
        modality=Modality.DRILLCORE,
        text_prompts=["gold vein", "auriferous vein", "mineralized vein", "quartz-gold vein"],
        description="Vein potentially hosting gold mineralization",
    ),
    "arsenopyrite_gold": GeologyConcept(
        name="arsenopyrite_gold",
        modality=Modality.DRILLCORE,
        text_prompts=["arsenopyrite", "arsenic sulfide", "gold-bearing arsenopyrite"],
        description="Arsenopyrite often associated with gold",
    ),
    "pyrite_gold_assoc": GeologyConcept(
        name="pyrite_gold_assoc",
        modality=Modality.DRILLCORE,
        text_prompts=["auriferous pyrite", "gold-bearing pyrite", "refractory gold host"],
        description="Pyrite hosting microscopic gold",
    ),
    "silicification": GeologyConcept(
        name="silicification",
        modality=Modality.DRILLCORE,
        text_prompts=["silicified rock", "silicification", "silica flooding", "jasperoid"],
        description="Silica alteration often associated with gold",
        synonyms=["jasperoid", "silica replacement"]
    ),
    "sericite_alteration": GeologyConcept(
        name="sericite_alteration",
        modality=Modality.DRILLCORE,
        text_prompts=["sericite alteration", "phyllic alteration", "white mica alteration"],
        description="Sericite alteration common in gold systems",
    ),
    "carbonaceous_material": GeologyConcept(
        name="carbonaceous_material",
        modality=Modality.DRILLCORE,
        text_prompts=["carbonaceous rock", "graphitic material", "black shale", "carbon seam"],
        description="Carbonaceous material that may host or rob gold",
    ),
    
    # Gold in thin section
    "gold_grain_ts": GeologyConcept(
        name="gold_grain_ts",
        modality=Modality.THIN_SECTION,
        text_prompts=["gold grain", "native gold", "gold inclusion", "electrum"],
        description="Gold grain in thin section",
        synonyms=["electrum", "native gold"]
    ),
    "gold_telluride": GeologyConcept(
        name="gold_telluride",
        modality=Modality.THIN_SECTION,
        text_prompts=["gold telluride", "calaverite", "sylvanite", "telluride mineral"],
        description="Gold telluride minerals",
        synonyms=["calaverite", "sylvanite", "petzite"]
    ),
    
    # Gold surface expressions (UAV/satellite)
    "orogenic_structure": GeologyConcept(
        name="orogenic_structure",
        modality=Modality.UAV_ORTHO,
        text_prompts=["shear zone", "fault structure", "orogenic structure", "deformation zone"],
        description="Structural features hosting orogenic gold",
    ),
    "iron_oxide_staining": GeologyConcept(
        name="iron_oxide_staining",
        modality=Modality.UAV_ORTHO,
        text_prompts=["iron staining", "limonite", "goethite staining", "rusty outcrop"],
        description="Iron oxide staining indicating sulfide weathering",
    ),
    "quartz_blow": GeologyConcept(
        name="quartz_blow",
        modality=Modality.UAV_ORTHO,
        text_prompts=["quartz blow", "quartz outcrop", "vein outcrop", "siliceous ridge"],
        description="Resistant quartz vein outcrop",
    ),
    
    # =========================================================================
    # LITHIUM EXPLORATION CONCEPTS
    # =========================================================================
    
    # Lithium pegmatite (drillcore)
    "spodumene": GeologyConcept(
        name="spodumene",
        modality=Modality.DRILLCORE,
        text_prompts=["spodumene", "lithium pyroxene", "spodumene crystal", "green spodumene"],
        description="Primary lithium ore mineral in pegmatites",
        synonyms=["kunzite", "hiddenite"]
    ),
    "lepidolite": GeologyConcept(
        name="lepidolite",
        modality=Modality.DRILLCORE,
        text_prompts=["lepidolite", "lithium mica", "purple mica", "lilac mica"],
        description="Lithium-bearing mica mineral",
    ),
    "petalite": GeologyConcept(
        name="petalite",
        modality=Modality.DRILLCORE,
        text_prompts=["petalite", "lithium feldspar", "castorite"],
        description="Lithium aluminum silicate mineral",
    ),
    "pegmatite_zone": GeologyConcept(
        name="pegmatite_zone",
        modality=Modality.DRILLCORE,
        text_prompts=["pegmatite", "coarse granite", "pegmatite zone", "LCT pegmatite"],
        description="Lithium-cesium-tantalum pegmatite zone",
        synonyms=["LCT pegmatite", "rare-element pegmatite"]
    ),
    "aplite_zone": GeologyConcept(
        name="aplite_zone",
        modality=Modality.DRILLCORE,
        text_prompts=["aplite", "fine-grained granite", "aplite dyke"],
        description="Fine-grained granite often associated with pegmatites",
    ),
    "greisen": GeologyConcept(
        name="greisen",
        modality=Modality.DRILLCORE,
        text_prompts=["greisen", "greisenized rock", "mica-quartz rock"],
        description="Greisen alteration associated with Li-Sn mineralization",
    ),
    
    # Lithium in thin section
    "spodumene_ts": GeologyConcept(
        name="spodumene_ts",
        modality=Modality.THIN_SECTION,
        text_prompts=["spodumene crystal", "lithium pyroxene", "prismatic spodumene"],
        description="Spodumene in thin section",
    ),
    "lepidolite_ts": GeologyConcept(
        name="lepidolite_ts",
        modality=Modality.THIN_SECTION,
        text_prompts=["lepidolite", "lithium mica", "purple mica flakes"],
        description="Lepidolite in thin section",
    ),
    "tourmaline_li": GeologyConcept(
        name="tourmaline_li",
        modality=Modality.THIN_SECTION,
        text_prompts=["tourmaline", "elbaite", "lithium tourmaline", "colored tourmaline"],
        description="Lithium-bearing tourmaline",
        synonyms=["elbaite", "liddicoatite"]
    ),
    
    # Lithium clay (sedimentary)
    "hectorite": GeologyConcept(
        name="hectorite",
        modality=Modality.DRILLCORE,
        text_prompts=["hectorite", "lithium clay", "smectite clay", "white clay"],
        description="Lithium-bearing smectite clay",
    ),
    "lithium_clay_zone": GeologyConcept(
        name="lithium_clay_zone",
        modality=Modality.DRILLCORE,
        text_prompts=["clay zone", "lithium clay horizon", "altered volcanic", "claystone"],
        description="Lithium-enriched clay horizon",
    ),
    
    # Lithium brine (surface)
    "salar_surface": GeologyConcept(
        name="salar_surface",
        modality=Modality.UAV_ORTHO,
        text_prompts=["salt flat", "salar", "evaporite surface", "brine pool"],
        description="Salt flat surface expression",
        synonyms=["playa", "salt pan"]
    ),
    "brine_pool": GeologyConcept(
        name="brine_pool",
        modality=Modality.UAV_ORTHO,
        text_prompts=["brine pool", "evaporation pond", "lithium pond", "salt pond"],
        description="Brine evaporation pond",
    ),
    
    # =========================================================================
    # RARE EARTH ELEMENTS (REE) CONCEPTS
    # =========================================================================
    
    # REE in drillcore
    "carbonatite": GeologyConcept(
        name="carbonatite",
        modality=Modality.DRILLCORE,
        text_prompts=["carbonatite", "carbite rock", "REE carbonatite", "calcite-rich rock"],
        description="Carbonatite hosting REE mineralization",
    ),
    "bastnaesite": GeologyConcept(
        name="bastnaesite",
        modality=Modality.DRILLCORE,
        text_prompts=["bastnaesite", "REE carbonate", "cerium mineral", "yellow-brown REE"],
        description="Primary REE ore mineral (Ce, La carbonate)",
    ),
    "monazite_core": GeologyConcept(
        name="monazite_core",
        modality=Modality.DRILLCORE,
        text_prompts=["monazite", "REE phosphate", "brown monazite", "radioactive mineral"],
        description="REE phosphate mineral",
    ),
    "xenotime": GeologyConcept(
        name="xenotime",
        modality=Modality.DRILLCORE,
        text_prompts=["xenotime", "yttrium phosphate", "heavy REE mineral"],
        description="Heavy REE phosphate mineral",
    ),
    "ion_adsorption_clay": GeologyConcept(
        name="ion_adsorption_clay",
        modality=Modality.DRILLCORE,
        text_prompts=["weathered granite", "ion adsorption clay", "lateritic clay", "REE clay"],
        description="Ion-adsorption type REE deposit in weathered granite",
    ),
    "fenitization": GeologyConcept(
        name="fenitization",
        modality=Modality.DRILLCORE,
        text_prompts=["fenite", "fenitized rock", "alkali metasomatism", "sodic alteration"],
        description="Fenite alteration around carbonatites",
    ),
    
    # REE in thin section
    "monazite_ts": GeologyConcept(
        name="monazite_ts",
        modality=Modality.THIN_SECTION,
        text_prompts=["monazite grain", "REE phosphate", "high relief mineral", "radioactive halo"],
        description="Monazite in thin section",
    ),
    "apatite_ree": GeologyConcept(
        name="apatite_ree",
        modality=Modality.THIN_SECTION,
        text_prompts=["apatite", "REE-bearing apatite", "phosphate mineral", "hexagonal crystal"],
        description="REE-bearing apatite",
    ),
    "allanite": GeologyConcept(
        name="allanite",
        modality=Modality.THIN_SECTION,
        text_prompts=["allanite", "REE epidote", "brown epidote", "metamict mineral"],
        description="REE-bearing epidote group mineral",
    ),
    "zircon_ree": GeologyConcept(
        name="zircon_ree",
        modality=Modality.THIN_SECTION,
        text_prompts=["zircon", "zirconium silicate", "high relief crystal", "tetragonal crystal"],
        description="Zircon potentially enriched in heavy REE",
    ),
    
    # REE surface expressions
    "carbonatite_outcrop": GeologyConcept(
        name="carbonatite_outcrop",
        modality=Modality.UAV_ORTHO,
        text_prompts=["carbonatite outcrop", "white outcrop", "carbonate rock", "circular intrusion"],
        description="Carbonatite surface expression",
    ),
    "alkaline_complex": GeologyConcept(
        name="alkaline_complex",
        modality=Modality.UAV_ORTHO,
        text_prompts=["alkaline complex", "ring structure", "circular feature", "intrusive complex"],
        description="Alkaline igneous complex hosting REE",
    ),
    "heavy_mineral_sand": GeologyConcept(
        name="heavy_mineral_sand",
        modality=Modality.UAV_ORTHO,
        text_prompts=["heavy mineral sand", "black sand", "beach placer", "mineral sand deposit"],
        description="Heavy mineral sand deposit with monazite/xenotime",
    ),
    
    # REE geophysics
    "radiometric_anomaly": GeologyConcept(
        name="radiometric_anomaly",
        modality=Modality.GEOPHYSICS,
        text_prompts=["radiometric anomaly", "thorium anomaly", "uranium anomaly", "radioactive zone"],
        description="Radiometric anomaly indicating REE mineralization",
    ),
}


@dataclass
class SegmentationResult:
    """Result from SAM3 segmentation."""
    masks: List[np.ndarray]
    scores: List[float]
    concept: str
    prompt_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mask_count": len(self.masks),
            "scores": self.scores,
            "concept": self.concept,
            "prompt_type": self.prompt_type,
            "metadata": self.metadata
        }


class SAM3Segmenter:
    """
    SAM3-based segmenter for geology/mining applications.
    
    Supports:
    - Text-based concept segmentation
    - Image exemplar prompts
    - Domain-specific fine-tuned adapters
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        device: str = "cuda",
        modality: Modality = Modality.DRILLCORE
    ):
        """
        Initialize SAM3 segmenter.
        
        Args:
            model_path: Path to SAM3 checkpoint
            adapter_path: Path to fine-tuned LoRA adapter
            device: Device to run inference on
            modality: Default imaging modality
        """
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.device = device
        self.modality = modality
        self.predictor = None
        self._initialized = False
        
        if not SAM3_AVAILABLE:
            logger.warning("SAM3 not available. Using mock segmenter.")
        
    def initialize(self) -> bool:
        """Initialize the SAM3 model."""
        if not SAM3_AVAILABLE:
            logger.info("SAM3 not installed. Segmentation will return default results.")
            self._initialized = True
            return True
            
        try:
            if self.model_path and Path(self.model_path).exists():
                self.predictor = build_sam3(checkpoint=self.model_path)
            else:
                self.predictor = build_sam3()
                
            if self.adapter_path and Path(self.adapter_path).exists():
                self._load_adapter(self.adapter_path)
                
            self.predictor.to(self.device)
            self._initialized = True
            logger.info("SAM3 initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SAM3: {e}")
            self._initialized = True
            return False
    
    def _load_adapter(self, adapter_path: str) -> None:
        """Load a fine-tuned LoRA adapter."""
        if not SAM3_AVAILABLE or self.predictor is None:
            return
        try:
            adapter_state = torch.load(adapter_path, map_location=self.device)
            self.predictor.load_adapter(adapter_state)
            logger.info(f"Loaded adapter from {adapter_path}")
        except Exception as e:
            logger.warning(f"Failed to load adapter: {e}")
    
    def segment_by_text(
        self,
        image: Union[np.ndarray, str, Path],
        text_prompt: str,
        concept: Optional[str] = None
    ) -> SegmentationResult:
        """
        Segment image using text prompt.
        
        Args:
            image: Image array or path
            text_prompt: Text description of target
            concept: Optional concept name for domain vocabulary lookup
            
        Returns:
            SegmentationResult with masks and scores
        """
        if not self._initialized:
            self.initialize()
            
        # Use domain vocabulary if concept provided
        if concept and concept in GEOLOGY_CONCEPTS:
            geo_concept = GEOLOGY_CONCEPTS[concept]
            prompts = geo_concept.text_prompts
            if text_prompt not in prompts:
                prompts = [text_prompt] + prompts
        else:
            prompts = [text_prompt]
        
        if not SAM3_AVAILABLE or self.predictor is None:
            return self._mock_segment(prompts[0], "text")
            
        try:
            if isinstance(image, (str, Path)):
                image = np.array(Image.open(image))
                
            self.predictor.set_image(image)
            
            all_masks = []
            all_scores = []
            
            for prompt in prompts:
                masks, scores, _ = self.predictor.predict(
                    text_prompt=prompt,
                    multimask_output=True
                )
                all_masks.extend(masks)
                all_scores.extend(scores.tolist())
            
            return SegmentationResult(
                masks=all_masks,
                scores=all_scores,
                concept=concept or text_prompt,
                prompt_type="text",
                metadata={"prompts": prompts}
            )
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return self._mock_segment(text_prompt, "text")
    
    def segment_by_exemplar(
        self,
        image: Union[np.ndarray, str, Path],
        exemplar_image: Union[np.ndarray, str, Path],
        exemplar_mask: Optional[np.ndarray] = None,
        exemplar_box: Optional[Tuple[int, int, int, int]] = None
    ) -> SegmentationResult:
        """
        Segment image using exemplar image prompt.
        
        Args:
            image: Target image to segment
            exemplar_image: Example image showing target concept
            exemplar_mask: Optional mask on exemplar
            exemplar_box: Optional bounding box on exemplar [x1, y1, x2, y2]
            
        Returns:
            SegmentationResult with masks and scores
        """
        if not self._initialized:
            self.initialize()
            
        if not SAM3_AVAILABLE or self.predictor is None:
            return self._mock_segment("exemplar", "exemplar")
            
        try:
            if isinstance(image, (str, Path)):
                image = np.array(Image.open(image))
            if isinstance(exemplar_image, (str, Path)):
                exemplar_image = np.array(Image.open(exemplar_image))
                
            self.predictor.set_image(image)
            
            masks, scores, _ = self.predictor.predict(
                exemplar_image=exemplar_image,
                exemplar_mask=exemplar_mask,
                exemplar_box=exemplar_box,
                multimask_output=True
            )
            
            return SegmentationResult(
                masks=list(masks),
                scores=scores.tolist(),
                concept="exemplar_match",
                prompt_type="exemplar",
                metadata={"has_mask": exemplar_mask is not None, "has_box": exemplar_box is not None}
            )
        except Exception as e:
            logger.error(f"Exemplar segmentation failed: {e}")
            return self._mock_segment("exemplar", "exemplar")
    
    def segment_by_point(
        self,
        image: Union[np.ndarray, str, Path],
        points: List[Tuple[int, int]],
        labels: List[int]
    ) -> SegmentationResult:
        """
        Segment image using point prompts.
        
        Args:
            image: Image to segment
            points: List of (x, y) coordinates
            labels: List of labels (1=foreground, 0=background)
            
        Returns:
            SegmentationResult with masks and scores
        """
        if not self._initialized:
            self.initialize()
            
        if not SAM3_AVAILABLE or self.predictor is None:
            return self._mock_segment("point", "point")
            
        try:
            if isinstance(image, (str, Path)):
                image = np.array(Image.open(image))
                
            self.predictor.set_image(image)
            
            point_coords = np.array(points)
            point_labels = np.array(labels)
            
            masks, scores, _ = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
            
            return SegmentationResult(
                masks=list(masks),
                scores=scores.tolist(),
                concept="point_selection",
                prompt_type="point",
                metadata={"point_count": len(points)}
            )
        except Exception as e:
            logger.error(f"Point segmentation failed: {e}")
            return self._mock_segment("point", "point")
    
    def get_concepts_for_modality(self, modality: Optional[Modality] = None) -> List[GeologyConcept]:
        """Get available concepts for a modality."""
        mod = modality or self.modality
        return [c for c in GEOLOGY_CONCEPTS.values() if c.modality == mod]
    
    def _mock_segment(self, concept: str, prompt_type: str) -> SegmentationResult:
        """Return default result when SAM3 not available."""
        return SegmentationResult(
            masks=[],
            scores=[],
            concept=concept,
            prompt_type=prompt_type,
            metadata={"mock": True, "reason": "SAM3 not available"}
        )


class SAM3VideoTracker:
    """
    SAM3-based video tracker for geology/mining applications.
    
    Tracks objects across video frames using text or visual prompts.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """Initialize video tracker."""
        self.model_path = model_path
        self.device = device
        self.predictor = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize the video predictor."""
        if not SAM3_AVAILABLE:
            self._initialized = True
            return True
            
        try:
            if self.model_path:
                self.predictor = SAM3VideoPredictor(checkpoint=self.model_path)
            else:
                self.predictor = SAM3VideoPredictor()
            self.predictor.to(self.device)
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize video tracker: {e}")
            self._initialized = True
            return False
    
    def track_concept(
        self,
        video_path: str,
        text_prompt: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Track a concept through video frames.
        
        Args:
            video_path: Path to input video
            text_prompt: Text description of target to track
            output_path: Optional path for output video with masks
            
        Returns:
            Dictionary with tracking results per frame
        """
        if not self._initialized:
            self.initialize()
            
        if not SAM3_AVAILABLE or self.predictor is None:
            return {
                "status": "unavailable",
                "message": "SAM3 video tracking not available",
                "frames": []
            }
            
        try:
            results = self.predictor.track(
                video_path=video_path,
                text_prompt=text_prompt,
                output_path=output_path
            )
            return {
                "status": "success",
                "frame_count": len(results),
                "frames": results
            }
        except Exception as e:
            logger.error(f"Video tracking failed: {e}")
            return {
                "status": "error",
                "message": str(e),
                "frames": []
            }


def create_sam3_segmenter(
    modality: str = "drillcore",
    model_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    device: str = "cuda"
) -> SAM3Segmenter:
    """
    Factory function to create SAM3 segmenter.
    
    Args:
        modality: Imaging modality (drillcore, thin_section, uav_ortho, etc.)
        model_path: Path to SAM3 checkpoint
        adapter_path: Path to fine-tuned adapter
        device: Device for inference
        
    Returns:
        Configured SAM3Segmenter instance
    """
    mod = Modality(modality) if modality in [m.value for m in Modality] else Modality.DRILLCORE
    
    segmenter = SAM3Segmenter(
        model_path=model_path,
        adapter_path=adapter_path,
        device=device,
        modality=mod
    )
    segmenter.initialize()
    return segmenter
