"""
Block Model Module for MineralVision Platform.

Comprehensive block modeling including:
1. Regular and sub-blocked model creation
2. Block model population from composites
3. Grade estimation (NN, ID, kriging)
4. Resource classification (measured, indicated, inferred)
5. Reporting by domain, bench, classification
6. Volume and tonnage calculations
7. Import/export (CSV, BMF, Datamine, Vulcan)
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Iterator
import math
import numpy as np
from collections import defaultdict


class BlockModelType(Enum):
    """Types of block models."""
    REGULAR = "regular"
    SUB_BLOCKED = "sub_blocked"
    VARIABLE = "variable"
    OCTREE = "octree"


class EstimationMethod(Enum):
    """Grade estimation methods."""
    NEAREST_NEIGHBOR = "nearest_neighbor"
    INVERSE_DISTANCE = "inverse_distance"
    INVERSE_DISTANCE_SQUARED = "inverse_distance_squared"
    KRIGING = "kriging"
    POLYGONAL = "polygonal"


class ResourceCategory(Enum):
    """Resource classification categories."""
    MEASURED = "measured"
    INDICATED = "indicated"
    INFERRED = "inferred"
    UNCLASSIFIED = "unclassified"


class ReserveCategory(Enum):
    """Reserve classification categories."""
    PROVEN = "proven"
    PROBABLE = "probable"
    NOT_RESERVE = "not_reserve"


@dataclass
class BlockModelOrigin:
    """Block model origin and orientation."""
    x_min: float
    y_min: float
    z_min: float
    rotation: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "z_min": self.z_min,
            "rotation": self.rotation
        }


@dataclass
class BlockSize:
    """Block dimensions."""
    x_size: float
    y_size: float
    z_size: float
    
    @property
    def volume(self) -> float:
        return self.x_size * self.y_size * self.z_size
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "x_size": self.x_size,
            "y_size": self.y_size,
            "z_size": self.z_size,
            "volume": self.volume
        }


@dataclass
class BlockModelExtent:
    """Block model extent (number of blocks)."""
    nx: int
    ny: int
    nz: int
    
    @property
    def total_blocks(self) -> int:
        return self.nx * self.ny * self.nz
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "total_blocks": self.total_blocks
        }


@dataclass
class Block:
    """Single block in the model."""
    ix: int
    iy: int
    iz: int
    x_centroid: float
    y_centroid: float
    z_centroid: float
    volume: float
    percent_fill: float = 100.0
    domain: int = 0
    rock_type: str = ""
    density: float = 2.7
    attributes: Dict[str, float] = field(default_factory=dict)
    variances: Dict[str, float] = field(default_factory=dict)
    n_samples: Dict[str, int] = field(default_factory=dict)
    classification: ResourceCategory = ResourceCategory.UNCLASSIFIED
    reserve_category: ReserveCategory = ReserveCategory.NOT_RESERVE
    
    @property
    def tonnage(self) -> float:
        return self.volume * self.percent_fill / 100 * self.density
    
    def get_metal(self, element: str) -> float:
        """Calculate contained metal for an element."""
        grade = self.attributes.get(element, 0)
        return self.tonnage * grade / 100 if grade < 100 else self.tonnage * grade / 1e6
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ix": self.ix,
            "iy": self.iy,
            "iz": self.iz,
            "x": self.x_centroid,
            "y": self.y_centroid,
            "z": self.z_centroid,
            "volume": self.volume,
            "percent_fill": self.percent_fill,
            "domain": self.domain,
            "rock_type": self.rock_type,
            "density": self.density,
            "tonnage": self.tonnage,
            "classification": self.classification.value,
            **self.attributes
        }


@dataclass
class SubBlock:
    """Sub-block for variable resolution models."""
    parent_ix: int
    parent_iy: int
    parent_iz: int
    sub_ix: int
    sub_iy: int
    sub_iz: int
    x_centroid: float
    y_centroid: float
    z_centroid: float
    volume: float
    attributes: Dict[str, float] = field(default_factory=dict)


@dataclass
class Domain:
    """Geological domain definition."""
    domain_id: int
    name: str
    description: str = ""
    rock_types: List[str] = field(default_factory=list)
    default_density: float = 2.7
    color: str = "#808080"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationCriteria:
    """Resource classification criteria."""
    category: ResourceCategory
    max_kriging_variance: float = float('inf')
    min_samples: int = 0
    max_distance: float = float('inf')
    min_kriging_efficiency: float = 0.0
    max_slope_of_regression: float = float('inf')
    min_slope_of_regression: float = 0.0
    custom_criteria: Optional[Dict[str, Any]] = None


@dataclass
class ReportingInterval:
    """Grade-tonnage reporting interval."""
    cutoff: float
    tonnage: float
    grade: float
    metal: float
    volume: float
    n_blocks: int
    classification: Optional[ResourceCategory] = None


@dataclass
class BenchReport:
    """Report for a single bench."""
    bench_rl: float
    bench_height: float
    tonnage: float
    volume: float
    n_blocks: int
    grades: Dict[str, float] = field(default_factory=dict)
    metals: Dict[str, float] = field(default_factory=dict)
    classification_breakdown: Dict[str, float] = field(default_factory=dict)


class BlockModel:
    """
    Main block model class.
    
    Provides comprehensive block model functionality including:
    - Model creation and management
    - Grade estimation
    - Resource classification
    - Reporting
    """
    
    def __init__(self, name: str, origin: BlockModelOrigin, 
                 block_size: BlockSize, extent: BlockModelExtent,
                 model_type: BlockModelType = BlockModelType.REGULAR):
        self.name = name
        self.origin = origin
        self.block_size = block_size
        self.extent = extent
        self.model_type = model_type
        
        self.blocks: Dict[Tuple[int, int, int], Block] = {}
        self.sub_blocks: Dict[Tuple[int, int, int], List[SubBlock]] = {}
        self.domains: Dict[int, Domain] = {}
        self.attributes: List[str] = []
        
        self.classification_criteria: List[ClassificationCriteria] = []
        
        self.created_date = datetime.now()
        self.modified_date = datetime.now()
        self.metadata: Dict[str, Any] = {}
    
    def _index_to_centroid(self, ix: int, iy: int, iz: int) -> Tuple[float, float, float]:
        """Convert block indices to centroid coordinates."""
        x = self.origin.x_min + (ix + 0.5) * self.block_size.x_size
        y = self.origin.y_min + (iy + 0.5) * self.block_size.y_size
        z = self.origin.z_min + (iz + 0.5) * self.block_size.z_size
        
        if self.origin.rotation != 0:
            rot_rad = math.radians(self.origin.rotation)
            cos_r = math.cos(rot_rad)
            sin_r = math.sin(rot_rad)
            
            dx = x - self.origin.x_min
            dy = y - self.origin.y_min
            
            x = self.origin.x_min + dx * cos_r - dy * sin_r
            y = self.origin.y_min + dx * sin_r + dy * cos_r
        
        return (x, y, z)
    
    def _point_to_index(self, x: float, y: float, z: float) -> Tuple[int, int, int]:
        """Convert point coordinates to block indices."""
        if self.origin.rotation != 0:
            rot_rad = math.radians(-self.origin.rotation)
            cos_r = math.cos(rot_rad)
            sin_r = math.sin(rot_rad)
            
            dx = x - self.origin.x_min
            dy = y - self.origin.y_min
            
            x = self.origin.x_min + dx * cos_r - dy * sin_r
            y = self.origin.y_min + dx * sin_r + dy * cos_r
        
        ix = int((x - self.origin.x_min) / self.block_size.x_size)
        iy = int((y - self.origin.y_min) / self.block_size.y_size)
        iz = int((z - self.origin.z_min) / self.block_size.z_size)
        
        return (ix, iy, iz)
    
    def create_block(self, ix: int, iy: int, iz: int, 
                    domain: int = 0, percent_fill: float = 100.0) -> Block:
        """Create a new block at the specified indices."""
        if not (0 <= ix < self.extent.nx and 
                0 <= iy < self.extent.ny and 
                0 <= iz < self.extent.nz):
            raise ValueError(f"Block indices ({ix}, {iy}, {iz}) out of range")
        
        x, y, z = self._index_to_centroid(ix, iy, iz)
        
        block = Block(
            ix=ix, iy=iy, iz=iz,
            x_centroid=x, y_centroid=y, z_centroid=z,
            volume=self.block_size.volume,
            percent_fill=percent_fill,
            domain=domain
        )
        
        self.blocks[(ix, iy, iz)] = block
        self.modified_date = datetime.now()
        
        return block
    
    def get_block(self, ix: int, iy: int, iz: int) -> Optional[Block]:
        """Get block at specified indices."""
        return self.blocks.get((ix, iy, iz))
    
    def get_block_at_point(self, x: float, y: float, z: float) -> Optional[Block]:
        """Get block containing the specified point."""
        ix, iy, iz = self._point_to_index(x, y, z)
        return self.get_block(ix, iy, iz)
    
    def set_attribute(self, ix: int, iy: int, iz: int, 
                     attribute: str, value: float,
                     variance: float = 0.0, n_samples: int = 0):
        """Set attribute value for a block."""
        block = self.get_block(ix, iy, iz)
        if block is None:
            block = self.create_block(ix, iy, iz)
        
        block.attributes[attribute] = value
        if variance > 0:
            block.variances[attribute] = variance
        if n_samples > 0:
            block.n_samples[attribute] = n_samples
        
        if attribute not in self.attributes:
            self.attributes.append(attribute)
        
        self.modified_date = datetime.now()
    
    def add_domain(self, domain: Domain):
        """Add a geological domain."""
        self.domains[domain.domain_id] = domain
    
    def set_domain(self, ix: int, iy: int, iz: int, domain_id: int):
        """Set domain for a block."""
        block = self.get_block(ix, iy, iz)
        if block:
            block.domain = domain_id
            if domain_id in self.domains:
                block.density = self.domains[domain_id].default_density
    
    def initialize_all_blocks(self, domain: int = 0, percent_fill: float = 100.0):
        """Initialize all blocks in the model."""
        for ix in range(self.extent.nx):
            for iy in range(self.extent.ny):
                for iz in range(self.extent.nz):
                    self.create_block(ix, iy, iz, domain, percent_fill)
    
    def iterate_blocks(self) -> Iterator[Block]:
        """Iterate over all blocks."""
        for block in self.blocks.values():
            yield block
    
    def iterate_blocks_in_domain(self, domain_id: int) -> Iterator[Block]:
        """Iterate over blocks in a specific domain."""
        for block in self.blocks.values():
            if block.domain == domain_id:
                yield block
    
    def iterate_blocks_in_bench(self, bench_rl: float, 
                               bench_height: Optional[float] = None) -> Iterator[Block]:
        """Iterate over blocks in a bench."""
        if bench_height is None:
            bench_height = self.block_size.z_size
        
        for block in self.blocks.values():
            block_base = block.z_centroid - self.block_size.z_size / 2
            block_top = block.z_centroid + self.block_size.z_size / 2
            
            if block_base >= bench_rl and block_base < bench_rl + bench_height:
                yield block
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get model statistics."""
        n_blocks = len(self.blocks)
        
        if n_blocks == 0:
            return {
                "name": self.name,
                "n_blocks": 0,
                "total_volume": 0,
                "total_tonnage": 0
            }
        
        total_volume = sum(b.volume * b.percent_fill / 100 for b in self.blocks.values())
        total_tonnage = sum(b.tonnage for b in self.blocks.values())
        
        attr_stats = {}
        for attr in self.attributes:
            values = [b.attributes.get(attr, 0) for b in self.blocks.values() 
                     if attr in b.attributes]
            if values:
                attr_stats[attr] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values, ddof=1) if len(values) > 1 else 0,
                    "n_estimated": len(values)
                }
        
        domain_counts = defaultdict(int)
        for block in self.blocks.values():
            domain_counts[block.domain] += 1
        
        classification_counts = defaultdict(int)
        for block in self.blocks.values():
            classification_counts[block.classification.value] += 1
        
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "n_blocks": n_blocks,
            "n_potential_blocks": self.extent.total_blocks,
            "fill_ratio": n_blocks / self.extent.total_blocks * 100,
            "total_volume": total_volume,
            "total_tonnage": total_tonnage,
            "attributes": attr_stats,
            "domains": dict(domain_counts),
            "classification": dict(classification_counts),
            "origin": self.origin.to_dict(),
            "block_size": self.block_size.to_dict(),
            "extent": self.extent.to_dict()
        }


class BlockModelEstimator:
    """Estimate grades into block model."""
    
    def __init__(self, block_model: BlockModel):
        self.model = block_model
    
    def estimate_nearest_neighbor(self, composites: List[Dict[str, Any]],
                                  element: str, max_distance: float = float('inf')):
        """Estimate grades using nearest neighbor."""
        for block in self.model.iterate_blocks():
            nearest_dist = float('inf')
            nearest_value = None
            
            for comp in composites:
                dx = comp.get('x', comp.get('easting', 0)) - block.x_centroid
                dy = comp.get('y', comp.get('northing', 0)) - block.y_centroid
                dz = comp.get('z', comp.get('elevation', 0)) - block.z_centroid
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                
                if dist < nearest_dist and dist <= max_distance:
                    nearest_dist = dist
                    nearest_value = comp.get(element, comp.get('value', 0))
            
            if nearest_value is not None:
                self.model.set_attribute(
                    block.ix, block.iy, block.iz,
                    element, nearest_value,
                    n_samples=1
                )
    
    def estimate_inverse_distance(self, composites: List[Dict[str, Any]],
                                  element: str, power: float = 2.0,
                                  max_distance: float = float('inf'),
                                  min_samples: int = 1, max_samples: int = 16):
        """Estimate grades using inverse distance weighting."""
        for block in self.model.iterate_blocks():
            neighbors = []
            
            for comp in composites:
                dx = comp.get('x', comp.get('easting', 0)) - block.x_centroid
                dy = comp.get('y', comp.get('northing', 0)) - block.y_centroid
                dz = comp.get('z', comp.get('elevation', 0)) - block.z_centroid
                dist = math.sqrt(dx**2 + dy**2 + dz**2)
                
                if dist <= max_distance and dist > 0:
                    value = comp.get(element, comp.get('value', 0))
                    neighbors.append((dist, value))
            
            neighbors.sort(key=lambda x: x[0])
            neighbors = neighbors[:max_samples]
            
            if len(neighbors) >= min_samples:
                total_weight = 0
                weighted_sum = 0
                
                for dist, value in neighbors:
                    weight = 1 / (dist ** power)
                    weighted_sum += weight * value
                    total_weight += weight
                
                if total_weight > 0:
                    estimate = weighted_sum / total_weight
                    self.model.set_attribute(
                        block.ix, block.iy, block.iz,
                        element, estimate,
                        n_samples=len(neighbors)
                    )
    
    def estimate_from_kriging_results(self, kriging_results: List[Dict[str, Any]],
                                      element: str):
        """Populate block model from kriging results."""
        for result in kriging_results:
            x = result.get('x', 0)
            y = result.get('y', 0)
            z = result.get('z', 0)
            
            ix, iy, iz = self.model._point_to_index(x, y, z)
            
            if 0 <= ix < self.model.extent.nx and \
               0 <= iy < self.model.extent.ny and \
               0 <= iz < self.model.extent.nz:
                
                estimate = result.get('estimate', 0)
                variance = result.get('variance', 0)
                n_samples = result.get('n_samples', 0)
                
                self.model.set_attribute(
                    ix, iy, iz, element, estimate,
                    variance=variance, n_samples=n_samples
                )


class ResourceClassifier:
    """Classify resources based on estimation quality."""
    
    def __init__(self, block_model: BlockModel):
        self.model = block_model
        self.criteria: List[ClassificationCriteria] = []
    
    def add_criteria(self, criteria: ClassificationCriteria):
        """Add classification criteria."""
        self.criteria.append(criteria)
    
    def set_default_criteria(self, element: str):
        """Set default classification criteria."""
        self.criteria = [
            ClassificationCriteria(
                category=ResourceCategory.MEASURED,
                max_kriging_variance=0.3,
                min_samples=8,
                max_distance=50,
                min_kriging_efficiency=0.7
            ),
            ClassificationCriteria(
                category=ResourceCategory.INDICATED,
                max_kriging_variance=0.6,
                min_samples=4,
                max_distance=100,
                min_kriging_efficiency=0.4
            ),
            ClassificationCriteria(
                category=ResourceCategory.INFERRED,
                max_kriging_variance=1.0,
                min_samples=1,
                max_distance=200,
                min_kriging_efficiency=0.1
            )
        ]
    
    def classify_block(self, block: Block, element: str) -> ResourceCategory:
        """Classify a single block."""
        variance = block.variances.get(element, float('inf'))
        n_samples = block.n_samples.get(element, 0)
        
        data_variance = 1.0
        efficiency = 1 - variance / data_variance if data_variance > 0 else 0
        
        for criteria in self.criteria:
            if (variance <= criteria.max_kriging_variance and
                n_samples >= criteria.min_samples and
                efficiency >= criteria.min_kriging_efficiency):
                return criteria.category
        
        return ResourceCategory.UNCLASSIFIED
    
    def classify_all(self, element: str):
        """Classify all blocks in the model."""
        for block in self.model.iterate_blocks():
            if element in block.attributes:
                block.classification = self.classify_block(block, element)
    
    def get_classification_summary(self) -> Dict[str, Any]:
        """Get classification summary."""
        summary = {cat.value: {"n_blocks": 0, "tonnage": 0, "volume": 0}
                  for cat in ResourceCategory}
        
        for block in self.model.iterate_blocks():
            cat = block.classification.value
            summary[cat]["n_blocks"] += 1
            summary[cat]["tonnage"] += block.tonnage
            summary[cat]["volume"] += block.volume * block.percent_fill / 100
        
        return summary


class BlockModelReporter:
    """Generate reports from block model."""
    
    def __init__(self, block_model: BlockModel):
        self.model = block_model
    
    def grade_tonnage_curve(self, element: str, 
                           cutoffs: Optional[List[float]] = None,
                           domain: Optional[int] = None,
                           classification: Optional[ResourceCategory] = None
                           ) -> List[ReportingInterval]:
        """Generate grade-tonnage curve."""
        if cutoffs is None:
            values = [b.attributes.get(element, 0) for b in self.model.iterate_blocks()
                     if element in b.attributes]
            if values:
                min_val = min(values)
                max_val = max(values)
                cutoffs = np.linspace(min_val, max_val, 20).tolist()
            else:
                cutoffs = [0]
        
        results = []
        
        for cutoff in cutoffs:
            tonnage = 0
            metal = 0
            volume = 0
            n_blocks = 0
            
            for block in self.model.iterate_blocks():
                if domain is not None and block.domain != domain:
                    continue
                if classification is not None and block.classification != classification:
                    continue
                
                grade = block.attributes.get(element, 0)
                if grade >= cutoff:
                    tonnage += block.tonnage
                    metal += block.get_metal(element)
                    volume += block.volume * block.percent_fill / 100
                    n_blocks += 1
            
            avg_grade = metal / tonnage * 100 if tonnage > 0 else 0
            
            results.append(ReportingInterval(
                cutoff=cutoff,
                tonnage=tonnage,
                grade=avg_grade,
                metal=metal,
                volume=volume,
                n_blocks=n_blocks,
                classification=classification
            ))
        
        return results
    
    def bench_report(self, element: str, bench_height: float,
                    domain: Optional[int] = None) -> List[BenchReport]:
        """Generate bench-by-bench report."""
        z_values = [b.z_centroid for b in self.model.iterate_blocks()]
        if not z_values:
            return []
        
        min_z = min(z_values) - self.model.block_size.z_size / 2
        max_z = max(z_values) + self.model.block_size.z_size / 2
        
        results = []
        bench_rl = min_z
        
        while bench_rl < max_z:
            tonnage = 0
            volume = 0
            n_blocks = 0
            grade_sum = 0
            metal_sum = 0
            classification_tonnage = defaultdict(float)
            
            for block in self.model.iterate_blocks_in_bench(bench_rl, bench_height):
                if domain is not None and block.domain != domain:
                    continue
                
                tonnage += block.tonnage
                volume += block.volume * block.percent_fill / 100
                n_blocks += 1
                
                grade = block.attributes.get(element, 0)
                grade_sum += grade * block.tonnage
                metal_sum += block.get_metal(element)
                
                classification_tonnage[block.classification.value] += block.tonnage
            
            if n_blocks > 0:
                avg_grade = grade_sum / tonnage if tonnage > 0 else 0
                
                results.append(BenchReport(
                    bench_rl=bench_rl,
                    bench_height=bench_height,
                    tonnage=tonnage,
                    volume=volume,
                    n_blocks=n_blocks,
                    grades={element: avg_grade},
                    metals={element: metal_sum},
                    classification_breakdown=dict(classification_tonnage)
                ))
            
            bench_rl += bench_height
        
        return results
    
    def domain_report(self, element: str) -> Dict[int, Dict[str, Any]]:
        """Generate report by domain."""
        results = {}
        
        for domain_id in self.model.domains:
            tonnage = 0
            volume = 0
            n_blocks = 0
            grade_sum = 0
            metal_sum = 0
            
            for block in self.model.iterate_blocks_in_domain(domain_id):
                tonnage += block.tonnage
                volume += block.volume * block.percent_fill / 100
                n_blocks += 1
                
                grade = block.attributes.get(element, 0)
                grade_sum += grade * block.tonnage
                metal_sum += block.get_metal(element)
            
            if n_blocks > 0:
                results[domain_id] = {
                    "domain_name": self.model.domains[domain_id].name,
                    "n_blocks": n_blocks,
                    "tonnage": tonnage,
                    "volume": volume,
                    "average_grade": grade_sum / tonnage if tonnage > 0 else 0,
                    "contained_metal": metal_sum
                }
        
        return results
    
    def classification_report(self, element: str) -> Dict[str, Dict[str, Any]]:
        """Generate report by resource classification."""
        results = {}
        
        for category in ResourceCategory:
            tonnage = 0
            volume = 0
            n_blocks = 0
            grade_sum = 0
            metal_sum = 0
            
            for block in self.model.iterate_blocks():
                if block.classification != category:
                    continue
                
                tonnage += block.tonnage
                volume += block.volume * block.percent_fill / 100
                n_blocks += 1
                
                grade = block.attributes.get(element, 0)
                grade_sum += grade * block.tonnage
                metal_sum += block.get_metal(element)
            
            if n_blocks > 0:
                results[category.value] = {
                    "n_blocks": n_blocks,
                    "tonnage": tonnage,
                    "volume": volume,
                    "average_grade": grade_sum / tonnage if tonnage > 0 else 0,
                    "contained_metal": metal_sum
                }
        
        return results
    
    def resource_statement(self, element: str, cutoff: float,
                          density_override: Optional[float] = None) -> Dict[str, Any]:
        """Generate resource statement at specified cutoff."""
        statement = {
            "element": element,
            "cutoff": cutoff,
            "unit": "g/t" if element.lower() in ["au", "ag", "pt", "pd"] else "%",
            "categories": {}
        }
        
        for category in [ResourceCategory.MEASURED, ResourceCategory.INDICATED, 
                        ResourceCategory.INFERRED]:
            tonnage = 0
            grade_sum = 0
            metal_sum = 0
            
            for block in self.model.iterate_blocks():
                if block.classification != category:
                    continue
                
                grade = block.attributes.get(element, 0)
                if grade < cutoff:
                    continue
                
                block_tonnage = block.tonnage
                if density_override:
                    block_tonnage = block.volume * block.percent_fill / 100 * density_override
                
                tonnage += block_tonnage
                grade_sum += grade * block_tonnage
                metal_sum += block.get_metal(element)
            
            if tonnage > 0:
                statement["categories"][category.value] = {
                    "tonnage_mt": tonnage / 1e6,
                    "grade": grade_sum / tonnage,
                    "contained_metal_kg": metal_sum * 1000 if element.lower() in ["au", "ag", "pt", "pd"] else metal_sum
                }
        
        measured = statement["categories"].get("measured", {})
        indicated = statement["categories"].get("indicated", {})
        
        if measured or indicated:
            m_tonnage = measured.get("tonnage_mt", 0)
            i_tonnage = indicated.get("tonnage_mt", 0)
            m_grade = measured.get("grade", 0)
            i_grade = indicated.get("grade", 0)
            m_metal = measured.get("contained_metal_kg", 0)
            i_metal = indicated.get("contained_metal_kg", 0)
            
            total_tonnage = m_tonnage + i_tonnage
            total_metal = m_metal + i_metal
            
            statement["categories"]["measured_indicated"] = {
                "tonnage_mt": total_tonnage,
                "grade": (m_grade * m_tonnage + i_grade * i_tonnage) / total_tonnage if total_tonnage > 0 else 0,
                "contained_metal_kg": total_metal
            }
        
        return statement


class BlockModelIO:
    """Import/export block models."""
    
    def __init__(self, block_model: BlockModel):
        self.model = block_model
    
    def export_to_csv(self, filepath: str, attributes: Optional[List[str]] = None):
        """Export block model to CSV."""
        import csv
        
        if attributes is None:
            attributes = self.model.attributes
        
        headers = ['ix', 'iy', 'iz', 'x', 'y', 'z', 'volume', 'percent_fill',
                  'domain', 'density', 'tonnage', 'classification'] + attributes
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            for block in self.model.iterate_blocks():
                row = [
                    block.ix, block.iy, block.iz,
                    block.x_centroid, block.y_centroid, block.z_centroid,
                    block.volume, block.percent_fill,
                    block.domain, block.density, block.tonnage,
                    block.classification.value
                ]
                
                for attr in attributes:
                    row.append(block.attributes.get(attr, ''))
                
                writer.writerow(row)
    
    def import_from_csv(self, filepath: str, 
                       x_col: str = 'x', y_col: str = 'y', z_col: str = 'z',
                       attribute_cols: Optional[List[str]] = None):
        """Import block model from CSV."""
        import csv
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                x = float(row[x_col])
                y = float(row[y_col])
                z = float(row[z_col])
                
                ix, iy, iz = self.model._point_to_index(x, y, z)
                
                if not (0 <= ix < self.model.extent.nx and
                       0 <= iy < self.model.extent.ny and
                       0 <= iz < self.model.extent.nz):
                    continue
                
                block = self.model.create_block(ix, iy, iz)
                
                if 'domain' in row:
                    block.domain = int(row['domain'])
                if 'density' in row:
                    block.density = float(row['density'])
                if 'percent_fill' in row:
                    block.percent_fill = float(row['percent_fill'])
                
                if attribute_cols:
                    for col in attribute_cols:
                        if col in row and row[col]:
                            try:
                                block.attributes[col] = float(row[col])
                                if col not in self.model.attributes:
                                    self.model.attributes.append(col)
                            except ValueError:
                                pass
    
    def export_to_json(self) -> Dict[str, Any]:
        """Export block model to JSON-serializable dict."""
        return {
            "name": self.model.name,
            "model_type": self.model.model_type.value,
            "origin": self.model.origin.to_dict(),
            "block_size": self.model.block_size.to_dict(),
            "extent": self.model.extent.to_dict(),
            "attributes": self.model.attributes,
            "domains": {k: {"name": v.name, "density": v.default_density}
                       for k, v in self.model.domains.items()},
            "blocks": [block.to_dict() for block in self.model.iterate_blocks()],
            "created": self.model.created_date.isoformat(),
            "modified": self.model.modified_date.isoformat()
        }


class BlockModelWorkflow:
    """
    Complete block modeling workflow.
    """
    
    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.model: Optional[BlockModel] = None
        self.estimator: Optional[BlockModelEstimator] = None
        self.classifier: Optional[ResourceClassifier] = None
        self.reporter: Optional[BlockModelReporter] = None
        self.io: Optional[BlockModelIO] = None
    
    def create_model(self, name: str,
                    x_min: float, y_min: float, z_min: float,
                    x_size: float, y_size: float, z_size: float,
                    nx: int, ny: int, nz: int,
                    rotation: float = 0.0) -> BlockModel:
        """Create a new block model."""
        origin = BlockModelOrigin(x_min, y_min, z_min, rotation)
        block_size = BlockSize(x_size, y_size, z_size)
        extent = BlockModelExtent(nx, ny, nz)
        
        self.model = BlockModel(name, origin, block_size, extent)
        self.estimator = BlockModelEstimator(self.model)
        self.classifier = ResourceClassifier(self.model)
        self.reporter = BlockModelReporter(self.model)
        self.io = BlockModelIO(self.model)
        
        return self.model
    
    def add_domain(self, domain_id: int, name: str, 
                  density: float = 2.7, color: str = "#808080"):
        """Add a geological domain."""
        if self.model:
            domain = Domain(domain_id, name, default_density=density, color=color)
            self.model.add_domain(domain)
    
    def estimate_grades(self, composites: List[Dict[str, Any]],
                       element: str, method: str = "inverse_distance",
                       **kwargs):
        """Estimate grades into the model."""
        if not self.estimator:
            raise ValueError("Model not created")
        
        if method == "nearest_neighbor":
            self.estimator.estimate_nearest_neighbor(composites, element, **kwargs)
        elif method == "inverse_distance":
            self.estimator.estimate_inverse_distance(composites, element, **kwargs)
        elif method == "inverse_distance_squared":
            self.estimator.estimate_inverse_distance(composites, element, power=2.0, **kwargs)
    
    def classify_resources(self, element: str):
        """Classify resources."""
        if not self.classifier:
            raise ValueError("Model not created")
        
        self.classifier.set_default_criteria(element)
        self.classifier.classify_all(element)
    
    def generate_report(self, element: str, cutoff: float = 0) -> Dict[str, Any]:
        """Generate comprehensive report."""
        if not self.reporter:
            raise ValueError("Model not created")
        
        return {
            "project": self.project_name,
            "model_statistics": self.model.get_statistics(),
            "resource_statement": self.reporter.resource_statement(element, cutoff),
            "classification_summary": self.classifier.get_classification_summary() if self.classifier else {},
            "grade_tonnage": [
                {"cutoff": r.cutoff, "tonnage": r.tonnage, "grade": r.grade, "metal": r.metal}
                for r in self.reporter.grade_tonnage_curve(element)
            ]
        }
    
    def export_model(self, filepath: str, format: str = "csv"):
        """Export the model."""
        if not self.io:
            raise ValueError("Model not created")
        
        if format == "csv":
            self.io.export_to_csv(filepath)
        elif format == "json":
            import json
            with open(filepath, 'w') as f:
                json.dump(self.io.export_to_json(), f, indent=2)


def create_block_model_workflow(project_name: str = "default") -> BlockModelWorkflow:
    """Factory function to create a block model workflow."""
    return BlockModelWorkflow(project_name)


def create_block_model(name: str, x_min: float, y_min: float, z_min: float,
                      x_size: float, y_size: float, z_size: float,
                      nx: int, ny: int, nz: int) -> BlockModel:
    """Factory function to create a block model."""
    origin = BlockModelOrigin(x_min, y_min, z_min)
    block_size = BlockSize(x_size, y_size, z_size)
    extent = BlockModelExtent(nx, ny, nz)
    
    return BlockModel(name, origin, block_size, extent)
