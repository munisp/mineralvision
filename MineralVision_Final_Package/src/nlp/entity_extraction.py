"""
Named Entity Recognition for geological text.

Extracts geological entities such as minerals, rock types,
structures, locations, and measurements from unstructured text.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set
import json

import numpy as np

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Types of geological entities."""
    MINERAL = "mineral"
    ROCK_TYPE = "rock_type"
    ALTERATION = "alteration"
    STRUCTURE = "structure"
    LOCATION = "location"
    DEPTH = "depth"
    GRADE = "grade"
    ORIENTATION = "orientation"
    AGE = "age"
    FORMATION = "formation"
    COMMODITY = "commodity"
    COMPANY = "company"
    PROJECT = "project"
    METHOD = "method"


@dataclass
class GeologyEntity:
    """A geological entity extracted from text."""
    text: str
    entity_type: EntityType
    start: int
    end: int
    confidence: float = 1.0
    normalized: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'type': self.entity_type.value,
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence,
            'normalized': self.normalized,
            'attributes': self.attributes
        }


class EntityDictionary:
    """Dictionary of known geological entities."""
    
    def __init__(self):
        self._build_dictionaries()
    
    def _build_dictionaries(self) -> None:
        """Build entity dictionaries."""
        self.minerals: Set[str] = {
            # Sulfides
            'pyrite', 'chalcopyrite', 'galena', 'sphalerite', 'arsenopyrite',
            'pyrrhotite', 'molybdenite', 'bornite', 'covellite', 'chalcocite',
            'pentlandite', 'millerite', 'cinnabar', 'stibnite', 'realgar',
            'orpiment', 'enargite', 'tennantite', 'tetrahedrite',
            # Oxides
            'magnetite', 'hematite', 'goethite', 'limonite', 'ilmenite',
            'chromite', 'cassiterite', 'rutile', 'uraninite', 'pitchblende',
            'columbite', 'tantalite', 'wolframite', 'scheelite',
            # Silicates
            'quartz', 'feldspar', 'plagioclase', 'orthoclase', 'microcline',
            'muscovite', 'biotite', 'chlorite', 'sericite', 'kaolinite',
            'illite', 'smectite', 'montmorillonite', 'talc', 'serpentine',
            'olivine', 'pyroxene', 'amphibole', 'hornblende', 'actinolite',
            'tremolite', 'garnet', 'epidote', 'zoisite', 'tourmaline',
            'beryl', 'topaz', 'zircon', 'titanite', 'allanite',
            'spodumene', 'lepidolite', 'petalite', 'amblygonite',
            # Carbonates
            'calcite', 'dolomite', 'siderite', 'ankerite', 'magnesite',
            'rhodochrosite', 'malachite', 'azurite', 'cerussite', 'smithsonite',
            # Phosphates
            'apatite', 'monazite', 'xenotime', 'autunite', 'torbernite',
            # Native elements
            'gold', 'silver', 'copper', 'platinum', 'palladium',
            # REE minerals
            'bastnaesite', 'synchysite', 'parisite', 'florencite',
            # Tellurides
            'calaverite', 'sylvanite', 'petzite', 'hessite'
        }
        
        self.rock_types: Set[str] = {
            # Igneous - Felsic
            'granite', 'granodiorite', 'tonalite', 'trondhjemite',
            'rhyolite', 'dacite', 'pegmatite', 'aplite', 'syenite',
            'monzonite', 'quartz monzonite', 'adamellite',
            # Igneous - Intermediate
            'diorite', 'andesite', 'latite', 'trachyte',
            # Igneous - Mafic
            'gabbro', 'basalt', 'dolerite', 'diabase', 'norite',
            # Igneous - Ultramafic
            'peridotite', 'dunite', 'harzburgite', 'lherzolite',
            'pyroxenite', 'hornblendite', 'komatiite', 'serpentinite',
            # Igneous - Alkaline
            'carbonatite', 'kimberlite', 'lamproite', 'lamprophyre',
            'nepheline syenite', 'phonolite', 'ijolite',
            # Sedimentary - Clastic
            'sandstone', 'siltstone', 'mudstone', 'shale', 'claystone',
            'conglomerate', 'breccia', 'arkose', 'greywacke', 'turbidite',
            # Sedimentary - Chemical
            'limestone', 'dolomite', 'dolostone', 'chert', 'ironstone',
            'banded iron formation', 'bif', 'evaporite', 'gypsum', 'halite',
            # Metamorphic
            'schist', 'gneiss', 'quartzite', 'marble', 'slate', 'phyllite',
            'amphibolite', 'granulite', 'eclogite', 'migmatite', 'hornfels',
            'skarn', 'greisen', 'mylonite', 'cataclasite',
            # Weathered
            'laterite', 'saprolite', 'regolith', 'gossan', 'duricrust'
        }
        
        self.alterations: Set[str] = {
            'silicification', 'silicified', 'sericitization', 'sericitic',
            'chloritization', 'chloritic', 'carbonation', 'carbonated',
            'propylitic', 'phyllic', 'argillic', 'advanced argillic',
            'potassic', 'sodic', 'calcic', 'sodic-calcic',
            'epidotization', 'epidote', 'albitization', 'albitic',
            'tourmalinization', 'greisenization', 'fenitization', 'fenite',
            'skarnification', 'serpentinization', 'talcification',
            'kaolinization', 'kaolinitic', 'illitization', 'illitic',
            'hematization', 'hematitic', 'limonitization', 'limonitic',
            'supergene', 'hypogene', 'oxidation', 'oxidized',
            'weathering', 'weathered', 'leaching', 'leached'
        }
        
        self.structures: Set[str] = {
            'fault', 'fracture', 'joint', 'shear', 'shear zone',
            'fold', 'anticline', 'syncline', 'monocline',
            'thrust', 'thrust fault', 'normal fault', 'reverse fault',
            'strike-slip', 'dextral', 'sinistral', 'transform',
            'lineament', 'contact', 'unconformity', 'disconformity',
            'foliation', 'cleavage', 'schistosity', 'gneissosity',
            'bedding', 'lamination', 'cross-bedding', 'graded bedding',
            'breccia', 'brecciation', 'cataclasis', 'mylonite zone',
            'vein', 'veinlet', 'stockwork', 'sheeted vein',
            'dyke', 'dike', 'sill', 'plug', 'stock', 'batholith'
        }
        
        self.commodities: Set[str] = {
            'gold', 'silver', 'copper', 'lead', 'zinc', 'nickel', 'cobalt',
            'platinum', 'palladium', 'rhodium', 'iridium', 'osmium', 'ruthenium',
            'iron', 'manganese', 'chromium', 'vanadium', 'titanium',
            'lithium', 'beryllium', 'cesium', 'rubidium', 'tantalum', 'niobium',
            'tungsten', 'molybdenum', 'tin', 'antimony', 'bismuth',
            'uranium', 'thorium', 'rare earth', 'ree', 'lree', 'hree',
            'diamond', 'emerald', 'ruby', 'sapphire',
            'bauxite', 'phosphate', 'potash', 'graphite', 'fluorspar'
        }
        
        self.age_terms: Set[str] = {
            # Eons
            'archean', 'proterozoic', 'phanerozoic',
            # Eras
            'paleozoic', 'mesozoic', 'cenozoic',
            # Periods
            'cambrian', 'ordovician', 'silurian', 'devonian',
            'carboniferous', 'permian', 'triassic', 'jurassic',
            'cretaceous', 'paleogene', 'neogene', 'quaternary',
            # Epochs
            'paleocene', 'eocene', 'oligocene', 'miocene', 'pliocene',
            'pleistocene', 'holocene',
            # Relative
            'precambrian', 'paleoproterozoic', 'mesoproterozoic',
            'neoproterozoic', 'neoarchean', 'mesoarchean', 'paleoarchean'
        }


class GeoEntityExtractor:
    """
    Extract geological entities from text.
    
    Uses rule-based and dictionary-based methods for
    entity recognition in geological documents.
    """
    
    def __init__(self):
        self.dictionary = EntityDictionary()
        self._build_patterns()
    
    def _build_patterns(self) -> None:
        """Build regex patterns for entity extraction."""
        # Depth patterns
        self.depth_pattern = re.compile(
            r'(\d+\.?\d*)\s*(?:to|-)\s*(\d+\.?\d*)\s*(m|meters?|ft|feet|\')',
            re.IGNORECASE
        )
        
        # Single depth
        self.single_depth_pattern = re.compile(
            r'(?:at|@|depth)\s*(\d+\.?\d*)\s*(m|meters?|ft|feet)',
            re.IGNORECASE
        )
        
        # Grade patterns
        self.grade_patterns = [
            # g/t Au format
            re.compile(
                r'(\d+\.?\d*)\s*(g/t|gpt|ppm|ppb|oz/t|opt)\s*(Au|Ag|Pt|Pd)?',
                re.IGNORECASE
            ),
            # % Cu format
            re.compile(
                r'(\d+\.?\d*)\s*%\s*(Cu|Pb|Zn|Ni|Co|Li2O|U3O8|WO3|MoS2)?',
                re.IGNORECASE
            ),
            # Au 2.5 g/t format
            re.compile(
                r'(Au|Ag|Cu|Pb|Zn|Ni|Co)\s*[:=]?\s*(\d+\.?\d*)\s*(g/t|%|ppm)?',
                re.IGNORECASE
            )
        ]
        
        # Orientation patterns
        self.orientation_pattern = re.compile(
            r'(\d{1,3})\s*[°/]\s*(\d{1,2})\s*([NSEW])?',
            re.IGNORECASE
        )
        
        # Age patterns (Ma, Ga)
        self.age_pattern = re.compile(
            r'(\d+\.?\d*)\s*(Ma|Ga|ka|mya|bya)',
            re.IGNORECASE
        )
        
        # Coordinate patterns
        self.coord_pattern = re.compile(
            r'(-?\d+\.?\d*)\s*[°,]\s*(-?\d+\.?\d*)',
            re.IGNORECASE
        )
    
    def extract(self, text: str) -> List[GeologyEntity]:
        """Extract all geological entities from text."""
        entities = []
        
        # Extract dictionary-based entities
        entities.extend(self._extract_dictionary_entities(text))
        
        # Extract pattern-based entities
        entities.extend(self._extract_depths(text))
        entities.extend(self._extract_grades(text))
        entities.extend(self._extract_orientations(text))
        entities.extend(self._extract_ages(text))
        
        # Sort by position
        entities.sort(key=lambda e: e.start)
        
        # Remove overlapping entities (keep higher confidence)
        entities = self._remove_overlaps(entities)
        
        return entities
    
    def _extract_dictionary_entities(self, text: str) -> List[GeologyEntity]:
        """Extract entities using dictionaries."""
        entities = []
        text_lower = text.lower()
        
        # Minerals
        for mineral in self.dictionary.minerals:
            for match in re.finditer(r'\b' + re.escape(mineral) + r'\b', text_lower):
                entities.append(GeologyEntity(
                    text=text[match.start():match.end()],
                    entity_type=EntityType.MINERAL,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                    normalized=mineral
                ))
        
        # Rock types
        for rock in self.dictionary.rock_types:
            for match in re.finditer(r'\b' + re.escape(rock) + r'\b', text_lower):
                entities.append(GeologyEntity(
                    text=text[match.start():match.end()],
                    entity_type=EntityType.ROCK_TYPE,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95,
                    normalized=rock
                ))
        
        # Alterations
        for alt in self.dictionary.alterations:
            for match in re.finditer(r'\b' + re.escape(alt) + r'\b', text_lower):
                entities.append(GeologyEntity(
                    text=text[match.start():match.end()],
                    entity_type=EntityType.ALTERATION,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                    normalized=alt
                ))
        
        # Structures
        for struct in self.dictionary.structures:
            for match in re.finditer(r'\b' + re.escape(struct) + r'\b', text_lower):
                entities.append(GeologyEntity(
                    text=text[match.start():match.end()],
                    entity_type=EntityType.STRUCTURE,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                    normalized=struct
                ))
        
        # Commodities
        for commodity in self.dictionary.commodities:
            for match in re.finditer(r'\b' + re.escape(commodity) + r'\b', text_lower):
                entities.append(GeologyEntity(
                    text=text[match.start():match.end()],
                    entity_type=EntityType.COMMODITY,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                    normalized=commodity
                ))
        
        # Ages
        for age in self.dictionary.age_terms:
            for match in re.finditer(r'\b' + re.escape(age) + r'\b', text_lower):
                entities.append(GeologyEntity(
                    text=text[match.start():match.end()],
                    entity_type=EntityType.AGE,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                    normalized=age
                ))
        
        return entities
    
    def _extract_depths(self, text: str) -> List[GeologyEntity]:
        """Extract depth intervals."""
        entities = []
        
        # Interval depths
        for match in self.depth_pattern.finditer(text):
            from_depth = float(match.group(1))
            to_depth = float(match.group(2))
            unit = match.group(3).lower()
            
            entities.append(GeologyEntity(
                text=match.group(0),
                entity_type=EntityType.DEPTH,
                start=match.start(),
                end=match.end(),
                confidence=0.95,
                attributes={
                    'from': from_depth,
                    'to': to_depth,
                    'unit': 'm' if unit.startswith('m') else 'ft'
                }
            ))
        
        # Single depths
        for match in self.single_depth_pattern.finditer(text):
            depth = float(match.group(1))
            unit = match.group(2).lower()
            
            entities.append(GeologyEntity(
                text=match.group(0),
                entity_type=EntityType.DEPTH,
                start=match.start(),
                end=match.end(),
                confidence=0.9,
                attributes={
                    'depth': depth,
                    'unit': 'm' if unit.startswith('m') else 'ft'
                }
            ))
        
        return entities
    
    def _extract_grades(self, text: str) -> List[GeologyEntity]:
        """Extract grade values."""
        entities = []
        
        for pattern in self.grade_patterns:
            for match in pattern.finditer(text):
                groups = match.groups()
                
                # Parse based on pattern
                if groups[0] and groups[0].replace('.', '').isdigit():
                    value = float(groups[0])
                    unit = groups[1] if len(groups) > 1 else None
                    element = groups[2] if len(groups) > 2 else None
                else:
                    element = groups[0]
                    value = float(groups[1]) if len(groups) > 1 and groups[1] else 0
                    unit = groups[2] if len(groups) > 2 else None
                
                entities.append(GeologyEntity(
                    text=match.group(0),
                    entity_type=EntityType.GRADE,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9,
                    attributes={
                        'value': value,
                        'unit': unit,
                        'element': element
                    }
                ))
        
        return entities
    
    def _extract_orientations(self, text: str) -> List[GeologyEntity]:
        """Extract structural orientations."""
        entities = []
        
        for match in self.orientation_pattern.finditer(text):
            azimuth = int(match.group(1))
            dip = int(match.group(2))
            direction = match.group(3) if match.group(3) else None
            
            if azimuth <= 360 and dip <= 90:
                entities.append(GeologyEntity(
                    text=match.group(0),
                    entity_type=EntityType.ORIENTATION,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.85,
                    attributes={
                        'azimuth': azimuth,
                        'dip': dip,
                        'direction': direction
                    }
                ))
        
        return entities
    
    def _extract_ages(self, text: str) -> List[GeologyEntity]:
        """Extract numerical ages."""
        entities = []
        
        for match in self.age_pattern.finditer(text):
            value = float(match.group(1))
            unit = match.group(2).lower()
            
            # Convert to Ma
            if unit == 'ga':
                value_ma = value * 1000
            elif unit == 'ka':
                value_ma = value / 1000
            else:
                value_ma = value
            
            entities.append(GeologyEntity(
                text=match.group(0),
                entity_type=EntityType.AGE,
                start=match.start(),
                end=match.end(),
                confidence=0.9,
                attributes={
                    'value': value,
                    'unit': unit,
                    'value_ma': value_ma
                }
            ))
        
        return entities
    
    def _remove_overlaps(self, entities: List[GeologyEntity]) -> List[GeologyEntity]:
        """Remove overlapping entities, keeping higher confidence."""
        if not entities:
            return entities
        
        result = []
        entities = sorted(entities, key=lambda e: (e.start, -e.confidence))
        
        current = entities[0]
        for entity in entities[1:]:
            if entity.start >= current.end:
                result.append(current)
                current = entity
            elif entity.confidence > current.confidence:
                current = entity
        
        result.append(current)
        return result
    
    def extract_by_type(
        self,
        text: str,
        entity_types: List[EntityType]
    ) -> List[GeologyEntity]:
        """Extract only specific entity types."""
        all_entities = self.extract(text)
        return [e for e in all_entities if e.entity_type in entity_types]
    
    def get_entity_summary(self, text: str) -> Dict[str, Any]:
        """Get summary of entities in text."""
        entities = self.extract(text)
        
        summary = {
            'total_entities': len(entities),
            'by_type': {},
            'minerals': [],
            'rock_types': [],
            'alterations': [],
            'structures': [],
            'grades': [],
            'depths': []
        }
        
        for entity in entities:
            type_name = entity.entity_type.value
            if type_name not in summary['by_type']:
                summary['by_type'][type_name] = 0
            summary['by_type'][type_name] += 1
            
            if entity.entity_type == EntityType.MINERAL:
                summary['minerals'].append(entity.normalized or entity.text)
            elif entity.entity_type == EntityType.ROCK_TYPE:
                summary['rock_types'].append(entity.normalized or entity.text)
            elif entity.entity_type == EntityType.ALTERATION:
                summary['alterations'].append(entity.normalized or entity.text)
            elif entity.entity_type == EntityType.STRUCTURE:
                summary['structures'].append(entity.normalized or entity.text)
            elif entity.entity_type == EntityType.GRADE:
                summary['grades'].append(entity.attributes)
            elif entity.entity_type == EntityType.DEPTH:
                summary['depths'].append(entity.attributes)
        
        # Deduplicate lists
        summary['minerals'] = list(set(summary['minerals']))
        summary['rock_types'] = list(set(summary['rock_types']))
        summary['alterations'] = list(set(summary['alterations']))
        summary['structures'] = list(set(summary['structures']))
        
        return summary


def extract_geological_entities(text: str) -> List[Dict[str, Any]]:
    """Convenience function to extract entities."""
    extractor = GeoEntityExtractor()
    entities = extractor.extract(text)
    return [e.to_dict() for e in entities]
