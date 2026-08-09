"""
Text-based Prospectivity Features for MineralVision.

Extracts NLP features from geological text that can be used
as inputs to mineral prospectivity models.
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


class ProspectivityIndicator(str, Enum):
    """Types of prospectivity indicators from text."""
    FAVORABLE_LITHOLOGY = "favorable_lithology"
    FAVORABLE_ALTERATION = "favorable_alteration"
    FAVORABLE_STRUCTURE = "favorable_structure"
    MINERALIZATION_PRESENT = "mineralization_present"
    PATHFINDER_ELEMENTS = "pathfinder_elements"
    GEOCHEMICAL_ANOMALY = "geochemical_anomaly"
    GEOPHYSICAL_ANOMALY = "geophysical_anomaly"
    HISTORICAL_WORKINGS = "historical_workings"
    PROXIMITY_TO_INTRUSION = "proximity_to_intrusion"
    FAVORABLE_HOST = "favorable_host"


@dataclass
class ProspectivityScore:
    """Prospectivity score derived from text analysis."""
    indicator: ProspectivityIndicator
    score: float  # 0-1
    confidence: float  # 0-1
    evidence: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'indicator': self.indicator.value,
            'score': self.score,
            'confidence': self.confidence,
            'evidence': self.evidence
        }


@dataclass
class TextProspectivityFeatures:
    """Features extracted from text for prospectivity modeling."""
    # Lithology features
    lithology_diversity: float = 0.0
    favorable_lithology_count: int = 0
    host_rock_score: float = 0.0
    
    # Alteration features
    alteration_intensity: float = 0.0
    alteration_diversity: float = 0.0
    favorable_alteration_count: int = 0
    
    # Mineralization features
    mineralization_mentions: int = 0
    ore_mineral_count: int = 0
    pathfinder_count: int = 0
    
    # Structure features
    structure_complexity: float = 0.0
    favorable_structure_count: int = 0
    
    # Grade features
    max_grade_au: float = 0.0
    max_grade_cu: float = 0.0
    grade_mention_count: int = 0
    
    # Sentiment/confidence
    positive_sentiment: float = 0.0
    uncertainty_score: float = 0.0
    
    # Embedding
    text_embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {
            'lithology_diversity': self.lithology_diversity,
            'favorable_lithology_count': self.favorable_lithology_count,
            'host_rock_score': self.host_rock_score,
            'alteration_intensity': self.alteration_intensity,
            'alteration_diversity': self.alteration_diversity,
            'favorable_alteration_count': self.favorable_alteration_count,
            'mineralization_mentions': self.mineralization_mentions,
            'ore_mineral_count': self.ore_mineral_count,
            'pathfinder_count': self.pathfinder_count,
            'structure_complexity': self.structure_complexity,
            'favorable_structure_count': self.favorable_structure_count,
            'max_grade_au': self.max_grade_au,
            'max_grade_cu': self.max_grade_cu,
            'grade_mention_count': self.grade_mention_count,
            'positive_sentiment': self.positive_sentiment,
            'uncertainty_score': self.uncertainty_score,
        }
        return d
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML models."""
        return np.array([
            self.lithology_diversity,
            self.favorable_lithology_count,
            self.host_rock_score,
            self.alteration_intensity,
            self.alteration_diversity,
            self.favorable_alteration_count,
            self.mineralization_mentions,
            self.ore_mineral_count,
            self.pathfinder_count,
            self.structure_complexity,
            self.favorable_structure_count,
            self.max_grade_au,
            self.max_grade_cu,
            self.grade_mention_count,
            self.positive_sentiment,
            self.uncertainty_score,
        ])


class CommodityConfig:
    """Configuration for commodity-specific prospectivity analysis."""
    
    def __init__(self, commodity: str = "gold"):
        self.commodity = commodity.lower()
        self._load_config()
    
    def _load_config(self) -> None:
        """Load commodity-specific configuration."""
        configs = {
            'gold': {
                'favorable_lithologies': [
                    'granite', 'granodiorite', 'greenstone', 'banded iron formation',
                    'bif', 'turbidite', 'black shale', 'carbonaceous shale',
                    'komatiite', 'basalt', 'dolerite', 'lamprophyre'
                ],
                'favorable_alterations': [
                    'silicification', 'sericitization', 'carbonation',
                    'sulfidation', 'potassic', 'propylitic', 'albitic'
                ],
                'favorable_structures': [
                    'shear zone', 'fault', 'fold hinge', 'anticline',
                    'thrust', 'breccia', 'stockwork', 'vein'
                ],
                'ore_minerals': [
                    'gold', 'native gold', 'electrum', 'calaverite',
                    'sylvanite', 'petzite'
                ],
                'pathfinders': [
                    'arsenopyrite', 'pyrite', 'pyrrhotite', 'stibnite',
                    'telluride', 'bismuth', 'tungsten', 'molybdenite'
                ],
                'grade_element': 'Au',
                'grade_unit': 'g/t'
            },
            'copper': {
                'favorable_lithologies': [
                    'porphyry', 'diorite', 'granodiorite', 'andesite',
                    'basalt', 'sedimentary', 'sandstone', 'shale'
                ],
                'favorable_alterations': [
                    'potassic', 'phyllic', 'propylitic', 'argillic',
                    'silicification', 'chloritization'
                ],
                'favorable_structures': [
                    'stockwork', 'breccia', 'fault', 'contact',
                    'unconformity'
                ],
                'ore_minerals': [
                    'chalcopyrite', 'bornite', 'chalcocite', 'covellite',
                    'malachite', 'azurite', 'chrysocolla', 'native copper'
                ],
                'pathfinders': [
                    'pyrite', 'molybdenite', 'magnetite', 'specularite'
                ],
                'grade_element': 'Cu',
                'grade_unit': '%'
            },
            'lithium': {
                'favorable_lithologies': [
                    'pegmatite', 'granite', 'aplite', 'greisen',
                    'clay', 'evaporite', 'salar'
                ],
                'favorable_alterations': [
                    'greisenization', 'albitization', 'muscovitization',
                    'kaolinization'
                ],
                'favorable_structures': [
                    'pegmatite', 'vein', 'contact', 'cupola'
                ],
                'ore_minerals': [
                    'spodumene', 'lepidolite', 'petalite', 'amblygonite',
                    'lithiophilite', 'hectorite', 'jadarite'
                ],
                'pathfinders': [
                    'tantalite', 'columbite', 'cassiterite', 'beryl',
                    'tourmaline', 'topaz'
                ],
                'grade_element': 'Li2O',
                'grade_unit': '%'
            },
            'ree': {
                'favorable_lithologies': [
                    'carbonatite', 'alkaline', 'syenite', 'granite',
                    'pegmatite', 'laterite', 'placer'
                ],
                'favorable_alterations': [
                    'fenitization', 'carbonation', 'silicification',
                    'lateritization'
                ],
                'favorable_structures': [
                    'carbonatite', 'alkaline complex', 'ring structure',
                    'placer', 'beach sand'
                ],
                'ore_minerals': [
                    'bastnaesite', 'monazite', 'xenotime', 'apatite',
                    'eudialyte', 'loparite', 'ion-adsorption clay'
                ],
                'pathfinders': [
                    'fluorite', 'barite', 'strontianite', 'thorium',
                    'uranium', 'niobium'
                ],
                'grade_element': 'TREO',
                'grade_unit': '%'
            }
        }
        
        config = configs.get(self.commodity, configs['gold'])
        self.favorable_lithologies = set(config['favorable_lithologies'])
        self.favorable_alterations = set(config['favorable_alterations'])
        self.favorable_structures = set(config['favorable_structures'])
        self.ore_minerals = set(config['ore_minerals'])
        self.pathfinders = set(config['pathfinders'])
        self.grade_element = config['grade_element']
        self.grade_unit = config['grade_unit']


class DocumentSimilarity:
    """Calculate similarity between geological documents."""
    
    def __init__(self):
        self._embeddings = None
    
    def _get_embeddings(self):
        """Lazy load embeddings."""
        if self._embeddings is None:
            from .geo_nlp import GeoEmbeddings
            self._embeddings = GeoEmbeddings()
        return self._embeddings
    
    def cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        embeddings = self._get_embeddings()
        return embeddings.similarity(text1, text2)
    
    def find_similar_documents(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float, str]]:
        """Find most similar documents to query."""
        embeddings = self._get_embeddings()
        return embeddings.find_similar(query, documents, top_k)
    
    def cluster_documents(
        self,
        documents: List[str],
        n_clusters: int = 5
    ) -> Dict[int, List[int]]:
        """Cluster documents by similarity."""
        embeddings = self._get_embeddings()
        emb_vectors = embeddings.embed(documents)
        
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(n_clusters, len(documents)), random_state=42)
            labels = kmeans.fit_predict(emb_vectors)
            
            clusters = {}
            for i, label in enumerate(labels):
                label = int(label)
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)
            
            return clusters
        except ImportError:
            return {0: list(range(len(documents)))}


class ProspectivityTextAnalyzer:
    """
    Analyze geological text for prospectivity indicators.
    
    Extracts features that can be used as inputs to
    mineral prospectivity models.
    """
    
    def __init__(self, commodity: str = "gold"):
        self.config = CommodityConfig(commodity)
        self._build_patterns()
    
    def _build_patterns(self) -> None:
        """Build regex patterns for analysis."""
        # Grade patterns
        self.grade_patterns = {
            'Au': re.compile(r'(\d+\.?\d*)\s*(?:g/t|gpt|ppm)\s*(?:Au|gold)?', re.IGNORECASE),
            'Cu': re.compile(r'(\d+\.?\d*)\s*%\s*(?:Cu|copper)?', re.IGNORECASE),
            'Li2O': re.compile(r'(\d+\.?\d*)\s*%\s*(?:Li2O|lithium)?', re.IGNORECASE),
            'TREO': re.compile(r'(\d+\.?\d*)\s*%\s*(?:TREO|REO|rare\s*earth)?', re.IGNORECASE),
        }
        
        # Intensity modifiers
        self.intensity_words = {
            'strong': 1.0, 'intense': 1.0, 'pervasive': 1.0, 'abundant': 0.9,
            'moderate': 0.6, 'common': 0.5, 'minor': 0.3, 'weak': 0.2,
            'trace': 0.1, 'rare': 0.1, 'occasional': 0.2, 'sporadic': 0.2
        }
        
        # Positive sentiment words
        self.positive_words = {
            'significant', 'excellent', 'high-grade', 'bonanza', 'rich',
            'promising', 'favorable', 'prospective', 'anomalous', 'elevated',
            'encouraging', 'substantial', 'impressive', 'exceptional'
        }
        
        # Uncertainty words
        self.uncertainty_words = {
            'possible', 'potential', 'may', 'might', 'could', 'uncertain',
            'unknown', 'unclear', 'inferred', 'interpreted', 'suggested',
            'preliminary', 'tentative', 'approximate'
        }
        
        # Historical workings indicators
        self.historical_patterns = [
            re.compile(r'(?:historical|historic|old|ancient)\s+(?:mine|working|pit|shaft|adit)', re.IGNORECASE),
            re.compile(r'(?:artisanal|alluvial|placer)\s+(?:mining|workings|diggings)', re.IGNORECASE),
            re.compile(r'(?:previously|formerly)\s+mined', re.IGNORECASE),
        ]
    
    def analyze(self, text: str) -> TextProspectivityFeatures:
        """Analyze text and extract prospectivity features."""
        features = TextProspectivityFeatures()
        text_lower = text.lower()
        words = set(re.findall(r'\b\w+\b', text_lower))
        
        # Lithology analysis
        found_lithologies = words & self.config.favorable_lithologies
        features.favorable_lithology_count = len(found_lithologies)
        features.lithology_diversity = len(found_lithologies) / max(len(self.config.favorable_lithologies), 1)
        features.host_rock_score = self._calculate_host_rock_score(text_lower)
        
        # Alteration analysis
        found_alterations = words & self.config.favorable_alterations
        features.favorable_alteration_count = len(found_alterations)
        features.alteration_diversity = len(found_alterations) / max(len(self.config.favorable_alterations), 1)
        features.alteration_intensity = self._calculate_intensity(text_lower, found_alterations)
        
        # Mineralization analysis
        found_ore_minerals = words & self.config.ore_minerals
        found_pathfinders = words & self.config.pathfinders
        features.ore_mineral_count = len(found_ore_minerals)
        features.pathfinder_count = len(found_pathfinders)
        features.mineralization_mentions = text_lower.count('mineralization') + text_lower.count('mineralisation')
        
        # Structure analysis
        found_structures = words & self.config.favorable_structures
        features.favorable_structure_count = len(found_structures)
        features.structure_complexity = self._calculate_structure_complexity(text_lower)
        
        # Grade analysis
        features.max_grade_au = self._extract_max_grade(text, 'Au')
        features.max_grade_cu = self._extract_max_grade(text, 'Cu')
        features.grade_mention_count = len(re.findall(r'\d+\.?\d*\s*(?:g/t|%|ppm)', text))
        
        # Sentiment analysis
        features.positive_sentiment = self._calculate_sentiment(words)
        features.uncertainty_score = self._calculate_uncertainty(words)
        
        return features
    
    def _calculate_host_rock_score(self, text: str) -> float:
        """Calculate host rock favorability score."""
        score = 0.0
        
        # Check for favorable host rock mentions
        host_patterns = [
            (r'hosted?\s+(?:in|by)\s+(\w+)', 0.8),
            (r'(\w+)\s+host(?:ed|ing)?', 0.7),
            (r'within\s+(\w+)', 0.5),
        ]
        
        for pattern, weight in host_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match in self.config.favorable_lithologies:
                    score = max(score, weight)
        
        return score
    
    def _calculate_intensity(self, text: str, terms: Set[str]) -> float:
        """Calculate alteration/mineralization intensity."""
        if not terms:
            return 0.0
        
        max_intensity = 0.0
        
        for term in terms:
            # Look for intensity modifiers near the term
            pattern = r'(\w+)\s+' + re.escape(term)
            matches = re.findall(pattern, text)
            
            for match in matches:
                if match in self.intensity_words:
                    max_intensity = max(max_intensity, self.intensity_words[match])
        
        return max_intensity if max_intensity > 0 else 0.5  # Default moderate
    
    def _calculate_structure_complexity(self, text: str) -> float:
        """Calculate structural complexity score."""
        complexity_indicators = [
            'multiple', 'complex', 'intersecting', 'conjugate',
            'anastomosing', 'en echelon', 'splaying', 'branching'
        ]
        
        count = sum(1 for ind in complexity_indicators if ind in text)
        return min(count / 3.0, 1.0)
    
    def _extract_max_grade(self, text: str, element: str) -> float:
        """Extract maximum grade value for element."""
        if element not in self.grade_patterns:
            return 0.0
        
        pattern = self.grade_patterns[element]
        matches = pattern.findall(text)
        
        if not matches:
            return 0.0
        
        try:
            return max(float(m) for m in matches)
        except ValueError:
            return 0.0
    
    def _calculate_sentiment(self, words: Set[str]) -> float:
        """Calculate positive sentiment score."""
        positive_count = len(words & self.positive_words)
        return min(positive_count / 5.0, 1.0)
    
    def _calculate_uncertainty(self, words: Set[str]) -> float:
        """Calculate uncertainty score."""
        uncertainty_count = len(words & self.uncertainty_words)
        return min(uncertainty_count / 5.0, 1.0)
    
    def get_prospectivity_scores(self, text: str) -> List[ProspectivityScore]:
        """Get detailed prospectivity scores with evidence."""
        scores = []
        features = self.analyze(text)
        text_lower = text.lower()
        
        # Favorable lithology
        if features.favorable_lithology_count > 0:
            evidence = [w for w in self.config.favorable_lithologies if w in text_lower]
            scores.append(ProspectivityScore(
                indicator=ProspectivityIndicator.FAVORABLE_LITHOLOGY,
                score=min(features.favorable_lithology_count / 3.0, 1.0),
                confidence=0.8,
                evidence=evidence[:5]
            ))
        
        # Favorable alteration
        if features.favorable_alteration_count > 0:
            evidence = [w for w in self.config.favorable_alterations if w in text_lower]
            scores.append(ProspectivityScore(
                indicator=ProspectivityIndicator.FAVORABLE_ALTERATION,
                score=features.alteration_intensity,
                confidence=0.75,
                evidence=evidence[:5]
            ))
        
        # Favorable structure
        if features.favorable_structure_count > 0:
            evidence = [w for w in self.config.favorable_structures if w in text_lower]
            scores.append(ProspectivityScore(
                indicator=ProspectivityIndicator.FAVORABLE_STRUCTURE,
                score=min(features.favorable_structure_count / 2.0, 1.0),
                confidence=0.7,
                evidence=evidence[:5]
            ))
        
        # Mineralization present
        if features.ore_mineral_count > 0:
            evidence = [w for w in self.config.ore_minerals if w in text_lower]
            scores.append(ProspectivityScore(
                indicator=ProspectivityIndicator.MINERALIZATION_PRESENT,
                score=min(features.ore_mineral_count / 2.0, 1.0),
                confidence=0.9,
                evidence=evidence[:5]
            ))
        
        # Pathfinder elements
        if features.pathfinder_count > 0:
            evidence = [w for w in self.config.pathfinders if w in text_lower]
            scores.append(ProspectivityScore(
                indicator=ProspectivityIndicator.PATHFINDER_ELEMENTS,
                score=min(features.pathfinder_count / 3.0, 1.0),
                confidence=0.7,
                evidence=evidence[:5]
            ))
        
        # Historical workings
        for pattern in self.historical_patterns:
            if pattern.search(text_lower):
                scores.append(ProspectivityScore(
                    indicator=ProspectivityIndicator.HISTORICAL_WORKINGS,
                    score=0.8,
                    confidence=0.85,
                    evidence=['historical mining activity detected']
                ))
                break
        
        return scores
    
    def calculate_overall_prospectivity(self, text: str) -> float:
        """Calculate overall prospectivity score (0-1)."""
        scores = self.get_prospectivity_scores(text)
        
        if not scores:
            return 0.0
        
        # Weighted average based on confidence
        total_weight = sum(s.confidence for s in scores)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(s.score * s.confidence for s in scores)
        return weighted_sum / total_weight


def extract_prospectivity_features(
    text: str,
    commodity: str = "gold"
) -> Dict[str, Any]:
    """Convenience function to extract prospectivity features."""
    analyzer = ProspectivityTextAnalyzer(commodity)
    features = analyzer.analyze(text)
    scores = analyzer.get_prospectivity_scores(text)
    overall = analyzer.calculate_overall_prospectivity(text)
    
    return {
        'features': features.to_dict(),
        'feature_vector': features.to_vector().tolist(),
        'scores': [s.to_dict() for s in scores],
        'overall_prospectivity': overall,
        'commodity': commodity
    }
