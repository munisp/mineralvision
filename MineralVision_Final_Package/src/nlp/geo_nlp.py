"""
Core NLP functionality for geological text processing.

Provides GeoSciBERT-style embeddings and text processing
specifically designed for geoscience terminology.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import hashlib

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class EmbeddingModel(str, Enum):
    """Available embedding models."""
    GEOSCIBERT = "geoscibert"
    SCIBERT = "scibert"
    MINILM = "minilm"
    CUSTOM = "custom"


@dataclass
class GeoVocabulary:
    """Geological vocabulary for domain-specific tokenization."""
    
    # Lithology terms
    lithology: List[str] = field(default_factory=lambda: [
        "granite", "granodiorite", "diorite", "gabbro", "basalt", "andesite",
        "rhyolite", "dacite", "pegmatite", "aplite", "syenite", "monzonite",
        "sandstone", "siltstone", "mudstone", "shale", "limestone", "dolomite",
        "conglomerate", "breccia", "arkose", "greywacke", "chert", "ironstone",
        "schist", "gneiss", "quartzite", "marble", "slate", "phyllite",
        "amphibolite", "granulite", "eclogite", "migmatite", "hornfels",
        "skarn", "greisen", "serpentinite", "talc", "chlorite", "epidote",
        "laterite", "saprolite", "regolith", "colluvium", "alluvium",
        "carbonatite", "kimberlite", "lamproite", "lamprophyre", "komatiite"
    ])
    
    # Alteration terms
    alteration: List[str] = field(default_factory=lambda: [
        "silicification", "sericitization", "chloritization", "carbonation",
        "propylitic", "phyllic", "argillic", "potassic", "sodic", "calcic",
        "epidotization", "albitization", "tourmalinization", "greisenization",
        "fenitization", "skarnification", "serpentinization", "talcification",
        "kaolinization", "illitization", "smectitization", "zeolitization",
        "hematization", "limonitization", "goethitization", "jarosite",
        "supergene", "hypogene", "oxidation", "reduction", "weathering"
    ])
    
    # Mineralization terms
    mineralization: List[str] = field(default_factory=lambda: [
        "pyrite", "chalcopyrite", "galena", "sphalerite", "arsenopyrite",
        "pyrrhotite", "magnetite", "hematite", "goethite", "limonite",
        "gold", "silver", "copper", "lead", "zinc", "molybdenite",
        "scheelite", "wolframite", "cassiterite", "columbite", "tantalite",
        "spodumene", "lepidolite", "petalite", "amblygonite", "lithiophilite",
        "monazite", "bastnaesite", "xenotime", "apatite", "zircon",
        "uraninite", "pitchblende", "carnotite", "autunite", "torbernite",
        "bornite", "covellite", "chalcocite", "malachite", "azurite",
        "native", "electrum", "telluride", "selenide", "sulfosalt",
        "disseminated", "massive", "vein", "stockwork", "breccia-hosted"
    ])
    
    # Structure terms
    structure: List[str] = field(default_factory=lambda: [
        "fault", "fracture", "joint", "shear", "fold", "anticline", "syncline",
        "thrust", "normal", "reverse", "strike-slip", "dextral", "sinistral",
        "lineament", "contact", "unconformity", "discordance", "foliation",
        "cleavage", "schistosity", "gneissosity", "banding", "layering",
        "bedding", "lamination", "cross-bedding", "graded", "massive",
        "brecciation", "cataclasis", "mylonite", "ultramylonite", "phyllonite",
        "boudin", "ptygmatic", "isoclinal", "recumbent", "overturned"
    ])
    
    # Texture terms
    texture: List[str] = field(default_factory=lambda: [
        "porphyritic", "aphanitic", "phaneritic", "glassy", "vesicular",
        "amygdaloidal", "pegmatitic", "graphic", "granophyric", "ophitic",
        "subophitic", "poikilitic", "cumulophyric", "trachytic", "pilotaxitic",
        "granoblastic", "lepidoblastic", "nematoblastic", "porphyroblastic",
        "clastic", "bioclastic", "oolitic", "pisolitic", "stromatolitic",
        "crystalline", "cryptocrystalline", "microcrystalline", "coarse",
        "medium", "fine", "very fine", "massive", "foliated", "lineated"
    ])
    
    # Grade/intensity terms
    grade_terms: List[str] = field(default_factory=lambda: [
        "strong", "moderate", "weak", "intense", "pervasive", "selective",
        "patchy", "spotty", "disseminated", "massive", "trace", "minor",
        "abundant", "common", "rare", "occasional", "frequent", "sporadic",
        "high-grade", "low-grade", "bonanza", "ore-grade", "sub-economic"
    ])
    
    def get_all_terms(self) -> List[str]:
        """Get all vocabulary terms."""
        return (
            self.lithology + self.alteration + self.mineralization +
            self.structure + self.texture + self.grade_terms
        )


class GeologyTokenizer:
    """
    Tokenizer specialized for geological text.
    
    Handles geological terminology, abbreviations, and
    domain-specific patterns.
    """
    
    def __init__(self, vocabulary: Optional[GeoVocabulary] = None):
        self.vocabulary = vocabulary or GeoVocabulary()
        self._build_patterns()
    
    def _build_patterns(self) -> None:
        """Build regex patterns for geological text."""
        # Depth patterns (e.g., "123.45m", "100-150m")
        self.depth_pattern = re.compile(
            r'(\d+\.?\d*)\s*-?\s*(\d+\.?\d*)?\s*(m|meters?|ft|feet)',
            re.IGNORECASE
        )
        
        # Grade patterns (e.g., "2.5 g/t Au", "1.2% Cu")
        self.grade_pattern = re.compile(
            r'(\d+\.?\d*)\s*(g/t|ppm|ppb|%|oz/t)\s*(Au|Ag|Cu|Pb|Zn|Li|REE|U)?',
            re.IGNORECASE
        )
        
        # Azimuth/dip patterns (e.g., "045/60", "N45E/60SE")
        self.orientation_pattern = re.compile(
            r'(\d{3}|\d{2})\s*/\s*(\d{2})',
            re.IGNORECASE
        )
        
        # Abbreviation patterns
        self.abbreviations = {
            'qtz': 'quartz', 'py': 'pyrite', 'cpy': 'chalcopyrite',
            'gn': 'galena', 'sp': 'sphalerite', 'asp': 'arsenopyrite',
            'mt': 'magnetite', 'hm': 'hematite', 'gt': 'goethite',
            'ser': 'sericite', 'chl': 'chlorite', 'carb': 'carbonate',
            'sil': 'silicification', 'alt': 'alteration', 'min': 'mineralization',
            'vn': 'vein', 'flt': 'fault', 'frac': 'fracture',
            'bx': 'breccia', 'cgl': 'conglomerate', 'ss': 'sandstone',
            'ls': 'limestone', 'dol': 'dolomite', 'sh': 'shale',
            'grt': 'granite', 'grd': 'granodiorite', 'dio': 'diorite',
            'peg': 'pegmatite', 'apl': 'aplite', 'por': 'porphyry'
        }
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand geological abbreviations."""
        words = text.split()
        expanded = []
        
        for word in words:
            lower = word.lower().strip('.,;:')
            if lower in self.abbreviations:
                expanded.append(self.abbreviations[lower])
            else:
                expanded.append(word)
        
        return ' '.join(expanded)
    
    def extract_depths(self, text: str) -> List[Dict[str, Any]]:
        """Extract depth intervals from text."""
        depths = []
        
        for match in self.depth_pattern.finditer(text):
            depth_info = {
                'from': float(match.group(1)),
                'to': float(match.group(2)) if match.group(2) else None,
                'unit': match.group(3).lower(),
                'span': (match.start(), match.end())
            }
            depths.append(depth_info)
        
        return depths
    
    def extract_grades(self, text: str) -> List[Dict[str, Any]]:
        """Extract grade values from text."""
        grades = []
        
        for match in self.grade_pattern.finditer(text):
            grade_info = {
                'value': float(match.group(1)),
                'unit': match.group(2),
                'element': match.group(3) if match.group(3) else None,
                'span': (match.start(), match.end())
            }
            grades.append(grade_info)
        
        return grades
    
    def tokenize(self, text: str, expand_abbrev: bool = True) -> List[str]:
        """Tokenize geological text."""
        if expand_abbrev:
            text = self.expand_abbreviations(text)
        
        # Basic tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        return tokens
    
    def identify_geological_terms(self, text: str) -> Dict[str, List[str]]:
        """Identify geological terms by category."""
        tokens = set(self.tokenize(text))
        
        found = {
            'lithology': [],
            'alteration': [],
            'mineralization': [],
            'structure': [],
            'texture': [],
            'grade_terms': []
        }
        
        for token in tokens:
            if token in self.vocabulary.lithology:
                found['lithology'].append(token)
            if token in self.vocabulary.alteration:
                found['alteration'].append(token)
            if token in self.vocabulary.mineralization:
                found['mineralization'].append(token)
            if token in self.vocabulary.structure:
                found['structure'].append(token)
            if token in self.vocabulary.texture:
                found['texture'].append(token)
            if token in self.vocabulary.grade_terms:
                found['grade_terms'].append(token)
        
        return found


class GeoEmbeddings:
    """
    Geological text embeddings using transformer models.
    
    Supports GeoSciBERT, SciBERT, and general-purpose models
    with domain-specific fine-tuning capabilities.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model_type: EmbeddingModel = EmbeddingModel.MINILM,
        cache_dir: Optional[str] = None
    ):
        self.model_name = model_name
        self.model_type = model_type
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        self._model = None
        self._tokenizer = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    def _load_model(self) -> None:
        """Load the embedding model."""
        if self._model is not None:
            return
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer: {e}")
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                logger.info(f"Loaded Transformers model: {self.model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load Transformers model: {e}")
        
        logger.warning("No transformer models available, using TF-IDF fallback")
        self._model = "tfidf_fallback"
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s)."""
        self._load_model()
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache
        cache_key = hashlib.md5(str(texts).encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        if self._model == "tfidf_fallback":
            embeddings = self._tfidf_embed(texts)
        elif SENTENCE_TRANSFORMERS_AVAILABLE and hasattr(self._model, 'encode'):
            embeddings = self._model.encode(texts, convert_to_numpy=True)
        elif TRANSFORMERS_AVAILABLE and self._tokenizer is not None:
            embeddings = self._transformer_embed(texts)
        else:
            embeddings = self._tfidf_embed(texts)
        
        self._embedding_cache[cache_key] = embeddings
        return embeddings
    
    def _transformer_embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using transformers."""
        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
        
        return embeddings
    
    def _tfidf_embed(self, texts: List[str]) -> np.ndarray:
        """Fallback TF-IDF embeddings."""
        # Simple bag-of-words with geological vocabulary weighting
        vocab = GeoVocabulary()
        all_terms = vocab.get_all_terms()
        
        embeddings = []
        for text in texts:
            tokens = set(text.lower().split())
            vector = np.zeros(len(all_terms) + 100)  # Extra dims for general terms
            
            for i, term in enumerate(all_terms):
                if term in tokens:
                    vector[i] = 1.0
            
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            embeddings.append(vector)
        
        return np.array(embeddings)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        
        if emb1.ndim == 2:
            emb1 = emb1[0]
        if emb2.ndim == 2:
            emb2 = emb2[0]
        
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    def find_similar(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float, str]]:
        """Find most similar documents to query."""
        query_emb = self.embed(query)
        doc_embs = self.embed(documents)
        
        if query_emb.ndim == 2:
            query_emb = query_emb[0]
        
        similarities = []
        for i, doc_emb in enumerate(doc_embs):
            sim = np.dot(query_emb, doc_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-8
            )
            similarities.append((i, float(sim), documents[i]))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class GeoNLPProcessor:
    """
    Main NLP processor for geological text.
    
    Combines tokenization, embedding, and analysis
    capabilities for geological documents.
    """
    
    def __init__(
        self,
        embedding_model: Optional[str] = None,
        vocabulary: Optional[GeoVocabulary] = None
    ):
        self.vocabulary = vocabulary or GeoVocabulary()
        self.tokenizer = GeologyTokenizer(self.vocabulary)
        
        model_name = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        self.embeddings = GeoEmbeddings(model_name=model_name)
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process geological text and extract features."""
        # Tokenize
        tokens = self.tokenizer.tokenize(text)
        
        # Identify geological terms
        geo_terms = self.tokenizer.identify_geological_terms(text)
        
        # Extract structured data
        depths = self.tokenizer.extract_depths(text)
        grades = self.tokenizer.extract_grades(text)
        
        # Generate embedding
        embedding = self.embeddings.embed(text)
        
        return {
            'tokens': tokens,
            'token_count': len(tokens),
            'geological_terms': geo_terms,
            'depths': depths,
            'grades': grades,
            'embedding': embedding[0] if embedding.ndim == 2 else embedding,
            'processed_at': datetime.now().isoformat()
        }
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compare two geological texts."""
        proc1 = self.process_text(text1)
        proc2 = self.process_text(text2)
        
        # Embedding similarity
        emb_sim = self.embeddings.similarity(text1, text2)
        
        # Term overlap
        terms1 = set()
        terms2 = set()
        for category in proc1['geological_terms'].values():
            terms1.update(category)
        for category in proc2['geological_terms'].values():
            terms2.update(category)
        
        if terms1 or terms2:
            jaccard = len(terms1 & terms2) / len(terms1 | terms2)
        else:
            jaccard = 0.0
        
        return {
            'embedding_similarity': emb_sim,
            'term_jaccard': jaccard,
            'shared_terms': list(terms1 & terms2),
            'unique_to_first': list(terms1 - terms2),
            'unique_to_second': list(terms2 - terms1)
        }
    
    def classify_lithology(self, text: str) -> Dict[str, float]:
        """Classify text by dominant lithology type."""
        terms = self.tokenizer.identify_geological_terms(text)
        lithologies = terms.get('lithology', [])
        
        if not lithologies:
            return {'unknown': 1.0}
        
        # Count occurrences
        counts = {}
        for lith in lithologies:
            counts[lith] = counts.get(lith, 0) + 1
        
        # Normalize to probabilities
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}
    
    def extract_mineralization_summary(self, text: str) -> Dict[str, Any]:
        """Extract mineralization summary from text."""
        terms = self.tokenizer.identify_geological_terms(text)
        grades = self.tokenizer.extract_grades(text)
        
        minerals = terms.get('mineralization', [])
        alterations = terms.get('alteration', [])
        grade_terms = terms.get('grade_terms', [])
        
        return {
            'minerals': minerals,
            'alterations': alterations,
            'intensity': grade_terms,
            'assay_values': grades,
            'has_economic_minerals': any(
                m in ['gold', 'silver', 'copper', 'lead', 'zinc', 'lithium']
                for m in minerals
            )
        }
    
    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process multiple texts."""
        return [self.process_text(text) for text in texts]
    
    def cluster_documents(
        self,
        documents: List[str],
        n_clusters: int = 5
    ) -> Dict[str, Any]:
        """Cluster documents by similarity."""
        embeddings = self.embeddings.embed(documents)
        
        # Simple k-means clustering
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(n_clusters, len(documents)), random_state=42)
            labels = kmeans.fit_predict(embeddings)
            
            clusters = {}
            for i, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(i)
            
            return {
                'labels': labels.tolist(),
                'clusters': clusters,
                'n_clusters': len(clusters)
            }
        except ImportError:
            logger.warning("sklearn not available for clustering")
            return {'labels': [0] * len(documents), 'clusters': {0: list(range(len(documents)))}}


def create_geo_nlp_processor(
    model_name: Optional[str] = None
) -> GeoNLPProcessor:
    """Factory function to create GeoNLPProcessor."""
    return GeoNLPProcessor(embedding_model=model_name)
