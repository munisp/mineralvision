"""
V-JEPA + LLM Integration Module for MineralVision.

Integrates V-JEPA visual understanding with LLMs (Ollama, OpenAI, etc.) to provide:
- Natural language explanations of V-JEPA findings
- Context-aware reasoning using retrieved documents and historical data
- Actionable recommendations grounded in evidence
- Interactive chat for asking questions about findings

Architecture:
1. V-JEPA provides embeddings, anomaly scores, similarity scores, change metrics
2. Retrieval layer pulls similar historical findings and relevant documents
3. LLM synthesizes human-readable explanations grounded in evidence
4. UI displays structured What/Why/Action with citations

Key principle: LLM increases INTERPRETABILITY and ACTIONABILITY of V-JEPA outputs,
not the underlying vision model's perception. All explanations must be grounded
in retrieved evidence, not speculation.

Usage:
    from api.jepa.llm_integration import create_explanation_service, create_jepa_chat
    
    # Create explanation service
    service = create_explanation_service(ollama_model="mistral")
    
    # Generate explanation for a V-JEPA finding
    explanation = service.explain_finding(
        finding_type="anomaly",
        scores={"anomaly_score": 94, "percentile": 98},
        neighbors=[{"id": "tile_123", "similarity": 0.92, "label": "alteration"}],
        metadata={"location": "Prospect Ridge A", "sensor": "drone_rgb"},
        context_docs=["Geology report mentions silicification in this zone"]
    )
    
    # Interactive chat about findings
    chat = create_jepa_chat(ollama_model="mistral")
    response = chat.ask("Why is this area flagged as anomalous?", finding_id="finding_001")
"""

import json
import logging
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # For testing/mock


class ExplanationType(Enum):
    """Types of explanations the LLM can generate."""
    ANOMALY = "anomaly"
    CHANGE = "change"
    SIMILARITY = "similarity"
    GENERAL = "general"


class DomainContext(Enum):
    """Domain contexts for specialized explanations."""
    GOLD_EXPLORATION = "gold_exploration"
    LITHIUM_EXPLORATION = "lithium_exploration"
    SOIL_ANALYSIS = "soil_analysis"
    GENERAL_MINING = "general_mining"


@dataclass
class RetrievedEvidence:
    """Evidence retrieved for grounding LLM explanations."""
    source_type: str  # "neighbor", "document", "historical_finding", "lab_result"
    source_id: str
    content: str
    relevance_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    citation: Optional[str] = None


@dataclass
class StructuredExplanation:
    """Structured explanation output from LLM."""
    what: str  # Clear observation in operational terms
    why_evidence: str  # Evidence-based explanation with citations
    uncertainty_reasons: List[str]  # Reasons for uncertainty
    recommended_actions: List[str]  # Specific actionable recommendations
    citations: List[str]  # References to evidence sources
    confidence: str  # "high", "medium", "low"
    domain_tags: List[str]  # Domain-specific tags
    raw_response: Optional[str] = None  # Original LLM response for debugging


@dataclass
class ChatMessage:
    """Message in a chat conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatResponse:
    """Response from the chat interface."""
    message: str
    citations: List[str]
    suggested_actions: List[str]
    confidence: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM is available."""
        pass


class OllamaClient(LLMClient):
    """Client for Ollama local LLM deployment."""
    
    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._http_client = None
    
    def _get_client(self):
        """Get or create HTTP client."""
        if self._http_client is None:
            try:
                import httpx
                self._http_client = httpx.Client(timeout=self.timeout)
            except ImportError:
                logger.warning("httpx not available, using urllib")
                self._http_client = "urllib"
        return self._http_client
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a response using Ollama."""
        client = self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        try:
            if client == "urllib":
                import urllib.request
                import urllib.error
                
                req = urllib.request.Request(
                    f"{self.base_url}/api/chat",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    return result.get("message", {}).get("content", "")
            else:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()
                return result.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return f"[LLM unavailable: {str(e)}]"
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            client = self._get_client()
            if client == "urllib":
                import urllib.request
                req = urllib.request.Request(f"{self.base_url}/api/tags")
                with urllib.request.urlopen(req, timeout=5) as response:
                    return response.status == 200
            else:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            client = self._get_client()
            if client == "urllib":
                import urllib.request
                req = urllib.request.Request(f"{self.base_url}/api/tags")
                with urllib.request.urlopen(req, timeout=5) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    return [m["name"] for m in result.get("models", [])]
            else:
                response = client.get(f"{self.base_url}/api/tags")
                result = response.json()
                return [m["name"] for m in result.get("models", [])]
        except Exception:
            return []


class MockLLMClient(LLMClient):
    """Mock LLM client for testing without actual LLM."""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a mock response."""
        self.call_count += 1
        
        # Check for specific response
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        if prompt_hash in self.responses:
            return self.responses[prompt_hash]
        
        # Generate contextual mock response
        if "anomaly" in prompt.lower():
            return json.dumps({
                "what": "Unusual texture pattern detected that differs from baseline imagery",
                "why_evidence": "The area shows 94% similarity to known alteration zones based on nearest neighbor analysis. Historical records indicate silicification in adjacent areas.",
                "uncertainty_reasons": ["Sensor conditions differ from baseline", "Limited historical data for this specific location"],
                "recommended_actions": ["Collect rock chip samples at 25m intervals", "Conduct portable XRF analysis", "Schedule follow-up drone survey at lower altitude"],
                "citations": ["Neighbor tile_123 (92% similar)", "Geology Report 2024-Q3"],
                "confidence": "high",
                "domain_tags": ["alteration", "silicification", "gold_target"]
            })
        elif "change" in prompt.lower():
            return json.dumps({
                "what": "Significant surface change detected over monitoring period",
                "why_evidence": "Time-series analysis shows 12% increase in affected area. Texture evolution suggests active geological or anthropogenic processes.",
                "uncertainty_reasons": ["Seasonal variation may contribute to observed changes"],
                "recommended_actions": ["Investigate change boundaries in field", "Compare with historical satellite imagery", "Document current conditions"],
                "citations": ["Baseline image 2024-01-15", "Current image 2024-04-15"],
                "confidence": "medium",
                "domain_tags": ["change_detection", "monitoring"]
            })
        else:
            return json.dumps({
                "what": "Feature of interest identified in imagery",
                "why_evidence": "Analysis indicates notable characteristics based on V-JEPA embedding comparison.",
                "uncertainty_reasons": ["Further field verification recommended"],
                "recommended_actions": ["Conduct field verification", "Collect additional samples"],
                "citations": ["V-JEPA analysis output"],
                "confidence": "medium",
                "domain_tags": ["general"]
            })
    
    def is_available(self) -> bool:
        """Mock is always available."""
        return True


class EvidenceRetriever:
    """Retrieves evidence for grounding LLM explanations."""
    
    def __init__(
        self,
        vector_index: Optional[Any] = None,
        document_store: Optional[Any] = None,
    ):
        self.vector_index = vector_index
        self.document_store = document_store
        self._cache: Dict[str, List[RetrievedEvidence]] = {}
    
    def retrieve_neighbors(
        self,
        embedding: List[float],
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedEvidence]:
        """Retrieve similar historical findings from vector index."""
        if self.vector_index is None:
            return []
        
        try:
            results = self.vector_index.search(embedding, k=k, filters=filters)
            evidence = []
            for item_id, score in results:
                evidence.append(RetrievedEvidence(
                    source_type="neighbor",
                    source_id=item_id,
                    content=f"Similar finding with {score:.1%} similarity",
                    relevance_score=score,
                    citation=f"Historical finding {item_id}",
                ))
            return evidence
        except Exception as e:
            logger.error(f"Neighbor retrieval failed: {e}")
            return []
    
    def retrieve_documents(
        self,
        query: str,
        k: int = 3,
        doc_types: Optional[List[str]] = None,
    ) -> List[RetrievedEvidence]:
        """Retrieve relevant documents (reports, logs, lab results)."""
        if self.document_store is None:
            return []
        
        try:
            # This would integrate with a document store like Elasticsearch, Milvus, etc.
            # For now, return empty list
            return []
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def retrieve_for_finding(
        self,
        finding_type: str,
        embedding: Optional[List[float]] = None,
        location: Optional[str] = None,
        project: Optional[str] = None,
    ) -> List[RetrievedEvidence]:
        """Retrieve all relevant evidence for a finding."""
        evidence = []
        
        # Retrieve similar neighbors
        if embedding:
            evidence.extend(self.retrieve_neighbors(embedding, k=5))
        
        # Retrieve relevant documents
        query = f"{finding_type} {location or ''} {project or ''}"
        evidence.extend(self.retrieve_documents(query, k=3))
        
        # Sort by relevance
        evidence.sort(key=lambda e: e.relevance_score, reverse=True)
        
        return evidence


class ExplanationService:
    """Service for generating LLM explanations of V-JEPA findings."""
    
    # System prompts for different contexts
    SYSTEM_PROMPTS = {
        DomainContext.GOLD_EXPLORATION: """You are a geological AI assistant helping with gold exploration.
Your role is to explain V-JEPA visual analysis findings in clear, actionable terms for field geologists.

IMPORTANT RULES:
1. NEVER claim to have found gold or ore - only describe visual patterns and similarities
2. Always ground explanations in the provided evidence (scores, neighbors, documents)
3. Clearly state uncertainty and limitations
4. Recommend specific, actionable field verification steps
5. Use geological terminology appropriately but explain for non-specialists

Output must be valid JSON with these fields:
- what: Clear observation in operational terms (1-2 sentences)
- why_evidence: Evidence-based explanation with specific citations
- uncertainty_reasons: List of reasons for uncertainty
- recommended_actions: List of specific actionable recommendations
- citations: List of evidence sources referenced
- confidence: "high", "medium", or "low"
- domain_tags: List of relevant geological/exploration tags""",

        DomainContext.LITHIUM_EXPLORATION: """You are a geological AI assistant helping with lithium exploration.
Your role is to explain V-JEPA visual analysis findings for brine, clay, and pegmatite lithium deposits.

IMPORTANT RULES:
1. NEVER claim to have found lithium deposits - only describe visual patterns and changes
2. Focus on change detection for brine evolution, texture analysis for clay/pegmatite
3. Always ground explanations in the provided evidence
4. Recommend specific sampling and monitoring actions
5. Consider seasonal and environmental factors in uncertainty

Output must be valid JSON with these fields:
- what: Clear observation in operational terms (1-2 sentences)
- why_evidence: Evidence-based explanation with specific citations
- uncertainty_reasons: List of reasons for uncertainty
- recommended_actions: List of specific actionable recommendations
- citations: List of evidence sources referenced
- confidence: "high", "medium", or "low"
- domain_tags: List of relevant lithium exploration tags""",

        DomainContext.SOIL_ANALYSIS: """You are an agricultural AI assistant helping with soil assessment.
Your role is to explain V-JEPA visual analysis findings for crop suitability and soil health.

IMPORTANT RULES:
1. NEVER make definitive soil quality claims without lab verification
2. Focus on visual indicators: texture, color patterns, vegetation stress, erosion
3. Always recommend appropriate soil tests (EC, pH, nutrients) for verification
4. Consider crop-specific requirements when making recommendations
5. Flag potential hazards (salinity, contamination, erosion) clearly

Output must be valid JSON with these fields:
- what: Clear observation in operational terms (1-2 sentences)
- why_evidence: Evidence-based explanation with specific citations
- uncertainty_reasons: List of reasons for uncertainty
- recommended_actions: List of specific actionable recommendations (include soil tests)
- citations: List of evidence sources referenced
- confidence: "high", "medium", or "low"
- domain_tags: List of relevant soil/agriculture tags""",

        DomainContext.GENERAL_MINING: """You are a mining AI assistant helping with visual analysis.
Your role is to explain V-JEPA findings in clear, actionable terms.

IMPORTANT RULES:
1. Ground all explanations in provided evidence
2. Clearly state uncertainty and limitations
3. Recommend specific verification actions
4. Never speculate beyond the evidence

Output must be valid JSON with these fields:
- what: Clear observation in operational terms (1-2 sentences)
- why_evidence: Evidence-based explanation with specific citations
- uncertainty_reasons: List of reasons for uncertainty
- recommended_actions: List of specific actionable recommendations
- citations: List of evidence sources referenced
- confidence: "high", "medium", or "low"
- domain_tags: List of relevant tags"""
    }
    
    def __init__(
        self,
        llm_client: LLMClient,
        evidence_retriever: Optional[EvidenceRetriever] = None,
        default_context: DomainContext = DomainContext.GENERAL_MINING,
    ):
        self.llm_client = llm_client
        self.evidence_retriever = evidence_retriever or EvidenceRetriever()
        self.default_context = default_context
    
    def _build_prompt(
        self,
        finding_type: str,
        scores: Dict[str, float],
        neighbors: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        context_docs: List[str],
        evidence: List[RetrievedEvidence],
    ) -> str:
        """Build the prompt for explanation generation."""
        prompt_parts = [
            f"## V-JEPA Finding Analysis Request",
            f"",
            f"**Finding Type:** {finding_type}",
            f"",
            f"**Scores:**",
        ]
        
        for key, value in scores.items():
            prompt_parts.append(f"- {key}: {value}")
        
        prompt_parts.extend([
            f"",
            f"**Metadata:**",
        ])
        for key, value in metadata.items():
            prompt_parts.append(f"- {key}: {value}")
        
        if neighbors:
            prompt_parts.extend([
                f"",
                f"**Similar Historical Findings (Nearest Neighbors):**",
            ])
            for i, neighbor in enumerate(neighbors[:5], 1):
                label = neighbor.get("label", "unlabeled")
                similarity = neighbor.get("similarity", 0)
                neighbor_id = neighbor.get("id", f"neighbor_{i}")
                prompt_parts.append(f"{i}. {neighbor_id}: {similarity:.1%} similar, labeled as '{label}'")
        
        if evidence:
            prompt_parts.extend([
                f"",
                f"**Retrieved Evidence:**",
            ])
            for i, ev in enumerate(evidence[:5], 1):
                prompt_parts.append(f"{i}. [{ev.source_type}] {ev.content} (relevance: {ev.relevance_score:.1%})")
        
        if context_docs:
            prompt_parts.extend([
                f"",
                f"**Relevant Context Documents:**",
            ])
            for i, doc in enumerate(context_docs[:3], 1):
                prompt_parts.append(f"{i}. {doc[:500]}...")
        
        prompt_parts.extend([
            f"",
            f"Based on the above evidence, provide a structured explanation in JSON format.",
            f"Remember: Ground all claims in the evidence. State uncertainty clearly. Recommend specific actions.",
        ])
        
        return "\n".join(prompt_parts)
    
    def explain_finding(
        self,
        finding_type: str,
        scores: Dict[str, float],
        neighbors: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context_docs: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        domain_context: Optional[DomainContext] = None,
    ) -> StructuredExplanation:
        """Generate a structured explanation for a V-JEPA finding."""
        neighbors = neighbors or []
        metadata = metadata or {}
        context_docs = context_docs or []
        domain_context = domain_context or self.default_context
        
        # Retrieve additional evidence
        evidence = []
        if embedding:
            evidence = self.evidence_retriever.retrieve_for_finding(
                finding_type=finding_type,
                embedding=embedding,
                location=metadata.get("location"),
                project=metadata.get("project"),
            )
        
        # Build prompt
        prompt = self._build_prompt(
            finding_type=finding_type,
            scores=scores,
            neighbors=neighbors,
            metadata=metadata,
            context_docs=context_docs,
            evidence=evidence,
        )
        
        # Get system prompt for domain
        system_prompt = self.SYSTEM_PROMPTS.get(
            domain_context,
            self.SYSTEM_PROMPTS[DomainContext.GENERAL_MINING]
        )
        
        # Generate explanation
        raw_response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=1024,
        )
        
        # Parse response
        try:
            # Try to extract JSON from response
            json_start = raw_response.find("{")
            json_end = raw_response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = raw_response[json_start:json_end]
                parsed = json.loads(json_str)
                
                return StructuredExplanation(
                    what=parsed.get("what", "Analysis complete"),
                    why_evidence=parsed.get("why_evidence", "See scores and neighbors above"),
                    uncertainty_reasons=parsed.get("uncertainty_reasons", []),
                    recommended_actions=parsed.get("recommended_actions", []),
                    citations=parsed.get("citations", []),
                    confidence=parsed.get("confidence", "medium"),
                    domain_tags=parsed.get("domain_tags", []),
                    raw_response=raw_response,
                )
        except json.JSONDecodeError:
            pass
        
        # Fallback if JSON parsing fails
        return StructuredExplanation(
            what=f"{finding_type.title()} detected in imagery",
            why_evidence=raw_response[:500] if raw_response else "Analysis complete",
            uncertainty_reasons=["LLM response parsing failed"],
            recommended_actions=["Review raw analysis output", "Conduct field verification"],
            citations=[],
            confidence="low",
            domain_tags=[finding_type],
            raw_response=raw_response,
        )
    
    def batch_explain(
        self,
        findings: List[Dict[str, Any]],
        domain_context: Optional[DomainContext] = None,
    ) -> List[StructuredExplanation]:
        """Generate explanations for multiple findings."""
        explanations = []
        for finding in findings:
            explanation = self.explain_finding(
                finding_type=finding.get("type", "general"),
                scores=finding.get("scores", {}),
                neighbors=finding.get("neighbors"),
                metadata=finding.get("metadata"),
                context_docs=finding.get("context_docs"),
                embedding=finding.get("embedding"),
                domain_context=domain_context,
            )
            explanations.append(explanation)
        return explanations


class JEPAChat:
    """Interactive chat interface for asking questions about V-JEPA findings."""
    
    SYSTEM_PROMPT = """You are an AI assistant helping users understand V-JEPA visual analysis findings.
You have access to the following tools:
- get_finding_details(finding_id): Get details about a specific finding
- search_similar(query): Search for similar historical findings
- get_baseline_info(location): Get baseline information for a location

IMPORTANT RULES:
1. Only answer questions based on available evidence and tool results
2. If you don't have enough information, say so clearly
3. Never speculate about geological interpretations without evidence
4. Recommend field verification for any actionable conclusions
5. Keep responses concise and actionable

When using tools, format as: [TOOL: tool_name(args)]
After tool results, provide a clear answer to the user's question."""
    
    def __init__(
        self,
        llm_client: LLMClient,
        explanation_service: Optional[ExplanationService] = None,
        max_history: int = 10,
    ):
        self.llm_client = llm_client
        self.explanation_service = explanation_service
        self.max_history = max_history
        self.conversations: Dict[str, List[ChatMessage]] = {}
    
    def _get_or_create_conversation(self, conversation_id: str) -> List[ChatMessage]:
        """Get or create a conversation history."""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        return self.conversations[conversation_id]
    
    def _build_chat_prompt(
        self,
        question: str,
        history: List[ChatMessage],
        finding_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the chat prompt with history and context."""
        prompt_parts = []
        
        # Add finding context if available
        if finding_context:
            prompt_parts.append("## Current Finding Context")
            prompt_parts.append(json.dumps(finding_context, indent=2))
            prompt_parts.append("")
        
        # Add conversation history
        if history:
            prompt_parts.append("## Conversation History")
            for msg in history[-self.max_history:]:
                prompt_parts.append(f"**{msg.role.title()}:** {msg.content}")
            prompt_parts.append("")
        
        # Add current question
        prompt_parts.append(f"## Current Question")
        prompt_parts.append(question)
        
        return "\n".join(prompt_parts)
    
    def ask(
        self,
        question: str,
        finding_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        finding_context: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        """Ask a question about V-JEPA findings."""
        conversation_id = conversation_id or "default"
        history = self._get_or_create_conversation(conversation_id)
        
        # Build prompt
        prompt = self._build_chat_prompt(
            question=question,
            history=history,
            finding_context=finding_context,
        )
        
        # Generate response
        raw_response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.5,
            max_tokens=512,
        )
        
        # Parse tool calls (if any)
        tool_calls = []
        response_text = raw_response
        
        # Simple tool call parsing
        import re
        tool_pattern = r'\[TOOL:\s*(\w+)\((.*?)\)\]'
        matches = re.findall(tool_pattern, raw_response)
        for tool_name, args in matches:
            tool_calls.append({"tool": tool_name, "args": args})
            response_text = response_text.replace(f"[TOOL: {tool_name}({args})]", "")
        
        # Extract citations
        citations = []
        citation_pattern = r'\[(?:cite|ref|source):\s*([^\]]+)\]'
        citation_matches = re.findall(citation_pattern, raw_response, re.IGNORECASE)
        citations.extend(citation_matches)
        
        # Extract suggested actions
        suggested_actions = []
        if "recommend" in raw_response.lower() or "suggest" in raw_response.lower():
            action_pattern = r'(?:recommend|suggest)[^.]*\.?'
            action_matches = re.findall(action_pattern, raw_response, re.IGNORECASE)
            suggested_actions.extend(action_matches[:3])
        
        # Determine confidence
        confidence = "medium"
        if "uncertain" in raw_response.lower() or "unclear" in raw_response.lower():
            confidence = "low"
        elif "confident" in raw_response.lower() or "clearly" in raw_response.lower():
            confidence = "high"
        
        # Update history
        history.append(ChatMessage(role="user", content=question))
        history.append(ChatMessage(role="assistant", content=response_text.strip()))
        
        return ChatResponse(
            message=response_text.strip(),
            citations=citations,
            suggested_actions=suggested_actions,
            confidence=confidence,
            tool_calls=tool_calls,
        )
    
    def clear_conversation(self, conversation_id: str = "default") -> None:
        """Clear conversation history."""
        if conversation_id in self.conversations:
            self.conversations[conversation_id] = []


class JEPAOrchestrator:
    """LLM-based orchestrator for V-JEPA analysis jobs."""
    
    SYSTEM_PROMPT = """You are an AI orchestrator for V-JEPA visual analysis.
Your role is to translate user requests into analysis job specifications.

Available analysis types:
- anomaly_scan: Scan an area for anomalies compared to baseline
- change_detection: Compare imagery from different dates
- similarity_search: Find areas similar to a reference

Required job parameters:
- analysis_type: One of the above types
- area_of_interest: Geographic bounds or location name
- baseline: Reference for comparison (date, dataset, or "site_baseline")
- thresholds: Sensitivity settings (anomaly_threshold, change_threshold)

Output must be valid JSON with the job specification."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
    
    def parse_request(self, user_request: str) -> Dict[str, Any]:
        """Parse a natural language request into a job specification."""
        prompt = f"""User request: {user_request}

Parse this into a V-JEPA analysis job specification.
Output valid JSON with: analysis_type, area_of_interest, baseline, thresholds, and any other relevant parameters."""
        
        response = self.llm_client.generate(
            prompt=prompt,
            system_prompt=self.SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=512,
        )
        
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except json.JSONDecodeError:
            pass
        
        return {
            "analysis_type": "anomaly_scan",
            "area_of_interest": "default",
            "baseline": "site_baseline",
            "thresholds": {"anomaly_threshold": 0.8},
            "raw_request": user_request,
            "parse_error": "Could not parse request",
        }


# Factory functions

def create_llm_client(
    provider: LLMProvider = LLMProvider.OLLAMA,
    model: str = "mistral",
    **kwargs,
) -> LLMClient:
    """Create an LLM client for the specified provider."""
    if provider == LLMProvider.OLLAMA:
        return OllamaClient(model=model, **kwargs)
    elif provider == LLMProvider.LOCAL:
        return MockLLMClient(**kwargs)
    else:
        # Default to mock for unsupported providers
        logger.warning(f"Provider {provider} not fully implemented, using mock")
        return MockLLMClient(**kwargs)


def create_explanation_service(
    ollama_model: str = "mistral",
    ollama_url: str = "http://localhost:11434",
    domain_context: DomainContext = DomainContext.GENERAL_MINING,
    use_mock: bool = False,
) -> ExplanationService:
    """Create an explanation service with Ollama backend."""
    if use_mock:
        llm_client = MockLLMClient()
    else:
        llm_client = OllamaClient(model=ollama_model, base_url=ollama_url)
    
    return ExplanationService(
        llm_client=llm_client,
        default_context=domain_context,
    )


def create_jepa_chat(
    ollama_model: str = "mistral",
    ollama_url: str = "http://localhost:11434",
    use_mock: bool = False,
) -> JEPAChat:
    """Create a chat interface for V-JEPA findings."""
    if use_mock:
        llm_client = MockLLMClient()
    else:
        llm_client = OllamaClient(model=ollama_model, base_url=ollama_url)
    
    return JEPAChat(llm_client=llm_client)


def create_orchestrator(
    ollama_model: str = "mistral",
    ollama_url: str = "http://localhost:11434",
    use_mock: bool = False,
) -> JEPAOrchestrator:
    """Create an orchestrator for V-JEPA analysis jobs."""
    if use_mock:
        llm_client = MockLLMClient()
    else:
        llm_client = OllamaClient(model=ollama_model, base_url=ollama_url)
    
    return JEPAOrchestrator(llm_client=llm_client)


# Convenience function for quick explanations
def explain_jepa_finding(
    finding_type: str,
    anomaly_score: Optional[float] = None,
    similarity_score: Optional[float] = None,
    change_percentage: Optional[float] = None,
    location: Optional[str] = None,
    domain: str = "general",
    use_mock: bool = True,
) -> StructuredExplanation:
    """Quick function to explain a V-JEPA finding."""
    scores = {}
    if anomaly_score is not None:
        scores["anomaly_score"] = anomaly_score
    if similarity_score is not None:
        scores["similarity_score"] = similarity_score
    if change_percentage is not None:
        scores["change_percentage"] = change_percentage
    
    metadata = {}
    if location:
        metadata["location"] = location
    
    domain_map = {
        "gold": DomainContext.GOLD_EXPLORATION,
        "lithium": DomainContext.LITHIUM_EXPLORATION,
        "soil": DomainContext.SOIL_ANALYSIS,
        "general": DomainContext.GENERAL_MINING,
    }
    
    service = create_explanation_service(
        domain_context=domain_map.get(domain, DomainContext.GENERAL_MINING),
        use_mock=use_mock,
    )
    
    return service.explain_finding(
        finding_type=finding_type,
        scores=scores,
        metadata=metadata,
    )
