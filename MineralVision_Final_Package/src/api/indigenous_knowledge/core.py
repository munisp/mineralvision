"""
Indigenous Knowledge Integration Core Module

This module provides the core functionality for integrating indigenous knowledge
into the MineralVision platform, enabling respectful incorporation of traditional
knowledge into the mineral exploration process.

The indigenous knowledge integration system enables:
1. Collaborative mapping interfaces for traditional knowledge
2. Cultural heritage site identification and protection
3. Benefit-sharing tracking mechanisms
4. Consultation workflow management
"""

import uuid
import datetime
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from shapely.geometry import Point, Polygon, shape
import geopandas as gpd

class KnowledgeHolder:
    """Representation of an indigenous knowledge holder."""
    
    def __init__(
        self, 
        holder_id: str = None, 
        name: str = None,
        community: str = None,
        role: str = None,
        contact_info: Dict = None,
        metadata: Dict = None
    ):
        """
        Initialize a knowledge holder.
        
        Args:
            holder_id: Unique identifier for the knowledge holder
            name: Name of the knowledge holder
            community: Community or nation of the knowledge holder
            role: Role or position of the knowledge holder
            contact_info: Contact information for the knowledge holder
            metadata: Additional metadata
        """
        self.holder_id = holder_id or str(uuid.uuid4())
        self.name = name or "Anonymous"
        self.community = community
        self.role = role
        self.contact_info = contact_info or {}
        self.metadata = metadata or {}
        self.created_at = datetime.datetime.now()
        self.updated_at = self.created_at
        self.knowledge_contributions = []
        
    def add_knowledge_contribution(self, knowledge_id: str) -> None:
        """
        Add a knowledge contribution to the knowledge holder.
        
        Args:
            knowledge_id: ID of the knowledge contribution
        """
        if knowledge_id not in self.knowledge_contributions:
            self.knowledge_contributions.append(knowledge_id)
            self.updated_at = datetime.datetime.now()
            
    def to_dict(self) -> Dict:
        """
        Convert the knowledge holder to a dictionary.
        
        Returns:
            Dictionary representation of the knowledge holder
        """
        return {
            "holder_id": self.holder_id,
            "name": self.name,
            "community": self.community,
            "role": self.role,
            "contact_info": self.contact_info,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "knowledge_contributions": self.knowledge_contributions
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeHolder':
        """
        Create a knowledge holder from a dictionary.
        
        Args:
            data: Dictionary representation of the knowledge holder
            
        Returns:
            New knowledge holder instance
        """
        holder = cls(
            holder_id=data.get("holder_id"),
            name=data.get("name"),
            community=data.get("community"),
            role=data.get("role"),
            contact_info=data.get("contact_info"),
            metadata=data.get("metadata")
        )
        
        holder.knowledge_contributions = data.get("knowledge_contributions", [])
        
        if "created_at" in data:
            holder.created_at = datetime.datetime.fromisoformat(data["created_at"])
        
        if "updated_at" in data:
            holder.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
            
        return holder


class TraditionalKnowledge:
    """Representation of traditional knowledge in the system."""
    
    def __init__(
        self, 
        knowledge_id: str = None, 
        title: str = None,
        description: str = None,
        knowledge_type: str = None,
        holder_id: str = None,
        community: str = None,
        geometry: Any = None,
        coordinate_system: str = "EPSG:4326",
        access_level: str = "public",
        metadata: Dict = None
    ):
        """
        Initialize a traditional knowledge entry.
        
        Args:
            knowledge_id: Unique identifier for the knowledge
            title: Title of the knowledge
            description: Description of the knowledge
            knowledge_type: Type of knowledge (e.g., cultural_site, resource_area)
            holder_id: ID of the knowledge holder
            community: Community or nation associated with the knowledge
            geometry: Shapely geometry object
            coordinate_system: Coordinate reference system
            access_level: Access level for the knowledge (public, restricted, confidential)
            metadata: Additional metadata
        """
        self.knowledge_id = knowledge_id or str(uuid.uuid4())
        self.title = title or f"Knowledge-{self.knowledge_id[:8]}"
        self.description = description or ""
        self.knowledge_type = knowledge_type
        self.holder_id = holder_id
        self.community = community
        self.geometry = geometry
        self.coordinate_system = coordinate_system
        self.access_level = access_level
        self.metadata = metadata or {}
        self.created_at = datetime.datetime.now()
        self.updated_at = self.created_at
        self.attachments = []
        self.verification_status = "unverified"
        self.verification_notes = ""
        
    def set_point_geometry(self, latitude: float, longitude: float) -> None:
        """
        Set the geometry to a point.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
        """
        self.geometry = Point(longitude, latitude)
        
    def set_polygon_geometry(self, coordinates: List[Tuple[float, float]]) -> None:
        """
        Set the geometry to a polygon.
        
        Args:
            coordinates: List of (longitude, latitude) tuples
        """
        self.geometry = Polygon(coordinates)
        
    def add_attachment(self, attachment_path: str, attachment_type: str, description: str = None) -> None:
        """
        Add an attachment to the knowledge.
        
        Args:
            attachment_path: Path to the attachment file
            attachment_type: Type of attachment (e.g., image, audio, document)
            description: Description of the attachment
        """
        attachment = {
            "id": str(uuid.uuid4()),
            "path": attachment_path,
            "type": attachment_type,
            "description": description or "",
            "added_at": datetime.datetime.now().isoformat()
        }
        
        self.attachments.append(attachment)
        self.updated_at = datetime.datetime.now()
        
    def verify(self, verified_by: str, notes: str = None) -> None:
        """
        Verify the knowledge.
        
        Args:
            verified_by: ID or name of the verifier
            notes: Verification notes
        """
        self.verification_status = "verified"
        self.verification_notes = notes or f"Verified by {verified_by}"
        self.metadata["verified_by"] = verified_by
        self.metadata["verified_at"] = datetime.datetime.now().isoformat()
        self.updated_at = datetime.datetime.now()
        
    def to_dict(self) -> Dict:
        """
        Convert the traditional knowledge to a dictionary.
        
        Returns:
            Dictionary representation of the traditional knowledge
        """
        data = {
            "knowledge_id": self.knowledge_id,
            "title": self.title,
            "description": self.description,
            "knowledge_type": self.knowledge_type,
            "holder_id": self.holder_id,
            "community": self.community,
            "access_level": self.access_level,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "attachments": self.attachments,
            "verification_status": self.verification_status,
            "verification_notes": self.verification_notes
        }
        
        if self.geometry:
            data["geometry"] = json.loads(gpd.GeoSeries([self.geometry]).to_json())["features"][0]["geometry"]
            data["coordinate_system"] = self.coordinate_system
            
        return data
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'TraditionalKnowledge':
        """
        Create a traditional knowledge entry from a dictionary.
        
        Args:
            data: Dictionary representation of the traditional knowledge
            
        Returns:
            New traditional knowledge instance
        """
        knowledge = cls(
            knowledge_id=data.get("knowledge_id"),
            title=data.get("title"),
            description=data.get("description"),
            knowledge_type=data.get("knowledge_type"),
            holder_id=data.get("holder_id"),
            community=data.get("community"),
            access_level=data.get("access_level"),
            metadata=data.get("metadata")
        )
        
        if "geometry" in data:
            knowledge.geometry = shape(data["geometry"])
            knowledge.coordinate_system = data.get("coordinate_system", "EPSG:4326")
            
        knowledge.attachments = data.get("attachments", [])
        knowledge.verification_status = data.get("verification_status", "unverified")
        knowledge.verification_notes = data.get("verification_notes", "")
        
        if "created_at" in data:
            knowledge.created_at = datetime.datetime.fromisoformat(data["created_at"])
        
        if "updated_at" in data:
            knowledge.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
            
        return knowledge


class CulturalHeritageSite(TraditionalKnowledge):
    """Representation of a cultural heritage site."""
    
    def __init__(
        self, 
        knowledge_id: str = None, 
        title: str = None,
        description: str = None,
        holder_id: str = None,
        community: str = None,
        geometry: Any = None,
        coordinate_system: str = "EPSG:4326",
        access_level: str = "public",
        metadata: Dict = None,
        site_type: str = None,
        significance: str = None,
        protection_level: str = "standard",
        seasonal_restrictions: List[str] = None
    ):
        """
        Initialize a cultural heritage site.
        
        Args:
            knowledge_id: Unique identifier for the knowledge
            title: Title of the knowledge
            description: Description of the knowledge
            holder_id: ID of the knowledge holder
            community: Community or nation associated with the knowledge
            geometry: Shapely geometry object
            coordinate_system: Coordinate reference system
            access_level: Access level for the knowledge (public, restricted, confidential)
            metadata: Additional metadata
            site_type: Type of cultural heritage site
            significance: Significance of the site
            protection_level: Level of protection required (standard, high, critical)
            seasonal_restrictions: List of seasonal restrictions
        """
        super().__init__(
            knowledge_id=knowledge_id,
            title=title,
            description=description,
            knowledge_type="cultural_heritage_site",
            holder_id=holder_id,
            community=community,
            geometry=geometry,
            coordinate_system=coordinate_system,
            access_level=access_level,
            metadata=metadata
        )
        
        self.site_type = site_type
        self.significance = significance
        self.protection_level = protection_level
        self.seasonal_restrictions = seasonal_restrictions or []
        
    def to_dict(self) -> Dict:
        """
        Convert the cultural heritage site to a dictionary.
        
        Returns:
            Dictionary representation of the cultural heritage site
        """
        data = super().to_dict()
        data["site_type"] = self.site_type
        data["significance"] = self.significance
        data["protection_level"] = self.protection_level
        data["seasonal_restrictions"] = self.seasonal_restrictions
        return data
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'CulturalHeritageSite':
        """
        Create a cultural heritage site from a dictionary.
        
        Args:
            data: Dictionary representation of the cultural heritage site
            
        Returns:
            New cultural heritage site instance
        """
        site = super().from_dict(data)
        site.site_type = data.get("site_type")
        site.significance = data.get("significance")
        site.protection_level = data.get("protection_level", "standard")
        site.seasonal_restrictions = data.get("seasonal_restrictions", [])
        return site


class ResourceArea(TraditionalKnowledge):
    """Representation of a traditional resource area."""
    
    def __init__(
        self, 
        knowledge_id: str = None, 
        title: str = None,
        description: str = None,
        holder_id: str = None,
        community: str = None,
        geometry: Any = None,
        coordinate_system: str = "EPSG:4326",
        access_level: str = "public",
        metadata: Dict = None,
        resource_type: str = None,
        seasonal_use: Dict[str, str] = None,
        current_use: bool = True,
        sustainability_practices: List[str] = None
    ):
        """
        Initialize a traditional resource area.
        
        Args:
            knowledge_id: Unique identifier for the knowledge
            title: Title of the knowledge
            description: Description of the knowledge
            holder_id: ID of the knowledge holder
            community: Community or nation associated with the knowledge
            geometry: Shapely geometry object
            coordinate_system: Coordinate reference system
            access_level: Access level for the knowledge (public, restricted, confidential)
            metadata: Additional metadata
            resource_type: Type of resource
            seasonal_use: Dictionary mapping seasons to use descriptions
            current_use: Whether the area is currently in use
            sustainability_practices: List of sustainability practices
        """
        super().__init__(
            knowledge_id=knowledge_id,
            title=title,
            description=description,
            knowledge_type="resource_area",
            holder_id=holder_id,
            community=community,
            geometry=geometry,
            coordinate_system=coordinate_system,
            access_level=access_level,
            metadata=metadata
        )
        
        self.resource_type = resource_type
        self.seasonal_use = seasonal_use or {}
        self.current_use = current_use
        self.sustainability_practices = sustainability_practices or []
        
    def to_dict(self) -> Dict:
        """
        Convert the resource area to a dictionary.
        
        Returns:
            Dictionary representation of the resource area
        """
        data = super().to_dict()
        data["resource_type"] = self.resource_type
        data["seasonal_use"] = self.seasonal_use
        data["current_use"] = self.current_use
        data["sustainability_practices"] = self.sustainability_practices
        return data
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'ResourceArea':
        """
        Create a resource area from a dictionary.
        
        Args:
            data: Dictionary representation of the resource area
            
        Returns:
            New resource area instance
        """
        area = super().from_dict(data)
        area.resource_type = data.get("resource_type")
        area.seasonal_use = data.get("seasonal_use", {})
        area.current_use = data.get("current_use", True)
        area.sustainability_practices = data.get("sustainability_practices", [])
        return area


class ConsultationRecord:
    """Record of a consultation with indigenous communities."""
    
    def __init__(
        self, 
        consultation_id: str = None, 
        title: str = None,
        description: str = None,
        community: str = None,
        consultation_date: datetime.date = None,
        participants: List[Dict] = None,
        status: str = "planned",
        outcomes: List[str] = None,
        follow_up_actions: List[Dict] = None,
        attachments: List[Dict] = None,
        metadata: Dict = None
    ):
        """
        Initialize a consultation record.
        
        Args:
            consultation_id: Unique identifier for the consultation
            title: Title of the consultation
            description: Description of the consultation
            community: Community or nation consulted
            consultation_date: Date of the consultation
            participants: List of participants
            status: Status of the consultation (planned, completed, cancelled)
            outcomes: List of outcomes
            follow_up_actions: List of follow-up actions
            attachments: List of attachments
            metadata: Additional metadata
        """
        self.consultation_id = consultation_id or str(uuid.uuid4())
        self.title = title or f"Consultation-{self.consultation_id[:8]}"
        self.description = description or ""
        self.community = community
        self.consultation_date = consultation_date or datetime.date.today()
        self.participants = participants or []
        self.status = status
        self.outcomes = outcomes or []
        self.follow_up_actions = follow_up_actions or []
        self.attachments = attachments or []
        self.metadata = metadata or {}
        self.created_at = datetime.datetime.now()
        self.updated_at = self.created_at
        
    def add_participant(self, name: str, role: str, organization: str = None, contact_info: Dict = None) -> None:
        """
        Add a participant to the consultation.
        
        Args:
            name: Name of the participant
            role: Role of the participant
            organization: Organization of the participant
            contact_info: Contact information for the participant
        """
        participant = {
            "id": str(uuid.uuid4()),
            "name": name,
            "role": role,
            "organization": organization,
            "contact_info": contact_info or {}
        }
        
        self.participants.append(participant)
        self.updated_at = datetime.datetime.now()
        
    def add_outcome(self, outcome: str) -> None:
        """
        Add an outcome to the consultation.
        
        Args:
            outcome: Outcome of the consultation
        """
        self.outcomes.append(outcome)
        self.updated_at = datetime.datetime.now()
        
    def add_follow_up_action(self, action: str, responsible: str, due_date: datetime.date = None, status: str = "pending") -> None:
        """
        Add a follow-up action to the consultation.
        
        Args:
            action: Description of the action
            responsible: Person or organization responsible for the action
            due_date: Due date for the action
            status: Status of the action (pending, in_progress, completed)
        """
        follow_up = {
            "id": str(uuid.uuid4()),
            "action": action,
            "responsible": responsible,
            "due_date": due_date.isoformat() if due_date else None,
            "status": status,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        self.follow_up_actions.append(follow_up)
        self.updated_at = datetime.datetime.now()
        
    def add_attachment(self, attachment_path: str, attachment_type: str, description: str = None) -> None:
        """
        Add an attachment to the consultation.
        
        Args:
            attachment_path: Path to the attachment file
            attachment_type: Type of attachment (e.g., minutes, presentation, agreement)
            description: Description of the attachment
        """
        attachment = {
            "id": str(uuid.uuid4()),
            "path": attachment_path,
            "type": attachment_type,
            "description": description or "",
            "added_at": datetime.datetime.now().isoformat()
        }
        
        self.attachments.append(attachment)
        self.updated_at = datetime.datetime.now()
        
    def complete_consultation(self) -> None:
        """Mark the consultation as completed."""
        self.status = "completed"
        self.updated_at = datetime.datetime.now()
        
    def cancel_consultation(self, reason: str = None) -> None:
        """
        Mark the consultation as cancelled.
        
        Args:
            reason: Reason for cancellation
        """
        self.status = "cancelled"
        if reason:
            self.metadata["cancellation_reason"] = reason
        self.updated_at = datetime.datetime.now()
        
    def to_dict(self) -> Dict:
        """
        Convert the consultation record to a dictionary.
        
        Returns:
            Dictionary representation of the consultation record
        """
        return {
            "consultation_id": self.consultation_id,
            "title": self.title,
            "description": self.description,
            "community": self.community,
            "consultation_date": self.consultation_date.isoformat(),
            "participants": self.participants,
            "status": self.status,
            "outcomes": self.outcomes,
            "follow_up_actions": self.follow_up_actions,
            "attachments": self.attachments,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConsultationRecord':
        """
        Create a consultation record from a dictionary.
        
        Args:
            data: Dictionary representation of the consultation record
            
        Returns:
            New consultation record instance
        """
        consultation = cls(
            consultation_id=data.get("consultation_id"),
            title=data.get("title"),
            description=data.get("description"),
            community=data.get("community"),
            consultation_date=datetime.date.fromisoformat(data["consultation_date"]) if "consultation_date" in data else None,
            participants=data.get("participants"),
            status=data.get("status"),
            outcomes=data.get("outcomes"),
            follow_up_actions=data.get("follow_up_actions"),
            attachments=data.get("attachments"),
            metadata=data.get("metadata")
        )
        
        if "created_at" in data:
            consultation.created_at = datetime.datetime.fromisoformat(data["created_at"])
        
        if "updated_at" in data:
            consultation.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
            
        return consultation


class BenefitSharingAgreement:
    """Representation of a benefit sharing agreement."""
    
    def __init__(
        self, 
        agreement_id: str = None, 
        title: str = None,
        description: str = None,
        community: str = None,
        start_date: datetime.date = None,
        end_date: datetime.date = None,
        status: str = "draft",
        agreement_type: str = None,
        benefits: List[Dict] = None,
        commitments: List[Dict] = None,
        attachments: List[Dict] = None,
        metadata: Dict = None
    ):
        """
        Initialize a benefit sharing agreement.
        
        Args:
            agreement_id: Unique identifier for the agreement
            title: Title of the agreement
            description: Description of the agreement
            community: Community or nation involved in the agreement
            start_date: Start date of the agreement
            end_date: End date of the agreement
            status: Status of the agreement (draft, active, expired, terminated)
            agreement_type: Type of agreement
            benefits: List of benefits
            commitments: List of commitments
            attachments: List of attachments
            metadata: Additional metadata
        """
        self.agreement_id = agreement_id or str(uuid.uuid4())
        self.title = title or f"Agreement-{self.agreement_id[:8]}"
        self.description = description or ""
        self.community = community
        self.start_date = start_date
        self.end_date = end_date
        self.status = status
        self.agreement_type = agreement_type
        self.benefits = benefits or []
        self.commitments = commitments or []
        self.attachments = attachments or []
        self.metadata = metadata or {}
        self.created_at = datetime.datetime.now()
        self.updated_at = self.created_at
        
    def add_benefit(self, benefit_type: str, description: str, value: float = None, schedule: str = None) -> None:
        """
        Add a benefit to the agreement.
        
        Args:
            benefit_type: Type of benefit (e.g., financial, employment, education)
            description: Description of the benefit
            value: Monetary value of the benefit
            schedule: Schedule for providing the benefit
        """
        benefit = {
            "id": str(uuid.uuid4()),
            "type": benefit_type,
            "description": description,
            "value": value,
            "schedule": schedule,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        self.benefits.append(benefit)
        self.updated_at = datetime.datetime.now()
        
    def add_commitment(self, commitment_type: str, description: str, responsible: str, due_date: datetime.date = None, status: str = "pending") -> None:
        """
        Add a commitment to the agreement.
        
        Args:
            commitment_type: Type of commitment
            description: Description of the commitment
            responsible: Person or organization responsible for the commitment
            due_date: Due date for the commitment
            status: Status of the commitment (pending, in_progress, completed)
        """
        commitment = {
            "id": str(uuid.uuid4()),
            "type": commitment_type,
            "description": description,
            "responsible": responsible,
            "due_date": due_date.isoformat() if due_date else None,
            "status": status,
            "created_at": datetime.datetime.now().isoformat()
        }
        
        self.commitments.append(commitment)
        self.updated_at = datetime.datetime.now()
        
    def add_attachment(self, attachment_path: str, attachment_type: str, description: str = None) -> None:
        """
        Add an attachment to the agreement.
        
        Args:
            attachment_path: Path to the attachment file
            attachment_type: Type of attachment (e.g., agreement, addendum, report)
            description: Description of the attachment
        """
        attachment = {
            "id": str(uuid.uuid4()),
            "path": attachment_path,
            "type": attachment_type,
            "description": description or "",
            "added_at": datetime.datetime.now().isoformat()
        }
        
        self.attachments.append(attachment)
        self.updated_at = datetime.datetime.now()
        
    def activate_agreement(self) -> None:
        """Mark the agreement as active."""
        self.status = "active"
        self.updated_at = datetime.datetime.now()
        
    def terminate_agreement(self, reason: str = None) -> None:
        """
        Mark the agreement as terminated.
        
        Args:
            reason: Reason for termination
        """
        self.status = "terminated"
        if reason:
            self.metadata["termination_reason"] = reason
        self.updated_at = datetime.datetime.now()
        
    def to_dict(self) -> Dict:
        """
        Convert the benefit sharing agreement to a dictionary.
        
        Returns:
            Dictionary representation of the benefit sharing agreement
        """
        data = {
            "agreement_id": self.agreement_id,
            "title": self.title,
            "description": self.description,
            "community": self.community,
            "status": self.status,
            "agreement_type": self.agreement_type,
            "benefits": self.benefits,
            "commitments": self.commitments,
            "attachments": self.attachments,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
        
        if self.start_date:
            data["start_date"] = self.start_date.isoformat()
            
        if self.end_date:
            data["end_date"] = self.end_date.isoformat()
            
        return data
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'BenefitSharingAgreement':
        """
        Create a benefit sharing agreement from a dictionary.
        
        Args:
            data: Dictionary representation of the benefit sharing agreement
            
        Returns:
            New benefit sharing agreement instance
        """
        agreement = cls(
            agreement_id=data.get("agreement_id"),
            title=data.get("title"),
            description=data.get("description"),
            community=data.get("community"),
            start_date=datetime.date.fromisoformat(data["start_date"]) if "start_date" in data else None,
            end_date=datetime.date.fromisoformat(data["end_date"]) if "end_date" in data else None,
            status=data.get("status"),
            agreement_type=data.get("agreement_type"),
            benefits=data.get("benefits"),
            commitments=data.get("commitments"),
            attachments=data.get("attachments"),
            metadata=data.get("metadata")
        )
        
        if "created_at" in data:
            agreement.created_at = datetime.datetime.fromisoformat(data["created_at"])
        
        if "updated_at" in data:
            agreement.updated_at = datetime.datetime.fromisoformat(data["updated_at"])
            
        return agreement


class IndigenousKnowledgeManager:
    """Manager for the indigenous knowledge integration system."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the indigenous knowledge manager.
        
        Args:
            data_dir: Directory for storing data files
        """
        self.knowledge_holders = {}
        self.traditional_knowledge = {}
        self.consultations = {}
        self.agreements = {}
        self.data_dir = data_dir or os.path.join(os.getcwd(), "indigenous_knowledge_data")
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
    def add_knowledge_holder(self, holder: KnowledgeHolder) -> str:
        """
        Add a knowledge holder to the system.
        
        Args:
            holder: Knowledge holder to add
            
        Returns:
            Holder ID
        """
        self.knowledge_holders[holder.holder_id] = holder
        return holder.holder_id
        
    def get_knowledge_holder(self, holder_id: str) -> Optional[KnowledgeHolder]:
        """
        Get a knowledge holder from the system.
        
        Args:
            holder_id: ID of the knowledge holder to get
            
        Returns:
            Knowledge holder if found, None otherwise
        """
        return self.knowledge_holders.get(holder_id)
        
    def add_traditional_knowledge(self, knowledge: TraditionalKnowledge) -> str:
        """
        Add traditional knowledge to the system.
        
        Args:
            knowledge: Traditional knowledge to add
            
        Returns:
            Knowledge ID
        """
        self.traditional_knowledge[knowledge.knowledge_id] = knowledge
        
        # Update knowledge holder's contributions if holder exists
        if knowledge.holder_id and knowledge.holder_id in self.knowledge_holders:
            self.knowledge_holders[knowledge.holder_id].add_knowledge_contribution(knowledge.knowledge_id)
            
        return knowledge.knowledge_id
        
    def get_traditional_knowledge(self, knowledge_id: str) -> Optional[TraditionalKnowledge]:
        """
        Get traditional knowledge from the system.
        
        Args:
            knowledge_id: ID of the traditional knowledge to get
            
        Returns:
            Traditional knowledge if found, None otherwise
        """
        return self.traditional_knowledge.get(knowledge_id)
        
    def add_consultation(self, consultation: ConsultationRecord) -> str:
        """
        Add a consultation record to the system.
        
        Args:
            consultation: Consultation record to add
            
        Returns:
            Consultation ID
        """
        self.consultations[consultation.consultation_id] = consultation
        return consultation.consultation_id
        
    def get_consultation(self, consultation_id: str) -> Optional[ConsultationRecord]:
        """
        Get a consultation record from the system.
        
        Args:
            consultation_id: ID of the consultation record to get
            
        Returns:
            Consultation record if found, None otherwise
        """
        return self.consultations.get(consultation_id)
        
    def add_agreement(self, agreement: BenefitSharingAgreement) -> str:
        """
        Add a benefit sharing agreement to the system.
        
        Args:
            agreement: Benefit sharing agreement to add
            
        Returns:
            Agreement ID
        """
        self.agreements[agreement.agreement_id] = agreement
        return agreement.agreement_id
        
    def get_agreement(self, agreement_id: str) -> Optional[BenefitSharingAgreement]:
        """
        Get a benefit sharing agreement from the system.
        
        Args:
            agreement_id: ID of the benefit sharing agreement to get
            
        Returns:
            Benefit sharing agreement if found, None otherwise
        """
        return self.agreements.get(agreement_id)
        
    def get_knowledge_by_community(self, community: str) -> List[TraditionalKnowledge]:
        """
        Get traditional knowledge by community.
        
        Args:
            community: Community or nation to filter by
            
        Returns:
            List of traditional knowledge entries for the community
        """
        return [k for k in self.traditional_knowledge.values() if k.community == community]
        
    def get_knowledge_by_type(self, knowledge_type: str) -> List[TraditionalKnowledge]:
        """
        Get traditional knowledge by type.
        
        Args:
            knowledge_type: Knowledge type to filter by
            
        Returns:
            List of traditional knowledge entries of the specified type
        """
        return [k for k in self.traditional_knowledge.values() if k.knowledge_type == knowledge_type]
        
    def get_knowledge_by_area(self, latitude: float, longitude: float, buffer_distance: float = 0.01) -> List[TraditionalKnowledge]:
        """
        Get traditional knowledge by geographic area.
        
        Args:
            latitude: Latitude of the point
            longitude: Longitude of the point
            buffer_distance: Buffer distance in degrees
            
        Returns:
            List of traditional knowledge entries in the area
        """
        point = Point(longitude, latitude)
        buffer = point.buffer(buffer_distance)
        
        return [
            k for k in self.traditional_knowledge.values() 
            if k.geometry and buffer.intersects(k.geometry)
        ]
        
    def export_to_json(self, file_path: str = None) -> None:
        """
        Export the indigenous knowledge system to a JSON file.
        
        Args:
            file_path: Path to the output file
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "indigenous_knowledge_export.json")
            
        data = {
            "knowledge_holders": {holder_id: holder.to_dict() for holder_id, holder in self.knowledge_holders.items()},
            "traditional_knowledge": {knowledge_id: knowledge.to_dict() for knowledge_id, knowledge in self.traditional_knowledge.items()},
            "consultations": {consultation_id: consultation.to_dict() for consultation_id, consultation in self.consultations.items()},
            "agreements": {agreement_id: agreement.to_dict() for agreement_id, agreement in self.agreements.items()}
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def import_from_json(self, file_path: str) -> None:
        """
        Import the indigenous knowledge system from a JSON file.
        
        Args:
            file_path: Path to the input file
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Import knowledge holders
        for holder_id, holder_data in data.get("knowledge_holders", {}).items():
            self.knowledge_holders[holder_id] = KnowledgeHolder.from_dict(holder_data)
            
        # Import traditional knowledge
        for knowledge_id, knowledge_data in data.get("traditional_knowledge", {}).items():
            knowledge_type = knowledge_data.get("knowledge_type")
            
            if knowledge_type == "cultural_heritage_site":
                knowledge = CulturalHeritageSite.from_dict(knowledge_data)
            elif knowledge_type == "resource_area":
                knowledge = ResourceArea.from_dict(knowledge_data)
            else:
                knowledge = TraditionalKnowledge.from_dict(knowledge_data)
                
            self.traditional_knowledge[knowledge_id] = knowledge
            
        # Import consultations
        for consultation_id, consultation_data in data.get("consultations", {}).items():
            self.consultations[consultation_id] = ConsultationRecord.from_dict(consultation_data)
            
        # Import agreements
        for agreement_id, agreement_data in data.get("agreements", {}).items():
            self.agreements[agreement_id] = BenefitSharingAgreement.from_dict(agreement_data)
