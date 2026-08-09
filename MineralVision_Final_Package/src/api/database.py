"""
MineralVision Database Layer - SQLAlchemy + SQLite

Provides persistent storage for all platform entities.
"""

import os
from datetime import datetime
from typing import Optional, List
from contextlib import contextmanager

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mineralvision.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# SQLAlchemy Models
class UserModel(Base):
    __tablename__ = "users"
    
    id = Column(String(36), primary_key=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100))
    last_name = Column(String(100))
    role = Column(String(50), default="user")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    projects = relationship("ProjectModel", back_populates="owner")


class ProjectModel(Base):
    __tablename__ = "projects"
    
    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False, index=True)
    description = Column(Text)
    location = Column(String(255))
    commodities = Column(JSON, default=list)
    status = Column(String(50), default="active")
    owner_id = Column(String(36), ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    owner = relationship("UserModel", back_populates="projects")
    drillholes = relationship("DrillholeModel", back_populates="project", cascade="all, delete-orphan")


class DrillholeModel(Base):
    __tablename__ = "drillholes"
    
    id = Column(String(36), primary_key=True)
    hole_id = Column(String(100), nullable=False, index=True)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    collar_x = Column(Float, nullable=False)
    collar_y = Column(Float, nullable=False)
    collar_z = Column(Float, nullable=False)
    total_depth = Column(Float, nullable=False)
    azimuth = Column(Float)
    dip = Column(Float)
    status = Column(String(50), default="planned")
    assay_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    project = relationship("ProjectModel", back_populates="drillholes")
    samples = relationship("SampleModel", back_populates="drillhole", cascade="all, delete-orphan")


class SampleModel(Base):
    __tablename__ = "samples"
    
    id = Column(String(36), primary_key=True)
    sample_id = Column(String(100), nullable=False, index=True)
    drillhole_id = Column(String(36), ForeignKey("drillholes.id"), nullable=False)
    from_depth = Column(Float, nullable=False)
    to_depth = Column(Float, nullable=False)
    sample_type = Column(String(50), default="core")
    lithology = Column(String(100))
    assay_data = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    drillhole = relationship("DrillholeModel", back_populates="samples")


class QAQCRecordModel(Base):
    __tablename__ = "qaqc_records"
    
    id = Column(String(36), primary_key=True)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    sample_id = Column(String(36), ForeignKey("samples.id"))
    qc_type = Column(String(50), nullable=False)  # standard, blank, duplicate
    expected_value = Column(Float)
    actual_value = Column(Float)
    tolerance = Column(Float)
    passed = Column(Boolean)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class ReportModel(Base):
    __tablename__ = "reports"
    
    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    report_type = Column(String(50), nullable=False)  # ni43-101, jorc, etc.
    project_id = Column(String(36), ForeignKey("projects.id"))
    status = Column(String(50), default="draft")
    content = Column(JSON, default=dict)
    created_by = Column(String(36), ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AuditLogModel(Base):
    __tablename__ = "audit_logs"
    
    id = Column(String(36), primary_key=True)
    user_id = Column(String(36), ForeignKey("users.id"))
    action = Column(String(100), nullable=False)
    entity_type = Column(String(50))
    entity_id = Column(String(36))
    details = Column(JSON, default=dict)
    ip_address = Column(String(45))
    created_at = Column(DateTime, default=datetime.utcnow)


# Database initialization
def init_db():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency for FastAPI to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """Context manager for database session."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# Seed demo data
def seed_demo_data():
    """Seed the database with demo data."""
    import uuid
    import hashlib
    
    with get_db_context() as db:
        # Check if data already exists
        if db.query(UserModel).first():
            return
        
        # Create admin user
        admin_id = str(uuid.uuid4())
        admin = UserModel(
            id=admin_id,
            username="admin",
            email="admin@mineralvision.com",
            password_hash=hashlib.sha256("admin123".encode()).hexdigest(),
            first_name="Admin",
            last_name="User",
            role="admin"
        )
        db.add(admin)
        
        # Create demo projects
        project1_id = str(uuid.uuid4())
        project1 = ProjectModel(
            id=project1_id,
            name="Gold Exploration Project",
            description="Demo gold exploration in West Africa",
            location="Nigeria",
            commodities=["Gold", "Copper"],
            status="active",
            owner_id=admin_id
        )
        db.add(project1)
        
        project2_id = str(uuid.uuid4())
        project2 = ProjectModel(
            id=project2_id,
            name="Lithium Assessment",
            description="Lithium brine assessment project",
            location="Chile",
            commodities=["Lithium"],
            status="active",
            owner_id=admin_id
        )
        db.add(project2)
        
        # Create demo drillholes
        dh1 = DrillholeModel(
            id=str(uuid.uuid4()),
            hole_id="DH-001",
            project_id=project1_id,
            collar_x=500000,
            collar_y=1000000,
            collar_z=350,
            total_depth=250,
            azimuth=45,
            dip=-60,
            status="completed",
            assay_count=50
        )
        db.add(dh1)
        
        dh2 = DrillholeModel(
            id=str(uuid.uuid4()),
            hole_id="DH-002",
            project_id=project1_id,
            collar_x=500100,
            collar_y=1000050,
            collar_z=345,
            total_depth=180,
            azimuth=45,
            dip=-55,
            status="in_progress",
            assay_count=30
        )
        db.add(dh2)
        
        # Create demo samples
        sample1 = SampleModel(
            id=str(uuid.uuid4()),
            sample_id="S-001",
            drillhole_id=dh1.id,
            from_depth=0,
            to_depth=2,
            sample_type="core",
            lithology="Granite",
            assay_data={"Au_ppm": 0.5, "Cu_pct": 0.02}
        )
        db.add(sample1)
        
        # Create demo report
        report1 = ReportModel(
            id=str(uuid.uuid4()),
            name="NI 43-101 Technical Report",
            report_type="ni43-101",
            project_id=project1_id,
            status="draft",
            created_by=admin_id
        )
        db.add(report1)
        
        db.commit()
