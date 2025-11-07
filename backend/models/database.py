"""
Database models for OPZ Product Matcher
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, JSON, Enum, Boolean
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import enum
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class DeviceCategory(str, enum.Enum):
    """Device categories for IT products"""
    SERVER = "server"
    PC = "pc"
    LAPTOP = "laptop"
    NETWORK_SWITCH = "network_switch"
    NETWORK_ROUTER = "network_router"
    FIREWALL = "firewall"
    STORAGE_NAS = "storage_nas"
    STORAGE_SAN = "storage_san"
    STORAGE_DAS = "storage_das"
    UPS = "ups"
    OTHER = "other"


class User(Base):
    """User model for authentication"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    opz_documents = relationship("OPZDocument", back_populates="owner")


class Vendor(Base):
    """Vendor/manufacturer information"""
    __tablename__ = "vendors"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True)
    full_name = Column(String)
    description = Column(Text)
    website = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    products = relationship("Product", back_populates="vendor")


class Product(Base):
    """Product information"""
    __tablename__ = "products"
    
    id = Column(Integer, primary_key=True, index=True)
    vendor_id = Column(Integer, ForeignKey("vendors.id"), nullable=False)
    name = Column(String, nullable=False, index=True)
    model = Column(String, index=True)
    category = Column(Enum(DeviceCategory), nullable=False, index=True)
    
    # Technical specifications (JSON field for flexibility)
    specifications = Column(JSON, default={})
    
    # Description and notes
    description = Column(Text)
    notes = Column(Text)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    vendor = relationship("Vendor", back_populates="products")
    documents = relationship("Document", back_populates="product")
    embeddings = relationship("ProductEmbedding", back_populates="product")


class Document(Base):
    """Uploaded documents (datasheets, manuals, etc.)"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    
    # File information
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # pdf, docx, xlsx, etc.
    file_path = Column(String, nullable=False)  # S3/MinIO path
    file_size = Column(Integer)
    
    # Document metadata
    title = Column(String)
    document_type = Column(String)  # datasheet, manual, specification, etc.
    
    # Extracted content
    extracted_text = Column(Text)
    extracted_specs = Column(JSON)
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_error = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True))
    
    # Relationships
    product = relationship("Product", back_populates="documents")


class ProductEmbedding(Base):
    """Vector embeddings for semantic search"""
    __tablename__ = "product_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)

    # Content that was embedded
    content = Column(Text, nullable=False)
    content_type = Column(String)  # specification, description, feature, etc.

    # Vector embedding using pgvector
    embedding = Column(Vector(768), nullable=False)  # nomic-embed-text-v1.5 produces 768 dimensions

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    product = relationship("Product", back_populates="embeddings")


class Benchmark(Base):
    """Benchmark data for product comparisons"""
    __tablename__ = "benchmarks"

    id = Column(Integer, primary_key=True, index=True)

    # Benchmark identification
    name = Column(String, nullable=False, index=True)
    category = Column(String, nullable=False, index=True)  # cpu, gpu, storage, network
    version = Column(String)

    # Component information
    component_name = Column(String, nullable=False, index=True)
    component_model = Column(String)
    vendor = Column(String)

    # Score
    score = Column(Float, nullable=False)
    score_type = Column(String)  # single_thread, multi_thread, sequential_read, etc.

    # Additional data
    benchmark_metadata = Column(JSON)

    # Version management
    content_hash = Column(String, nullable=False, index=True)  # Hash of benchmark content for duplicate detection
    version_group_id = Column(String, nullable=False, index=True)  # Groups versions of the same benchmark
    is_latest_version = Column(Boolean, default=True)  # Indicates if this is the latest version in the group

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class OPZDocument(Base):
    """Generated OPZ documents"""
    __tablename__ = "opz_documents"
    
    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # OPZ information
    title = Column(String, nullable=False)
    category = Column(Enum(DeviceCategory), nullable=False)
    
    # Requirements
    requirements = Column(JSON, nullable=False)  # Structured requirements
    selected_vendors = Column(JSON)  # List of vendor IDs
    
    # Generated content
    generated_text = Column(Text)
    
    # File storage
    file_path = Column(String)  # Path to generated DOCX
    
    # Status
    status = Column(String, default="draft")  # draft, generated, downloaded
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    generated_at = Column(DateTime(timezone=True))
    
    # Relationships
    owner = relationship("User", back_populates="opz_documents")


class DocumentChunk(Base):
    """Document chunks for unstructured knowledge base"""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)

    # Chunk content
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Position in document
    chunk_type = Column(String)  # paragraph, heading, table, etc.

    # Vector embedding
    embedding = Column(Vector(768), nullable=False)  # nomic-embed-text-v1.5 produces 768 dimensions

    # Metadata (JSONB for flexible metadata)
    chunk_metadata = Column(JSON, default={})

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document = relationship("Document")


class KnowledgeEntity(Base):
    """Knowledge entities extracted from documents"""
    __tablename__ = "knowledge_entities"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)

    # Entity information
    entity_type = Column(String, nullable=False)  # product, specification, feature, etc.
    entity_name = Column(String, nullable=False)
    entity_value = Column(String)  # Optional value for the entity

    # Context and description
    context = Column(Text)
    description = Column(Text)

    # Vector embedding for semantic search
    embedding = Column(Vector(768), nullable=False)  # nomic-embed-text-v1.5 produces 768 dimensions

    # Metadata (JSONB for flexible metadata)
    entity_metadata = Column(JSON, default={})

    # Confidence score from extraction
    confidence_score = Column(Float)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    document = relationship("Document")


class SearchCache(Base):
    """Cache for vector search results"""
    __tablename__ = "search_cache"

    id = Column(Integer, primary_key=True, index=True)

    # Search query
    query_text = Column(Text, nullable=False)
    query_embedding = Column(Vector(768), nullable=False)  # nomic-embed-text-v1.5 produces 768 dimensions

    # Search parameters
    search_type = Column(String, nullable=False)  # document_chunks, knowledge_entities, products
    filters = Column(JSON, default={})  # Additional filters applied

    # Cached results
    results = Column(JSON, nullable=False)  # List of results with scores and metadata
    result_count = Column(Integer, nullable=False)

    # Cache metadata
    expires_at = Column(DateTime(timezone=True), nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SearchHistory(Base):
    """Search history for analytics and improvement"""
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, index=True)

    # Search query
    query_text = Column(Text, nullable=False)
    requirements = Column(JSON)
    category = Column(Enum(DeviceCategory))

    # Results
    results = Column(JSON)  # List of matched product IDs with scores

    # User feedback (optional)
    selected_product_id = Column(Integer, ForeignKey("products.id"))
    feedback_score = Column(Integer)  # 1-5 rating

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
