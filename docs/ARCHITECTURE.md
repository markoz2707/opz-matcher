# OPZ Product Matcher - System Architecture

## Overview

The OPZ Product Matcher is an AI-powered application designed to streamline IT procurement processes in Poland. It leverages Claude API to extract product specifications, match products to tender requirements, and generate OPZ (Opis Przedmiotu Zamówienia) documents.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Layer                          │
│  (Web Browser / API Client / Mobile App)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ HTTPS/REST API
                         │
┌────────────────────────▼────────────────────────────────────┐
│                     FastAPI Backend                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Auth       │  │ Data Import  │  │   Product    │      │
│  │   Routes     │  │   Routes     │  │   Search     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐                                           │
│  │     OPZ      │         API Layer                         │
│  │  Creation    │                                           │
│  └──────────────┘                                           │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Claude     │  │  Document    │  │   Storage    │      │
│  │   Service    │  │  Processor   │  │   Service    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                    Service Layer                            │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
┌───────────▼────────┐      ┌──────────▼──────────┐
│   PostgreSQL       │      │   Claude API         │
│   + pgvector       │      │   (Anthropic)        │
└───────────┬────────┘      └─────────────────────┘
            │
   ┌────────┴────────┐
   │                 │
┌──▼────┐    ┌──────▼──────┐
│ Redis │    │ MinIO/S3    │
│       │    │  Storage    │
└───────┘    └─────────────┘
```

## Core Components

### 1. API Layer (FastAPI)

The REST API provides endpoints for all three modes of operation:

#### Authentication (`/api/auth`)
- JWT-based authentication
- User registration and login
- Role-based access control

#### Data Import (`/api/import`)
- Vendor management
- Product creation
- Document upload and processing
- Benchmark data import

#### Product Search (`/api/search`)
- Semantic product search
- Requirement matching with flexibility analysis
- Benchmark-based comparisons

#### OPZ Creation (`/api/opz`)
- Template-based OPZ generation
- Multi-vendor specifications
- Document refinement
- DOCX export

### 2. Service Layer

#### Claude Service
Handles all interactions with Anthropic's Claude API:

**Key Functions:**
- `extract_product_specifications()`: Extracts structured data from datasheets
- `match_products()`: Matches products to OPZ requirements
- `generate_opz()`: Creates OPZ documents
- `refine_opz()`: Refines OPZ based on feedback

**Features:**
- Multilingual support (Polish & English)
- JSON-structured responses
- Error handling and retry logic
- Token optimization

#### Document Processor
Processes various document formats:

**Supported Formats:**
- PDF (with and without text layer)
- DOCX (Word documents)
- XLSX (Excel spreadsheets)
- TXT (plain text)
- Images (PNG, JPG, JPEG)

**Processing Pipeline:**
1. File validation
2. Text extraction (pdfplumber, python-docx, openpyxl)
3. OCR for scanned documents (Tesseract)
4. Text chunking for embeddings
5. Metadata extraction

#### Storage Service
Manages file storage using MinIO or S3:

**Features:**
- Upload/download operations
- Presigned URL generation
- Bucket management
- File existence checks

### 3. Data Layer

#### PostgreSQL + pgvector
Primary database for structured data:

**Tables:**
- `users`: User accounts and authentication
- `vendors`: Product manufacturers
- `products`: Product catalog with specifications
- `documents`: Uploaded datasheets and manuals
- `benchmarks`: Performance benchmark data
- `product_embeddings`: Vector embeddings for semantic search
- `opz_documents`: Generated OPZ documents
- `search_history`: Search analytics

**Features:**
- ACID compliance
- Vector similarity search (pgvector)
- Full-text search capabilities
- Relationship management

#### Redis
Used for caching and background tasks:

**Use Cases:**
- Session management
- Rate limiting
- Celery task queue (optional)
- Caching frequently accessed data

#### MinIO/S3
Object storage for files:

**Stored Files:**
- Product datasheets (PDF, DOCX, etc.)
- Generated OPZ documents
- Extracted images
- Temporary processing files

## Data Flow

### Mode 1: Data Import

```
User uploads document
    ↓
FastAPI validates file
    ↓
Storage Service saves file
    ↓
Document Processor extracts text
    ↓
Claude Service extracts specifications
    ↓
Database stores structured data
    ↓
Background: Generate embeddings
    ↓
User receives confirmation
```

### Mode 2: Product Search

```
User submits OPZ requirements
    ↓
FastAPI receives request
    ↓
Database queries products (filtered)
    ↓
Database retrieves benchmark data
    ↓
Claude Service analyzes requirements
    ↓
Claude matches products with scores
    ↓
API returns ranked results
    ↓
User reviews matches
```

### Mode 3: OPZ Creation

```
User defines configuration
    ↓
FastAPI creates OPZ record
    ↓
Background: Claude generates OPZ text
    ↓
API updates status to "generated"
    ↓
User requests download
    ↓
System creates DOCX file
    ↓
User downloads OPZ document
```

## Key Technologies

### Backend
- **FastAPI**: Modern, fast web framework
- **SQLAlchemy**: ORM with async support
- **Pydantic**: Data validation
- **Anthropic SDK**: Claude API client

### Document Processing
- **PyPDF2 / pdfplumber**: PDF text extraction
- **python-docx**: Word document handling
- **openpyxl**: Excel file processing
- **Tesseract**: OCR engine
- **Pillow**: Image processing

### Storage & Database
- **PostgreSQL**: Relational database
- **pgvector**: Vector similarity search
- **Redis**: Caching and queuing
- **MinIO**: S3-compatible object storage

### AI & ML
- **Claude Sonnet 4.5**: Primary LLM
- **sentence-transformers**: Text embeddings (optional)

## Security Considerations

### Authentication
- JWT tokens with configurable expiration
- Bcrypt password hashing
- OAuth2 password flow

### Data Protection
- Environment-based configuration
- Secure secret management
- HTTPS in production
- Database connection encryption

### Access Control
- Role-based permissions (user, superuser)
- Resource ownership verification
- Rate limiting (future enhancement)

## Scalability

### Horizontal Scaling
- Stateless API design
- Database connection pooling
- Redis for session management
- Load balancer ready

### Performance Optimization
- Async/await throughout
- Background task processing
- Database indexing
- Query optimization
- Caching strategies

### Resource Management
- File size limits
- Request timeouts
- Rate limiting
- Token usage optimization

## Monitoring & Logging

### Logging
- Structured logging with loguru
- Log levels: DEBUG, INFO, WARNING, ERROR
- Request/response logging
- Error tracking

### Health Checks
- `/health` endpoint
- Database connectivity
- Storage availability
- Service status

### Metrics (Future)
- API response times
- Document processing times
- Claude API usage
- Search accuracy metrics

## Deployment

### Development
- Local setup with virtual environment
- SQLite for quick testing
- Hot reload enabled

### Production
- Docker containers
- Docker Compose orchestration
- Environment-based configuration
- Persistent volumes for data

### CI/CD (Recommended)
- Automated testing
- Code quality checks
- Container image building
- Automated deployment

## API Design

### RESTful Principles
- Resource-based URLs
- HTTP methods (GET, POST, PUT, DELETE)
- Status codes (200, 201, 400, 401, 404, 500)
- JSON request/response format

### Error Handling
- Consistent error response format
- Detailed error messages
- HTTP status codes
- Validation errors

### Pagination
- Limit/offset for lists
- Default limits
- Total count in response

### Versioning
- URL-based versioning (future)
- Backward compatibility

## Future Enhancements

### Short Term
1. Email notifications for completed tasks
2. Advanced search filters
3. Product comparison view
4. OPZ templates library
5. User preferences

### Medium Term
1. Real-time collaboration
2. Advanced analytics dashboard
3. Batch document processing
4. API rate limiting
5. Multi-language UI

### Long Term
1. Machine learning for better matching
2. Automated tender monitoring
3. Integration with procurement systems
4. Mobile applications
5. Advanced reporting
