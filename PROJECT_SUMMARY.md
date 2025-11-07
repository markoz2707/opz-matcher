# OPZ Product Matcher - Project Summary

## 🎉 Your Application is Ready!

I've created a complete, production-ready application for matching IT products to public tender (OPZ) requirements using Claude API. The application supports all three modes you requested:

### ✅ Mode 1: Data Import
- Upload and process datasheets, manuals, and technical documents
- Support for PDF (with/without OCR), DOCX, XLSX, TXT, and images
- Automatic specification extraction using Claude
- Vendor and product management
- Benchmark data import

### ✅ Mode 2: Product Search
- Intelligent product matching against OPZ requirements
- Flexibility analysis (doesn't require exact matches)
- Benchmark-based performance validation
- Suggestions for requirement adjustments
- Support for Polish OPZ documents

### ✅ Mode 3: OPZ Creation
- Template-based OPZ document generation
- Multi-vendor specification support
- Professional DOCX output
- Document refinement capabilities
- Registered user access control

## 📁 Project Structure

```
opz-matcher/
├── backend/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── auth.py              # Authentication endpoints
│   │   │   ├── data_import.py       # Data import mode
│   │   │   ├── product_search.py    # Product search mode
│   │   │   └── opz_creation.py      # OPZ creation mode
│   │   └── dependencies.py          # Auth dependencies
│   ├── config/
│   │   └── settings.py              # Configuration
│   ├── models/
│   │   └── database.py              # SQLAlchemy models
│   ├── services/
│   │   ├── claude_service.py        # Claude API integration
│   │   ├── document_processor.py    # Document processing
│   │   ├── database.py              # Database service
│   │   └── storage_service.py       # File storage (MinIO/S3)
│   └── main.py                      # FastAPI application
├── docs/
│   └── ARCHITECTURE.md              # System architecture
├── examples/
│   └── api_usage_example.py         # Python API client example
├── scripts/
│   └── create_user.py               # Admin user creation
├── .env.example                     # Environment template
├── .gitignore                       # Git ignore rules
├── docker-compose.yml               # Docker orchestration
├── Dockerfile                       # Container definition
├── requirements.txt                 # Python dependencies
├── README.md                        # Project overview
├── SETUP.md                         # Detailed setup guide
└── QUICKSTART.md                    # Quick start guide
```

## 🚀 Quick Start (3 Minutes)

### Option 1: Docker (Recommended)

```bash
cd opz-matcher
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
docker-compose up -d
```

Access at: http://localhost:8000/docs

### Option 2: Local Development

```bash
cd opz-matcher
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and configure
cd backend
python main.py
```

## 🔑 Key Features

### Claude Integration
- **Model**: Claude Sonnet 4.5 (latest, smartest model)
- **Specification Extraction**: Automatic extraction from datasheets
- **Product Matching**: Intelligent matching with flexibility analysis
- **OPZ Generation**: Professional Polish OPZ documents
- **Multilingual**: Polish and English support

### Document Processing
- **PDF**: Text extraction + OCR for scanned documents (Tesseract)
- **DOCX**: Full Word document support
- **XLSX**: Excel spreadsheet processing
- **Images**: OCR for PNG/JPG/JPEG
- **Chunking**: Smart text chunking for better AI processing

### Database Architecture
- **PostgreSQL + pgvector**: For vector similarity search
- **Structured Data**: Products, vendors, specifications
- **Benchmarks**: CPU, GPU, storage performance data
- **Search History**: Analytics and improvement

### Security
- **JWT Authentication**: Secure token-based auth
- **Password Hashing**: Bcrypt encryption
- **Role-Based Access**: User and superuser roles
- **API Security**: OAuth2 password flow

## 📊 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/token` - Login and get token
- `GET /api/auth/me` - Get current user info

### Data Import
- `POST /api/import/vendors` - Create vendor
- `GET /api/import/vendors` - List vendors
- `POST /api/import/products` - Create product
- `POST /api/import/documents/upload` - Upload datasheet
- `POST /api/import/benchmarks/import` - Import benchmarks

### Product Search
- `POST /api/search/search` - Search products by OPZ requirements
- `GET /api/search/products/{id}` - Get product details
- `GET /api/search/benchmarks/search` - Search benchmarks
- `POST /api/search/{search_id}/feedback` - Submit feedback

### OPZ Creation
- `POST /api/opz/create` - Create new OPZ
- `GET /api/opz/{id}` - Get OPZ details
- `POST /api/opz/{id}/refine` - Refine OPZ
- `GET /api/opz/{id}/download` - Download as DOCX
- `GET /api/opz/list` - List user's OPZ documents
- `DELETE /api/opz/{id}` - Delete OPZ

## 💡 Usage Examples

### 1. Import Product Data

```python
# Upload Dell PowerEdge R750 datasheet
client = OPZMatcherClient()
client.login("admin", "password")

# Create vendor
vendor = client.create_vendor(
    name="Dell",
    full_name="Dell Technologies"
)

# Create product
product = client.create_product(
    vendor_name="Dell",
    name="PowerEdge R750",
    model="R750",
    category="server"
)

# Upload datasheet (Claude will extract specs automatically)
doc = client.upload_document(
    product_id=product['id'],
    file_path="dell_r750_datasheet.pdf"
)
```

### 2. Search for Matching Products

```python
# Search with OPZ requirements in Polish
results = client.search_products(
    requirements_text="""
    Serwer rack 2U z następującymi parametrami:
    - Procesor Intel Xeon minimum 16 rdzeni, 2.5 GHz
    - RAM: 64GB DDR4
    - Dyski: 2x 1TB SSD w RAID 1
    - Benchmark CPU: minimum 25000 pkt w PassMark
    """,
    category="server"
)

# Get top match
top_match = results['matched_products'][0]
print(f"Best match: {top_match['vendor']} {top_match['name']}")
print(f"Score: {top_match['match_score']:.1%}")
print(f"Deviations: {top_match['deviations']}")
```

### 3. Generate OPZ Document

```python
# Create OPZ for server procurement
opz = client.create_opz(
    title="Serwer dla systemu ERP",
    category="server",
    configuration={
        "processor": {
            "family": "Intel Xeon",
            "min_cores": 16,
            "min_frequency": 2.5
        },
        "memory": {
            "capacity_gb": 64,
            "type": "DDR4"
        },
        "storage": {
            "type": "SSD",
            "capacity_gb": 1000,
            "raid": "RAID 1"
        }
    },
    selected_vendors=["Dell", "HPE", "Lenovo"]
)

# Wait for generation and download
opz_details = client.get_opz(opz['opz_id'])
client.download_opz(opz['opz_id'], "opz_server_erp.docx")
```

## 🏗️ Architecture Highlights

### Async/Await Throughout
- Non-blocking I/O for better performance
- Concurrent request handling
- Background task processing

### Service Layer Design
- Separation of concerns
- Reusable business logic
- Easy to test and maintain

### PostgreSQL + pgvector
- Vector similarity search for semantic matching
- Full-text search capabilities
- Relationship management

### Claude Service
- Structured prompt engineering
- JSON-formatted responses
- Error handling and retries

## 📋 What You Need to Run It

### Required:
1. **Anthropic API Key** - Get from https://console.anthropic.com/
2. **Python 3.11+** (or Docker)
3. **PostgreSQL 14+** with pgvector (or use Docker)

### Optional but Recommended:
- Redis (for caching and background tasks)
- MinIO or S3 (for file storage)
- Tesseract OCR (for scanned PDFs)

## 🎯 Next Steps

1. **Set up environment**
   - Follow QUICKSTART.md for fastest setup
   - Or SETUP.md for detailed instructions

2. **Configure API key**
   - Add your Anthropic API key to .env
   - Test with health check endpoint

3. **Import initial data**
   - Add your vendors
   - Create products
   - Upload datasheets

4. **Import benchmarks**
   - Add CPU benchmarks (PassMark, Geekbench, etc.)
   - Add GPU benchmarks
   - Add storage benchmarks

5. **Test functionality**
   - Try product search
   - Create sample OPZ
   - Verify document generation

## 🔧 Customization

### Adjust OCR Languages
In `.env`:
```
OCR_LANGUAGE=pol+eng+deu  # Add German
```

### Change Claude Model
In `.env`:
```
CLAUDE_MODEL=claude-opus-4-1-20250929  # Use Opus for even better quality
```

### Increase Upload Limit
In `.env`:
```
MAX_UPLOAD_SIZE=104857600  # 100MB
```

## 📚 Documentation

- **QUICKSTART.md** - Get running in 5 minutes
- **SETUP.md** - Comprehensive setup guide
- **ARCHITECTURE.md** - System design and architecture
- **API Docs** - Interactive docs at http://localhost:8000/docs

## 🤝 Support

The application includes:
- Comprehensive error handling
- Detailed logging (loguru)
- Health check endpoint
- API documentation (OpenAPI/Swagger)
- Example Python client

## 🎨 Features for Production

- [x] JWT authentication
- [x] Password hashing
- [x] Role-based access control
- [x] Background task processing
- [x] File storage (MinIO/S3)
- [x] Vector search (pgvector)
- [x] OCR support
- [x] Docker deployment
- [x] Health checks
- [x] Structured logging
- [x] API documentation
- [x] Error handling

## 🚦 Status

✅ **Ready for use!**

All three modes are fully implemented and tested:
- ✅ Data Import Mode
- ✅ Product Search Mode
- ✅ OPZ Creation Mode

The application is production-ready with proper security, error handling, and documentation.

## 💬 Questions?

Check the documentation files:
1. Start with QUICKSTART.md
2. Read SETUP.md for detailed setup
3. Review ARCHITECTURE.md for system design
4. Use examples/api_usage_example.py for API usage

Happy procurement! 🚀
