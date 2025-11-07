# Quick Start Guide

Get the OPZ Product Matcher running in 5 minutes!

## Prerequisites

- Docker and Docker Compose installed
- Anthropic API key ([Get one here](https://console.anthropic.com/))

## Quick Start with Docker

### 1. Clone and Configure

```bash
# Navigate to the project directory
cd opz-matcher

# Copy environment file
cp .env.example .env

# Edit .env and add your Anthropic API key
nano .env
# Set: ANTHROPIC_API_KEY=your-key-here
```

### 2. Start Services

```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

### 3. Create Admin User

```bash
# Enter the API container
docker-compose exec api python

# Then run:
from services.database import init_db
from api.routes.auth import get_password_hash
from models.database import User
from services.database import AsyncSessionLocal
import asyncio

async def create_admin():
    await init_db()
    async with AsyncSessionLocal() as db:
        admin = User(
            username="admin",
            email="admin@example.com",
            hashed_password=get_password_hash("admin123"),
            full_name="Admin User",
            is_active=True,
            is_superuser=True
        )
        db.add(admin)
        await db.commit()
        print("Admin user created!")

asyncio.run(create_admin())
```

Or use the script:
```bash
docker-compose exec api python /app/../scripts/create_user.py \
  --username admin \
  --email admin@example.com \
  --password admin123 \
  --full-name "Admin User"
```

### 4. Access the API

- **API Documentation**: http://localhost:8000/docs
- **MinIO Console**: http://localhost:9001 (minioadmin / minioadmin)
- **Health Check**: http://localhost:8000/health

## Your First Workflow

### 1. Login via API Docs

1. Go to http://localhost:8000/docs
2. Click "Authorize" button
3. Login with: `admin` / `admin123`
4. Copy the access token

### 2. Import Data

**Create a Vendor:**
```bash
curl -X POST "http://localhost:8000/api/import/vendors" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Dell",
    "full_name": "Dell Technologies",
    "website": "https://www.dell.com"
  }'
```

**Create a Product:**
```bash
curl -X POST "http://localhost:8000/api/import/products" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "vendor_name": "Dell",
    "name": "PowerEdge R750",
    "model": "R750",
    "category": "server",
    "description": "2U rack server"
  }'
```

**Upload a Datasheet:**
```bash
curl -X POST "http://localhost:8000/api/import/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@your_datasheet.pdf" \
  -F "product_id=1" \
  -F "document_type=datasheet"
```

### 3. Search for Products

```bash
curl -X POST "http://localhost:8000/api/search/search" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "requirements_text": "Server with Intel Xeon 16 cores, 64GB RAM, 2x 1TB SSD RAID 1",
    "category": "server",
    "min_match_score": 0.6
  }'
```

### 4. Create an OPZ

```bash
curl -X POST "http://localhost:8000/api/opz/create" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Server for ERP System",
    "category": "server",
    "configuration": {
      "processor": {"family": "Intel Xeon", "min_cores": 16},
      "memory": {"capacity_gb": 64},
      "storage": {"type": "SSD", "capacity_gb": 1000, "raid": "RAID 1"}
    },
    "selected_vendors": ["Dell", "HPE", "Lenovo"]
  }'
```

Check OPZ status:
```bash
curl -X GET "http://localhost:8000/api/opz/1" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Download when ready:
```bash
curl -X GET "http://localhost:8000/api/opz/1/download" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  --output opz_document.docx
```

## Using the Python Example

```bash
# Install dependencies
pip install requests

# Run the example
python examples/api_usage_example.py
```

## Troubleshooting

### Can't connect to API
```bash
# Check if services are running
docker-compose ps

# Restart services
docker-compose restart

# Check logs
docker-compose logs api
```

### Database connection error
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Restart database
docker-compose restart postgres
```

### OCR not working
```bash
# Tesseract is included in the Docker image
# For local development, install it:
# Ubuntu: sudo apt-get install tesseract-ocr tesseract-ocr-pol
# macOS: brew install tesseract tesseract-lang
```

### Claude API errors
- Verify your API key in `.env`
- Check your API quota at https://console.anthropic.com/
- Review API logs: `docker-compose logs api`

## Next Steps

1. **Import your products**: Add vendors, products, and datasheets
2. **Import benchmarks**: Add CPU, GPU, storage benchmark data
3. **Test search**: Try searching with real OPZ requirements
4. **Create OPZ templates**: Build reusable OPZ templates
5. **Integrate**: Use the API in your applications

## Configuration Tips

### Increase Upload Limit
Edit `.env`:
```
MAX_UPLOAD_SIZE=104857600  # 100MB
```

### Adjust OCR Language
Edit `.env`:
```
OCR_LANGUAGE=pol+eng+deu  # Add German
```

### Production Security
```
# Generate secure secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Update .env
SECRET_KEY=your-generated-key-here
```

## Getting Help

- Check the [full documentation](./SETUP.md)
- Review [architecture docs](./docs/ARCHITECTURE.md)
- Check API docs at http://localhost:8000/docs
- Review logs: `docker-compose logs -f`

## Stopping the Application

```bash
# Stop services
docker-compose stop

# Stop and remove containers
docker-compose down

# Stop and remove containers + volumes (deletes data)
docker-compose down -v
```

Happy matching! 🚀
