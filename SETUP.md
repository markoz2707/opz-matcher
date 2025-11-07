# OPZ Product Matcher - Setup Guide

This guide will help you set up and run the OPZ Product Matcher application.

## Prerequisites

- Python 3.11 or higher
- PostgreSQL 14+ with pgvector extension
- Redis (optional, for production)
- MinIO or S3-compatible storage (optional, for production)
- Tesseract OCR (for image and PDF OCR)
- Anthropic API key

## Quick Start (Development)

### 1. Install Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-pol tesseract-ocr-eng
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
Download from: https://github.com/UB-Mannheim/tesseract/wiki

### 2. Install PostgreSQL and pgvector

**Ubuntu/Debian:**
```bash
sudo apt-get install postgresql postgresql-contrib
sudo apt-get install postgresql-14-pgvector
```

**macOS:**
```bash
brew install postgresql pgvector
```

Create database:
```bash
sudo -u postgres psql
CREATE DATABASE opzdb;
CREATE USER opzuser WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE opzdb TO opzuser;
\c opzdb
CREATE EXTENSION vector;
\q
```

### 3. Install Redis (optional for development)

**Ubuntu/Debian:**
```bash
sudo apt-get install redis-server
sudo systemctl start redis
```

**macOS:**
```bash
brew install redis
brew services start redis
```

### 4. Install MinIO (optional for development)

**Using Docker:**
```bash
docker run -d \
  -p 9000:9000 \
  -p 9001:9001 \
  --name minio \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  minio/minio server /data --console-address ":9001"
```

Or download from: https://min.io/download

### 5. Set Up Python Environment

```bash
cd opz-matcher
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 6. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your configuration
nano .env
```

**Important:** Set your Anthropic API key:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 7. Initialize Database

```bash
cd backend
python -c "from services.database import init_db; import asyncio; asyncio.run(init_db())"
```

### 8. Create First User (Optional)

```bash
python scripts/create_user.py --username admin --email admin@example.com --password your_password --superuser
```

### 9. Run the Application

```bash
cd backend
python main.py
```

The API will be available at: http://localhost:8000
API documentation: http://localhost:8000/docs

## Docker Deployment

For production deployment with Docker:

```bash
docker-compose up -d
```

This will start:
- PostgreSQL with pgvector
- Redis
- MinIO
- The OPZ Matcher API

## Production Considerations

### Security

1. **Change SECRET_KEY**: Generate a secure random key
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Use HTTPS**: Set up SSL/TLS certificates

3. **Firewall**: Restrict access to database and Redis

4. **Environment Variables**: Never commit .env file to git

### Performance

1. **Database Connection Pooling**: Configure in DATABASE_URL

2. **Redis for Caching**: Enable Redis for production

3. **Worker Processes**: Use multiple Uvicorn workers
   ```bash
   uvicorn main:app --workers 4 --host 0.0.0.0 --port 8000
   ```

4. **Background Tasks**: Use Celery for long-running tasks
   ```bash
   celery -A tasks worker --loglevel=info
   ```

### Monitoring

1. **Logging**: Configure loguru for production logging

2. **Health Checks**: Use `/health` endpoint

3. **Metrics**: Add Prometheus metrics (optional)

## Testing

Run tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=backend tests/
```

## API Usage Examples

### 1. Register and Login

```bash
# Register
curl -X POST "http://localhost:8000/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "username": "testuser",
    "password": "securepassword",
    "full_name": "Test User"
  }'

# Login
curl -X POST "http://localhost:8000/api/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=securepassword"

# Save the access_token from response
export TOKEN="your-access-token-here"
```

### 2. Import Data

```bash
# Create vendor
curl -X POST "http://localhost:8000/api/import/vendors" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Dell",
    "full_name": "Dell Technologies",
    "website": "https://www.dell.com"
  }'

# Create product
curl -X POST "http://localhost:8000/api/import/products" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "vendor_name": "Dell",
    "name": "PowerEdge R750",
    "model": "R750",
    "category": "server"
  }'

# Upload document (assuming product_id=1)
curl -X POST "http://localhost:8000/api/import/documents/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@datasheet.pdf" \
  -F "product_id=1" \
  -F "document_type=datasheet"
```

### 3. Search Products

```bash
curl -X POST "http://localhost:8000/api/search/search" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "requirements_text": "Serwer z procesorem Intel Xeon min 2.5GHz, 64GB RAM, dyski SSD 2x 1TB w RAID 1",
    "category": "server",
    "min_match_score": 0.6
  }'
```

### 4. Create OPZ

```bash
curl -X POST "http://localhost:8000/api/opz/create" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Serwer dla systemu ERP",
    "category": "server",
    "configuration": {
      "processor": {
        "family": "Intel Xeon",
        "min_cores": 16,
        "min_frequency": 2.5
      },
      "memory": {
        "capacity": "64GB",
        "type": "DDR4"
      },
      "storage": {
        "type": "SSD",
        "capacity": "2x 1TB",
        "raid": "RAID 1"
      }
    },
    "selected_vendors": ["Dell", "HPE", "Lenovo"]
  }'
```

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection
psql -U opzuser -d opzdb -h localhost
```

### OCR Not Working

```bash
# Verify Tesseract installation
tesseract --version

# Check language support
tesseract --list-langs
```

### MinIO Connection Issues

```bash
# Check MinIO is running
docker ps | grep minio

# Access MinIO console
# http://localhost:9001
# Login: minioadmin / minioadmin
```

### Claude API Issues

1. Verify API key is correct
2. Check API quota and usage
3. Review error messages in logs

## Support

For issues and questions:
- Check the documentation
- Review API docs at http://localhost:8000/docs
- Check logs in `logs/` directory

## Next Steps

1. Import your product data and documents
2. Import benchmark data for comparisons
3. Test product search with sample OPZ requirements
4. Create your first OPZ document
5. Set up automated backups for the database
