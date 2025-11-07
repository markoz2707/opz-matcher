# OPZ Product Matcher

AI-powered application for IT procurement (public tender) product matching and OPZ creation using Claude API.

## Features

### 1. Data Import Mode
- Upload product datasheets, manuals, and technical documentation
- Support for PDF (with/without OCR), DOCX, XLSX, TXT, and images
- Automatic extraction of technical specifications
- Vendor and product categorization
- Benchmark data management

### 2. Product Search Mode
- Input tender requirements (OPZ)
- Intelligent product matching with flexibility analysis
- Benchmark comparison (CPU, performance metrics)
- Requirement adjustment suggestions
- Multi-criteria scoring

### 3. OPZ Creation Mode
- Template-based OPZ generation
- Multi-vendor requirement specification
- Configurable product parameters
- DOCX export with professional formatting

## Technology Stack

- **Backend**: Python 3.11+ with FastAPI
- **AI**: Anthropic Claude API (Sonnet 4.5)
- **Database**: PostgreSQL + pgvector
- **Cache/Queue**: Redis
- **Storage**: MinIO/S3 for documents
- **Frontend**: React with TypeScript (optional)

## Project Structure

```
opz-matcher/
├── backend/
│   ├── api/              # FastAPI routes
│   ├── services/         # Business logic
│   ├── models/           # Database models
│   ├── utils/            # Utilities
│   └── config/           # Configuration
├── frontend/             # React frontend (optional)
├── docs/                 # Documentation
└── tests/               # Test suites
```

## Setup

See [SETUP.md](./SETUP.md) for detailed installation instructions.

## Environment Variables

```env
ANTHROPIC_API_KEY=your_api_key
DATABASE_URL=postgresql://user:pass@localhost/opzdb
REDIS_URL=redis://localhost:6379
MINIO_ENDPOINT=localhost:9000
```

## License

MIT License
