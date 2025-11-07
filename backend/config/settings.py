"""
Application configuration settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    APP_NAME: str = "OPZ Product Matcher"
    DEBUG: bool = False
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # OpenRouter API
    OPENROUTER_API_KEY: str
    OPENROUTER_MODEL: str = "x-ai/grok-code-fast-1"
    MAX_TOKENS: int = 8192
    
    # Database
    DATABASE_URL: str
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # Storage (MinIO/S3)
    STORAGE_ENDPOINT: str = "localhost:9000"
    STORAGE_ACCESS_KEY: str
    STORAGE_SECRET_KEY: str
    STORAGE_BUCKET: str = "opz-documents"
    STORAGE_SECURE: bool = False
    
    # CORS - stored as string from env
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8080"
    
    @computed_field
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS_ORIGINS from comma-separated string to list"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(',')]
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = [".pdf", ".docx", ".xlsx", ".txt", ".png", ".jpg", ".jpeg"]
    
    # Document Processing
    OCR_LANGUAGE: str = "pol+eng"  # Polish and English
    CHUNK_SIZE: int = 1000  # Characters per chunk
    CHUNK_OVERLAP: int = 200
    
    # Embeddings
    EMBEDDING_MODEL: str = "text-embedding-nomic-embed-text-v1.5"
    EMBEDDING_DIMENSION: int = 768  # nomic-embed-text-v1.5 produces 768 dimensions
    
    # Product Matching
    SIMILARITY_THRESHOLD: float = 0.7
    MAX_SEARCH_RESULTS: int = 10
    
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)


settings = Settings()
