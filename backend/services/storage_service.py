"""
Storage service for managing file uploads and downloads (MinIO/S3)
"""
from minio import Minio
from minio.error import S3Error
from typing import BinaryIO, Optional
import io
from loguru import logger

from config.settings import settings


class StorageService:
    """Service for managing file storage in MinIO/S3"""
    
    def __init__(self):
        self.client = Minio(
            settings.STORAGE_ENDPOINT,
            access_key=settings.STORAGE_ACCESS_KEY,
            secret_key=settings.STORAGE_SECRET_KEY,
            secure=settings.STORAGE_SECURE
        )
        self.bucket = settings.STORAGE_BUCKET
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self):
        """Ensure the storage bucket exists"""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
        except S3Error as e:
            logger.error(f"Error ensuring bucket exists: {e}")
            raise
    
    async def upload_file(
        self,
        file_content: bytes,
        filename: str,
        folder: str = ""
    ) -> str:
        """
        Upload a file to storage
        
        Args:
            file_content: File content as bytes
            filename: Name of the file
            folder: Optional folder path
            
        Returns:
            File path in storage
        """
        try:
            # Construct full path
            if folder:
                file_path = f"{folder}/{filename}"
            else:
                file_path = filename
            
            # Upload file
            file_stream = io.BytesIO(file_content)
            self.client.put_object(
                self.bucket,
                file_path,
                file_stream,
                length=len(file_content)
            )
            
            logger.info(f"Uploaded file: {file_path}")
            return file_path
            
        except S3Error as e:
            logger.error(f"Error uploading file {filename}: {e}")
            raise
    
    async def download_file(self, file_path: str) -> bytes:
        """
        Download a file from storage
        
        Args:
            file_path: Path to file in storage
            
        Returns:
            File content as bytes
        """
        try:
            response = self.client.get_object(self.bucket, file_path)
            file_content = response.read()
            response.close()
            response.release_conn()
            
            return file_content
            
        except S3Error as e:
            logger.error(f"Error downloading file {file_path}: {e}")
            raise
    
    async def delete_file(self, file_path: str):
        """Delete a file from storage"""
        try:
            self.client.remove_object(self.bucket, file_path)
            logger.info(f"Deleted file: {file_path}")
        except S3Error as e:
            logger.error(f"Error deleting file {file_path}: {e}")
            raise
    
    async def file_exists(self, file_path: str) -> bool:
        """Check if a file exists in storage"""
        try:
            self.client.stat_object(self.bucket, file_path)
            return True
        except S3Error:
            return False
    
    def get_file_url(self, file_path: str, expires: int = 3600) -> str:
        """
        Get a presigned URL for a file
        
        Args:
            file_path: Path to file in storage
            expires: URL expiration time in seconds (default 1 hour)
            
        Returns:
            Presigned URL
        """
        try:
            url = self.client.presigned_get_object(
                self.bucket,
                file_path,
                expires=expires
            )
            return url
        except S3Error as e:
            logger.error(f"Error generating presigned URL for {file_path}: {e}")
            raise


# Singleton instance
storage_service = StorageService()
