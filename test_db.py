#!/usr/bin/env python3
"""
Database testing script for OPZ Matcher
"""
import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from sqlalchemy import select, func
from backend.services.database import get_db
from backend.models.database import Product, Document, ProductEmbedding, DocumentChunk, Vendor


async def test_database():
    """Test database operations and check document processing status"""
    async for db in get_db():
        try:
            print("=== Database Test Results ===\n")

            # Check vendors
            result = await db.execute(select(func.count(Vendor.id)))
            vendor_count = result.scalar()
            print(f"Vendors: {vendor_count}")

            # Check products
            result = await db.execute(select(func.count(Product.id)))
            product_count = result.scalar()
            print(f"Products: {product_count}")

            # Check documents
            result = await db.execute(select(func.count(Document.id)))
            doc_count = result.scalar()
            print(f"Documents: {doc_count}")

            # Check processed documents
            result = await db.execute(
                select(func.count(Document.id)).where(Document.is_processed == True)
            )
            processed_count = result.scalar()
            print(f"Processed documents: {processed_count}")

            # Check product embeddings
            result = await db.execute(select(func.count(ProductEmbedding.id)))
            embedding_count = result.scalar()
            print(f"Product embeddings: {embedding_count}")

            # Check document chunks
            result = await db.execute(select(func.count(DocumentChunk.id)))
            chunk_count = result.scalar()
            print(f"Document chunks: {chunk_count}")

            print("\n=== Recent Documents ===")
            # Get recent documents
            result = await db.execute(
                select(Document, Product.name.label('product_name'), Vendor.name.label('vendor_name'))
                .join(Product, Document.product_id == Product.id)
                .join(Vendor, Product.vendor_id == Vendor.id)
                .order_by(Document.created_at.desc())
                .limit(5)
            )
            recent_docs = result.all()

            for doc, product_name, vendor_name in recent_docs:
                status = "✓ Processed" if doc.is_processed else "⏳ Processing" if not doc.processing_error else "✗ Error"
                print(f"- {doc.filename} ({vendor_name} {product_name}) - {status}")
                if doc.processing_error:
                    print(f"  Error: {doc.processing_error[:100]}...")

            print("\n=== Recent Product Embeddings ===")
            # Get recent embeddings
            result = await db.execute(
                select(ProductEmbedding, Product.name.label('product_name'), Vendor.name.label('vendor_name'))
                .join(Product, ProductEmbedding.product_id == Product.id)
                .join(Vendor, Product.vendor_id == Vendor.id)
                .order_by(ProductEmbedding.created_at.desc())
                .limit(5)
            )
            recent_embeddings = result.all()

            for embedding, product_name, vendor_name in recent_embeddings:
                print(f"- {vendor_name} {product_name} - {embedding.content_type}")

            print("\n=== Test Complete ===")

        except Exception as e:
            print(f"Database test failed: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_database())