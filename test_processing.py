#!/usr/bin/env python3
"""
Test document processing and embeddings
"""
import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.services.document_processor import document_processor
from backend.services.claude_service import openrouter_service


async def test_processing():
    """Test document processing and embedding generation"""
    pdf_path = 'poweredge-r570-spec-sheet.pdf'
    if os.path.exists(pdf_path):
        with open(pdf_path, 'rb') as f:
            content = f.read()

        print('Testing document processing...')
        result = await document_processor.process_file(content, pdf_path, 'pdf')
        print(f'Extracted text length: {len(result["text"])}')
        print(f'Number of chunks: {len(result["chunks"])}')

        if result['chunks']:
            print('Testing embedding generation...')
            embedding = await openrouter_service.generate_embedding(result['chunks'][0])
            print(f'Embedding dimension: {len(embedding)}')
            print('Processing test successful!')
        else:
            print('No chunks generated')
    else:
        print(f'PDF file not found: {pdf_path}')


if __name__ == "__main__":
    asyncio.run(test_processing())