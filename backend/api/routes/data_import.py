"""
API routes for data import mode - uploading and processing product documents
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, validator
import uuid
from datetime import datetime
import json
import asyncio

from services.database import get_db
from services.document_processor import document_processor
from services.claude_service import openrouter_service
from services.storage_service import storage_service
from services.vector_search_service import vector_search_service
from services.benchmark_import_service import benchmark_import_service, BenchmarkVersionInfo, VersionManagementResult
from models.database import Product, Document, Vendor, DeviceCategory, ProductEmbedding, Benchmark
from api.dependencies import get_current_user
from loguru import logger
from config.settings import settings


router = APIRouter()


class VendorCreate(BaseModel):
    name: str
    full_name: Optional[str] = None
    description: Optional[str] = None
    website: Optional[str] = None

    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Vendor name cannot be empty')
        return v.strip()

    @validator('website')
    def website_must_be_valid_url(cls, v):
        if v and v.strip():
            from urllib.parse import urlparse
            parsed = urlparse(v.strip())
            if not parsed.scheme or not parsed.netloc:
                raise ValueError('Website must be a valid URL')
            return v.strip()
        return v


class ProductCreate(BaseModel):
    vendor_name: str
    name: str
    model: Optional[str] = None
    category: DeviceCategory
    description: Optional[str] = None
    notes: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    document_id: int
    filename: str
    status: str
    message: str


class BenchmarkImport(BaseModel):
    name: str
    category: str
    version: Optional[str] = None
    data: List[dict]  # List of benchmark entries
    score_type: Optional[str] = None  # New field for benchmark type (speed_integer, rate_floating_point, passmark_cpu, etc.)
    component_name: Optional[str] = None  # New field for component name (e.g., CPU model)
    component_model: Optional[str] = None  # New field for component model details
    vendor: Optional[str] = None  # New field for vendor information
    benchmark_metadata: Optional[Dict[str, Any]] = None  # New field for additional benchmark metadata


class SPECImportRequest(BaseModel):
    version: Optional[str] = None  # SPEC version (e.g., "2023")


class BatchDocumentUploadRequest(BaseModel):
    product_id: int
    document_type: str = "datasheet"
    validate_specifications: bool = True
    generate_embeddings: bool = True
    processing_options: Optional[Dict[str, Any]] = None


class BatchDocumentUploadResponse(BaseModel):
    batch_id: str
    total_files: int
    processed_files: int
    failed_files: int
    results: List[DocumentUploadResponse]
    validation_summary: Optional[Dict[str, Any]] = None


class SpecificationValidationRequest(BaseModel):
    specifications: Dict[str, Any]
    category: DeviceCategory
    strict_validation: bool = False


@router.post("/vendors", response_model=dict)
async def create_vendor(
    vendor: VendorCreate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a new vendor"""
    from sqlalchemy import select
    from sqlalchemy.exc import IntegrityError

    logger.info(f"Creating vendor: {vendor.name} by user: {current_user.username}")

    try:
        # Check if vendor exists
        result = await db.execute(
            select(Vendor).where(Vendor.name == vendor.name)
        )
        existing = result.scalar_one_or_none()

        if existing:
            logger.warning(f"Vendor creation failed: {vendor.name} already exists")
            raise HTTPException(status_code=400, detail="Vendor already exists")

        # Create vendor
        new_vendor = Vendor(
            name=vendor.name,
            full_name=vendor.full_name,
            description=vendor.description,
            website=vendor.website
        )

        db.add(new_vendor)
        await db.commit()
        await db.refresh(new_vendor)

        logger.info(f"Vendor created successfully: {new_vendor.id} - {new_vendor.name}")

        return {
            "id": new_vendor.id,
            "name": new_vendor.name,
            "message": "Vendor created successfully"
        }

    except IntegrityError as e:
        logger.error(f"Database integrity error creating vendor {vendor.name}: {e}")
        await db.rollback()
        raise HTTPException(status_code=400, detail="Vendor with this name already exists")

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error creating vendor {vendor.name}: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error while creating vendor")


@router.get("/vendors", response_model=List[dict])
async def list_vendors(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List all vendors"""
    from sqlalchemy import select, func

    result = await db.execute(
        select(Vendor, func.count(Product.id).label('product_count'))
        .outerjoin(Product, Vendor.id == Product.vendor_id)
        .group_by(Vendor.id)
        .order_by(Vendor.name)
    )
    vendor_rows = result.all()

    return [
        {
            "id": v.id,
            "name": v.name,
            "full_name": v.full_name,
            "description": v.description,
            "website": v.website,
            "product_count": count
        }
        for v, count in vendor_rows
    ]


@router.get("/products", response_model=List[dict])
async def list_products(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List all products"""
    from sqlalchemy import select

    result = await db.execute(
        select(Product, Vendor.name.label('vendor_name'))
        .join(Vendor, Product.vendor_id == Vendor.id)
        .order_by(Product.name)
    )
    product_rows = result.all()

    products = []
    for p, vendor_name in product_rows:
        # Get documents for this product
        doc_result = await db.execute(
            select(Document).where(Document.product_id == p.id)
        )
        documents = doc_result.scalars().all()

        products.append({
            "id": p.id,
            "name": p.name,
            "model": p.model,
            "category": p.category.value,
            "vendor": vendor_name,
            "description": p.description,
            "notes": p.notes,
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "document_type": doc.document_type,
                    "is_processed": doc.is_processed,
                    "created_at": doc.created_at
                }
                for doc in documents
            ]
        })

    return products


@router.get("/documents", response_model=List[dict])
async def list_documents(
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List all documents with product information"""
    from sqlalchemy import select

    result = await db.execute(
        select(Document, Product.name.label('product_name'), Vendor.name.label('vendor_name'))
        .join(Product, Document.product_id == Product.id)
        .join(Vendor, Product.vendor_id == Vendor.id)
        .order_by(Document.created_at.desc())
    )
    document_rows = result.all()

    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "document_type": doc.document_type,
            "is_processed": doc.is_processed,
            "processing_error": doc.processing_error,
            "product_id": doc.product_id,
            "product_name": product_name,
            "vendor_name": vendor_name,
            "created_at": doc.created_at,
            "processed_at": doc.processed_at
        }
        for doc, product_name, vendor_name in document_rows
    ]


@router.post("/products", response_model=dict)
async def create_product(
    product: ProductCreate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a new product"""
    from sqlalchemy import select
    
    # Get vendor
    result = await db.execute(
        select(Vendor).where(Vendor.name == product.vendor_name)
    )
    vendor = result.scalar_one_or_none()
    
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")
    
    # Create product
    new_product = Product(
        vendor_id=vendor.id,
        name=product.name,
        model=product.model,
        category=product.category,
        description=product.description,
        notes=product.notes,
        specifications={}
    )
    
    db.add(new_product)
    await db.commit()
    await db.refresh(new_product)
    
    return {
        "id": new_product.id,
        "name": new_product.name,
        "model": new_product.model,
        "vendor": vendor.name,
        "message": "Product created successfully"
    }


@router.post("/documents/upload", response_model=List[DocumentUploadResponse])
async def upload_documents(
    files: List[UploadFile] = File(...),
    product_id: int = Form(...),
    document_type: str = Form("datasheet"),
    validate_specs: bool = Form(True),
    generate_embeddings: bool = Form(True),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Upload multiple documents for a product with enhanced validation and processing

    Features:
    - Batch processing with validation
    - Specification validation before processing
    - Automatic embedding generation
    - Background processing with progress tracking
    """
    from sqlalchemy import select

    # Get product
    result = await db.execute(
        select(Product).where(Product.id == product_id)
    )
    product = result.scalar_one_or_none()

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    responses = []
    validation_errors = []

    for file in files:
        try:
            # Validate file
            file_ext = file.filename.split('.')[-1].lower()
            if f".{file_ext}" not in settings.ALLOWED_EXTENSIONS:
                responses.append(DocumentUploadResponse(
                    document_id=0,
                    filename=file.filename,
                    status="error",
                    message=f"File type .{file_ext} not allowed"
                ))
                continue

            # Read file content
            file_content = await file.read()

            # Check file size
            if len(file_content) > settings.MAX_UPLOAD_SIZE:
                responses.append(DocumentUploadResponse(
                    document_id=0,
                    filename=file.filename,
                    status="error",
                    message="File too large"
                ))
                continue

            # Pre-validate document content if requested
            if validate_specs:
                validation_result = await _pre_validate_document(
                    file_content, file.filename, file_ext, product.category
                )
                if not validation_result["is_valid"]:
                    validation_errors.extend(validation_result["errors"])
                    if validation_result["is_critical"]:
                        responses.append(DocumentUploadResponse(
                            document_id=0,
                            filename=file.filename,
                            status="error",
                            message=f"Validation failed: {', '.join(validation_result['errors'])}"
                        ))
                        continue

            # Generate unique filename
            unique_filename = f"{uuid.uuid4()}_{file.filename}"

            # Upload to storage
            file_path = await storage_service.upload_file(
                file_content,
                unique_filename,
                f"products/{product_id}"
            )

            # Create document record
            document = Document(
                product_id=product_id,
                filename=file.filename,
                file_type=file_ext,
                file_path=file_path,
                file_size=len(file_content),
                document_type=document_type,
                is_processed=False
            )

            db.add(document)
            await db.commit()
            await db.refresh(document)

            # Process document in background with enhanced options
            processing_options = {
                "validate_specs": validate_specs,
                "generate_embeddings": generate_embeddings,
                "pre_validation_errors": validation_errors
            }

            # Process document synchronously to avoid threading issues
            processing_result = await process_document_with_validation(
                document.id,
                file_content,
                file.filename,
                file_ext,
                product.id,
                processing_options
            )

            if processing_result["success"]:
                # Update document with successful processing results
                document.extracted_text = processing_result["extracted_text"]
                document.extracted_specs = processing_result["extracted_specs"]
                document.is_processed = True
                document.processed_at = datetime.utcnow()

                # Update product specifications
                if processing_result["extracted_specs"] and 'specifications' in processing_result["extracted_specs"]:
                    product.specifications = {
                        **product.specifications,
                        **processing_result["extracted_specs"]['specifications']
                    }

                # Generate embeddings if requested
                if processing_options.get("generate_embeddings", True):
                    await generate_product_embeddings(product.id, processing_result["extracted_specs"], db)

                await db.commit()
                logger.info(f"Document {document.id} processing completed successfully")
            else:
                # Update document with error
                document.processing_error = processing_result["error"]
                document.is_processed = False
                await db.commit()
                logger.error(f"Document {document.id} processing failed: {processing_result['error']}")

            responses.append(DocumentUploadResponse(
                document_id=document.id,
                filename=file.filename,
                status="processing",
                message="Document uploaded and queued for processing with validation"
            ))

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            responses.append(DocumentUploadResponse(
                document_id=0,
                filename=file.filename,
                status="error",
                message=f"Upload failed: {str(e)}"
            ))

    return responses


@router.post("/documents/batch-upload", response_model=BatchDocumentUploadResponse)
async def batch_upload_documents(
    request: BatchDocumentUploadRequest,
    files: List[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Batch upload documents with comprehensive validation and processing

    Supports:
    - Batch validation before processing
    - Parallel processing
    - Progress tracking
    - Comprehensive error reporting
    """
    from sqlalchemy import select

    batch_id = str(uuid.uuid4())
    logger.info(f"Starting batch upload {batch_id} with {len(files)} files")

    # Get product
    result = await db.execute(
        select(Product).where(Product.id == request.product_id)
    )
    product = result.scalar_one_or_none()

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Pre-validate all files
    validation_summary = await _batch_validate_documents(files, product.category)

    # Filter out invalid files
    valid_files = [f for f, v in zip(files, validation_summary["validations"]) if v["is_valid"]]
    invalid_files = [f for f, v in zip(files, validation_summary["validations"]) if not v["is_valid"]]

    results = []
    processed_count = 0
    failed_count = len(invalid_files)

    # Process valid files
    for file in valid_files:
        try:
            file_ext = file.filename.split('.')[-1].lower()
            file_content = await file.read()

            # Generate unique filename
            unique_filename = f"{uuid.uuid4()}_{file.filename}"

            # Upload to storage
            file_path = await storage_service.upload_file(
                file_content,
                unique_filename,
                f"products/{request.product_id}"
            )

            # Create document record
            document = Document(
                product_id=request.product_id,
                filename=file.filename,
                file_type=file_ext,
                file_path=file_path,
                file_size=len(file_content),
                document_type=request.document_type,
                is_processed=False
            )

            db.add(document)
            await db.commit()
            await db.refresh(document)

            # Process document in background with enhanced options
            processing_options = request.processing_options or {}

            # Process document synchronously to avoid threading issues
            processing_result = await process_document_with_validation(
                document.id,
                file_content,
                file.filename,
                file_ext,
                request.product_id,
                processing_options
            )

            if processing_result["success"]:
                # Update document with successful processing results
                document.extracted_text = processing_result["extracted_text"]
                document.extracted_specs = processing_result["extracted_specs"]
                document.is_processed = True
                document.processed_at = datetime.utcnow()

                # Update product specifications
                if processing_result["extracted_specs"] and 'specifications' in processing_result["extracted_specs"]:
                    product.specifications = {
                        **product.specifications,
                        **processing_result["extracted_specs"]['specifications']
                    }

                # Generate embeddings if requested
                if processing_options.get("generate_embeddings", True):
                    await generate_product_embeddings(request.product_id, processing_result["extracted_specs"], db)

                await db.commit()
                logger.info(f"Document {document.id} batch processing completed successfully")
            else:
                # Update document with error
                document.processing_error = processing_result["error"]
                document.is_processed = False
                await db.commit()
                logger.error(f"Document {document.id} batch processing failed: {processing_result['error']}")

            results.append(DocumentUploadResponse(
                document_id=document.id,
                filename=file.filename,
                status="processing",
                message="Document queued for batch processing"
            ))

            processed_count += 1

        except Exception as e:
            logger.error(f"Error in batch upload for {file.filename}: {e}")
            results.append(DocumentUploadResponse(
                document_id=0,
                filename=file.filename,
                status="error",
                message=f"Batch upload failed: {str(e)}"
            ))
            failed_count += 1

    # Add error responses for invalid files
    for file in invalid_files:
        results.append(DocumentUploadResponse(
            document_id=0,
            filename=file.filename,
            status="error",
            message="File failed pre-validation"
        ))

    return BatchDocumentUploadResponse(
        batch_id=batch_id,
        total_files=len(files),
        processed_files=processed_count,
        failed_files=failed_count,
        results=results,
        validation_summary=validation_summary
    )


async def generate_product_embeddings(product_id: int, specs: dict, db):
    """
    Generate and store embeddings for product specifications and features

    Args:
        product_id: Product ID to associate embeddings with
        specs: Extracted specifications from Claude
        db: Database session
    """
    try:
        logger.info(f"Generating embeddings for product {product_id}")

        # Generate embedding for product specifications
        if specs and 'specifications' in specs:
            specs_text = json.dumps(specs['specifications'], indent=2)
            specs_embedding = await openrouter_service.generate_embedding(specs_text)

            # Create ProductEmbedding record for specifications
            specs_embedding_record = ProductEmbedding(
                product_id=product_id,
                content=specs_text,
                content_type="specifications",
                embedding=specs_embedding
            )
            db.add(specs_embedding_record)
            logger.info(f"Created specifications embedding for product {product_id}")

        # Generate embedding for product description if available
        if specs and 'description' in specs and specs['description']:
            desc_embedding = await openrouter_service.generate_embedding(specs['description'])

            # Create ProductEmbedding record for description
            desc_embedding_record = ProductEmbedding(
                product_id=product_id,
                content=specs['description'],
                content_type="description",
                embedding=desc_embedding
            )
            db.add(desc_embedding_record)
            logger.info(f"Created description embedding for product {product_id}")

        # Generate embeddings for key features if available
        if specs and 'features' in specs and specs['features']:
            features_text = "\n".join(specs['features'])
            features_embedding = await openrouter_service.generate_embedding(features_text)

            # Create ProductEmbedding record for features
            features_embedding_record = ProductEmbedding(
                product_id=product_id,
                content=features_text,
                content_type="features",
                embedding=features_embedding
            )
            db.add(features_embedding_record)
            logger.info(f"Created features embedding for product {product_id}")

        logger.info(f"Successfully generated embeddings for product {product_id}")

    except Exception as e:
        logger.error(f"Error generating embeddings for product {product_id}: {e}")
        # Don't raise exception - embedding generation failure shouldn't block document processing
        # The error is logged but processing continues


async def process_document_with_validation(
    document_id: int,
    file_content: bytes,
    filename: str,
    file_type: str,
    product_id: int,
    processing_options: Dict[str, Any]
) -> Dict[str, Any]:
    """Enhanced document processing with validation - returns processing results"""
    try:
        logger.info(f"Processing document {document_id}: Starting enhanced processing")

        # Process document (without database operations)
        result = await document_processor.process_file(
            file_content, filename, file_type, document_id, None  # No db session
        )
        extracted_text = result['text']

        # Extract specifications (will need product info from caller)
        specs = await openrouter_service.extract_product_specifications(
            extracted_text,
            document_type="datasheet",
            vendor_name=None,  # Will be set by caller
            product_name=None  # Will be set by caller
        )

        # Return processing results for caller to handle database operations
        return {
            "success": True,
            "extracted_text": extracted_text,
            "extracted_specs": specs,
            "chunks": result.get('chunks', []),
            "metadata": result.get('metadata', {})
        }

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        return {
            "success": False,
            "error": str(e),
            "extracted_text": None,
            "extracted_specs": None,
            "chunks": [],
            "metadata": {}
        }


# process_document_batch removed - now using process_document_with_validation directly


async def _pre_validate_document(
    file_content: bytes,
    filename: str,
    file_type: str,
    category: DeviceCategory
) -> Dict[str, Any]:
    """Pre-validate document before processing"""
    try:
        validation_result = {
            "is_valid": True,
            "is_critical": False,
            "errors": [],
            "warnings": []
        }

        # Check file size (already done in upload, but double-check)
        if len(file_content) == 0:
            validation_result["errors"].append("File is empty")
            validation_result["is_critical"] = True
            validation_result["is_valid"] = False

        # Basic content validation
        if file_type.lower() in ['pdf', 'docx']:
            # For PDFs and DOCX, check if they contain text
            try:
                text_sample = await document_processor.extract_text_sample(file_content, filename, file_type)
                if not text_sample or len(text_sample.strip()) < 10:
                    validation_result["errors"].append("Document appears to contain insufficient text content")
                    validation_result["is_critical"] = True
                    validation_result["is_valid"] = False
            except Exception as e:
                validation_result["errors"].append(f"Cannot extract text from document: {str(e)}")
                validation_result["is_critical"] = True
                validation_result["is_valid"] = False

        return validation_result

    except Exception as e:
        logger.error(f"Error in document pre-validation: {e}")
        return {
            "is_valid": False,
            "is_critical": True,
            "errors": [f"Pre-validation failed: {str(e)}"],
            "warnings": []
        }


async def _batch_validate_documents(
    files: List[UploadFile],
    category: DeviceCategory
) -> Dict[str, Any]:
    """Batch validate multiple documents"""
    validations = []
    total_errors = 0
    total_warnings = 0

    for file in files:
        file_content = await file.read()
        file_ext = file.filename.split('.')[-1].lower()

        validation = await _pre_validate_document(
            file_content, file.filename, file_ext, category
        )
        validations.append(validation)

        if not validation["is_valid"]:
            total_errors += 1
        if validation["warnings"]:
            total_warnings += len(validation["warnings"])

    return {
        "total_files": len(files),
        "valid_files": len([v for v in validations if v["is_valid"]]),
        "invalid_files": total_errors,
        "total_warnings": total_warnings,
        "validations": validations
    }


async def _validate_extracted_specs(specs: Dict[str, Any], category: DeviceCategory) -> Dict[str, Any]:
    """Validate extracted specifications"""
    try:
        validation_result = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }

        if not specs or 'specifications' not in specs:
            validation_result["errors"].append("No specifications extracted")
            validation_result["is_valid"] = False
            return validation_result

        extracted_specs = specs['specifications']

        # Category-specific validation
        if category == DeviceCategory.SERVER:
            required_fields = ['cpu', 'memory', 'storage']
        elif category == DeviceCategory.PC:
            required_fields = ['cpu', 'memory', 'storage', 'gpu']
        elif category == DeviceCategory.LAPTOP:
            required_fields = ['cpu', 'memory', 'storage', 'display']
        else:
            required_fields = []

        missing_fields = [field for field in required_fields if field not in extracted_specs]
        if missing_fields:
            validation_result["warnings"].append(f"Missing recommended fields: {', '.join(missing_fields)}")

        # Validate data types and ranges
        for key, value in extracted_specs.items():
            if key in ['cores', 'threads', 'memory_gb', 'storage_gb']:
                if isinstance(value, str):
                    # Try to extract numeric value
                    import re
                    numbers = re.findall(r'\d+', value)
                    if not numbers:
                        validation_result["warnings"].append(f"Could not extract numeric value from {key}: {value}")

        return validation_result

    except Exception as e:
        logger.error(f"Error validating extracted specs: {e}")
        return {
            "is_valid": False,
            "errors": [f"Validation error: {str(e)}"],
            "warnings": []
        }


async def process_document_task(
    document_id: int,
    file_content: bytes,
    filename: str,
    file_type: str,
    product_id: int,
    db: AsyncSession
):
    """Legacy background task - now delegates to enhanced version"""
    processing_options = {
        "validate_specs": True,
        "generate_embeddings": True
    }

    await process_document_with_validation(
        document_id, file_content, filename, file_type, product_id, processing_options
    )


@router.get("/documents/{document_id}", response_model=dict)
async def get_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get document details"""
    from sqlalchemy import select

    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": document.id,
        "filename": document.filename,
        "file_type": document.file_type,
        "document_type": document.document_type,
        "is_processed": document.is_processed,
        "processing_error": document.processing_error,
        "extracted_specs": document.extracted_specs,
        "created_at": document.created_at,
        "processed_at": document.processed_at
    }


@router.delete("/vendors/{vendor_id}", response_model=dict)
async def delete_vendor(
    vendor_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Delete a vendor if it has no associated products"""
    from sqlalchemy import select, delete
    from sqlalchemy.exc import IntegrityError

    logger.info(f"DEBUG: DELETE /api/import/vendors/{vendor_id} called by user: {current_user.username}")
    logger.info(f"DEBUG: Vendor ID parameter: {vendor_id} (type: {type(vendor_id)})")

    try:
        # Get vendor first to check if it exists
        result = await db.execute(
            select(Vendor).where(Vendor.id == vendor_id)
        )
        vendor = result.scalar_one_or_none()

        if not vendor:
            logger.warning(f"Vendor deletion failed: vendor {vendor_id} not found")
            raise HTTPException(status_code=404, detail="Vendor not found")

        # Check if vendor has associated products
        products_result = await db.execute(
            select(Product).where(Product.vendor_id == vendor_id).limit(1)
        )
        has_products = products_result.scalar_one_or_none() is not None

        if has_products:
            logger.warning(f"Vendor deletion failed: vendor {vendor_id} has associated products")
            raise HTTPException(status_code=400, detail="Cannot delete vendor with associated products. Delete products first.")

        # Delete the vendor record
        try:
            await db.execute(
                delete(Vendor).where(Vendor.id == vendor_id)
            )
            logger.info(f"Vendor record {vendor_id} deleted from database")
        except IntegrityError as e:
            logger.error(f"Database integrity error deleting vendor {vendor_id}: {e}")
            await db.rollback()
            raise HTTPException(status_code=400, detail="Cannot delete vendor due to database constraints")
        except Exception as e:
            logger.error(f"Error deleting vendor record {vendor_id}: {e}")
            await db.rollback()
            raise HTTPException(status_code=500, detail="Failed to delete vendor from database")

        await db.commit()
        logger.info(f"Vendor {vendor_id} deletion completed successfully")

        return {
            "message": "Vendor deleted successfully",
            "vendor_id": vendor_id
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error deleting vendor {vendor_id}: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error while deleting vendor")


@router.delete("/products/{product_id}", response_model=dict)
async def delete_product(
    product_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Delete a product and all its associated documents and data"""
    from sqlalchemy import select, delete
    from sqlalchemy.exc import IntegrityError

    logger.info(f"DEBUG: DELETE /api/import/products/{product_id} called by user: {current_user.username}")
    logger.info(f"DEBUG: Product ID parameter: {product_id} (type: {type(product_id)})")

    try:
        # Get product first to check if it exists
        result = await db.execute(
            select(Product).where(Product.id == product_id)
        )
        product = result.scalar_one_or_none()

        if not product:
            logger.warning(f"Product deletion failed: product {product_id} not found")
            raise HTTPException(status_code=404, detail="Product not found")

        # Get all documents for this product to delete associated data
        documents_result = await db.execute(
            select(Document).where(Document.product_id == product_id)
        )
        documents = documents_result.scalars().all()

        # Delete associated document chunks first
        try:
            from models.database import DocumentChunk
            deleted_chunks = await db.execute(
                delete(DocumentChunk).where(DocumentChunk.document_id.in_([doc.id for doc in documents]))
            )
            logger.info(f"Deleted {deleted_chunks.rowcount} document chunks for product {product_id}")
        except Exception as e:
            logger.error(f"Error deleting document chunks for product {product_id}: {e}")
            # Continue with deletion even if chunks fail

        # Delete associated embeddings
        try:
            deleted_embeddings = await db.execute(
                delete(ProductEmbedding).where(ProductEmbedding.product_id == product_id)
            )
            logger.info(f"Deleted {deleted_embeddings.rowcount} embeddings for product {product_id}")
        except Exception as e:
            logger.error(f"Error deleting embeddings for product {product_id}: {e}")
            # Continue with deletion even if embeddings fail

        # Delete document files from storage
        for document in documents:
            try:
                await storage_service.delete_file(document.file_path)
                logger.info(f"Deleted file {document.file_path} from storage")
            except Exception as e:
                logger.warning(f"Could not delete file from storage: {e}")
                # Don't fail the whole operation if file deletion fails

        # Delete all documents for this product
        try:
            deleted_docs = await db.execute(
                delete(Document).where(Document.product_id == product_id)
            )
            logger.info(f"Deleted {deleted_docs.rowcount} documents for product {product_id}")
        except Exception as e:
            logger.error(f"Error deleting documents for product {product_id}: {e}")
            await db.rollback()
            raise HTTPException(status_code=500, detail="Failed to delete associated documents")

        # Delete the product record
        try:
            await db.execute(
                delete(Product).where(Product.id == product_id)
            )
            logger.info(f"Product record {product_id} deleted from database")
        except IntegrityError as e:
            logger.error(f"Database integrity error deleting product {product_id}: {e}")
            await db.rollback()
            raise HTTPException(status_code=400, detail="Cannot delete product due to database constraints")
        except Exception as e:
            logger.error(f"Error deleting product record {product_id}: {e}")
            await db.rollback()
            raise HTTPException(status_code=500, detail="Failed to delete product from database")

        await db.commit()
        logger.info(f"Product {product_id} deletion completed successfully")

        return {
            "message": "Product deleted successfully",
            "product_id": product_id
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"Unexpected error deleting product {product_id}: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error while deleting product")


@router.delete("/documents/{document_id}", response_model=dict)
async def delete_document(
    document_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Delete a document and its associated data"""
    from sqlalchemy import select, delete
    from sqlalchemy.exc import IntegrityError

    logger.info(f"DEBUG: DELETE /api/import/documents/{document_id} called by user: {current_user.username}")
    logger.info(f"DEBUG: Document ID parameter: {document_id} (type: {type(document_id)})")

    try:
        # Get document first to check if it exists and get file path
        logger.info(f"DEBUG: Fetching document {document_id} from database")
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()

        if not document:
            logger.warning(f"DEBUG: Document deletion failed: document {document_id} not found")
            raise HTTPException(status_code=404, detail="Document not found")

        logger.info(f"DEBUG: Found document {document_id}, filename: {document.filename}, product_id: {document.product_id}")

        # Delete associated document chunks first
        try:
            logger.info(f"DEBUG: Deleting document chunks for document {document_id}")
            from models.database import DocumentChunk
            deleted_chunks = await db.execute(
                delete(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )
            logger.info(f"DEBUG: Deleted {deleted_chunks.rowcount} document chunks for document {document_id}")
        except Exception as e:
            logger.error(f"DEBUG: Error deleting document chunks for document {document_id}: {e}")
            await db.rollback()
            raise HTTPException(status_code=500, detail="Failed to delete associated document chunks")

        # Delete the document record
        try:
            logger.info(f"DEBUG: Deleting document record {document_id} from database")
            result = await db.execute(
                delete(Document).where(Document.id == document_id)
            )
            logger.info(f"DEBUG: Document record {document_id} deleted from database (affected: {result.rowcount})")
            if result.rowcount == 0:
                logger.warning(f"DEBUG: No document found with ID {document_id} to delete")
                raise HTTPException(status_code=404, detail="Document not found")
        except IntegrityError as e:
            logger.error(f"DEBUG: Database integrity error deleting document {document_id}: {e}")
            await db.rollback()
            raise HTTPException(status_code=400, detail="Cannot delete document due to database constraints")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"DEBUG: Error deleting document record {document_id}: {e}")
            await db.rollback()
            raise HTTPException(status_code=500, detail="Failed to delete document from database")

        # Try to delete file from storage (don't fail if it doesn't exist)
        try:
            logger.info(f"DEBUG: Deleting file from storage: {document.file_path}")
            await storage_service.delete_file(document.file_path)
            logger.info(f"DEBUG: File {document.file_path} deleted from storage")
        except Exception as e:
            logger.warning(f"DEBUG: Could not delete file from storage: {e}")
            # Don't fail the whole operation if file deletion fails

        # Try to delete file from storage (don't fail if it doesn't exist)
        try:
            logger.info(f"DEBUG: Deleting file from storage: {document.file_path}")
            await storage_service.delete_file(document.file_path)
            logger.info(f"DEBUG: File {document.file_path} deleted from storage")
        except Exception as e:
            logger.warning(f"DEBUG: Could not delete file from storage: {e}")
            # Don't fail the whole operation if file deletion fails

        await db.commit()
        logger.info(f"DEBUG: Document {document_id} deletion completed successfully")

        return {
            "message": "Document deleted successfully",
            "document_id": document_id
        }

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except Exception as e:
        logger.error(f"DEBUG: Unexpected error deleting document {document_id}: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Internal server error while deleting document")


# Benchmark Version Management API Routes

@router.get("/benchmarks/versions", response_model=List[dict])
async def list_benchmark_version_groups(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """List all benchmark version groups with their latest versions"""
    try:
        groups = await benchmark_import_service.get_benchmark_version_groups(db, limit=limit, offset=offset)

        return groups

    except Exception as e:
        logger.error(f"Error listing benchmark version groups: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list version groups: {str(e)}")


@router.get("/benchmarks/versions/{version_group_id}", response_model=dict)
async def get_benchmark_versions(
    version_group_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get all versions of a specific benchmark"""
    try:
        version_info = await benchmark_import_service.get_benchmark_versions(version_group_id, db)

        return {
            "version_group_id": version_info.version_group_id,
            "version_count": version_info.version_count,
            "latest_version_id": version_info.latest_version_id,
            "latest_version_created_at": version_info.latest_version_created_at,
            "versions": version_info.versions
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting benchmark versions for group {version_group_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get versions: {str(e)}")


@router.post("/benchmarks/versions/{benchmark_id}/set-active", response_model=dict)
async def set_active_benchmark_version(
    benchmark_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Set a specific benchmark version as the active (latest) version"""
    try:
        result = await benchmark_import_service.set_active_version(benchmark_id, db)

        if not result.success:
            raise HTTPException(status_code=400, detail=result.message)

        return {
            "message": result.message,
            "benchmark_id": benchmark_id,
            "affected_versions": result.affected_versions
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting active version for benchmark {benchmark_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set active version: {str(e)}")


@router.delete("/benchmarks/versions/{benchmark_id}", response_model=dict)
async def delete_benchmark_version(
    benchmark_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Delete a specific benchmark version"""
    try:
        result = await benchmark_import_service.delete_benchmark_version(benchmark_id, db)

        if not result.success:
            raise HTTPException(status_code=404, detail=result.message)

        return {
            "message": result.message,
            "benchmark_id": benchmark_id,
            "affected_versions": result.affected_versions
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting benchmark version {benchmark_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete version: {str(e)}")


@router.post("/benchmarks/versions/cleanup", response_model=dict)
async def cleanup_old_benchmark_versions(
    max_versions_per_group: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Clean up old benchmark versions across all groups"""
    try:
        result = await benchmark_import_service.cleanup_old_versions(db, max_versions_per_group=max_versions_per_group)

        return {
            "message": result.message,
            "operation": result.operation,
            "success": result.success,
            "affected_versions": result.affected_versions,
            "errors": result.errors
        }

    except Exception as e:
        logger.error(f"Error during benchmark version cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.post("/benchmarks/versions/create", response_model=dict)
async def create_benchmark_version(
    benchmark_data: dict,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Create a new benchmark version manually"""
    try:
        # Validate required fields
        required_fields = ['name', 'category', 'component_name', 'score', 'score_type']
        missing_fields = [field for field in required_fields if field not in benchmark_data]

        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required fields: {', '.join(missing_fields)}"
            )

        benchmark = await benchmark_import_service.create_benchmark_version(benchmark_data, db)

        return {
            "message": "Benchmark version created successfully",
            "benchmark_id": benchmark.id,
            "version_group_id": benchmark.version_group_id,
            "content_hash": benchmark.content_hash,
            "is_latest_version": benchmark.is_latest_version
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating benchmark version: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create version: {str(e)}")


@router.post("/benchmarks/import", response_model=dict)
async def import_benchmarks(
    benchmark_data: BenchmarkImport,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Import benchmark data"""
    from models.database import Benchmark

    imported_count = 0

    for entry in benchmark_data.data:
        # Use new fields from request if provided, otherwise fall back to entry data
        benchmark = Benchmark(
            name=benchmark_data.name,
            category=benchmark_data.category,
            version=benchmark_data.version,
            component_name=benchmark_data.component_name or entry.get('component_name'),
            component_model=benchmark_data.component_model or entry.get('component_model'),
            vendor=benchmark_data.vendor or entry.get('vendor'),
            score=entry.get('score'),
            score_type=benchmark_data.score_type or entry.get('score_type'),
            benchmark_metadata=benchmark_data.benchmark_metadata or entry.get('metadata', {}),
            content_hash=None,  # Will be set by service if needed
            version_group_id=None,  # Will be set by service if needed
            is_latest_version=True  # Default for manual imports
        )

        # Use the service to create benchmark with version management if it has the required fields
        if benchmark.score_type and benchmark.component_name:
            try:
                # Convert to dict for service method
                benchmark_dict = {
                    'name': benchmark.name,
                    'category': benchmark.category,
                    'version': benchmark.version,
                    'component_name': benchmark.component_name,
                    'component_model': benchmark.component_model,
                    'vendor': benchmark.vendor,
                    'score': benchmark.score,
                    'score_type': benchmark.score_type,
                    'benchmark_metadata': benchmark.benchmark_metadata
                }
                created_benchmark = await benchmark_import_service.create_benchmark_version(benchmark_dict, db)
                imported_count += 1
            except ValueError as e:
                logger.warning(f"Skipping duplicate benchmark entry: {e}")
                continue
        else:
            # Fallback to direct database insertion for legacy compatibility
            db.add(benchmark)
            imported_count += 1

    await db.commit()

    return {
        "message": f"Imported {imported_count} benchmark entries with version management",
        "benchmark_name": benchmark_data.name,
        "category": benchmark_data.category,
        "score_type": benchmark_data.score_type,
        "component_name": benchmark_data.component_name,
        "vendor": benchmark_data.vendor,
        "version_management": {
            "versions_created": imported_count,
            "history_maintained": "10 latest versions per benchmark group"
        }
    }


@router.post("/benchmarks/import/spec-csv", response_model=dict)
async def import_spec_csv(
    file: UploadFile = File(...),
    version: str = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Import SPEC benchmark data from CSV file with version management

    Supports all SPEC CPU benchmark types:
    - SPECspeed Integer
    - SPECspeed Floating Point
    - SPECrate Integer
    - SPECrate Floating Point

    Handles all required columns: Test Sponsor, System Name, Parallel (for Speed),
    Base Threads/Copies, Processor specs, Results (Base/Peak), Energy (Base/Peak)

    Features:
    - Automatic version management
    - Duplicate detection based on content hash
    - History management (keeps 10 latest versions per benchmark)
    """
    # Validate file type
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")

    # Read file content
    file_content = await file.read()

    # Check file size (limit to 50MB)
    if len(file_content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    try:
        # Import SPEC data using the service (now includes version management)
        result = await benchmark_import_service.import_spec_csv(
            csv_content=file_content,
            filename=file.filename,
            db=db,
            version=version
        )

        # Prepare response with version management info
        response = {
            "message": f"SPEC benchmark import completed with version management",
            "filename": file.filename,
            "total_rows": result.total_rows,
            "imported_entries": result.imported_entries,
            "benchmark_types": result.benchmark_types,
            "errors": result.errors,
            "warnings": result.warnings,
            "version_management": {
                "duplicates_skipped": result.total_rows - len(result.errors) - result.imported_entries,
                "versions_created": result.imported_entries,
                "history_maintained": "10 latest versions per benchmark group"
            }
        }

        # Add success/failure status
        if result.errors:
            response["status"] = "partial_success" if result.imported_entries > 0 else "failed"
        else:
            response["status"] = "success"

        logger.info(f"SPEC CSV import completed: {result.imported_entries} entries imported from {result.total_rows} rows with version management")

        return response

    except Exception as e:
        logger.error(f"Error importing SPEC CSV {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")


@router.post("/benchmarks/import/passmark-csv", response_model=dict)
async def import_passmark_csv(
    file: UploadFile = File(...),
    benchmark_type: str = Form("PASSMARK_CPU"),  # PASSMARK_CPU or PASSMARK_GPU
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Import PassMark benchmark data from CSV file with version management

    Supports:
    - PASSMARK_CPU: CPU benchmark scores
    - PASSMARK_GPU: GPU benchmark scores

    Features:
    - Automatic version management
    - Duplicate detection based on content hash
    - History management (keeps 10 latest versions per benchmark)
    """
    # Validate file type
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV file")

    # Validate benchmark type
    valid_types = ["PASSMARK_CPU", "PASSMARK_GPU"]
    if benchmark_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid benchmark type: {benchmark_type}. Must be one of: {', '.join(valid_types)}"
        )

    # Read file content
    file_content = await file.read()

    # Check file size (limit to 50MB)
    if len(file_content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")

    try:
        # Import PassMark data using the service (includes version management)
        result = await benchmark_import_service.import_passmark_csv(
            csv_content=file_content,
            filename=file.filename,
            db=db,
            benchmark_type=benchmark_type
        )

        # Prepare response with version management info
        response = {
            "message": f"PassMark benchmark import completed with version management",
            "filename": file.filename,
            "benchmark_type": benchmark_type,
            "total_rows": result.total_rows,
            "imported_entries": result.imported_entries,
            "benchmark_types": result.benchmark_types,
            "errors": result.errors,
            "warnings": result.warnings,
            "version_management": {
                "duplicates_skipped": result.total_rows - len(result.errors) - result.imported_entries,
                "versions_created": result.imported_entries,
                "history_maintained": "10 latest versions per benchmark group"
            }
        }

        # Add success/failure status
        if result.errors:
            response["status"] = "partial_success" if result.imported_entries > 0 else "failed"
        else:
            response["status"] = "success"

        logger.info(f"PassMark CSV import completed: {result.imported_entries} entries imported from {result.total_rows} rows with version management")

        return response

    except Exception as e:
        logger.error(f"Error importing PassMark CSV {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")
