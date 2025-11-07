"""
API routes for product search mode - finding matching products for OPZ requirements
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import json

from services.database import get_db
from services.claude_service import openrouter_service
from services.vector_search_service import vector_search_service
from models.database import Product, DeviceCategory, Benchmark, SearchHistory
from api.dependencies import get_current_user
from loguru import logger


router = APIRouter()


class ProductSearchRequest(BaseModel):
    requirements_text: str
    category: Optional[DeviceCategory] = None
    vendor_filter: Optional[List[str]] = None
    min_match_score: float = 0.6
    use_vector_search: bool = True  # New parameter for backward compatibility


class ProductMatch(BaseModel):
    product_id: int
    vendor: str
    name: str
    model: Optional[str]
    match_score: float
    exact_matches: List[str]
    close_matches: List[str]
    deviations: List[dict]
    adjustable_requirements: List[dict]
    benchmark_analysis: Optional[dict]
    recommendation: str


class SearchResponse(BaseModel):
    matched_products: List[ProductMatch]
    questions_for_customer: List[str]
    general_analysis: str
    search_id: int


class AdvancedSearchRequest(BaseModel):
    query_text: str
    category: Optional[DeviceCategory] = None
    vendor_filter: Optional[List[str]] = None
    min_match_score: float = 0.6
    search_mode: str = "hybrid"  # hybrid, vector, text, benchmark
    include_benchmarks: bool = True
    include_specifications: bool = True
    max_results: int = 50
    multimodal_filters: Optional[Dict[str, Any]] = None  # Additional filters for multimodal search


class AdvancedSearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_found: int
    search_metadata: Dict[str, Any]
    multimodal_analysis: Optional[Dict[str, Any]] = None


class SearchSuggestionsRequest(BaseModel):
    partial_query: str
    category: Optional[DeviceCategory] = None
    max_suggestions: int = 10


class SearchSuggestionsResponse(BaseModel):
    suggestions: List[str]
    category_suggestions: List[str]
    vendor_suggestions: List[str]


class SpecificationValidationRequest(BaseModel):
    specifications: Dict[str, Any]
    category: DeviceCategory
    validation_rules: Optional[Dict[str, Any]] = None


class SpecificationValidationResponse(BaseModel):
    is_valid: bool
    validation_errors: List[str]
    validation_warnings: List[str]
    normalized_specifications: Dict[str, Any]
    compliance_score: float


@router.post("/search", response_model=SearchResponse)
async def search_products(
    search_request: ProductSearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Search for products matching OPZ requirements

    This endpoint supports both traditional Claude-based matching and new vector similarity search:
    - Vector search: Multi-stage retrieval (embedding → vector search → reranking → response)
    - Traditional search: Claude analysis with benchmark data (backward compatibility)

    The search method is determined by the use_vector_search parameter.
    """
    if search_request.use_vector_search:
        # Use new vector similarity search
        return await _vector_search_products(search_request, db)
    else:
        # Use traditional Claude-based search for backward compatibility
        return await _traditional_search_products(search_request, db)


async def _vector_search_products(
    search_request: ProductSearchRequest,
    db: AsyncSession
) -> SearchResponse:
    """
    Perform vector similarity search with multi-stage retrieval
    """
    try:
        # Stage 1-4: Vector search pipeline
        search_result = await vector_search_service.search_products(
            query_text=search_request.requirements_text,
            db=db,
            category=search_request.category,
            vendor_filter=search_request.vendor_filter,
            min_match_score=search_request.min_match_score,
            limit=50
        )

        # Convert vector search results to ProductMatch format
        matched_products = []
        for result in search_result["results"]:
            if result["final_score"] >= search_request.min_match_score:
                # Get full product details with vendor relationship loaded
                from sqlalchemy.orm import joinedload
                product_result = await db.execute(
                    select(Product).options(joinedload(Product.vendor)).where(Product.id == result["product_id"])
                )
                product = product_result.scalar_one()

                matched_products.append(
                    ProductMatch(
                        product_id=product.id,
                        vendor=product.vendor.name,
                        name=product.name,
                        model=product.model,
                        match_score=result["final_score"],
                        exact_matches=[],  # Vector search doesn't provide detailed matches
                        close_matches=[],
                        deviations=[],
                        adjustable_requirements=[],
                        benchmark_analysis=None,
                        recommendation=f"Vector similarity score: {result['vector_score']:.3f}"
                    )
                )

        # Sort by match score
        matched_products.sort(key=lambda x: x.match_score, reverse=True)

        # Get benchmark data for additional analysis (optional)
        benchmark_data = await _get_benchmark_data(
            db,
            search_request.requirements_text,
            search_request.category
        )

        # Use Claude for additional analysis and questions
        analysis_result = await _analyze_search_results(
            search_request.requirements_text,
            matched_products,
            benchmark_data
        )

        # Save search history
        search_history = SearchHistory(
            query_text=search_request.requirements_text,
            requirements=search_request.dict(),
            category=search_request.category,
            results=[
                {"product_id": m.product_id, "match_score": m.match_score}
                for m in matched_products
            ]
        )

        db.add(search_history)
        await db.commit()
        await db.refresh(search_history)

        # Extract questions_for_customer as list of strings
        questions_for_customer = []
        raw_questions = analysis_result.get('questions_for_customer', [])
        if isinstance(raw_questions, list):
            for q in raw_questions:
                if isinstance(q, str):
                    questions_for_customer.append(q)
                elif isinstance(q, dict) and 'question' in q:
                    questions_for_customer.append(q['question'])
                elif isinstance(q, dict):
                    # Handle dict format by extracting the first string value
                    for key, value in q.items():
                        if isinstance(value, str):
                            questions_for_customer.append(value)
                            break
                else:
                    questions_for_customer.append(str(q))

        return SearchResponse(
            matched_products=matched_products,
            questions_for_customer=questions_for_customer,
            general_analysis=analysis_result.get('general_analysis',
                f'Vector search found {len(matched_products)} matching products.'),
            search_id=search_history.id
        )

    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        # Rollback any pending transaction to prevent session corruption
        await _rollback_on_error(db, e)

        # Fallback to traditional search
        return await _traditional_search_products(search_request, db)


async def _traditional_search_products(
    search_request: ProductSearchRequest,
    db: AsyncSession
) -> SearchResponse:
    """
    Traditional Claude-based product search (backward compatibility)
    """
    from sqlalchemy import select, and_
    from sqlalchemy.orm import joinedload

    # Build query for products
    query = select(Product).options(joinedload(Product.vendor))

    filters = []
    if search_request.category:
        filters.append(Product.category == search_request.category)

    if search_request.vendor_filter:
        from models.database import Vendor
        vendor_result = await db.execute(
            select(Vendor.id).where(Vendor.name.in_(search_request.vendor_filter))
        )
        vendor_ids = [row[0] for row in vendor_result.all()]
        if vendor_ids:
            filters.append(Product.vendor_id.in_(vendor_ids))

    if filters:
        query = query.where(and_(*filters))

    # Execute query
    result = await db.execute(query.limit(50))  # Limit for performance
    products = result.scalars().all()

    if not products:
        raise HTTPException(
            status_code=404,
            detail="No products found matching the filters"
        )

    # Prepare products context for Claude
    products_context = []
    for product in products:
        products_context.append({
            "id": product.id,
            "vendor": product.vendor.name,
            "name": product.name,
            "model": product.model,
            "category": product.category.value,
            "specifications": product.specifications,
            "description": product.description
        })

    # Get relevant benchmark data
    benchmark_data = await _get_benchmark_data(
        db,
        search_request.requirements_text,
        search_request.category
    )

    # Use OpenRouter to match products
    match_result = await openrouter_service.match_products(
        requirements=search_request.requirements_text,
        products_context=products_context,
        benchmark_data=benchmark_data
    )

    # Parse results
    matched_products = []

    if 'matched_products' in match_result:
        for match in match_result['matched_products']:
            try:
                product_idx = int(match.get('product_id', 0))
                if 0 <= product_idx < len(products):
                    product = products[product_idx]

                    match_score = float(match.get('match_score', 0)) / 100.0

                    if match_score >= search_request.min_match_score:
                        matched_products.append(
                            ProductMatch(
                                product_id=product.id,
                                vendor=product.vendor.name,
                                name=product.name,
                                model=product.model,
                                match_score=match_score,
                                exact_matches=match.get('exact_matches', []),
                                close_matches=match.get('close_matches', []),
                                deviations=match.get('deviations', []),
                                adjustable_requirements=match.get('adjustable_requirements', []),
                                benchmark_analysis=match.get('benchmark_analysis'),
                                recommendation=match.get('recommendation', '')
                            )
                        )
            except (ValueError, IndexError) as e:
                logger.warning(f"Error parsing match result: {e}")
                continue

    # Sort by match score
    matched_products.sort(key=lambda x: x.match_score, reverse=True)

    # Save search history
    search_history = SearchHistory(
        query_text=search_request.requirements_text,
        requirements=search_request.dict(),
        category=search_request.category,
        results=[
            {"product_id": m.product_id, "match_score": m.match_score}
            for m in matched_products
        ]
    )

    db.add(search_history)
    await db.commit()
    await db.refresh(search_history)

    # Extract questions_for_customer as list of strings
    questions_for_customer = []
    raw_questions = match_result.get('questions_for_customer', [])
    if isinstance(raw_questions, list):
        for q in raw_questions:
            if isinstance(q, str):
                questions_for_customer.append(q)
            elif isinstance(q, dict) and 'question' in q:
                questions_for_customer.append(q['question'])
            else:
                questions_for_customer.append(str(q))

    return SearchResponse(
        matched_products=matched_products,
        questions_for_customer=questions_for_customer,
        general_analysis=match_result.get('general_analysis', ''),
        search_id=search_history.id
    )


async def _get_benchmark_data(
    db: AsyncSession,
    requirements_text: str,
    category: Optional[DeviceCategory]
) -> dict:
    """Retrieve relevant benchmark data based on requirements"""
    from sqlalchemy import select

    # Extract potential benchmark names from requirements
    # (simplified - in production, use NLP or Claude to extract)
    benchmark_keywords = ['passmark', 'geekbench', 'cinebench', 'spec', '3dmark']

    benchmark_data = {}

    # Query benchmarks
    query = select(Benchmark)

    # Add category filter if applicable
    if category:
        if category in [DeviceCategory.SERVER, DeviceCategory.PC, DeviceCategory.LAPTOP]:
            query = query.where(Benchmark.category.in_(['cpu', 'memory', 'storage']))

    result = await db.execute(query.limit(100))
    benchmarks = result.scalars().all()

    # Organize by category
    for benchmark in benchmarks:
        if benchmark.category not in benchmark_data:
            benchmark_data[benchmark.category] = {}

        if benchmark.name not in benchmark_data[benchmark.category]:
            benchmark_data[benchmark.category][benchmark.name] = []

        benchmark_data[benchmark.category][benchmark.name].append({
            'component': benchmark.component_name,
            'model': benchmark.component_model,
            'vendor': benchmark.vendor,
            'score': benchmark.score,
            'score_type': benchmark.score_type
        })

    return benchmark_data


async def _analyze_search_results(
    requirements_text: str,
    matched_products: List[ProductMatch],
    benchmark_data: dict
) -> dict:
    """
    Use Claude to analyze search results and provide additional insights

    This is used in vector search to provide questions and general analysis
    """
    try:
        # Prepare context for Claude
        products_summary = []
        for i, product in enumerate(matched_products[:5]):  # Limit for token efficiency
            products_summary.append({
                "index": i,
                "vendor": product.vendor,
                "name": product.name,
                "model": product.model,
                "match_score": product.match_score
            })

        analysis_prompt = f"""Analyze these search results for OPZ requirements and provide insights:

OPZ Requirements:
{requirements_text}

Top Matching Products:
{json.dumps(products_summary, indent=2, ensure_ascii=False)}

Benchmark Data Available:
{json.dumps(benchmark_data, indent=2, ensure_ascii=False) if benchmark_data else "None"}

Provide a JSON response with:
{{
    "questions_for_customer": ["List any clarifying questions for the customer"],
    "general_analysis": "Brief analysis of the search results and recommendations"
}}"""

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{openrouter_service.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_service.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": openrouter_service.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert OPZ analyst. Provide concise, actionable analysis."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    "max_tokens": openrouter_service.max_tokens,
                    "temperature": 0.1
                },
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]

        # Parse JSON response
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
            else:
                result = {"raw_response": content}
        except json.JSONDecodeError:
            result = {"raw_response": content}

        return result

    except Exception as e:
        logger.error(f"Error analyzing search results: {e}")
        return {
            "questions_for_customer": [],
            "general_analysis": "Analysis not available due to processing error."
        }


async def _rollback_on_error(db: AsyncSession, error: Exception):
    """Helper function to rollback database transactions on error"""
    try:
        await db.rollback()
        logger.info("Database transaction rolled back successfully")
    except Exception as rollback_error:
        logger.error(f"Error during rollback: {rollback_error}")


async def _safe_db_operation(db: AsyncSession, operation_func, *args, **kwargs):
    """Wrapper for database operations with automatic rollback on failure"""
    try:
        return await operation_func(db, *args, **kwargs)
    except Exception as e:
        logger.error(f"Database operation failed: {e}")
        await _rollback_on_error(db, e)
        raise


@router.get("/products/{product_id}", response_model=dict)
async def get_product_details(
    product_id: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Get detailed product information"""
    from sqlalchemy import select

    logger.info(f"Fetching product details for product_id: {product_id}")

    try:
        # Load product data
        product_result = await db.execute(
            select(
                Product.id,
                Product.name,
                Product.model,
                Product.category,
                Product.specifications,
                Product.description,
                Product.notes,
                Product.created_at,
                Product.updated_at,
                Product.vendor_id
            ).where(Product.id == product_id)
        )
        product_row = product_result.first()

        if not product_row:
            logger.warning(f"Product not found: {product_id}")
            raise HTTPException(status_code=404, detail="Product not found")

        # Load vendor data separately
        from models.database import Vendor
        vendor_result = await db.execute(
            select(Vendor.name).where(Vendor.id == product_row.vendor_id)
        )
        vendor_row = vendor_result.first()
        vendor_name = vendor_row.name if vendor_row else "Unknown"

        # Load documents data separately
        from models.database import Document
        documents_result = await db.execute(
            select(
                Document.id,
                Document.filename,
                Document.document_type,
                Document.is_processed
            ).join(Product.documents).where(Product.id == product_id)
        )
        documents = documents_result.all()

        # Build documents list
        documents_list = [
            {
                "id": doc.id,
                "filename": doc.filename,
                "document_type": doc.document_type,
                "is_processed": doc.is_processed
            }
            for doc in documents
        ]

        logger.info(f"Returning product details for: {product_row.name}")

        return {
            "id": product_row.id,
            "vendor": vendor_name,
            "name": product_row.name,
            "model": product_row.model,
            "category": product_row.category.value,
            "specifications": product_row.specifications,
            "description": product_row.description,
            "notes": product_row.notes,
            "documents": documents_list,
            "created_at": product_row.created_at,
            "updated_at": product_row.updated_at
        }
    except Exception as e:
        logger.error(f"Error in get_product_details: {e}")
        raise


@router.post("/search/{search_id}/feedback", response_model=dict)
async def submit_search_feedback(
    search_id: int,
    selected_product_id: int,
    feedback_score: int,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Submit feedback on search results
    
    This helps improve future searches
    """
    from sqlalchemy import select
    
    if not 1 <= feedback_score <= 5:
        raise HTTPException(status_code=400, detail="Feedback score must be 1-5")
    
    result = await db.execute(
        select(SearchHistory).where(SearchHistory.id == search_id)
    )
    search = result.scalar_one_or_none()
    
    if not search:
        raise HTTPException(status_code=404, detail="Search not found")
    
    search.selected_product_id = selected_product_id
    search.feedback_score = feedback_score
    
    await db.commit()
    
    return {
        "message": "Feedback submitted successfully",
        "search_id": search_id
    }


@router.post("/advanced", response_model=AdvancedSearchResponse)
async def advanced_search(
    search_request: AdvancedSearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Advanced multimodal hybrid search endpoint

    Supports multiple search modes:
    - hybrid: Combines vector, text, and benchmark search
    - vector: Pure vector similarity search
    - text: Full-text search
    - benchmark: Benchmark-based search
    """
    try:
        # Determine search parameters based on mode
        use_hybrid = search_request.search_mode == "hybrid"
        benchmark_filter = None

        if search_request.search_mode == "benchmark":
            # Extract benchmark type from multimodal filters
            benchmark_filter = search_request.multimodal_filters.get("benchmark_type") if search_request.multimodal_filters else None

        # Perform search using vector search service
        search_result = await vector_search_service.search_products(
            query_text=search_request.query_text,
            db=db,
            category=search_request.category,
            vendor_filter=search_request.vendor_filter,
            min_match_score=search_request.min_match_score,
            limit=search_request.max_results,
            use_hybrid_search=use_hybrid,
            benchmark_filter=benchmark_filter
        )

        # Add multimodal analysis if requested
        multimodal_analysis = None
        if search_request.multimodal_filters:
            multimodal_analysis = await _analyze_multimodal_results(
                search_request.query_text,
                search_result["results"],
                search_request.multimodal_filters
            )

        # Filter results based on additional criteria
        filtered_results = await _apply_multimodal_filters(
            search_result["results"],
            search_request.multimodal_filters
        )

        return AdvancedSearchResponse(
            results=filtered_results,
            total_found=len(filtered_results),
            search_metadata=search_result["search_metadata"],
            multimodal_analysis=multimodal_analysis
        )

    except Exception as e:
        logger.error(f"Error in advanced search: {e}")
        raise HTTPException(status_code=500, detail=f"Advanced search failed: {str(e)}")


@router.get("/suggestions", response_model=SearchSuggestionsResponse)
async def get_search_suggestions(
    partial_query: str,
    category: Optional[DeviceCategory] = None,
    max_suggestions: int = 10,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get search suggestions based on partial query

    Provides autocomplete suggestions for queries, categories, and vendors
    """
    try:
        suggestions = []
        category_suggestions = []
        vendor_suggestions = []

        # Get query suggestions from existing search history
        from sqlalchemy import select
        history_query = select(SearchHistory.query_text).where(
            SearchHistory.query_text.ilike(f"{partial_query}%")
        ).distinct().limit(max_suggestions)

        history_result = await db.execute(history_query)
        history_queries = [row[0] for row in history_result.all()]
        suggestions.extend(history_queries)

        # Get product name suggestions
        product_query = select(Product.name).where(
            Product.name.ilike(f"{partial_query}%")
        ).distinct().limit(max_suggestions // 2)

        product_result = await db.execute(product_query)
        product_names = [row[0] for row in product_result.all()]
        suggestions.extend(product_names)

        # Get category suggestions if no category specified
        if not category:
            from models.database import Vendor
            category_query = select(Product.category).distinct().limit(10)
            category_result = await db.execute(category_query)
            categories = [row[0].value for row in category_result.all()]
            category_suggestions = [cat for cat in categories if partial_query.lower() in cat.lower()]

        # Get vendor suggestions
        vendor_query = select(Vendor.name).where(
            Vendor.name.ilike(f"{partial_query}%")
        ).distinct().limit(max_suggestions // 2)

        vendor_result = await db.execute(vendor_query)
        vendor_names = [row[0] for row in vendor_result.all()]
        vendor_suggestions.extend(vendor_names)

        # Remove duplicates and limit results
        suggestions = list(set(suggestions))[:max_suggestions]
        vendor_suggestions = list(set(vendor_suggestions))[:max_suggestions // 2]

        return SearchSuggestionsResponse(
            suggestions=suggestions,
            category_suggestions=category_suggestions,
            vendor_suggestions=vendor_suggestions
        )

    except Exception as e:
        logger.error(f"Error getting search suggestions: {e}")
        return SearchSuggestionsResponse(
            suggestions=[],
            category_suggestions=[],
            vendor_suggestions=[]
        )


@router.post("/validation/specifications", response_model=SpecificationValidationResponse)
async def validate_specifications(
    validation_request: SpecificationValidationRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Validate product specifications against standards and requirements

    Performs comprehensive validation including:
    - Format validation
    - Range checking
    - Compliance verification
    - Cross-reference validation
    """
    try:
        validation_errors = []
        validation_warnings = []
        normalized_specs = {}

        # Basic validation rules
        required_fields = {
            DeviceCategory.SERVER: ["cpu_cores", "memory_gb", "storage_gb", "network_ports"],
            DeviceCategory.PC: ["cpu_cores", "memory_gb", "storage_gb", "gpu"],
            DeviceCategory.LAPTOP: ["cpu_cores", "memory_gb", "storage_gb", "battery_wh", "display"],
            DeviceCategory.NETWORK_SWITCH: ["network_ports", "network_speed", "protocol"],
            DeviceCategory.STORAGE_NAS: ["storage_capacity_gb", "storage_interface", "storage_speed"]
        }

        # Check required fields
        category_required = required_fields.get(validation_request.category, [])
        for field in category_required:
            if field not in validation_request.specifications:
                validation_errors.append(f"Missing required field: {field}")

        # Validate and normalize specifications
        for key, value in validation_request.specifications.items():
            try:
                normalized_value, warnings = await _validate_specification_field(key, value, validation_request.category)
                normalized_specs[key] = normalized_value
                validation_warnings.extend(warnings)
            except ValueError as e:
                validation_errors.append(f"Invalid {key}: {str(e)}")

        # Cross-reference validation
        cross_validation_errors = await _cross_validate_specifications(
            normalized_specs, validation_request.category, db
        )
        validation_errors.extend(cross_validation_errors)

        # Calculate compliance score
        compliance_score = max(0.0, 1.0 - (len(validation_errors) * 0.2) - (len(validation_warnings) * 0.1))

        return SpecificationValidationResponse(
            is_valid=len(validation_errors) == 0,
            validation_errors=validation_errors,
            validation_warnings=validation_warnings,
            normalized_specifications=normalized_specs,
            compliance_score=compliance_score
        )

    except Exception as e:
        logger.error(f"Error validating specifications: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


async def _analyze_multimodal_results(
    query_text: str,
    results: List[Dict[str, Any]],
    multimodal_filters: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze multimodal search results"""
    try:
        # Prepare analysis prompt
        analysis_prompt = f"""Analyze multimodal search results for query: {query_text}

Results summary:
{json.dumps([{
    'product': r.get('name', ''),
    'vendor': r.get('vendor', ''),
    'score': r.get('final_score', 0),
    'category': r.get('category', '')
} for r in results[:5]], indent=2)}

Multimodal filters applied:
{json.dumps(multimodal_filters, indent=2)}

Provide analysis of:
1. Search effectiveness across modalities
2. Key matching criteria
3. Recommendations for refinement"""

        # Use Claude for analysis
        response = await openrouter_service._make_request("chat/completions", {
            "model": openrouter_service.model,
            "messages": [{"role": "user", "content": analysis_prompt}],
            "max_tokens": 500
        })

        content = response["choices"][0]["message"]["content"]

        return {
            "analysis": content,
            "modalities_used": list(multimodal_filters.keys()) if multimodal_filters else [],
            "results_count": len(results)
        }

    except Exception as e:
        logger.error(f"Error in multimodal analysis: {e}")
        return {"error": str(e)}


async def _apply_multimodal_filters(
    results: List[Dict[str, Any]],
    multimodal_filters: Optional[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Apply additional multimodal filters to results"""
    if not multimodal_filters:
        return results

    filtered_results = []

    for result in results:
        include_result = True

        # Apply price range filter
        if "price_range" in multimodal_filters:
            price_range = multimodal_filters["price_range"]
            # Note: Price filtering would require price data in database
            # This is a placeholder for future implementation

        # Apply performance filters
        if "performance_requirements" in multimodal_filters:
            perf_reqs = multimodal_filters["performance_requirements"]
            # Check against benchmarks and specifications
            if not await _check_performance_requirements(result, perf_reqs):
                include_result = False

        # Apply compatibility filters
        if "compatibility" in multimodal_filters:
            compat_reqs = multimodal_filters["compatibility"]
            if not await _check_compatibility(result, compat_reqs):
                include_result = False

        if include_result:
            filtered_results.append(result)

    return filtered_results


async def _validate_specification_field(key: str, value: Any, category: DeviceCategory) -> tuple:
    """Validate and normalize a specification field"""
    warnings = []

    # Type validation and normalization
    if key in ["cpu_cores", "memory_gb", "storage_gb", "battery_wh"]:
        if isinstance(value, str):
            # Try to extract number
            import re
            numbers = re.findall(r'\d+', value)
            if numbers:
                value = int(numbers[0])
            else:
                raise ValueError(f"Cannot extract numeric value from {value}")
        elif not isinstance(value, (int, float)):
            raise ValueError(f"Expected numeric value, got {type(value)}")

    # Range validation
    if key == "cpu_cores" and isinstance(value, (int, float)):
        if value < 1 or value > 128:
            warnings.append(f"CPU cores ({value}) outside typical range (1-128)")

    if key == "memory_gb" and isinstance(value, (int, float)):
        if value < 1 or value > 10000:
            warnings.append(f"Memory ({value}GB) outside typical range (1-10000GB)")

    return value, warnings


async def _cross_validate_specifications(
    specs: Dict[str, Any],
    category: DeviceCategory,
    db: AsyncSession
) -> List[str]:
    """Cross-validate specifications against database knowledge"""
    errors = []

    # Check CPU-memory compatibility
    if "cpu_cores" in specs and "memory_gb" in specs:
        cpu_cores = specs["cpu_cores"]
        memory_gb = specs["memory_gb"]

        # Basic rule: memory per core should be reasonable
        memory_per_core = memory_gb / cpu_cores
        if memory_per_core < 0.5 or memory_per_core > 100:
            errors.append(f"Memory per CPU core ({memory_per_core:.1f}GB) seems unreasonable")

    # Check against known benchmarks if available
    if "cpu" in specs:
        # Could validate against benchmark data
        pass

    return errors


async def _check_performance_requirements(result: Dict[str, Any], requirements: Dict[str, Any]) -> bool:
    """Check if product meets performance requirements"""
    # Placeholder for performance checking logic
    # Would compare against benchmarks and specifications
    return True


async def _check_compatibility(result: Dict[str, Any], requirements: Dict[str, Any]) -> bool:
    """Check product compatibility"""
    # Placeholder for compatibility checking logic
    return True


@router.get("/benchmarks/search", response_model=List[dict])
async def search_benchmarks(
    component_name: str,
    category: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """Search for benchmark data by component name"""
    from sqlalchemy import select, or_

    query = select(Benchmark).where(
        or_(
            Benchmark.component_name.ilike(f"%{component_name}%"),
            Benchmark.component_model.ilike(f"%{component_name}%")
        )
    )

    if category:
        query = query.where(Benchmark.category == category)

    result = await db.execute(query.limit(50))
    benchmarks = result.scalars().all()

    return [
        {
            "id": b.id,
            "name": b.name,
            "category": b.category,
            "component": b.component_name,
            "model": b.component_model,
            "vendor": b.vendor,
            "score": b.score,
            "score_type": b.score_type
        }
        for b in benchmarks
    ]
