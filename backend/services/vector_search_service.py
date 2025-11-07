"""
Vector search service for semantic product search using pgvector
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, and_, or_, func
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
import json

from models.database import Product, ProductEmbedding, DeviceCategory, Vendor, Benchmark
from services.claude_service import openrouter_service


class VectorSearchService:
    """Service for vector-based semantic search of products"""

    def __init__(self):
        self.embedding_dimension = 1536  # Claude embedding dimension

    async def search_products(
        self,
        query_text: str,
        db: AsyncSession,
        category: Optional[DeviceCategory] = None,
        vendor_filter: Optional[List[str]] = None,
        min_match_score: float = 0.6,
        limit: int = 50,
        include_embeddings: bool = True,
        use_hybrid_search: bool = True,
        benchmark_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform multi-stage vector similarity search with hybrid capabilities

        Args:
            query_text: Search query/requirements text
            db: Database session
            category: Optional category filter
            vendor_filter: Optional vendor name filters
            min_match_score: Minimum match score threshold
            limit: Maximum number of results
            include_embeddings: Whether to include embedding data in results
            use_hybrid_search: Whether to use hybrid search (vector + text)
            benchmark_filter: Optional benchmark type filter

        Returns:
            Dictionary with search results and metadata
        """
        try:
            # Stage 1: Generate query embedding
            logger.info("Vector search: Generating query embedding")
            query_embedding = await self._generate_query_embedding(query_text)

            # Stage 2: Perform search based on type
            if use_hybrid_search:
                logger.info("Vector search: Performing hybrid search")
                search_results = await self._hybrid_search(
                    db, query_text, query_embedding, category, vendor_filter, benchmark_filter, limit * 2
                )
            else:
                logger.info("Vector search: Performing vector similarity search")
                search_results = await self._vector_similarity_search(
                    db, query_embedding, category, vendor_filter, limit * 2
                )

            # Stage 3: Rerank results
            logger.info("Vector search: Reranking results")
            reranked_results = await self._rerank_results(
                query_text, search_results, min_match_score
            )

            # Stage 4: Format response
            logger.info("Vector search: Formatting response")
            formatted_results = await self._format_search_response(
                reranked_results[:limit], query_text, db
            )

            return {
                "results": formatted_results,
                "total_found": len(reranked_results),
                "search_metadata": {
                    "query_embedding_generated": True,
                    "search_type": "hybrid" if use_hybrid_search else "vector",
                    "reranking_applied": True,
                    "filters_applied": {
                        "category": category.value if category else None,
                        "vendors": vendor_filter,
                        "benchmark_filter": benchmark_filter,
                        "min_score": min_match_score
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            # Rollback any pending transaction to prevent session corruption
            try:
                await db.rollback()
            except Exception as rollback_error:
                logger.error(f"Error during rollback in vector search: {rollback_error}")

            # Fallback to basic search if vector search fails
            return await self._fallback_search(query_text, db, category, vendor_filter, limit)

    async def _generate_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding vector for the search query"""
        try:
            # Use Claude to generate embedding for the query
            embedding = await openrouter_service.generate_embedding(query_text)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * self.embedding_dimension

    async def _vector_similarity_search(
        self,
        db: AsyncSession,
        query_embedding: List[float],
        category: Optional[DeviceCategory] = None,
        vendor_filter: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using pgvector

        Returns list of dicts with product info and similarity scores
        """
        try:
            # Build the query with vector similarity
            base_query = select(
                ProductEmbedding,
                Product,
                Vendor,
                # Calculate cosine similarity
                (1 - func.cosine_distance(
                    ProductEmbedding.embedding,
                    text(f"'[{','.join(map(str, query_embedding))}]'::vector")
                )).label('similarity_score')
            ).select_from(
                ProductEmbedding
            ).join(
                Product, ProductEmbedding.product_id == Product.id
            ).join(
                Vendor, Product.vendor_id == Vendor.id
            )

            # Apply filters
            filters = []

            if category:
                filters.append(Product.category == category)

            if vendor_filter:
                filters.append(Vendor.name.in_(vendor_filter))

            if filters:
                base_query = base_query.where(and_(*filters))

            # Order by similarity score descending
            query = base_query.order_by(text('similarity_score DESC')).limit(limit)

            # Execute query
            result = await db.execute(query)
            rows = result.all()

            # Convert to list of dicts
            results = []
            for row in rows:
                embedding, product, vendor, similarity_score = row

                results.append({
                    "product_id": product.id,
                    "product": product,
                    "vendor": vendor,
                    "embedding": embedding,
                    "vector_similarity": float(similarity_score),
                    "content": embedding.content,
                    "content_type": embedding.content_type
                })

            logger.info(f"Vector search found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in vector similarity search: {e}")
            # Rollback any pending transaction to prevent session corruption
            try:
                await db.rollback()
            except Exception as rollback_error:
                logger.error(f"Error during rollback in vector similarity search: {rollback_error}")
            return []

    async def _hybrid_search(
        self,
        db: AsyncSession,
        query_text: str,
        query_embedding: List[float],
        category: Optional[DeviceCategory] = None,
        vendor_filter: Optional[List[str]] = None,
        benchmark_filter: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector similarity and text-based search

        Returns combined and scored results
        """
        try:
            # Get vector search results
            vector_results = await self._vector_similarity_search(
                db, query_embedding, category, vendor_filter, limit // 2
            )

            # Get text-based search results
            text_results = await self._text_similarity_search(
                db, query_text, category, vendor_filter, limit // 2
            )

            # Get benchmark-enhanced results if filter specified
            benchmark_results = []
            if benchmark_filter:
                benchmark_results = await self._benchmark_enhanced_search(
                    db, query_text, benchmark_filter, category, vendor_filter, limit // 4
                )

            # Combine and deduplicate results
            combined_results = self._combine_hybrid_results(
                vector_results, text_results, benchmark_results
            )

            logger.info(f"Hybrid search found {len(combined_results)} combined results")
            return combined_results[:limit]

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to vector search only
            return await self._vector_similarity_search(db, query_embedding, category, vendor_filter, limit)

    async def _text_similarity_search(
        self,
        db: AsyncSession,
        query_text: str,
        category: Optional[DeviceCategory] = None,
        vendor_filter: Optional[List[str]] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Perform text-based similarity search using PostgreSQL full-text search
        """
        try:
            # Prepare search terms
            search_terms = query_text.lower().split()
            search_vector = " | ".join(search_terms[:10])  # Limit terms for performance

            # Build query with text search
            base_query = select(
                ProductEmbedding,
                Product,
                Vendor,
                # Calculate text similarity using ts_rank
                func.ts_rank(
                    func.to_tsvector('english', ProductEmbedding.content),
                    func.to_tsquery('english', search_vector)
                ).label('text_similarity')
            ).select_from(
                ProductEmbedding
            ).join(
                Product, ProductEmbedding.product_id == Product.id
            ).join(
                Vendor, Product.vendor_id == Vendor.id
            )

            # Apply filters
            filters = []

            if category:
                filters.append(Product.category == category)

            if vendor_filter:
                filters.append(Vendor.name.in_(vendor_filter))

            if filters:
                base_query = base_query.where(and_(*filters))

            # Order by text similarity
            query = base_query.order_by(text('text_similarity DESC')).limit(limit)

            # Execute query
            result = await db.execute(query)
            rows = result.all()

            # Convert to list of dicts
            results = []
            for row in rows:
                embedding, product, vendor, text_similarity = row

                results.append({
                    "product_id": product.id,
                    "product": product,
                    "vendor": vendor,
                    "embedding": embedding,
                    "text_similarity": float(text_similarity),
                    "vector_similarity": 0.0,  # No vector similarity in text search
                    "content": embedding.content,
                    "content_type": embedding.content_type
                })

            logger.info(f"Text search found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in text similarity search: {e}")
            return []

    async def _benchmark_enhanced_search(
        self,
        db: AsyncSession,
        query_text: str,
        benchmark_type: str,
        category: Optional[DeviceCategory] = None,
        vendor_filter: Optional[List[str]] = None,
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Perform benchmark-enhanced search
        """
        try:
            # Get benchmark data for the specified type
            benchmark_query = select(Benchmark).where(
                Benchmark.benchmark_type == benchmark_type
            ).order_by(Benchmark.score.desc()).limit(limit * 2)

            benchmark_result = await db.execute(benchmark_query)
            benchmarks = benchmark_result.scalars().all()

            # Convert benchmarks to search results format
            results = []
            for benchmark in benchmarks:
                # Get associated product
                product_query = select(Product, Vendor).join(
                    Vendor, Product.vendor_id == Vendor.id
                ).where(Product.id == benchmark.product_id)

                if category:
                    product_query = product_query.where(Product.category == category)
                if vendor_filter:
                    product_query = product_query.where(Vendor.name.in_(vendor_filter))

                product_result = await db.execute(product_query)
                product_row = product_result.first()

                if product_row:
                    product, vendor = product_row

                    results.append({
                        "product_id": product.id,
                        "product": product,
                        "vendor": vendor,
                        "embedding": None,  # No embedding for benchmark search
                        "benchmark_score": benchmark.score,
                        "benchmark_type": benchmark.benchmark_type,
                        "vector_similarity": 0.0,
                        "text_similarity": 0.0,
                        "content": f"Benchmark: {benchmark.benchmark_type} - Score: {benchmark.score}",
                        "content_type": "benchmark"
                    })

            logger.info(f"Benchmark search found {len(results)} results for type {benchmark_type}")
            return results

        except Exception as e:
            logger.error(f"Error in benchmark enhanced search: {e}")
            return []

    def _combine_hybrid_results(
        self,
        vector_results: List[Dict[str, Any]],
        text_results: List[Dict[str, Any]],
        benchmark_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine results from different search methods with deduplication
        """
        combined = {}
        weights = {
            "vector": 0.5,
            "text": 0.3,
            "benchmark": 0.2
        }

        # Process vector results
        for result in vector_results:
            product_id = result["product_id"]
            if product_id not in combined:
                combined[product_id] = result.copy()
                combined[product_id]["search_sources"] = ["vector"]
                combined[product_id]["combined_score"] = result["vector_similarity"] * weights["vector"]
            else:
                combined[product_id]["search_sources"].append("vector")
                combined[product_id]["combined_score"] += result["vector_similarity"] * weights["vector"]

        # Process text results
        for result in text_results:
            product_id = result["product_id"]
            if product_id not in combined:
                combined[product_id] = result.copy()
                combined[product_id]["search_sources"] = ["text"]
                combined[product_id]["combined_score"] = result["text_similarity"] * weights["text"]
            else:
                if "text" not in combined[product_id]["search_sources"]:
                    combined[product_id]["search_sources"].append("text")
                combined[product_id]["combined_score"] += result["text_similarity"] * weights["text"]

        # Process benchmark results
        for result in benchmark_results:
            product_id = result["product_id"]
            benchmark_score = result.get("benchmark_score", 0.0)
            if product_id not in combined:
                combined[product_id] = result.copy()
                combined[product_id]["search_sources"] = ["benchmark"]
                combined[product_id]["combined_score"] = benchmark_score * weights["benchmark"]
            else:
                if "benchmark" not in combined[product_id]["search_sources"]:
                    combined[product_id]["search_sources"].append("benchmark")
                combined[product_id]["combined_score"] += benchmark_score * weights["benchmark"]

        # Convert back to list and sort by combined score
        results_list = list(combined.values())
        results_list.sort(key=lambda x: x.get("combined_score", 0.0), reverse=True)

        return results_list

    async def _rerank_results(
        self,
        query_text: str,
        search_results: List[Dict[str, Any]],
        min_score_threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Rerank vector search results using additional criteria

        This could include:
        - Content relevance scoring
        - Recency weighting
        - Popularity boosting
        - Semantic coherence
        """
        try:
            reranked = []

            for result in search_results:
                # Get base score based on search type
                if "combined_score" in result:
                    # Hybrid search result
                    base_score = result["combined_score"]
                elif "vector_similarity" in result:
                    # Vector search result
                    base_score = result["vector_similarity"]
                else:
                    # Fallback
                    base_score = 0.0

                # Apply content relevance scoring
                content_relevance = await self._calculate_content_relevance(
                    query_text, result["content"]
                )

                # Combine scores (weighted average)
                final_score = (base_score * 0.7) + (content_relevance * 0.3)

                # Only include results above threshold
                if final_score >= min_score_threshold:
                    result["final_score"] = final_score
                    result["content_relevance"] = content_relevance
                    reranked.append(result)

            # Sort by final score
            reranked.sort(key=lambda x: x["final_score"], reverse=True)

            logger.info(f"Reranking: {len(reranked)} results above threshold")
            return reranked

        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Return original results if reranking fails
            return search_results

    async def _calculate_content_relevance(self, query: str, content: str) -> float:
        """
        Calculate content relevance score between query and content

        Simple implementation - could be enhanced with more sophisticated NLP
        """
        try:
            # Simple text overlap scoring
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())

            # Jaccard similarity
            intersection = len(query_words.intersection(content_words))
            union = len(query_words.union(content_words))

            if union == 0:
                return 0.0

            return intersection / union

        except Exception:
            return 0.0

    async def _format_search_response(
        self,
        results: List[Dict[str, Any]],
        query_text: str,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """
        Format search results for API response

        Includes product details and search metadata
        """
        try:
            formatted_results = []

            for result in results:
                product = result["product"]
                vendor = result["vendor"]

                # Get benchmark data if available
                benchmark_data = await self._get_product_benchmarks(db, product.id)

                # Get additional product details if needed
                product_data = {
                    "product_id": product.id,
                    "vendor": vendor.name,
                    "name": product.name,
                    "model": product.model,
                    "category": product.category.value,
                    "specifications": product.specifications,
                    "description": product.description,
                    "vector_score": result.get("vector_similarity", 0.0),
                    "final_score": result.get("final_score", 0.0),
                    "matched_content": result.get("content", ""),
                    "content_type": result.get("content_type", ""),
                    "benchmarks": benchmark_data,
                    "search_sources": result.get("search_sources", []),
                    "search_metadata": {
                        "similarity_score": result.get("vector_similarity", 0.0),
                        "content_relevance": result.get("content_relevance", 0.0),
                        "text_similarity": result.get("text_similarity", 0.0),
                        "benchmark_score": result.get("benchmark_score", 0.0),
                        "combined_score": result.get("combined_score", 0.0),
                        "reranked": True
                    }
                }

                formatted_results.append(product_data)

            return formatted_results

        except Exception as e:
            logger.error(f"Error formatting search response: {e}")
            # Rollback any pending transaction to prevent session corruption
            try:
                await db.rollback()
            except Exception as rollback_error:
                logger.error(f"Error during rollback in format search response: {rollback_error}")
            return []

    async def _get_product_benchmarks(
        self,
        db: AsyncSession,
        product_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get benchmark data for a product

        Args:
            db: Database session
            product_id: Product ID

        Returns:
            List of benchmark data
        """
        try:
            from sqlalchemy import select

            query = select(Benchmark).where(Benchmark.product_id == product_id)
            result = await db.execute(query)
            benchmarks = result.scalars().all()

            benchmark_data = []
            for benchmark in benchmarks:
                benchmark_data.append({
                    "benchmark_type": benchmark.benchmark_type,
                    "score": benchmark.score,
                    "unit": benchmark.unit,
                    "description": benchmark.description,
                    "test_date": benchmark.test_date.isoformat() if benchmark.test_date else None
                })

            return benchmark_data

        except Exception as e:
            logger.error(f"Error getting benchmarks for product {product_id}: {e}")
            return []

    async def _fallback_search(
        self,
        query_text: str,
        db: AsyncSession,
        category: Optional[DeviceCategory] = None,
        vendor_filter: Optional[List[str]] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Fallback search method when vector search fails
        Uses traditional text-based search
        """
        try:
            logger.warning("Using fallback search method")

            # Simple text-based search
            query = select(Product).join(Vendor)

            filters = []

            if category:
                filters.append(Product.category == category)

            if vendor_filter:
                filters.append(Vendor.name.in_(vendor_filter))

            # Add text search on name, description, and specifications
            search_terms = query_text.lower().split()
            text_filters = []

            for term in search_terms[:5]:  # Limit terms for performance
                text_filters.append(Product.name.ilike(f"%{term}%"))
                text_filters.append(Product.description.ilike(f"%{term}%"))

            if text_filters:
                filters.append(or_(*text_filters))

            if filters:
                query = query.where(and_(*filters))

            result = await db.execute(query.limit(limit))
            products = result.scalars().all()

            # Format results
            formatted_results = []
            for product in products:
                formatted_results.append({
                    "product_id": product.id,
                    "vendor": product.vendor.name,
                    "name": product.name,
                    "model": product.model,
                    "category": product.category.value,
                    "specifications": product.specifications,
                    "description": product.description,
                    "vector_score": 0.0,  # No vector score in fallback
                    "final_score": 0.5,   # Default score
                    "matched_content": "",
                    "content_type": "",
                    "search_metadata": {
                        "fallback_search": True,
                        "similarity_score": 0.0,
                        "content_relevance": 0.0
                    }
                })

            return {
                "results": formatted_results,
                "total_found": len(formatted_results),
                "search_metadata": {
                    "fallback_mode": True,
                    "query_embedding_generated": False,
                    "vector_search_performed": False,
                    "reranking_applied": False
                }
            }

        except Exception as e:
            logger.error(f"Error in fallback search: {e}")
            # Rollback any pending transaction to prevent session corruption
            try:
                await db.rollback()
            except Exception as rollback_error:
                logger.error(f"Error during rollback in fallback search: {rollback_error}")

            return {
                "results": [],
                "total_found": 0,
                "search_metadata": {
                    "error": str(e),
                    "fallback_mode": True
                }
            }


    async def batch_generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 10
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"Batch generating embeddings for {len(texts)} texts with batch size {batch_size}")

            all_embeddings = []
            total_batches = (len(texts) + batch_size - 1) // batch_size

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_num = (i // batch_size) + 1

                logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch_texts)} texts")

                # Generate embeddings for this batch
                batch_embeddings = []
                for text in batch_texts:
                    try:
                        embedding = await openrouter_service.generate_embedding(text)
                        batch_embeddings.append(embedding)
                    except Exception as e:
                        logger.error(f"Failed to generate embedding for text in batch {batch_num}: {e}")
                        # Use zero vector as fallback
                        batch_embeddings.append([0.0] * self.embedding_dimension)

                all_embeddings.extend(batch_embeddings)

                # Small delay between batches to avoid rate limits
                if batch_num < total_batches:
                    await asyncio.sleep(0.1)

            logger.info(f"Successfully generated {len(all_embeddings)} embeddings in {total_batches} batches")
            return all_embeddings

        except Exception as e:
            logger.error(f"Error in batch embedding generation: {e}")
            # Return zero vectors as fallback
            return [[0.0] * self.embedding_dimension] * len(texts)


# Singleton instance
vector_search_service = VectorSearchService()