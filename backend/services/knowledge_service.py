"""
Knowledge service for storing and retrieving learned patterns, user feedback,
and improving search accuracy and OPZ quality over time.
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_, or_, func, text, desc
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import asyncio
from loguru import logger

from models.database import (
    KnowledgeEntity, DocumentChunk, SearchHistory, Product,
    Benchmark, ProductEmbedding, DeviceCategory, Vendor
)
from services.claude_service import openrouter_service


class KnowledgeService:
    """Service for managing knowledge acquisition and integration"""

    def __init__(self):
        self.embedding_dimension = 768  # nomic-embed-text-v1.5 dimension

    async def store_user_feedback(
        self,
        search_id: int,
        selected_product_id: int,
        feedback_score: int,
        additional_feedback: Optional[str] = None,
        db: AsyncSession = None
    ) -> bool:
        """
        Store user feedback on search results to improve future searches

        Args:
            search_id: Search history ID
            selected_product_id: Product selected by user
            feedback_score: Rating 1-5
            additional_feedback: Optional text feedback
            db: Database session

        Returns:
            Success status
        """
        try:
            # Update search history with feedback
            stmt = (
                update(SearchHistory)
                .where(SearchHistory.id == search_id)
                .values(
                    selected_product_id=selected_product_id,
                    feedback_score=feedback_score,
                    updated_at=datetime.utcnow()
                )
            )
            await db.execute(stmt)

            # Extract patterns from successful searches for learning
            if feedback_score >= 4:  # High-quality feedback
                await self._learn_from_successful_search(search_id, selected_product_id, db)

            # Store additional feedback if provided
            if additional_feedback:
                await self._store_feedback_text(additional_feedback, search_id, db)

            await db.commit()
            logger.info(f"Stored feedback for search {search_id}: score {feedback_score}")
            return True

        except Exception as e:
            logger.error(f"Error storing user feedback: {e}")
            await db.rollback()
            return False

    async def _learn_from_successful_search(
        self,
        search_id: int,
        selected_product_id: int,
        db: AsyncSession
    ):
        """Extract learning patterns from successful searches"""
        try:
            # Get search details
            search_result = await db.execute(
                select(SearchHistory).where(SearchHistory.id == search_id)
            )
            search = search_result.scalar_one()

            # Get selected product details
            product_result = await db.execute(
                select(Product, Vendor).join(Vendor).where(Product.id == selected_product_id)
            )
            product_row = product_result.first()
            if not product_row:
                return

            product, vendor = product_row

            # Create knowledge entity for successful match pattern
            success_pattern = {
                "search_query": search.query_text,
                "requirements": search.requirements,
                "category": search.category.value if search.category else None,
                "selected_product": {
                    "id": product.id,
                    "name": product.name,
                    "vendor": vendor.name,
                    "category": product.category.value,
                    "specifications": product.specifications
                },
                "feedback_score": 5,  # Assuming high score for learning
                "pattern_type": "successful_match"
            }

            # Store as knowledge entity
            await self._store_knowledge_entity(
                entity_type="search_pattern",
                entity_name=f"successful_search_{search_id}",
                entity_value=json.dumps(success_pattern),
                context=f"Successful product match from search {search_id}",
                confidence_score=0.9,
                db=db
            )

        except Exception as e:
            logger.error(f"Error learning from successful search: {e}")

    async def _store_feedback_text(
        self,
        feedback_text: str,
        search_id: int,
        db: AsyncSession
    ):
        """Store textual feedback for analysis"""
        try:
            # Create knowledge entity for feedback
            await self._store_knowledge_entity(
                entity_type="user_feedback",
                entity_name=f"feedback_search_{search_id}",
                entity_value=feedback_text,
                context=f"User feedback for search {search_id}",
                confidence_score=0.8,
                db=db
            )
        except Exception as e:
            logger.error(f"Error storing feedback text: {e}")

    async def _store_knowledge_entity(
        self,
        entity_type: str,
        entity_name: str,
        entity_value: str,
        context: str,
        confidence_score: float,
        db: AsyncSession,
        document_id: Optional[int] = None
    ):
        """Store a knowledge entity with embedding"""
        try:
            # Generate embedding for the entity
            embedding_text = f"{entity_name} {entity_value} {context}"
            embedding = await openrouter_service.generate_embedding(embedding_text)

            # Create knowledge entity
            entity = KnowledgeEntity(
                document_id=document_id,
                entity_type=entity_type,
                entity_name=entity_name,
                entity_value=entity_value,
                context=context,
                description=f"Auto-generated {entity_type} entity",
                embedding=embedding,
                confidence_score=confidence_score,
                entity_metadata={
                    "source": "learning_system",
                    "created_via": "user_feedback"
                }
            )

            db.add(entity)
            await db.commit()

        except Exception as e:
            logger.error(f"Error storing knowledge entity: {e}")
            await db.rollback()

    async def get_learning_insights(
        self,
        category: Optional[DeviceCategory] = None,
        limit: int = 10,
        db: AsyncSession = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve learning insights from accumulated knowledge

        Args:
            category: Optional category filter
            limit: Maximum number of insights
            db: Database session

        Returns:
            List of learning insights
        """
        try:
            query = select(KnowledgeEntity).where(
                KnowledgeEntity.entity_type == "search_pattern"
            )

            if category:
                # Filter by category in entity metadata
                query = query.where(
                    KnowledgeEntity.entity_metadata.contains({"category": category.value})
                )

            query = query.order_by(desc(KnowledgeEntity.confidence_score)).limit(limit)

            result = await db.execute(query)
            entities = result.scalars().all()

            insights = []
            for entity in entities:
                try:
                    pattern_data = json.loads(entity.entity_value)
                    insights.append({
                        "pattern_id": entity.id,
                        "search_query": pattern_data.get("search_query"),
                        "selected_product": pattern_data.get("selected_product"),
                        "category": pattern_data.get("category"),
                        "confidence": entity.confidence_score,
                        "created_at": entity.created_at
                    })
                except json.JSONDecodeError:
                    continue

            return insights

        except Exception as e:
            logger.error(f"Error retrieving learning insights: {e}")
            return []

    async def normalize_specifications(
        self,
        specifications: Dict[str, Any],
        category: DeviceCategory,
        db: AsyncSession = None
    ) -> Dict[str, Any]:
        """
        Normalize product specifications using learned patterns

        Args:
            specifications: Raw specifications
            category: Product category
            db: Database session

        Returns:
            Normalized specifications
        """
        try:
            normalized = specifications.copy()

            # Get normalization patterns from knowledge base
            patterns = await self._get_normalization_patterns(category, db)

            # Apply normalization rules
            for key, value in specifications.items():
                normalized_key = await self._normalize_spec_key(key, patterns)
                normalized_value = await self._normalize_spec_value(key, value, patterns)

                if normalized_key != key:
                    del normalized[key]
                normalized[normalized_key] = normalized_value

            # Cross-reference with known products
            cross_referenced = await self._cross_reference_specifications(
                normalized, category, db
            )

            return cross_referenced

        except Exception as e:
            logger.error(f"Error normalizing specifications: {e}")
            return specifications  # Return original if normalization fails

    async def _get_normalization_patterns(
        self,
        category: DeviceCategory,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Get normalization patterns from knowledge base"""
        try:
            query = select(KnowledgeEntity).where(
                and_(
                    KnowledgeEntity.entity_type == "normalization_pattern",
                    KnowledgeEntity.entity_metadata.contains({"category": category.value})
                )
            )

            result = await db.execute(query)
            entities = result.scalars().all()

            patterns = {}
            for entity in entities:
                try:
                    pattern_data = json.loads(entity.entity_value)
                    patterns.update(pattern_data)
                except json.JSONDecodeError:
                    continue

            return patterns

        except Exception as e:
            logger.error(f"Error getting normalization patterns: {e}")
            return {}

    async def _normalize_spec_key(self, key: str, patterns: Dict[str, Any]) -> str:
        """Normalize specification key using patterns"""
        # Simple key normalization - can be enhanced with ML
        key_lower = key.lower().strip()

        # Common key mappings
        key_mappings = {
            "cpu": "cpu_cores",
            "cores": "cpu_cores",
            "ram": "memory_gb",
            "memory": "memory_gb",
            "storage": "storage_gb",
            "disk": "storage_gb",
            "hdd": "storage_gb",
            "ssd": "storage_gb",
            "gpu": "gpu_model",
            "graphics": "gpu_model",
            "battery": "battery_wh",
            "power": "battery_wh",
            "display": "display_size",
            "screen": "display_size"
        }

        return key_mappings.get(key_lower, key_lower)

    async def _normalize_spec_value(self, key: str, value: Any, patterns: Dict[str, Any]) -> Any:
        """Normalize specification value"""
        try:
            if isinstance(value, str):
                value = value.strip()

                # Numeric extraction for common specs
                if key in ["cpu_cores", "memory_gb", "storage_gb", "battery_wh"]:
                    import re
                    numbers = re.findall(r'\d+(?:\.\d+)?', value)
                    if numbers:
                        return float(numbers[0]) if '.' in numbers[0] else int(numbers[0])

                # Unit normalization
                if key == "memory_gb":
                    value_lower = value.lower()
                    if "tb" in value_lower:
                        # Convert TB to GB
                        numbers = re.findall(r'\d+(?:\.\d+)?', value)
                        if numbers:
                            return float(numbers[0]) * 1000

            return value

        except Exception:
            return value

    async def _cross_reference_specifications(
        self,
        specifications: Dict[str, Any],
        category: DeviceCategory,
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Cross-reference specifications with known products"""
        try:
            # Find similar products
            similar_products = await self._find_similar_products_by_specs(
                specifications, category, db, limit=5
            )

            # Extract common spec ranges
            spec_ranges = {}
            for product in similar_products:
                for key, value in product.get("specifications", {}).items():
                    if key not in spec_ranges:
                        spec_ranges[key] = []
                    if isinstance(value, (int, float)):
                        spec_ranges[key].append(value)

            # Add range information to specifications
            enhanced_specs = specifications.copy()
            for key, values in spec_ranges.items():
                if len(values) >= 3:  # Need at least 3 data points
                    enhanced_specs[f"{key}_range"] = {
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "count": len(values)
                    }

            return enhanced_specs

        except Exception as e:
            logger.error(f"Error in cross-referencing: {e}")
            return specifications

    async def _find_similar_products_by_specs(
        self,
        specifications: Dict[str, Any],
        category: DeviceCategory,
        db: AsyncSession,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Find products with similar specifications"""
        try:
            # Build query for products in same category
            query = select(Product, Vendor).join(Vendor).where(
                Product.category == category
            )

            # Simple similarity scoring based on spec matches
            result = await db.execute(query.limit(limit * 2))
            products = result.all()

            scored_products = []
            for product_row in products:
                product, vendor = product_row
                similarity_score = self._calculate_spec_similarity(
                    specifications, product.specifications
                )

                scored_products.append({
                    "id": product.id,
                    "name": product.name,
                    "vendor": vendor.name,
                    "specifications": product.specifications,
                    "similarity_score": similarity_score
                })

            # Sort by similarity and return top results
            scored_products.sort(key=lambda x: x["similarity_score"], reverse=True)
            return scored_products[:limit]

        except Exception as e:
            logger.error(f"Error finding similar products: {e}")
            return []

    def _calculate_spec_similarity(
        self,
        spec1: Dict[str, Any],
        spec2: Dict[str, Any]
    ) -> float:
        """Calculate similarity score between two specification sets"""
        if not spec1 or not spec2:
            return 0.0

        matches = 0
        total_keys = len(set(spec1.keys()) | set(spec2.keys()))

        for key in spec1.keys():
            if key in spec2:
                val1, val2 = spec1[key], spec2[key]
                if val1 == val2:
                    matches += 1
                elif isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Numeric similarity
                    if max(val1, val2) > 0:
                        ratio = min(val1, val2) / max(val1, val2)
                        matches += ratio

        return matches / total_keys if total_keys > 0 else 0.0

    async def improve_search_accuracy(
        self,
        search_query: str,
        category: Optional[DeviceCategory],
        current_results: List[Dict[str, Any]],
        db: AsyncSession = None
    ) -> List[Dict[str, Any]]:
        """
        Improve search results using learned patterns

        Args:
            search_query: Original search query
            category: Product category
            current_results: Current search results
            db: Database session

        Returns:
            Improved search results
        """
        try:
            # Get learning insights for this category
            insights = await self.get_learning_insights(category, limit=20, db=db)

            # Find similar successful searches
            similar_patterns = await self._find_similar_search_patterns(
                search_query, insights, db
            )

            # Boost results based on successful patterns
            boosted_results = await self._boost_results_by_patterns(
                current_results, similar_patterns
            )

            # Re-rank results
            reranked_results = await self._rerank_with_learning(
                boosted_results, similar_patterns
            )

            return reranked_results

        except Exception as e:
            logger.error(f"Error improving search accuracy: {e}")
            return current_results

    async def _find_similar_search_patterns(
        self,
        search_query: str,
        insights: List[Dict[str, Any]],
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Find similar successful search patterns"""
        try:
            similar_patterns = []

            # Simple text similarity for now - could be enhanced with embeddings
            query_words = set(search_query.lower().split())

            for insight in insights:
                pattern_query = insight.get("search_query", "").lower()
                pattern_words = set(pattern_query.split())

                # Jaccard similarity
                intersection = len(query_words.intersection(pattern_words))
                union = len(query_words.union(pattern_words))

                if union > 0:
                    similarity = intersection / union
                    if similarity > 0.3:  # Similarity threshold
                        insight["query_similarity"] = similarity
                        similar_patterns.append(insight)

            return similar_patterns

        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return []

    async def _boost_results_by_patterns(
        self,
        results: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Boost results based on successful patterns"""
        try:
            boosted_results = []

            for result in results:
                boost_score = 0.0
                boost_factors = []

                # Check if this product was successfully selected in similar searches
                for pattern in patterns:
                    selected_product = pattern.get("selected_product", {})
                    if selected_product.get("id") == result.get("product_id"):
                        boost_score += pattern.get("query_similarity", 0) * 0.2
                        boost_factors.append("previous_success")

                    # Check vendor preference
                    if selected_product.get("vendor") == result.get("vendor"):
                        boost_score += 0.1
                        boost_factors.append("vendor_preference")

                    # Check category match
                    if selected_product.get("category") == result.get("category"):
                        boost_score += 0.05
                        boost_factors.append("category_match")

                result["learning_boost"] = boost_score
                result["boost_factors"] = boost_factors
                result["adjusted_score"] = result.get("final_score", 0) + boost_score

                boosted_results.append(result)

            return boosted_results

        except Exception as e:
            logger.error(f"Error boosting results: {e}")
            return results

    async def _rerank_with_learning(
        self,
        results: List[Dict[str, Any]],
        patterns: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Re-rank results incorporating learning insights"""
        try:
            # Sort by adjusted score (original + learning boost)
            reranked = sorted(
                results,
                key=lambda x: x.get("adjusted_score", x.get("final_score", 0)),
                reverse=True
            )

            return reranked

        except Exception as e:
            logger.error(f"Error reranking with learning: {e}")
            return results

    async def improve_opz_quality(
        self,
        requirements: Dict[str, Any],
        category: DeviceCategory,
        current_opz: str,
        db: AsyncSession = None
    ) -> str:
        """
        Improve OPZ quality using learned patterns

        Args:
            requirements: OPZ requirements
            category: Product category
            current_opz: Current OPZ text
            db: Database session

        Returns:
            Improved OPZ text
        """
        try:
            # Get successful OPZ patterns for this category
            opz_patterns = await self._get_opz_patterns(category, db)

            # Analyze current OPZ against patterns
            improvements = await self._analyze_opz_improvements(
                current_opz, opz_patterns, requirements
            )

            # Apply improvements
            if improvements:
                improved_opz = await self._apply_opz_improvements(
                    current_opz, improvements
                )
                return improved_opz

            return current_opz

        except Exception as e:
            logger.error(f"Error improving OPZ quality: {e}")
            return current_opz

    async def _get_opz_patterns(
        self,
        category: DeviceCategory,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Get successful OPZ patterns from knowledge base"""
        try:
            query = select(KnowledgeEntity).where(
                and_(
                    KnowledgeEntity.entity_type == "opz_pattern",
                    KnowledgeEntity.entity_metadata.contains({"category": category.value})
                )
            ).order_by(desc(KnowledgeEntity.confidence_score)).limit(10)

            result = await db.execute(query)
            entities = result.scalars().all()

            patterns = []
            for entity in entities:
                try:
                    pattern_data = json.loads(entity.entity_value)
                    patterns.append(pattern_data)
                except json.JSONDecodeError:
                    continue

            return patterns

        except Exception as e:
            logger.error(f"Error getting OPZ patterns: {e}")
            return []

    async def _analyze_opz_improvements(
        self,
        opz_text: str,
        patterns: List[Dict[str, Any]],
        requirements: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze potential improvements for OPZ"""
        try:
            improvements = []

            # Check for common successful elements
            for pattern in patterns:
                successful_elements = pattern.get("successful_elements", [])
                for element in successful_elements:
                    if element not in opz_text:
                        improvements.append({
                            "type": "add_element",
                            "element": element,
                            "reason": pattern.get("reason", "Based on successful OPZ patterns")
                        })

            # Use Claude to analyze improvements
            if patterns:
                analysis_prompt = f"""Analyze this OPZ document and suggest improvements based on successful patterns:

OPZ Requirements:
{json.dumps(requirements, indent=2)}

Current OPZ Text:
{opz_text[:2000]}...

Successful Patterns:
{json.dumps(patterns[:3], indent=2)}

Provide specific improvement suggestions in JSON format:
{{
    "improvements": [
        {{
            "type": "add_section|modify_section|remove_section",
            "target": "section_name",
            "content": "suggested_content",
            "reason": "explanation"
        }}
    ]
}}"""

                response = await openrouter_service._make_request("chat/completions", {
                    "model": openrouter_service.model,
                    "messages": [{"role": "user", "content": analysis_prompt}],
                    "max_tokens": 1000
                })

                content = response["choices"][0]["message"]["content"]

                try:
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        analysis_result = json.loads(json_str)
                        improvements.extend(analysis_result.get("improvements", []))
                except json.JSONDecodeError:
                    pass

            return improvements

        except Exception as e:
            logger.error(f"Error analyzing OPZ improvements: {e}")
            return []

    async def _apply_opz_improvements(
        self,
        opz_text: str,
        improvements: List[Dict[str, Any]]
    ) -> str:
        """Apply improvements to OPZ text"""
        try:
            improved_text = opz_text

            for improvement in improvements:
                imp_type = improvement.get("type")
                target = improvement.get("target", "")
                content = improvement.get("content", "")

                if imp_type == "add_section" and content:
                    # Add new section at the end
                    improved_text += f"\n\n{content}"

            return improved_text

        except Exception as e:
            logger.error(f"Error applying OPZ improvements: {e}")
            return opz_text


# Singleton instance
knowledge_service = KnowledgeService()