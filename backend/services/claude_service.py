"""
OpenRouter API service for document processing, product matching, and OPZ generation
"""
import httpx
from typing import List, Dict, Any, Optional
from loguru import logger
import json

from config.settings import settings
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, ValidationError
import re


class SpecificationValidator:
    """Validator for product specifications extracted from documents"""

    # Required fields for different product categories
    REQUIRED_SPECS = {
        "server": {
            "processor": ["model", "cores", "frequency"],
            "memory": ["capacity", "type", "speed"],
            "storage": ["type", "capacity"],
            "network": ["ports", "speed"],
            "power": ["consumption", "supply"]
        },
        "pc": {
            "processor": ["model", "cores"],
            "memory": ["capacity", "type"],
            "storage": ["type", "capacity"],
            "graphics": ["model"],
            "power": ["supply"]
        },
        "network": {
            "network": ["ports", "speed", "protocols"],
            "power": ["consumption", "supply"],
            "physical": ["dimensions", "weight"]
        },
        "storage": {
            "storage": ["type", "capacity", "interface"],
            "network": ["ports", "speed"],
            "power": ["consumption", "supply"]
        }
    }

    COMMON_REQUIRED = ["vendor", "product_name", "model", "category"]

    def __init__(self):
        self.validation_rules = {
            "vendor": self._validate_vendor,
            "product_name": self._validate_product_name,
            "model": self._validate_model,
            "category": self._validate_category,
            "specifications": self._validate_specifications
        }

    def validate_specifications(self, specs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extracted product specifications

        Args:
            specs: Extracted specifications dictionary

        Returns:
            Validation result with errors and completeness score
        """
        errors = []
        warnings = []
        completeness_score = 0.0

        # Check common required fields
        for field in self.COMMON_REQUIRED:
            if not specs.get(field):
                errors.append(f"Missing required field: {field}")
            else:
                completeness_score += 0.2

        # Validate category-specific requirements
        category = specs.get("category", "").lower()
        if category in self.REQUIRED_SPECS:
            category_specs = self.REQUIRED_SPECS[category]
            spec_score = self._validate_category_specs(specs.get("specifications", {}), category_specs)
            completeness_score += spec_score * 0.8
        else:
            warnings.append(f"Unknown or missing category: {category}")

        # Run field-specific validations
        for field, validator in self.validation_rules.items():
            try:
                field_errors = validator(specs)
                if field_errors:
                    errors.extend(field_errors)
            except Exception as e:
                errors.append(f"Validation error for {field}: {str(e)}")

        # Calculate final completeness score
        completeness_score = min(completeness_score, 1.0)

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "completeness_score": completeness_score,
            "validation_timestamp": "2025-11-02T10:38:52.179Z"
        }

    def _validate_vendor(self, specs: Dict[str, Any]) -> List[str]:
        """Validate vendor field"""
        errors = []
        vendor = specs.get("vendor", "")

        if not vendor or not isinstance(vendor, str):
            errors.append("Vendor must be a non-empty string")
            return errors

        # Check for common vendor name patterns
        if len(vendor.strip()) < 2:
            errors.append("Vendor name too short")

        # Check for placeholder values
        if vendor.lower() in ["unknown", "n/a", "tbd"]:
            errors.append("Vendor appears to be a placeholder value")

        return errors

    def _validate_product_name(self, specs: Dict[str, Any]) -> List[str]:
        """Validate product name field"""
        errors = []
        name = specs.get("product_name", "")

        if not name or not isinstance(name, str):
            errors.append("Product name must be a non-empty string")
            return errors

        if len(name.strip()) < 3:
            errors.append("Product name too short")

        return errors

    def _validate_model(self, specs: Dict[str, Any]) -> List[str]:
        """Validate model field"""
        errors = []
        model = specs.get("model", "")

        if not model or not isinstance(model, str):
            errors.append("Model must be a non-empty string")
            return errors

        if len(model.strip()) < 2:
            errors.append("Model identifier too short")

        return errors

    def _validate_category(self, specs: Dict[str, Any]) -> List[str]:
        """Validate category field"""
        errors = []
        category = specs.get("category", "")

        if not category:
            errors.append("Category is required")
            return errors

        valid_categories = ["server", "pc", "network", "storage", "other"]
        if category.lower() not in valid_categories:
            errors.append(f"Invalid category. Must be one of: {', '.join(valid_categories)}")

        return errors

    def _validate_specifications(self, specs: Dict[str, Any]) -> List[str]:
        """Validate specifications structure"""
        errors = []
        specifications = specs.get("specifications", {})

        if not isinstance(specifications, dict):
            errors.append("Specifications must be a dictionary")
            return errors

        # Check for empty specifications
        if not specifications:
            errors.append("Specifications dictionary is empty")

        return errors

    def _validate_category_specs(self, specs: Dict[str, Any], required_specs: Dict[str, List[str]]) -> float:
        """Validate category-specific specifications and return completeness score"""
        total_required = 0
        found_required = 0

        for spec_category, fields in required_specs.items():
            total_required += len(fields)
            if spec_category in specs:
                spec_data = specs[spec_category]
                if isinstance(spec_data, dict):
                    for field in fields:
                        if field in spec_data and spec_data[field]:
                            found_required += 1

        return found_required / total_required if total_required > 0 else 0.0


# Global validator instance
specification_validator = SpecificationValidator()


class OpenRouterService:
    """Service for interacting with OpenRouter API or local LM Studio"""

    def __init__(self):
        # Try local LM Studio first, fallback to OpenRouter
        self.use_local_lm_studio = True  # Prefer local LM Studio
        self.local_lm_studio_url = "http://host.docker.internal:1234/v1"  # Use host.docker.internal for Docker containers

        if self.use_local_lm_studio:
            self.api_key = None
            self.model = "bielik-11b-v2.6-instruct"  # Use the chat model for completions
            self.embedding_model = settings.EMBEDDING_MODEL  # Use the configured embedding model
            self.base_url = self.local_lm_studio_url
            self.provider_name = "Local LM Studio"
        else:
            self.api_key = settings.OPENROUTER_API_KEY
            self.model = settings.OPENROUTER_MODEL
            self.base_url = "https://openrouter.ai/api/v1"
            self.provider_name = "OpenRouter API"

        # Always set max_tokens for both modes
        self.max_tokens = settings.MAX_TOKENS
    
    async def extract_product_specifications(
        self,
        document_text: str,
        document_type: str = "datasheet",
        vendor_name: Optional[str] = None,
        product_name: Optional[str] = None,
        validate_specs: bool = True
    ) -> Dict[str, Any]:
        """
        Extract structured product specifications from document text using chunked processing

        Args:
            document_text: Raw text from document
            document_type: Type of document (datasheet, manual, etc.)
            vendor_name: Known vendor name (if any)
            product_name: Known product name (if any)
            validate_specs: Whether to validate extracted specifications

        Returns:
            Dictionary with extracted specifications and validation results
        """
        # For large documents, use chunked processing
        if len(document_text) > 50000:  # ~12k tokens threshold
            return await self._extract_specifications_chunked(
                document_text, document_type, vendor_name, product_name, validate_specs
            )

        # For smaller documents, use single extraction
        return await self._extract_specifications_single(
            document_text, document_type, vendor_name, product_name, validate_specs
        )

    async def _extract_specifications_single(
        self,
        document_text: str,
        document_type: str,
        vendor_name: Optional[str],
        product_name: Optional[str],
        validate_specs: bool
    ) -> Dict[str, Any]:
        """Extract specifications from single text chunk"""
        system_prompt = """You are an expert IT product analyst specializing in extracting technical specifications from datasheets and manuals.

Your task is to extract structured product specifications from the provided document text. Focus on:
- Product identification (vendor, model, name)
- Technical specifications (CPU, RAM, storage, network, etc.)
- Performance metrics and benchmarks
- Supported standards and protocols
- Physical characteristics (dimensions, power consumption, etc.)
- Warranty and support information

Return a structured JSON response with all extracted information."""

        user_prompt = f"""Extract product specifications from this {document_type}.

Document text:
{document_text[:45000]}  # Limited to fit within context window

{"Vendor: " + vendor_name if vendor_name else ""}
{"Product: " + product_name if product_name else ""}

Return a JSON object with the following structure:
{{
    "vendor": "vendor name",
    "product_name": "product name",
    "model": "model number",
    "category": "device category (server/pc/network/storage/etc.)",
    "specifications": {{
        "processor": {{}},
        "memory": {{}},
        "storage": {{}},
        "network": {{}},
        "graphics": {{}},
        "power": {{}},
        "physical": {{}},
        "other": {{}}
    }},
    "features": [],
    "certifications": [],
    "warranty": {{}},
    "notes": ""
}}

Fill in as many fields as possible based on the document. Use null for unavailable information."""

        try:
            logger.info(f"Extracting specifications: Sending single chunk request to {self.provider_name}")
            async with httpx.AsyncClient() as client:
                headers = {
                    "Content-Type": "application/json"
                }
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                # LM Studio uses /chat/completions endpoint
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": self.max_tokens,
                        "temperature": 0.1
                    },
                    timeout=120  # Longer timeout for local processing
                )

                response.raise_for_status()
                data = response.json()
            logger.info(f"Extracting specifications: Received response from {self.provider_name}")

            # Extract JSON from response
            content = data["choices"][0]["message"]["content"]

            # Try to parse JSON
            try:
                logger.info(f"Extracting specifications: Parsing JSON from {self.provider_name} response")
                # Look for JSON in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    specs = json.loads(json_str)
                else:
                    specs = {"raw_response": content}
                logger.info("Extracting specifications: JSON parsing complete")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from {self.provider_name} response")
                specs = {"raw_response": content}

            # Validate specifications if requested
            if validate_specs and specs:
                validation_result = specification_validator.validate_specifications(specs)
                specs["validation"] = validation_result
                logger.info(f"Specifications validation: valid={validation_result['is_valid']}, score={validation_result['completeness_score']:.2f}")

            return specs

        except Exception as e:
            logger.error(f"Error extracting specifications: {e}")
            raise

    async def _extract_specifications_chunked(
        self,
        document_text: str,
        document_type: str,
        vendor_name: Optional[str],
        product_name: Optional[str],
        validate_specs: bool
    ) -> Dict[str, Any]:
        """Extract specifications using chunked processing for large documents"""
        logger.info(f"Using chunked processing for large document ({len(document_text)} chars)")

        # Split document into chunks of ~30k characters (should fit in 32k context)
        chunk_size = 30000
        overlap = 2000  # Small overlap to maintain context
        chunks = []

        start = 0
        while start < len(document_text):
            end = start + chunk_size

            # Try to break at paragraph boundary
            if end < len(document_text):
                paragraph_break = document_text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + chunk_size * 0.7:
                    end = paragraph_break + 2

            chunk = document_text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

            # Prevent infinite loop
            if start >= len(document_text):
                break

        logger.info(f"Split document into {len(chunks)} chunks for processing")

        # Process each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):  # Process all chunks
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")

            try:
                chunk_specs = await self._extract_specifications_single(
                    chunk, document_type, vendor_name, product_name, validate_specs=False
                )
                if chunk_specs and 'raw_response' not in chunk_specs:
                    chunk_results.append(chunk_specs)
            except Exception as e:
                logger.warning(f"Failed to process chunk {i+1}: {e}")
                continue

        # Merge results from all chunks
        if not chunk_results:
            return {"error": "Failed to extract specifications from any chunk"}

        # Use the first valid result as base and merge additional info
        merged_specs = chunk_results[0].copy()

        # Merge specifications from other chunks
        for additional_specs in chunk_results[1:]:
            if 'specifications' in additional_specs:
                for spec_category, spec_data in additional_specs['specifications'].items():
                    if spec_category not in merged_specs.get('specifications', {}):
                        merged_specs.setdefault('specifications', {})[spec_category] = spec_data
                    elif isinstance(spec_data, dict):
                        # Merge dictionaries
                        base_spec = merged_specs['specifications'].setdefault(spec_category, {})
                        for key, value in spec_data.items():
                            if key not in base_spec or not base_spec[key]:
                                base_spec[key] = value

        # Validate final merged specifications
        if validate_specs:
            validation_result = specification_validator.validate_specifications(merged_specs)
            merged_specs["validation"] = validation_result
            logger.info(f"Chunked specifications validation: valid={validation_result['is_valid']}, score={validation_result['completeness_score']:.2f}")

        logger.info(f"Completed chunked extraction with {len(chunk_results)} successful chunks")
        return merged_specs

        try:
            logger.info(f"Extracting specifications: Sending request to {self.provider_name}")
            async with httpx.AsyncClient() as client:
                headers = {
                    "Content-Type": "application/json"
                }
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                # LM Studio uses /chat/completions endpoint
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": self.max_tokens,
                        "temperature": 0.1
                    },
                    timeout=120  # Longer timeout for local processing
                )

                response.raise_for_status()
                data = response.json()
            logger.info(f"Extracting specifications: Received response from {self.provider_name}")
            
            # Extract JSON from response
            content = data["choices"][0]["message"]["content"]
            
            # Try to parse JSON
            try:
                logger.info(f"Extracting specifications: Parsing JSON from {self.provider_name} response")
                # Look for JSON in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    specs = json.loads(json_str)
                else:
                    specs = {"raw_response": content}
                logger.info("Extracting specifications: JSON parsing complete")
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from {self.provider_name} response")
                specs = {"raw_response": content}
            
            # Validate specifications if requested
            if validate_specs and specs:
                validation_result = specification_validator.validate_specifications(specs)
                specs["validation"] = validation_result
                logger.info(f"Specifications validation: valid={validation_result['is_valid']}, score={validation_result['completeness_score']:.2f}")

            return specs

        except Exception as e:
            logger.error(f"Error extracting specifications: {e}")
            raise
    
    async def match_products(
        self,
        requirements: str,
        products_context: List[Dict[str, Any]],
        benchmark_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Match products to OPZ requirements
        
        Args:
            requirements: OPZ requirements text (in Polish or English)
            products_context: List of products with their specifications
            benchmark_data: Benchmark data for performance comparisons
            
        Returns:
            Dictionary with matched products and analysis
        """
        system_prompt = """You are an expert IT procurement analyst specializing in Polish public tenders (zamówienia publiczne).

Your task is to analyze OPZ (Opis Przedmiotu Zamówienia) requirements and match them with available products. You understand that:

1. Perfect matches are rare - some flexibility is allowed
2. Polish law allows asking clarifying questions to adjust requirements
3. Benchmark scores can be used to verify performance requirements
4. Some requirements may be negotiable with the contracting authority

Provide detailed analysis including:
- Exact matches
- Close matches with minor deviations
- Requirements that could be adjusted
- Benchmark comparisons
- Suggestions for questions to ask the customer"""

        # Prepare products context
        products_text = "\n\n".join([
            f"Product {i+1}:\n" + 
            f"Vendor: {p.get('vendor', 'Unknown')}\n" +
            f"Model: {p.get('model', 'Unknown')}\n" +
            f"Specifications: {json.dumps(p.get('specifications', {}), indent=2, ensure_ascii=False)}"
            for i, p in enumerate(products_context[:10])  # Limit to top 10
        ])
        
        benchmark_text = ""
        if benchmark_data:
            benchmark_text = f"\n\nBenchmark data:\n{json.dumps(benchmark_data, indent=2, ensure_ascii=False)}"
        
        user_prompt = f"""Analyze these OPZ requirements and match them with available products.

OPZ Requirements:
{requirements}

Available Products:
{products_text}
{benchmark_text}

Provide a JSON response with:
{{
    "matched_products": [
        {{
            "product_id": "index in the provided list",
            "match_score": "0-100",
            "exact_matches": [],
            "close_matches": [],
            "deviations": [],
            "adjustable_requirements": [],
            "benchmark_analysis": {{}},
            "recommendation": ""
        }}
    ],
    "questions_for_customer": [],
    "general_analysis": ""
}}"""

        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Content-Type": "application/json"
                }
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": self.max_tokens,
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
            logger.error(f"Error matching products: {e}")
            raise
    
    async def generate_opz(
        self,
        category: str,
        configuration: Dict[str, Any],
        vendors: List[str],
        template_type: str = "standard"
    ) -> str:
        """
        Generate OPZ document text
        
        Args:
            category: Device category (server, pc, network, etc.)
            configuration: Product configuration and requirements
            vendors: List of vendors that should meet requirements
            template_type: Type of OPZ template to use
            
        Returns:
            Generated OPZ text
        """
        system_prompt = """You are an expert in creating OPZ (Opis Przedmiotu Zamówienia) documents for Polish public tenders in IT procurement.

Generate professional, legally compliant OPZ documents that:
- Follow Polish public procurement law requirements
- Are vendor-neutral but specify technical requirements
- Include clear, measurable criteria
- Allow multiple vendors to participate
- Specify verification methods
- Include warranty and support requirements

The document should be in professional Polish."""

        config_text = json.dumps(configuration, indent=2, ensure_ascii=False)
        vendors_text = ", ".join(vendors)
        
        user_prompt = f"""Create an OPZ document for {category} procurement.

Configuration requirements:
{config_text}

The requirements should be achievable by these vendors: {vendors_text}
However, the OPZ must remain vendor-neutral and allow other vendors to participate if they meet the technical requirements.

Generate a complete OPZ document in Polish including:
1. Opis ogólny (General description)
2. Wymagania techniczne (Technical requirements)
3. Wymagania funkcjonalne (Functional requirements)
4. Standardy i certyfikaty (Standards and certificates)
5. Gwarancja i wsparcie (Warranty and support)
6. Metody weryfikacji (Verification methods)

Format the document professionally with clear sections and numbering."""

        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Content-Type": "application/json"
                }
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": self.max_tokens,
                        "temperature": 0.1
                    },
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()

            opz_text = data["choices"][0]["message"]["content"]
            return opz_text
            
        except Exception as e:
            logger.error(f"Error generating OPZ: {e}")
            raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate vector embedding for text using local LM Studio or OpenRouter API

        Args:
            text: Text to embed

        Returns:
            List of float values representing the embedding vector
        """
        try:
            logger.info(f"Generating embedding: Sending request to {self.provider_name}")

            if self.use_local_lm_studio:
                # Use local LM Studio API for embeddings
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/embeddings",
                        headers={
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": self.embedding_model,  # Use embedding model for embeddings
                            "input": text[:8000]  # Limit text length
                        },
                        timeout=60  # Increased timeout for local processing
                    )
                    response.raise_for_status()
                    data = response.json()

                # Extract embedding from LM Studio response
                embedding = data["data"][0]["embedding"]
                logger.info(f"Embedding generation: Successfully generated embedding with {len(embedding)} dimensions from {self.provider_name}")
                return embedding

            else:
                # Use OpenRouter's embedding capability through chat completions API
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": self.model,
                            "messages": [
                                {"role": "system", "content": "You are an AI assistant that generates semantic embeddings. Respond only with a JSON array of 1536 float values representing the semantic embedding of the input text."},
                                {"role": "user", "content": f"Generate a semantic embedding vector for this text: {text[:8000]}"}  # Limit text length
                            ],
                            "max_tokens": 2000,  # Enough for JSON array
                            "temperature": 0.0
                        },
                        timeout=60
                    )
                    response.raise_for_status()
                    data = response.json()

                # Parse the response as JSON array
                content = data["choices"][0]["message"]["content"].strip()

                # Log the raw response for debugging
                logger.debug(f"Raw embedding response from {self.provider_name}: {content[:200]}...")

                # Try to extract JSON array from response
                import json
                try:
                    # Look for array in response
                    start_idx = content.find('[')
                    end_idx = content.rfind(']') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        embedding_json = content[start_idx:end_idx]
                        embedding = json.loads(embedding_json)
                        if isinstance(embedding, list) and len(embedding) == 1536:
                            logger.info("Embedding generation: Successfully parsed embedding vector")
                            return embedding
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse embedding JSON from {self.provider_name} response: {e}")

                # Try alternative parsing - maybe the model returned just the array
                try:
                    if content.startswith('[') and content.endswith(']'):
                        embedding = json.loads(content)
                        if isinstance(embedding, list) and len(embedding) >= 100:  # Accept reasonable length
                            logger.info(f"Embedding generation: Successfully parsed embedding vector (alternative parsing, length: {len(embedding)})")
                            return embedding
                    # Try to find and parse the first valid JSON array in the response
                    elif '[' in content and ']' in content:
                        # Extract the first complete array
                        start = content.find('[')
                        bracket_count = 0
                        end = start
                        for i, char in enumerate(content[start:], start):
                            if char == '[':
                                bracket_count += 1
                            elif char == ']':
                                bracket_count -= 1
                                if bracket_count == 0:
                                    end = i + 1
                                    break
                        if end > start:
                            array_content = content[start:end]
                            embedding = json.loads(array_content)
                            if isinstance(embedding, list) and len(embedding) >= 100:
                                logger.info(f"Embedding generation: Successfully parsed embedding vector (bracket parsing, length: {len(embedding)})")
                                return embedding
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed alternative embedding parsing from {self.provider_name}: {e}")

                # Fallback: generate a mock embedding for now (should be replaced with proper embedding)
                logger.warning(f"Failed to parse embedding from {self.provider_name} response, using fallback")
                import numpy as np
                # Create a normalized random vector as fallback
                np.random.seed(hash(text) % 2**32)
                embedding = np.random.normal(0, 1, 1536).tolist()
                norm = np.linalg.norm(embedding)
                embedding = (np.array(embedding) / norm).tolist()

                return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Try fallback to OpenRouter if local LM Studio fails
            if self.use_local_lm_studio:
                logger.warning("Local LM Studio failed, trying OpenRouter as fallback")
                self.use_local_lm_studio = False
                self.api_key = settings.OPENROUTER_API_KEY
                self.model = settings.OPENROUTER_MODEL
                self.max_tokens = settings.MAX_TOKENS
                self.base_url = "https://openrouter.ai/api/v1"
                self.provider_name = "OpenRouter API (fallback)"
                return await self.generate_embedding(text)
            # Return zero vector as last resort
            return [0.0] * 1536

    async def refine_opz(
        self,
        current_opz: str,
        feedback: str
    ) -> str:
        """
        Refine OPZ based on user feedback

        Args:
            current_opz: Current OPZ text
            feedback: User feedback and change requests

        Returns:
            Refined OPZ text
        """
        system_prompt = """You are an expert OPZ editor. Refine the document based on user feedback while maintaining legal compliance and professional quality."""

        user_prompt = f"""Current OPZ:
{current_opz}

User feedback:
{feedback}

Update the OPZ document based on this feedback. Maintain the professional structure and ensure compliance with Polish public procurement law."""

        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "Content-Type": "application/json"
                }
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": self.max_tokens,
                        "temperature": 0.1
                    },
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()

            refined_opz = data["choices"][0]["message"]["content"]
            return refined_opz

        except Exception as e:
            logger.error(f"Error refining OPZ: {e}")
            raise


# Singleton instance
openrouter_service = OpenRouterService()
