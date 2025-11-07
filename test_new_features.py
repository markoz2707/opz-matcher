#!/usr/bin/env python3
"""
Test new features for OPZ Matcher - validation, hybrid search, batch processing, benchmarks, new APIs
"""
import asyncio
import sys
import os
import requests
import json
from pathlib import Path
from typing import Dict, Any

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.services.database import get_db
from backend.models.database import Product, Vendor, ProductEmbedding, DocumentChunk
from sqlalchemy import select, func


class NewFeaturesTestSuite:
    """Test suite for new features"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.token = None
        self.session = requests.Session()

    def login(self, username: str = "demo_user1", password: str = "demo_password1"):
        """Login and get access token"""
        response = self.session.post(
            f"{self.base_url}/api/auth/token",
            data={"username": username, "password": password}
        )
        if response.status_code == 200:
            data = response.json()
            self.token = data["access_token"]
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
            print("✓ Logged in successfully")
            return True
        else:
            print(f"✗ Login failed: {response.status_code}")
            return False

    def test_specification_validation(self):
        """Test specification validation API"""
        print("\n1. Testing specification validation...")

        test_specs = {
            "vendor": "Dell",
            "product_name": "PowerEdge R750",
            "category": "server",
            "specifications": {
                "cpu_cores": 16,
                "ram_gb": 64,
                "storage_gb": 1000,
                "network_ports": 4
            }
        }

        try:
            response = self.session.post(
                f"{self.base_url}/api/search/validation/specifications",
                json={"specifications": test_specs, "category": "server"}
            )

            if response.status_code == 200:
                data = response.json()
                print("✓ Specification validation successful")
                print(f"  Valid: {data['is_valid']}")
                if data.get('issues'):
                    print(f"  Issues found: {len(data['issues'])}")
                return True
            else:
                print(f"✗ Specification validation failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False

        except Exception as e:
            print(f"✗ Specification validation error: {e}")
            return False

    def test_hybrid_search(self):
        """Test hybrid search functionality"""
        print("\n2. Testing hybrid search...")

        requirements_text = """
        Serwer rack 2U z następującymi parametrami:
        - Procesor: Intel Xeon minimum 16 rdzeni, 2.5 GHz
        - RAM: 64GB DDR4
        - Dyski: 2x 1TB SSD w RAID 1
        - Sieć: 4x 1GbE
        - Zasilacz: redundantny
        """

        try:
            response = self.session.post(
                f"{self.base_url}/api/search/search",
                json={
                    "requirements_text": requirements_text,
                    "category": "server",
                    "use_vector_search": True,
                    "use_hybrid_search": True,
                    "min_match_score": 0.3
                }
            )

            if response.status_code == 200:
                data = response.json()
                print("✓ Hybrid search successful")
                print(f"  Found {len(data['matched_products'])} matches")
                print(f"  Search ID: {data['search_id']}")

                if data['matched_products']:
                    top_match = data['matched_products'][0]
                    print(f"  Top match: {top_match['vendor']} {top_match['name']}")
                    print(f"  Match score: {top_match['match_score']:.2%}")

                return True
            else:
                print(f"✗ Hybrid search failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False

        except Exception as e:
            print(f"✗ Hybrid search error: {e}")
            return False

    def test_batch_processing(self):
        """Test batch processing of embeddings"""
        print("\n3. Testing batch processing...")

        try:
            # Check if batch processing endpoint exists
            response = self.session.get(f"{self.base_url}/api/import/documents/batch-status")

            if response.status_code == 200:
                data = response.json()
                print("✓ Batch processing status check successful")
                print(f"  Processing queue: {data.get('queue_size', 'N/A')}")
                print(f"  Active tasks: {data.get('active_tasks', 'N/A')}")
                return True
            else:
                print(f"✗ Batch processing status failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"✗ Batch processing error: {e}")
            return False

    def test_benchmarks_api(self):
        """Test benchmarks API"""
        print("\n4. Testing benchmarks API...")

        try:
            response = self.session.get(
                f"{self.base_url}/api/search/benchmarks/search",
                params={"component_name": "cpu", "limit": 10}
            )

            if response.status_code == 200:
                data = response.json()
                print("✓ Benchmarks search successful")
                print(f"  Found {len(data)} benchmark entries")
                return True
            else:
                print(f"✗ Benchmarks search failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False

        except Exception as e:
            print(f"✗ Benchmarks API error: {e}")
            return False

    def test_advanced_search(self):
        """Test advanced search features"""
        print("\n5. Testing advanced search...")

        try:
            response = self.session.post(
                f"{self.base_url}/api/search/advanced",
                json={
                    "query_text": "server rack 2U",
                    "filters": {
                        "category": "server",
                        "vendor": "Dell",
                        "price_range": {"min": 1000, "max": 10000}
                    },
                    "sort_by": "relevance",
                    "limit": 10
                }
            )

            if response.status_code == 200:
                data = response.json()
                print("✓ Advanced search successful")
                print(f"  Found {len(data['results'])} results")
                print(f"  Search time: {data.get('search_time', 'N/A')}ms")
                return True
            else:
                print(f"✗ Advanced search failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False

        except Exception as e:
            print(f"✗ Advanced search error: {e}")
            return False

    def test_search_suggestions(self):
        """Test search suggestions API"""
        print("\n6. Testing search suggestions...")

        try:
            response = self.session.get(
                f"{self.base_url}/api/search/suggestions",
                params={"partial_query": "server", "max_suggestions": 5}
            )

            if response.status_code == 200:
                data = response.json()
                print("✓ Search suggestions successful")
                print(f"  Found {len(data['suggestions'])} suggestions")
                return True
            else:
                print(f"✗ Search suggestions failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"✗ Search suggestions error: {e}")
            return False

    async def check_enhanced_database_state(self):
        """Check enhanced database state for new features"""
        print("\n7. Checking enhanced database state...")

        async for db in get_db():
            try:
                # Check product embeddings with content types
                result = await db.execute(
                    select(ProductEmbedding.content_type, func.count(ProductEmbedding.id))
                    .group_by(ProductEmbedding.content_type)
                )
                embedding_types = result.all()
                print(f"  Product embeddings by type: {dict(embedding_types)}")

                # Check document chunks
                result = await db.execute(select(func.count(DocumentChunk.id)))
                chunk_count = result.scalar()
                print(f"  Document chunks: {chunk_count}")

                # Check for benchmark data (if column exists)
                try:
                    result = await db.execute(
                        select(func.count(Product.id)).where(Product.benchmark_data.isnot(None))
                    )
                    benchmark_count = result.scalar()
                    print(f"  Products with benchmarks: {benchmark_count}")
                except AttributeError:
                    print("  Benchmark data column not available")
                    benchmark_count = 0

                if chunk_count > 0 or benchmark_count > 0:
                    print("✓ Enhanced database features available")
                    return True
                else:
                    print("! Enhanced features may be limited without data")
                    return False

            except Exception as e:
                print(f"✗ Enhanced database check failed: {e}")
                return False

    async def run_new_features_tests(self):
        """Run all new features tests"""
        print("=== New Features Test Suite ===")
        print("Testing enhanced OPZ Matcher functionality...")

        # Login
        if not self.login():
            return False

        # Run tests
        validation_success = self.test_specification_validation()
        hybrid_success = self.test_hybrid_search()
        batch_success = self.test_batch_processing()
        benchmarks_success = self.test_benchmarks_api()
        advanced_success = self.test_advanced_search()
        suggestions_success = self.test_search_suggestions()

        # Enhanced database check
        enhanced_db = await self.check_enhanced_database_state()

        print("\n=== New Features Test Results ===")
        print(f"Specification Validation: {'✓' if validation_success else '✗'}")
        print(f"Hybrid Search: {'✓' if hybrid_success else '✗'}")
        print(f"Batch Processing: {'✓' if batch_success else '✗'}")
        print(f"Benchmarks API: {'✓' if benchmarks_success else '✗'}")
        print(f"Advanced Search: {'✓' if advanced_success else '✗'}")
        print(f"Search Suggestions: {'✓' if suggestions_success else '✗'}")
        print(f"Enhanced Database: {'✓' if enhanced_db else '!'}")

        success_count = sum([
            validation_success, hybrid_success, batch_success,
            benchmarks_success, advanced_success, suggestions_success
        ])

        if success_count >= 4:  # At least 4 out of 6 tests pass
            print(f"\n✓ New features test passed ({success_count}/6)")
            return True
        else:
            print(f"\n✗ New features test failed ({success_count}/6)")
            return False


async def main():
    """Main test function"""
    print("Testing new OPZ Matcher features...")
    print("Make sure the API server is running on http://localhost:8000")

    test_suite = NewFeaturesTestSuite()
    success = await test_suite.run_new_features_tests()

    if success:
        print("\n🎉 New features tests completed successfully!")
    else:
        print("\n❌ New features tests failed!")


if __name__ == "__main__":
    asyncio.run(main())