#!/usr/bin/env python3
"""
API test suite for OPZ Matcher - testing vector search functionality
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


class APITestSuite:
    """Test suite for API endpoints"""

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

    def register_user(self, username: str = "demo_user1", email: str = "demo1@example.com",
                     password: str = "demo_password1", full_name: str = "Demo User 1"):
        """Register a new user"""
        response = self.session.post(
            f"{self.base_url}/api/auth/register",
            json={
                "username": username,
                "email": email,
                "password": password,
                "full_name": full_name
            }
        )
        if response.status_code in [200, 201]:
            print("✓ User registered")
            return True
        elif response.status_code == 400:
            print("! User already exists")
            return True  # Still proceed
        else:
            print(f"✗ Registration failed: {response.status_code}")
            return False

    def test_health_check(self):
        """Test health check endpoint"""
        print("\n1. Testing health check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Health check passed: {data}")
                return True
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Health check error: {e}")
            return False

    def test_vector_search(self):
        """Test vector search functionality"""
        print("\n2. Testing vector search...")

        # Test requirements for server
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
                    "min_match_score": 0.5,
                    "use_vector_search": True
                }
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✓ Vector search successful")
                print(f"  Found {len(data['matched_products'])} matches")
                print(f"  Search ID: {data['search_id']}")

                if data['matched_products']:
                    top_match = data['matched_products'][0]
                    print(f"  Top match: {top_match['vendor']} {top_match['name']}")
                    print(f"  Match score: {top_match['match_score']:.2%}")

                return True
            else:
                print(f"✗ Vector search failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False

        except Exception as e:
            print(f"✗ Vector search error: {e}")
            return False

    def test_traditional_search(self):
        """Test traditional Claude-based search"""
        print("\n3. Testing traditional search...")

        requirements_text = """
        Serwer rack 2U z następującymi parametrami:
        - Procesor: Intel Xeon minimum 16 rdzeni, 2.5 GHz
        - RAM: 64GB DDR4
        """

        try:
            response = self.session.post(
                f"{self.base_url}/api/search/search",
                json={
                    "requirements_text": requirements_text,
                    "category": "server",
                    "min_match_score": 0.1,  # Lower threshold for testing
                    "use_vector_search": False
                }
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✓ Traditional search successful")
                print(f"  Found {len(data['matched_products'])} matches")
                print(f"  Search ID: {data['search_id']}")

                if data['matched_products']:
                    top_match = data['matched_products'][0]
                    print(f"  Top match: {top_match['vendor']} {top_match['name']}")
                    print(f"  Match score: {top_match['match_score']:.2%}")

                return True
            else:
                print(f"✗ Traditional search failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False

        except Exception as e:
            print(f"✗ Traditional search error: {e}")
            return False

    async def check_database_state(self):
        """Check database state for vector search readiness"""
        print("\n4. Checking database state...")

        async for db in get_db():
            try:
                # Check vendors
                result = await db.execute(select(func.count(Vendor.id)))
                vendor_count = result.scalar()
                print(f"  Vendors: {vendor_count}")

                # Check products
                result = await db.execute(select(func.count(Product.id)))
                product_count = result.scalar()
                print(f"  Products: {product_count}")

                # Check product embeddings
                result = await db.execute(select(func.count(ProductEmbedding.id)))
                embedding_count = result.scalar()
                print(f"  Product embeddings: {embedding_count}")

                # Check document chunks
                result = await db.execute(select(func.count(DocumentChunk.id)))
                chunk_count = result.scalar()
                print(f"  Document chunks: {chunk_count}")

                if embedding_count > 0 or chunk_count > 0:
                    print("✓ Database has embeddings for vector search")
                    return True
                else:
                    print("! No embeddings found - vector search may not work optimally")
                    return False

            except Exception as e:
                print(f"✗ Database check failed: {e}")
                return False

    async def run_tests_async(self):
        """Run all API tests asynchronously"""
        print("=== OPZ Matcher API Test Suite ===")
        print("Testing vector search functionality...")

        # Test health check
        if not self.test_health_check():
            print("✗ Health check failed - API may not be running")
            return False

        # Register/login
        print("\n5. Setting up authentication...")
        if not self.register_user():
            return False
        if not self.login():
            return False

        # Check database state
        db_ready = await self.check_database_state()

        # Test searches
        vector_success = self.test_vector_search()
        traditional_success = self.test_traditional_search()

        print("\n=== Test Results ===")
        print(f"Health Check: ✓")
        print(f"Authentication: ✓")
        print(f"Database Ready: {'✓' if db_ready else '!'}")
        print(f"Vector Search: {'✓' if vector_success else '✗'}")
        print(f"Traditional Search: {'✓' if traditional_success else '✗'}")

        if vector_success and traditional_success:
            print("\n✓ All tests passed! Vector search is working correctly.")
            return True
        else:
            print("\n✗ Some tests failed. Check the output above for details.")
            return False


async def main():
    """Main test function"""
    print("Note: Please start the API server manually with:")
    print("python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload")
    print("Then run this test script.\n")

    # Run tests
    test_suite = APITestSuite()
    success = await test_suite.run_tests_async()

    if success:
        print("\n🎉 Vector search API tests completed successfully!")
    else:
        print("\n❌ Vector search API tests failed!")


if __name__ == "__main__":
    asyncio.run(main())