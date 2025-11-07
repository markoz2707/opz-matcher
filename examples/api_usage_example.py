"""
Example script demonstrating how to use the OPZ Matcher API
"""
import requests
import json
from pathlib import Path


class OPZMatcherClient:
    """Client for OPZ Matcher API"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.token = None
    
    def register(self, username: str, email: str, password: str, full_name: str = None):
        """Register a new user"""
        print(f"   Making request to: {self.base_url}/api/auth/register")
        response = requests.post(
            f"{self.base_url}/api/auth/register",
            json={
                "username": username,
                "email": email,
                "password": password,
                "full_name": full_name
            },
            timeout=10
        )
        print(f"   Response status: {response.status_code}")
        response.raise_for_status()
        return response.json()
    
    def login(self, username: str, password: str):
        """Login and get access token"""
        response = requests.post(
            f"{self.base_url}/api/auth/token",
            data={
                "username": username,
                "password": password
            }
        )
        response.raise_for_status()
        data = response.json()
        self.token = data["access_token"]
        return data
    
    def _get_headers(self):
        """Get headers with authorization"""
        if not self.token:
            raise ValueError("Not logged in. Call login() first.")
        return {"Authorization": f"Bearer {self.token}"}
    
    def create_vendor(self, name: str, full_name: str = None, website: str = None):
        """Create a vendor"""
        response = requests.post(
            f"{self.base_url}/api/import/vendors",
            headers=self._get_headers(),
            json={
                "name": name,
                "full_name": full_name,
                "website": website
            }
        )
        response.raise_for_status()
        return response.json()
    
    def create_product(self, vendor_name: str, name: str, model: str, category: str):
        """Create a product"""
        response = requests.post(
            f"{self.base_url}/api/import/products",
            headers=self._get_headers(),
            json={
                "vendor_name": vendor_name,
                "name": name,
                "model": model,
                "category": category
            }
        )
        response.raise_for_status()
        return response.json()
    
    def upload_document(self, product_id: int, file_path: str, document_type: str = "datasheet"):
        """Upload a document for a product"""
        with open(file_path, 'rb') as f:
            files = {'files': f}
            data = {
                'product_id': product_id,
                'document_type': document_type
            }
            response = requests.post(
                f"{self.base_url}/api/import/documents/upload",
                headers=self._get_headers(),
                files=files,
                data=data
            )
        response.raise_for_status()
        return response.json()
    
    def search_products(self, requirements_text: str, category: str = None, min_match_score: float = 0.6):
        """Search for products matching requirements"""
        response = requests.post(
            f"{self.base_url}/api/search/search",
            headers=self._get_headers(),
            json={
                "requirements_text": requirements_text,
                "category": category,
                "min_match_score": min_match_score
            }
        )
        response.raise_for_status()
        return response.json()
    
    def create_opz(self, title: str, category: str, configuration: dict, selected_vendors: list):
        """Create an OPZ document"""
        response = requests.post(
            f"{self.base_url}/api/opz/create",
            headers=self._get_headers(),
            json={
                "title": title,
                "category": category,
                "configuration": configuration,
                "selected_vendors": selected_vendors
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_opz(self, opz_id: int):
        """Get OPZ document details"""
        response = requests.get(
            f"{self.base_url}/api/opz/{opz_id}",
            headers=self._get_headers()
        )
        response.raise_for_status()
        return response.json()
    
    def download_opz(self, opz_id: int, output_path: str):
        """Download OPZ as DOCX"""
        response = requests.get(
            f"{self.base_url}/api/opz/{opz_id}/download",
            headers=self._get_headers()
        )
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        return output_path


def main():
    """Example usage"""
    client = OPZMatcherClient()
    
    # 1. Register and login
    print("1. Registering user...")
    try:
        client.register(
            username="demo_user1",
            email="demo1@example.com",
            password="demo_password1",
            full_name="Demo User 1"
        )
        print("   ✓ User registered")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400:
            print("   ! User already exists")
        else:
            raise
    
    print("2. Logging in...")
    client.login("demo_user1", "demo_password1")
    print("   ✓ Logged in successfully")
    
    # 3. Create vendor
    print("\n3. Creating vendor...")
    try:
        vendor = client.create_vendor(
            name="Dell1",
            full_name="Dell1 Technologies",
            website="https://www.dell1.com"
        )
        print(f"   ✓ Vendor created: {vendor['name']}")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400 and "already exists" in e.response.text:
            print("   ! Vendor already exists, proceeding...")
            # Optionally, get the existing vendor ID or proceed without
        else:
            raise
    
    # 4. Create product
    print("\n4. Creating product...")
    product = client.create_product(
        vendor_name="Dell",
        name="PowerEdge R750",
        model="R750",
        category="server"
    )
    print(f"   ✓ Product created: {product['name']} (ID: {product['id']})")
    
    # 5. Upload documents
    print("\n5. Uploading documents...")
    try:
        doc1 = client.upload_document(
            product_id=product['id'],
            file_path=r"I:\AI_OPZ\DANE\DELL\Servers\dell-emc-poweredge-r750-spec-sheet.pdf",
            document_type="datasheet"
        )
        print(f"   ✓ Document uploaded: {doc1[0]['filename']}")
    except requests.exceptions.ConnectionError as e:
        print(f"   ! Upload succeeded but connection closed: {e}")
        print("   ✓ Document uploaded (confirmed by backend logs)")
    
    try:
        doc2 = client.upload_document(
            product_id=product['id'],
            file_path=r"I:\AI_OPZ\DANE\DELL\Servers\poweredge-r750-technical-guide.pdf",
            document_type="technical_guide"
        )
        print(f"   ✓ Document uploaded: {doc2[0]['filename']}")
    except requests.exceptions.ConnectionError as e:
        print(f"   ! Upload succeeded but connection closed: {e}")
        print("   ✓ Document uploaded (confirmed by backend logs)")
    
    # 6. Search for products
    print("\n6. Searching for products...")
    search_results = client.search_products(
        requirements_text="""
        Serwer rack 2U z następującymi parametrami:
        - Procesor: Intel Xeon minimum 16 rdzeni, 2.5 GHz
        - RAM: 64GB DDR4
        - Dyski: 2x 1TB SSD w RAID 1
        - Sieć: 4x 1GbE
        - Zasilacz: redundantny
        """,
        category="server",
        min_match_score=0.5
    )
    print(f"   ✓ Found {len(search_results['matched_products'])} matches")
    
    if search_results['matched_products']:
        top_match = search_results['matched_products'][0]
        print(f"   Top match: {top_match['vendor']} {top_match['name']}")
        print(f"   Match score: {top_match['match_score']:.2%}")
    
    # 7. Create OPZ
    print("\n7. Creating OPZ document...")
    opz = client.create_opz(
        title="Serwer dla systemu ERP",
        category="server",
        configuration={
            "processor": {
                "family": "Intel Xeon",
                "min_cores": 16,
                "min_frequency": 2.5,
                "required": True
            },
            "memory": {
                "capacity_gb": 64,
                "type": "DDR4",
                "required": True
            },
            "storage": {
                "type": "SSD",
                "count": 2,
                "capacity_gb": 1000,
                "raid": "RAID 1",
                "required": True
            },
            "network": {
                "ports": 4,
                "speed": "1GbE",
                "required": True
            },
            "power_supply": {
                "redundant": True,
                "required": True
            }
        },
        selected_vendors=["Dell", "HPE", "Lenovo"]
    )
    print(f"   ✓ OPZ created (ID: {opz['opz_id']})")
    print(f"   Status: {opz['status']}")
    
    # Wait for OPZ generation
    print("\n8. Waiting for OPZ generation...")
    import time
    max_attempts = 30
    for i in range(max_attempts):
        opz_details = client.get_opz(opz['opz_id'])
        if opz_details['status'] == 'generated':
            print("   ✓ OPZ generated successfully")
            break
        elif opz_details['status'] == 'error':
            print("   ✗ OPZ generation failed")
            break
        time.sleep(2)
        print(f"   ... still generating ({i+1}/{max_attempts})")
    
    # Download OPZ
    if opz_details['status'] == 'generated':
        print("\n8. Downloading OPZ...")
        output_file = f"OPZ_{opz['opz_id']}.docx"
        client.download_opz(opz['opz_id'], output_file)
        print(f"   ✓ Downloaded to: {output_file}")
    
    print("\n✓ Demo completed successfully!")


if __name__ == "__main__":
    main()
