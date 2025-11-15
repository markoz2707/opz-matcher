import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = import.meta.env.DEV ? '' : import.meta.env.VITE_API_URL;

class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add auth token to requests
    this.client.interceptors.request.use((config) => {
      const token = localStorage.getItem('token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    });

    // Handle auth errors
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          localStorage.removeItem('token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  // Auth
  async register(data: { username: string; email: string; password: string; full_name?: string }) {
    const response = await this.client.post('/api/auth/register', data);
    return response.data;
  }

  async login(username: string, password: string) {
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    
    const response = await this.client.post('/api/auth/token', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });
    
    if (response.data.access_token) {
      localStorage.setItem('token', response.data.access_token);
    }
    
    return response.data;
  }

  async getCurrentUser() {
    const response = await this.client.get('/api/auth/me');
    return response.data;
  }

  // Vendors
  async createVendor(data: { name: string; full_name?: string; website?: string }) {
    const response = await this.client.post('/api/import/vendors', data);
    return response.data;
  }

  async getVendors() {
    const response = await this.client.get('/api/import/vendors');
    return response.data;
  }

  async deleteVendor(vendorId: number) {
    const response = await this.client.delete(`/api/import/vendors/${vendorId}`);
    return response.data;
  }

  async updateVendor(vendorId: number, data: { name: string; full_name?: string; website?: string }) {
    const response = await this.client.put(`/api/import/vendors/${vendorId}`, data);
    return response.data;
  }

  // Products
  async createProduct(data: {
    vendor_name: string;
    name: string;
    model?: string;
    category: string;
    description?: string;
  }) {
    const response = await this.client.post('/api/import/products', data);
    return response.data;
  }

  async getProducts() {
    const response = await this.client.get('/api/import/products');
    return response.data;
  }

  async deleteProduct(product_id: number) {
    const response = await this.client.delete(`/api/import/products/${product_id}`);
    return response.data;
  }

  async updateProduct(productId: number, data: {
    vendor_name: string;
    name: string;
    model?: string;
    category: string;
    description?: string;
  }) {
    const response = await this.client.put(`/api/import/products/${productId}`, data);
    return response.data;
  }

  async getDocuments() {
    const response = await this.client.get('/api/import/documents');
    return response.data;
  }

  // Documents
  async uploadDocuments(productId: number, files: File[], documentType: string = 'datasheet') {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    formData.append('product_id', productId.toString());
    formData.append('document_type', documentType);

    const response = await this.client.post('/api/import/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }

  async getDocument(documentId: number) {
    const response = await this.client.get(`/api/import/documents/${documentId}`);
    return response.data;
  }

  async deleteDocument(documentId: number) {
    const response = await this.client.delete(`/api/import/documents/${documentId}`);
    return response.data;
  }

  // Benchmarks
  async importBenchmarks(data: {
    name: string;
    category: string;
    version?: string;
    data: any[];
  }) {
    const response = await this.client.post('/api/import/benchmarks/import', data);
    return response.data;
  }

  async importSpecCsv(file: File, version?: string) {
    const formData = new FormData();
    formData.append('file', file);
    if (version) {
      formData.append('version', version);
    }

    const response = await this.client.post('/api/import/benchmarks/import/spec-csv', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }

  async importPassmarkCsv(file: File, benchmarkType: 'PASSMARK_CPU' | 'PASSMARK_GPU') {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('benchmark_type', benchmarkType);

    const response = await this.client.post('/api/import/benchmarks/import/passmark-csv', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }

  // Product Search
  async searchProducts(data: {
    requirements_text: string;
    category?: string;
    vendor_filter?: string[];
    min_match_score?: number;
  }) {
    const response = await this.client.post('/api/search/search', data);
    return response.data;
  }

  async getProduct(productId: number) {
    const response = await this.client.get(`/api/search/products/${productId}`);
    return response.data;
  }

  async submitSearchFeedback(searchId: number, selectedProductId: number, feedbackScore: number) {
    const response = await this.client.post(`/api/search/${searchId}/feedback`, {
      selected_product_id: selectedProductId,
      feedback_score: feedbackScore,
    });
    return response.data;
  }

  async searchBenchmarks(componentName: string, category?: string) {
    const params = new URLSearchParams();
    params.append('component_name', componentName);
    if (category) params.append('category', category);
    
    const response = await this.client.get(`/api/search/benchmarks/search?${params}`);
    return response.data;
  }

  // OPZ Creation
  async createOPZ(data: {
    title: string;
    category: string;
    configuration: any;
    selected_vendors: string[];
    template_type?: string;
  }) {
    const response = await this.client.post('/api/opz/create', data);
    return response.data;
  }

  async getOPZ(opzId: number) {
    const response = await this.client.get(`/api/opz/${opzId}`);
    return response.data;
  }

  async refineOPZ(opzId: number, feedback: string) {
    const response = await this.client.post(`/api/opz/${opzId}/refine`, { feedback });
    return response.data;
  }

  async downloadOPZ(opzId: number) {
    const response = await this.client.get(`/api/opz/${opzId}/download`, {
      responseType: 'blob',
    });
    return response.data;
  }

  async listUserOPZs() {
    const response = await this.client.get('/api/opz/list');
    return response.data;
  }

  async deleteOPZ(opzId: number) {
    const response = await this.client.delete(`/api/opz/${opzId}`);
    return response.data;
  }

  // Statistics
  async getStatistics() {
    try {
      const [products, documents, vendors, opzs] = await Promise.all([
        this.getProducts(),
        this.getDocuments(),
        this.getVendors(),
        this.listUserOPZs(),
      ]);

      return {
        totalProducts: products.length || 0,
        totalDocuments: documents.length || 0,
        totalVendors: vendors.length || 0,
        totalOPZs: opzs.length || 0,
      };
    } catch (error) {
      return {
        totalProducts: 0,
        totalDocuments: 0,
        totalVendors: 0,
        totalOPZs: 0,
      };
    }
  }
}

export const apiClient = new ApiClient();
