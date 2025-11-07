import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  Box,
  TextField,
  InputAdornment,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  CircularProgress,
  Alert,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  List,
  ListItem,
  ListItemText,
  Divider,
  Paper,
} from '@mui/material';
import { Search, Info, Description, Assessment } from '@mui/icons-material';
import { apiClient } from '../services/api';
import Layout from '../components/Layout';

interface Product {
  id: number;
  vendor: string;
  name: string;
  model?: string;
  category: string;
  specifications?: any;
  description?: string;
  notes?: string;
  documents: Document[];
  created_at: string;
  updated_at: string;
}

interface Document {
  id: number;
  filename: string;
  document_type: string;
  is_processed: boolean;
}

const KnowledgeBrowser: React.FC = () => {
  const [products, setProducts] = useState<Product[]>([]);
  const [filteredProducts, setFilteredProducts] = useState<Product[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [selectedVendor, setSelectedVendor] = useState<string>('');
  const [selectedProduct, setSelectedProduct] = useState<Product | null>(null);
  const [productDetailsOpen, setProductDetailsOpen] = useState(false);

  // Get unique categories and vendors for filters
  const categories = [...new Set(products.map(p => p.category))];
  const vendors = [...new Set(products.map(p => p.vendor))];

  useEffect(() => {
    loadProducts();
  }, []);

  useEffect(() => {
    filterProducts();
  }, [products, searchTerm, selectedCategory, selectedVendor]);

  const loadProducts = async () => {
    try {
      setLoading(true);
      const response = await apiClient.getProducts();
      setProducts(response);
      setError(null);
    } catch (err) {
      setError('Failed to load products');
      console.error('Error loading products:', err);
    } finally {
      setLoading(false);
    }
  };

  const filterProducts = () => {
    let filtered = products;

    if (searchTerm) {
      filtered = filtered.filter(product =>
        product.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        product.vendor.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (product.model && product.model.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (product.description && product.description.toLowerCase().includes(searchTerm.toLowerCase()))
      );
    }

    if (selectedCategory) {
      filtered = filtered.filter(product => product.category === selectedCategory);
    }

    if (selectedVendor) {
      filtered = filtered.filter(product => product.vendor === selectedVendor);
    }

    setFilteredProducts(filtered);
  };

  const handleProductClick = async (product: Product) => {
    try {
      // Get full product details
      const details = await apiClient.getProduct(product.id);
      setSelectedProduct(details);
      setProductDetailsOpen(true);
    } catch (err) {
      console.error('Error loading product details:', err);
    }
  };

  const handleCloseProductDetails = () => {
    setProductDetailsOpen(false);
    setSelectedProduct(null);
  };

  const formatSpecifications = (specs: any) => {
    if (!specs) return 'No specifications available';

    if (typeof specs === 'object') {
      return Object.entries(specs)
        .map(([key, value]) => `${key}: ${value}`)
        .join(', ');
    }

    return specs;
  };

  if (loading) {
    return (
      <Layout>
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
          <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
            <CircularProgress />
          </Box>
        </Container>
      </Layout>
    );
  }

  return (
    <Layout>
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Knowledge Browser
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
          Browse and explore product knowledge base with specifications, benchmarks, and documentation
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {/* Filters */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                label="Search products"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <Search />
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Category</InputLabel>
                <Select
                  value={selectedCategory}
                  label="Category"
                  onChange={(e) => setSelectedCategory(e.target.value)}
                >
                  <MenuItem value="">All Categories</MenuItem>
                  {categories.map(category => (
                    <MenuItem key={category} value={category}>
                      {category}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={3}>
              <FormControl fullWidth>
                <InputLabel>Vendor</InputLabel>
                <Select
                  value={selectedVendor}
                  label="Vendor"
                  onChange={(e) => setSelectedVendor(e.target.value)}
                >
                  <MenuItem value="">All Vendors</MenuItem>
                  {vendors.map(vendor => (
                    <MenuItem key={vendor} value={vendor}>
                      {vendor}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={2}>
              <Button
                fullWidth
                variant="outlined"
                onClick={() => {
                  setSearchTerm('');
                  setSelectedCategory('');
                  setSelectedVendor('');
                }}
              >
                Clear Filters
              </Button>
            </Grid>
          </Grid>
        </Paper>

        {/* Results count */}
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Showing {filteredProducts.length} of {products.length} products
        </Typography>

        {/* Products Grid */}
        <Grid container spacing={3}>
          {filteredProducts.map((product) => (
            <Grid item xs={12} md={6} lg={4} key={product.id}>
              <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
                <CardContent sx={{ flexGrow: 1 }}>
                  <Typography variant="h6" component="h2" gutterBottom>
                    {product.name}
                  </Typography>
                  {product.model && (
                    <Typography variant="subtitle1" color="text.secondary" gutterBottom>
                      Model: {product.model}
                    </Typography>
                  )}
                  <Box sx={{ mb: 2 }}>
                    <Chip
                      label={product.vendor}
                      size="small"
                      color="primary"
                      sx={{ mr: 1, mb: 1 }}
                    />
                    <Chip
                      label={product.category}
                      size="small"
                      color="secondary"
                    />
                  </Box>
                  {product.description && (
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {product.description.length > 100
                        ? `${product.description.substring(0, 100)}...`
                        : product.description}
                    </Typography>
                  )}
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Specifications:</strong>
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {formatSpecifications(product.specifications).length > 80
                      ? `${formatSpecifications(product.specifications).substring(0, 80)}...`
                      : formatSpecifications(product.specifications)}
                  </Typography>
                  {product.documents && product.documents.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <Description sx={{ mr: 0.5, fontSize: 16, verticalAlign: 'middle' }} />
                        {product.documents.length} document{product.documents.length !== 1 ? 's' : ''}
                      </Typography>
                    </Box>
                  )}
                </CardContent>
                <CardActions>
                  <Button
                    size="small"
                    startIcon={<Info />}
                    onClick={() => handleProductClick(product)}
                  >
                    View Details
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>

        {filteredProducts.length === 0 && !loading && (
          <Box sx={{ textAlign: 'center', mt: 4 }}>
            <Typography variant="h6" color="text.secondary">
              No products found matching your criteria
            </Typography>
          </Box>
        )}

        {/* Product Details Dialog */}
        <Dialog
          open={productDetailsOpen}
          onClose={handleCloseProductDetails}
          maxWidth="md"
          fullWidth
        >
          <DialogTitle>
            {selectedProduct?.name} {selectedProduct?.model && `(${selectedProduct.model})`}
          </DialogTitle>
          <DialogContent>
            {selectedProduct && (
              <Box>
                <Grid container spacing={2} sx={{ mb: 3 }}>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2">Vendor</Typography>
                    <Typography>{selectedProduct.vendor}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2">Category</Typography>
                    <Typography>{selectedProduct.category}</Typography>
                  </Grid>
                </Grid>

                {selectedProduct.description && (
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>Description</Typography>
                    <Typography>{selectedProduct.description}</Typography>
                  </Box>
                )}

                {selectedProduct.specifications && (
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>Specifications</Typography>
                    <Typography component="pre" sx={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>
                      {typeof selectedProduct.specifications === 'object'
                        ? JSON.stringify(selectedProduct.specifications, null, 2)
                        : selectedProduct.specifications}
                    </Typography>
                  </Box>
                )}

                {selectedProduct.notes && (
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>Notes</Typography>
                    <Typography>{selectedProduct.notes}</Typography>
                  </Box>
                )}

                {selectedProduct.documents && selectedProduct.documents.length > 0 && (
                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>Documents</Typography>
                    <List dense>
                      {selectedProduct.documents.map((doc) => (
                        <ListItem key={doc.id}>
                          <ListItemText
                            primary={doc.filename}
                            secondary={`Type: ${doc.document_type} | Processed: ${doc.is_processed ? 'Yes' : 'No'}`}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )}

                <Divider sx={{ my: 2 }} />
                <Typography variant="caption" color="text.secondary">
                  Created: {new Date(selectedProduct.created_at).toLocaleDateString()} |
                  Updated: {new Date(selectedProduct.updated_at).toLocaleDateString()}
                </Typography>
              </Box>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={handleCloseProductDetails}>Close</Button>
          </DialogActions>
        </Dialog>
      </Container>
    </Layout>
  );
};

export default KnowledgeBrowser;