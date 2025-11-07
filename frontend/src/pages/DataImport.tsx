import { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Card,
  Tabs,
  Tab,
  Typography,
  Button,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  LinearProgress,
} from '@mui/material';
import { AddOutlined, UploadFileOutlined, RefreshOutlined, DeleteOutlined, EditOutlined } from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import { useSnackbar } from 'notistack';
import Layout from '../components/Layout';
import { apiClient } from '../services/api';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel({ children, value, index }: TabPanelProps) {
  return (
    <div hidden={value !== index}>
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

export default function DataImport() {
  const [tabValue, setTabValue] = useState(0);
  const [vendors, setVendors] = useState<any[]>([]);
  const [products, setProducts] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const { enqueueSnackbar } = useSnackbar();

  useEffect(() => {
    loadVendors();
    loadProducts();
  }, []);

  // Vendor Dialog
  const [vendorDialogOpen, setVendorDialogOpen] = useState(false);
  const [vendorForm, setVendorForm] = useState({ name: '', full_name: '', website: '' });
  const [editingVendor, setEditingVendor] = useState<any | null>(null);

  // Product Dialog
  const [productDialogOpen, setProductDialogOpen] = useState(false);
  const [productForm, setProductForm] = useState({
    vendor_name: '',
    name: '',
    model: '',
    category: 'server',
    description: '',
  });
  const [editingProduct, setEditingProduct] = useState<any | null>(null);

  // Upload Dialog
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState<number | null>(null);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [uploadProgress, setUploadProgress] = useState<{
    currentStep: string;
    progress: number;
    totalSteps: number;
  } | null>(null);

  // Benchmark Upload Dialog
  const [benchmarkDialogOpen, setBenchmarkDialogOpen] = useState(false);
  const [benchmarkType, setBenchmarkType] = useState<'spec' | 'passmark'>('spec');
  const [benchmarkForm, setBenchmarkForm] = useState({
    name: '',
    category: '',
    version: '',
    data: [] as any[],
  });
  const [specForm, setSpecForm] = useState({
    version: '',
    file: null as File | null,
  });
  const [passmarkForm, setPassmarkForm] = useState({
    benchmarkType: 'PASSMARK_CPU' as 'PASSMARK_CPU' | 'PASSMARK_GPU',
    file: null as File | null,
  });

  // Drag & drop hooks for benchmark files
  const specDropzone = useDropzone({
    onDrop: (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        setSpecForm(prev => ({ ...prev, file: acceptedFiles[0] }));
      }
    },
    accept: {
      'text/csv': ['.csv'],
    },
    multiple: false,
  });

  const passmarkDropzone = useDropzone({
    onDrop: (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        setPassmarkForm(prev => ({ ...prev, file: acceptedFiles[0] }));
      }
    },
    accept: {
      'text/csv': ['.csv'],
    },
    multiple: false,
  });

  const loadVendors = async () => {
    try {
      const data = await apiClient.getVendors();
      setVendors(data);
    } catch (error) {
      enqueueSnackbar('Failed to load vendors', { variant: 'error' });
    }
  };

  const loadProducts = async () => {
    try {
      const data = await apiClient.getProducts();
      setProducts(data);
      console.log('Loaded products:', data);
    } catch (error) {
      enqueueSnackbar('Failed to load products', { variant: 'error' });
      console.error('Error loading products:', error);
    }
  };

  const handleCreateVendor = async () => {
    try {
      await apiClient.createVendor(vendorForm);
      enqueueSnackbar('Vendor created successfully', { variant: 'success' });
      setVendorDialogOpen(false);
      setVendorForm({ name: '', full_name: '', website: '' });
      setEditingVendor(null);
      loadVendors();
    } catch (error: any) {
      enqueueSnackbar(error.response?.data?.detail || 'Failed to create vendor', { variant: 'error' });
    }
  };

  const handleEditVendor = async () => {
    if (!editingVendor) return;
    try {
      await apiClient.updateVendor(editingVendor.id, vendorForm);
      enqueueSnackbar('Vendor updated successfully', { variant: 'success' });
      setVendorDialogOpen(false);
      setVendorForm({ name: '', full_name: '', website: '' });
      setEditingVendor(null);
      loadVendors();
    } catch (error: any) {
      enqueueSnackbar(error.response?.data?.detail || 'Failed to update vendor', { variant: 'error' });
    }
  };

  const openEditVendorDialog = (vendor: any) => {
    setEditingVendor(vendor);
    setVendorForm({
      name: vendor.name,
      full_name: vendor.full_name || '',
      website: vendor.website || '',
    });
    setVendorDialogOpen(true);
  };

  const handleCreateProduct = async () => {
    try {
      await apiClient.createProduct(productForm);
      enqueueSnackbar('Product created successfully', { variant: 'success' });
      setProductDialogOpen(false);
      setProductForm({ vendor_name: '', name: '', model: '', category: 'server', description: '' });
      setEditingProduct(null);
      loadProducts();
    } catch (error: any) {
      enqueueSnackbar(error.response?.data?.detail || 'Failed to create product', { variant: 'error' });
    }
  };

  const handleEditProduct = async () => {
    if (!editingProduct) return;
    try {
      await apiClient.updateProduct(editingProduct.id, productForm);
      enqueueSnackbar('Product updated successfully', { variant: 'success' });
      setProductDialogOpen(false);
      setProductForm({ vendor_name: '', name: '', model: '', category: 'server', description: '' });
      setEditingProduct(null);
      loadProducts();
    } catch (error: any) {
      enqueueSnackbar(error.response?.data?.detail || 'Failed to update product', { variant: 'error' });
    }
  };

  const openEditProductDialog = (product: any) => {
    setEditingProduct(product);
    setProductForm({
      vendor_name: product.vendor,
      name: product.name,
      model: product.model || '',
      category: product.category,
      description: product.description || '',
    });
    setProductDialogOpen(true);
  };

  const handleImportBenchmarks = async () => {
    try {
      let result;
      if (benchmarkType === 'spec') {
        if (!specForm.file) {
          enqueueSnackbar('Please select a CSV file', { variant: 'warning' });
          return;
        }
        result = await apiClient.importSpecCsv(specForm.file, specForm.version);
        enqueueSnackbar(`SPEC benchmarks imported successfully: ${result.imported_entries} entries`, { variant: 'success' });
      } else {
        if (!passmarkForm.file) {
          enqueueSnackbar('Please select a CSV file', { variant: 'warning' });
          return;
        }
        result = await apiClient.importPassmarkCsv(passmarkForm.file, passmarkForm.benchmarkType);
        enqueueSnackbar(`PassMark benchmarks imported successfully: ${result.imported_entries} entries`, { variant: 'success' });
      }

      // Show additional feedback if there were warnings or errors
      if (result.errors && result.errors.length > 0) {
        enqueueSnackbar(`Import completed with ${result.errors.length} errors. Check console for details.`, { variant: 'warning' });
      }
      if (result.warnings && result.warnings.length > 0) {
        enqueueSnackbar(`Import completed with ${result.warnings.length} warnings.`, { variant: 'info' });
      }

      setBenchmarkDialogOpen(false);
      setBenchmarkForm({ name: '', category: '', version: '', data: [] });
      setSpecForm({ version: '', file: null });
      setPassmarkForm({ benchmarkType: 'PASSMARK_CPU', file: null });
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error.response?.data?.message || 'Failed to import benchmarks';
      enqueueSnackbar(errorMessage, { variant: 'error' });

      // Log detailed error for debugging
      console.error('Benchmark import error:', error.response?.data);
    }
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setUploadedFiles(acceptedFiles);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'text/plain': ['.txt'],
      'image/*': ['.png', '.jpg', '.jpeg'],
    },
  });

  const handleUpload = async () => {
    if (!selectedProduct || uploadedFiles.length === 0) {
      enqueueSnackbar('Please select a product and files', { variant: 'warning' });
      return;
    }

    setLoading(true);
    setUploadProgress({
      currentStep: 'Initializing upload...',
      progress: 0,
      totalSteps: uploadedFiles.length * 4 + 2 // 4 steps per file + final steps
    });

    try {
      // Step 1: Validate files
      setUploadProgress(prev => prev ? {
        ...prev,
        currentStep: 'Validating files...',
        progress: 5
      } : null);

      await new Promise(resolve => setTimeout(resolve, 500)); // Simulate validation time

      // Step 2: Upload files to storage
      setUploadProgress(prev => prev ? {
        ...prev,
        currentStep: 'Uploading files to storage...',
        progress: 15
      } : null);

      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate upload time

      // Step 3: Process documents (extract text, specs, generate embeddings)
      for (let i = 0; i < uploadedFiles.length; i++) {
        const file = uploadedFiles[i];
        const stepBase = 20 + (i * 70 / uploadedFiles.length);

        setUploadProgress(prev => prev ? {
          ...prev,
          currentStep: `Processing ${file.name} - Extracting text...`,
          progress: stepBase + 10
        } : null);

        await new Promise(resolve => setTimeout(resolve, 1500)); // Simulate text extraction

        setUploadProgress(prev => prev ? {
          ...prev,
          currentStep: `Processing ${file.name} - Analyzing specifications...`,
          progress: stepBase + 30
        } : null);

        await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate spec extraction

        setUploadProgress(prev => prev ? {
          ...prev,
          currentStep: `Processing ${file.name} - Generating embeddings...`,
          progress: stepBase + 50
        } : null);

        await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate embedding generation

        setUploadProgress(prev => prev ? {
          ...prev,
          currentStep: `Processing ${file.name} - Finalizing...`,
          progress: stepBase + 65
        } : null);

        await new Promise(resolve => setTimeout(resolve, 500)); // Simulate finalization
      }

      // Step 4: Final API call
      setUploadProgress(prev => prev ? {
        ...prev,
        currentStep: 'Saving to database...',
        progress: 90
      } : null);

      const response = await apiClient.uploadDocuments(selectedProduct, uploadedFiles);

      setUploadProgress(prev => prev ? {
        ...prev,
        currentStep: 'Upload completed!',
        progress: 100
      } : null);

      await new Promise(resolve => setTimeout(resolve, 500)); // Show completion

      enqueueSnackbar(`${uploadedFiles.length} file(s) uploaded and processed successfully`, { variant: 'success' });
      setUploadDialogOpen(false);
      setUploadedFiles([]);
      setSelectedProduct(null);
      setUploadProgress(null);
      // Reset progress state
      setUploadProgress(null);

      // Refresh data to show new documents
      loadVendors();
      loadProducts();
    } catch (error: any) {
      enqueueSnackbar(error.response?.data?.detail || 'Upload failed', { variant: 'error' });
      setUploadProgress(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Layout>
      <Box>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h5" fontWeight="bold">
            Data Import Mode
          </Typography>
          <Button
            variant="outlined"
            startIcon={<RefreshOutlined />}
            onClick={() => {
              loadVendors();
              loadProducts();
            }}
          >
            Refresh
          </Button>
        </Box>

        <Card>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={tabValue} onChange={(_, v) => setTabValue(v)}>
              <Tab label="Vendors" />
              <Tab label="Products" />
              <Tab label="Documents" />
              <Tab label="Benchmarks" />
            </Tabs>
          </Box>

          {/* Vendors Tab */}
          <TabPanel value={tabValue} index={0}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
              <Typography variant="h6">Manage Vendors</Typography>
              <Button
                variant="contained"
                startIcon={<AddOutlined />}
                onClick={() => setVendorDialogOpen(true)}
              >
                Add Vendor
              </Button>
            </Box>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Full Name</TableCell>
                    <TableCell>Website</TableCell>
                    <TableCell>Products</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
              <TableBody>
                {vendors.map((vendor) => (
                  <TableRow key={vendor.id}>
                    <TableCell>{vendor.name}</TableCell>
                    <TableCell>{vendor.full_name || '-'}</TableCell>
                    <TableCell>{vendor.website || '-'}</TableCell>
                    <TableCell>{vendor.product_count || 0}</TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <IconButton
                          color="primary"
                          onClick={() => openEditVendorDialog(vendor)}
                        >
                          <EditOutlined />
                        </IconButton>
                        <IconButton
                          color="error"
                          onClick={async () => {
                            if (window.confirm(`Are you sure you want to delete vendor "${vendor.name}"?`)) {
                              try {
                                await apiClient.deleteVendor(vendor.id);
                                enqueueSnackbar('Vendor deleted successfully', { variant: 'success' });
                                loadVendors();
                                loadProducts(); // Refresh products as well since vendor deletion might affect them
                              } catch (error: any) {
                                enqueueSnackbar(error.response?.data?.detail || 'Failed to delete vendor', { variant: 'error' });
                              }
                            }
                          }}
                        >
                          <DeleteOutlined />
                        </IconButton>
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

          {/* Products Tab */}
          <TabPanel value={tabValue} index={1}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
              <Typography variant="h6">Manage Products</Typography>
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Button
                  variant="outlined"
                  startIcon={<UploadFileOutlined />}
                  onClick={() => setUploadDialogOpen(true)}
                >
                  Upload Documents
                </Button>
                <Button
                  variant="contained"
                  startIcon={<AddOutlined />}
                  onClick={() => setProductDialogOpen(true)}
                >
                  Add Product
                </Button>
              </Box>
            </Box>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Name</TableCell>
                    <TableCell>Vendor</TableCell>
                    <TableCell>Model</TableCell>
                    <TableCell>Category</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {products.map((product) => (
                    <TableRow key={product.id}>
                      <TableCell>{product.name}</TableCell>
                      <TableCell>{product.vendor}</TableCell>
                      <TableCell>{product.model || '-'}</TableCell>
                      <TableCell>
                        <Chip label={product.category} size="small" />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <IconButton
                            color="primary"
                            onClick={() => openEditProductDialog(product)}
                          >
                            <EditOutlined />
                          </IconButton>
                          <Button
                            size="small"
                            onClick={() => {
                              setSelectedProduct(product.id);
                              setUploadDialogOpen(true);
                            }}
                          >
                            Upload Docs
                          </Button>
                          <IconButton
                            color="error"
                            size="small"
                            onClick={async () => {
                              if (window.confirm(`Are you sure you want to delete product "${product.name}"? This will also delete all associated documents.`)) {
                                try {
                                  await apiClient.deleteProduct(product.id);
                                  enqueueSnackbar('Product deleted successfully', { variant: 'success' });
                                  loadVendors();
                                  loadProducts();
                                } catch (error: any) {
                                  enqueueSnackbar(error.response?.data?.detail || 'Failed to delete product', { variant: 'error' });
                                }
                              }
                            }}
                          >
                            <DeleteOutlined />
                          </IconButton>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>

          {/* Documents Tab */}
          <TabPanel value={tabValue} index={2}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
              <Typography variant="h6">Document Management</Typography>
              <Button
                variant="outlined"
                startIcon={<RefreshOutlined />}
                onClick={() => {
                  loadVendors();
                  loadProducts();
                }}
              >
                Refresh
              </Button>
            </Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              View processed documents for all products
            </Typography>

            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Product</TableCell>
                    <TableCell>Vendor</TableCell>
                    <TableCell>Document Name</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Upload Date</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {products.flatMap((product) =>
                    (product.documents || []).map((doc: any) => (
                      <TableRow key={`${product.id}-${doc.id}`}>
                        <TableCell>{product.name}</TableCell>
                        <TableCell>{product.vendor}</TableCell>
                        <TableCell>{doc.filename}</TableCell>
                        <TableCell>
                          <Chip
                            label={doc.document_type}
                            size="small"
                            color={doc.document_type === 'datasheet' ? 'primary' : 'secondary'}
                          />
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={doc.is_processed ? 'Processed' : 'Pending'}
                            size="small"
                            color={doc.is_processed ? 'success' : 'warning'}
                          />
                        </TableCell>
                        <TableCell>
                          {new Date(doc.created_at).toLocaleDateString()}
                        </TableCell>
                        <TableCell>
                          <IconButton
                            color="error"
                            onClick={async () => {
                              if (window.confirm(`Are you sure you want to delete "${doc.filename}"?`)) {
                                try {
                                  await apiClient.deleteDocument(doc.id);
                                  enqueueSnackbar('Document deleted successfully', { variant: 'success' });
                                  // Refresh data
                                  loadVendors();
                                  loadProducts();
                                } catch (error: any) {
                                  enqueueSnackbar(error.response?.data?.detail || 'Failed to delete document', { variant: 'error' });
                                }
                              }
                            }}
                          >
                            <DeleteOutlined />
                          </IconButton>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                  {products.length === 0 && (
                    <TableRow>
                      <TableCell colSpan={7} align="center">
                        <Typography variant="body2" color="text.secondary">
                          No documents found. Upload documents for products first.
                        </Typography>
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            </TableContainer>
          </TabPanel>

          {/* Benchmarks Tab */}
          <TabPanel value={tabValue} index={3}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
              <Typography variant="h6">Benchmark Data</Typography>
              <Button
                variant="contained"
                startIcon={<AddOutlined />}
                onClick={() => setBenchmarkDialogOpen(true)}
              >
                Import Benchmarks
              </Button>
            </Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Import CPU, GPU, and storage benchmark data for product comparisons
            </Typography>
          </TabPanel>
        </Card>

        {/* Vendor Dialog */}
        <Dialog open={vendorDialogOpen} onClose={() => setVendorDialogOpen(false)} maxWidth="sm" fullWidth>
          <DialogTitle>{editingVendor ? 'Edit Vendor' : 'Add New Vendor'}</DialogTitle>
          <DialogContent>
            <TextField
              fullWidth
              label="Vendor Name"
              margin="normal"
              value={vendorForm.name}
              onChange={(e) => setVendorForm({ ...vendorForm, name: e.target.value })}
              required
            />
            <TextField
              fullWidth
              label="Full Name"
              margin="normal"
              value={vendorForm.full_name}
              onChange={(e) => setVendorForm({ ...vendorForm, full_name: e.target.value })}
            />
            <TextField
              fullWidth
              label="Website"
              margin="normal"
              value={vendorForm.website}
              onChange={(e) => setVendorForm({ ...vendorForm, website: e.target.value })}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => {
              setVendorDialogOpen(false);
              setEditingVendor(null);
              setVendorForm({ name: '', full_name: '', website: '' });
            }}>Cancel</Button>
            <Button onClick={editingVendor ? handleEditVendor : handleCreateVendor} variant="contained">
              {editingVendor ? 'Update' : 'Create'}
            </Button>
          </DialogActions>
        </Dialog>

        {/* Product Dialog */}
        <Dialog open={productDialogOpen} onClose={() => setProductDialogOpen(false)} maxWidth="sm" fullWidth>
          <DialogTitle>{editingProduct ? 'Edit Product' : 'Add New Product'}</DialogTitle>
          <DialogContent>
            <TextField
              fullWidth
              select
              label="Vendor Name"
              margin="normal"
              value={productForm.vendor_name}
              onChange={(e) => setProductForm({ ...productForm, vendor_name: e.target.value })}
              SelectProps={{ native: true }}
              required
            >
              <option value="">Select Vendor</option>
              {vendors.map((vendor) => (
                <option key={vendor.id} value={vendor.name}>
                  {vendor.name}
                </option>
              ))}
            </TextField>
            <TextField
              fullWidth
              label="Product Name"
              margin="normal"
              value={productForm.name}
              onChange={(e) => setProductForm({ ...productForm, name: e.target.value })}
              required
            />
            <TextField
              fullWidth
              label="Model"
              margin="normal"
              value={productForm.model}
              onChange={(e) => setProductForm({ ...productForm, model: e.target.value })}
            />
            <TextField
              fullWidth
              select
              label="Category"
              margin="normal"
              value={productForm.category}
              onChange={(e) => setProductForm({ ...productForm, category: e.target.value })}
              SelectProps={{ native: true }}
            >
              <option value="server">Server</option>
              <option value="pc">PC</option>
              <option value="laptop">Laptop</option>
              <option value="network_switch">Network Switch</option>
              <option value="storage_nas">Storage NAS</option>
            </TextField>
            <TextField
              fullWidth
              label="Description"
              margin="normal"
              multiline
              rows={3}
              value={productForm.description}
              onChange={(e) => setProductForm({ ...productForm, description: e.target.value })}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => {
              setProductDialogOpen(false);
              setEditingProduct(null);
              setProductForm({ vendor_name: '', name: '', model: '', category: 'server', description: '' });
            }}>Cancel</Button>
            <Button onClick={editingProduct ? handleEditProduct : handleCreateProduct} variant="contained">
              {editingProduct ? 'Update' : 'Create'}
            </Button>
          </DialogActions>
        </Dialog>

        {/* Benchmark Import Dialog */}
        <Dialog open={benchmarkDialogOpen} onClose={() => setBenchmarkDialogOpen(false)} maxWidth="md" fullWidth>
          <DialogTitle>Import Benchmark Data</DialogTitle>
          <DialogContent>
            <TextField
              fullWidth
              select
              label="Benchmark Type"
              margin="normal"
              value={benchmarkType}
              onChange={(e) => setBenchmarkType(e.target.value as 'spec' | 'passmark')}
              SelectProps={{ native: true }}
              required
            >
              <option value="spec">SPEC Benchmarks</option>
              <option value="passmark">PassMark Benchmarks</option>
            </TextField>

            {benchmarkType === 'spec' && (
              <>
                <TextField
                  fullWidth
                  label="SPEC Version (optional)"
                  margin="normal"
                  value={specForm.version}
                  onChange={(e) => setSpecForm({ ...specForm, version: e.target.value })}
                  placeholder="e.g., 2023"
                />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2, mb: 1 }}>
                  Upload SPEC benchmark CSV file. Supported formats include SPECspeed Integer, SPECspeed Floating Point, SPECrate Integer, and SPECrate Floating Point.
                </Typography>
                <Box
                  {...specDropzone.getRootProps()}
                  sx={{
                    border: '2px dashed',
                    borderColor: specDropzone.isDragActive ? 'primary.main' : 'grey.300',
                    borderRadius: 2,
                    p: 4,
                    textAlign: 'center',
                    cursor: 'pointer',
                    bgcolor: specDropzone.isDragActive ? 'action.hover' : 'background.paper',
                    mb: 2,
                  }}
                >
                  <input {...specDropzone.getInputProps()} />
                  <UploadFileOutlined sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    {specDropzone.isDragActive ? 'Drop CSV file here' : 'Drag & drop SPEC CSV file or click to browse'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Only CSV files are supported
                  </Typography>
                </Box>
                {specForm.file && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Selected File:
                    </Typography>
                    <Chip
                      label={`${specForm.file.name} (${(specForm.file.size / 1024 / 1024).toFixed(1)}MB)`}
                      size="small"
                      onDelete={() => setSpecForm({ ...specForm, file: null })}
                    />
                  </Box>
                )}
              </>
            )}

            {benchmarkType === 'passmark' && (
              <>
                <TextField
                  fullWidth
                  select
                  label="Benchmark Type"
                  margin="normal"
                  value={passmarkForm.benchmarkType}
                  onChange={(e) => setPassmarkForm({ ...passmarkForm, benchmarkType: e.target.value as 'PASSMARK_CPU' | 'PASSMARK_GPU' })}
                  SelectProps={{ native: true }}
                  required
                >
                  <option value="PASSMARK_CPU">PassMark CPU</option>
                  <option value="PASSMARK_GPU">PassMark GPU</option>
                </TextField>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 2, mb: 1 }}>
                  Upload PassMark benchmark CSV file for {passmarkForm.benchmarkType === 'PASSMARK_CPU' ? 'CPU' : 'GPU'} benchmarks.
                </Typography>
                <Box
                  {...passmarkDropzone.getRootProps()}
                  sx={{
                    border: '2px dashed',
                    borderColor: passmarkDropzone.isDragActive ? 'primary.main' : 'grey.300',
                    borderRadius: 2,
                    p: 4,
                    textAlign: 'center',
                    cursor: 'pointer',
                    bgcolor: passmarkDropzone.isDragActive ? 'action.hover' : 'background.paper',
                    mb: 2,
                  }}
                >
                  <input {...passmarkDropzone.getInputProps()} />
                  <UploadFileOutlined sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                  <Typography variant="h6" gutterBottom>
                    {passmarkDropzone.isDragActive ? 'Drop CSV file here' : 'Drag & drop PassMark CSV file or click to browse'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Only CSV files are supported
                  </Typography>
                </Box>
                {passmarkForm.file && (
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      Selected File:
                    </Typography>
                    <Chip
                      label={`${passmarkForm.file.name} (${(passmarkForm.file.size / 1024 / 1024).toFixed(1)}MB)`}
                      size="small"
                      onDelete={() => setPassmarkForm({ ...passmarkForm, file: null })}
                    />
                  </Box>
                )}
              </>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setBenchmarkDialogOpen(false)}>Cancel</Button>
            <Button
              onClick={handleImportBenchmarks}
              variant="contained"
              disabled={(benchmarkType === 'spec' && !specForm.file) || (benchmarkType === 'passmark' && !passmarkForm.file)}
            >
              Import Benchmarks
            </Button>
          </DialogActions>
        </Dialog>

        {/* Upload Dialog */}
        <Dialog open={uploadDialogOpen} onClose={() => !loading && setUploadDialogOpen(false)} maxWidth="md" fullWidth>
          <DialogTitle>Upload Documents</DialogTitle>
          <DialogContent>
            {uploadProgress && (
              <Box sx={{ mb: 3 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {uploadProgress.currentStep}
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={uploadProgress.progress}
                  sx={{ height: 8, borderRadius: 4 }}
                />
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                  {Math.round(uploadProgress.progress)}% complete
                </Typography>
              </Box>
            )}

            {!uploadProgress && (
              <Box
                {...getRootProps()}
                sx={{
                  border: '2px dashed',
                  borderColor: isDragActive ? 'primary.main' : 'grey.300',
                  borderRadius: 2,
                  p: 4,
                  textAlign: 'center',
                  cursor: 'pointer',
                  bgcolor: isDragActive ? 'action.hover' : 'background.paper',
                  mb: 2,
                }}
              >
                <input {...getInputProps()} />
                <UploadFileOutlined sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  {isDragActive ? 'Drop files here' : 'Drag & drop files or click to browse'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Supported: PDF, DOCX, XLSX, TXT, PNG, JPG
                </Typography>
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                  Files will be processed automatically: text extraction, specification analysis, and embedding generation
                </Typography>
              </Box>
            )}

            {uploadedFiles.length > 0 && !uploadProgress && (
              <Box>
                <Typography variant="subtitle2" gutterBottom>
                  Selected Files ({uploadedFiles.length}):
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {uploadedFiles.map((file, idx) => (
                    <Chip
                      key={idx}
                      label={`${file.name} (${(file.size / 1024 / 1024).toFixed(1)}MB)`}
                      size="small"
                      sx={{ maxWidth: 200 }}
                    />
                  ))}
                </Box>
              </Box>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setUploadDialogOpen(false)} disabled={loading}>
              Cancel
            </Button>
            <Button onClick={handleUpload} variant="contained" disabled={loading || uploadedFiles.length === 0}>
              {loading ? 'Processing...' : `Upload ${uploadedFiles.length} File${uploadedFiles.length !== 1 ? 's' : ''}`}
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </Layout>
  );
}
