import { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  Paper,
  Chip,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  Divider,
  Alert,
} from '@mui/material';
import {
  SearchOutlined,
  ExpandMoreOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  InfoOutlined,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import Layout from '../components/Layout';
import { apiClient } from '../services/api';

interface ProductMatch {
  product_id: number;
  vendor: string;
  name: string;
  model: string | null;
  match_score: number;
  exact_matches: string[];
  close_matches: string[];
  deviations: any[];
  adjustable_requirements: any[];
  benchmark_analysis: any;
  recommendation: string;
}

export default function ProductSearch() {
  const [requirements, setRequirements] = useState('');
  const [category, setCategory] = useState('');
  const [loading, setLoading] = useState(false);
  const [searchResults, setSearchResults] = useState<ProductMatch[]>([]);
  const [questions, setQuestions] = useState<string[]>([]);
  const [analysis, setAnalysis] = useState('');
  const { enqueueSnackbar } = useSnackbar();

  const handleSearch = async () => {
    if (!requirements.trim()) {
      enqueueSnackbar('Please enter OPZ requirements', { variant: 'warning' });
      return;
    }

    setLoading(true);
    try {
      const data = await apiClient.searchProducts({
        requirements_text: requirements,
        category: category || undefined,
        min_match_score: 0.6,
      });

      setSearchResults(data.matched_products);
      setQuestions(data.questions_for_customer || []);
      setAnalysis(data.general_analysis || '');
      enqueueSnackbar(`Found ${data.matched_products.length} matching products`, { variant: 'success' });
    } catch (error: any) {
      enqueueSnackbar(error.response?.data?.detail || 'Search failed', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  };

  return (
    <Layout>
      <Box>
        <Typography variant="h5" fontWeight="bold" gutterBottom>
          Product Search Mode
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Enter your OPZ requirements to find matching products. The AI will analyze specifications and suggest adjustments.
        </Typography>

        {/* Search Input */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Grid container spacing={2}>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  multiline
                  rows={8}
                  label="OPZ Requirements (Polish or English)"
                  placeholder="Paste your OPZ requirements here...

Example:
Serwer rack 2U z następującymi parametrami:
- Procesor: Intel Xeon minimum 16 rdzeni, 2.5 GHz
- RAM: 64GB DDR4
- Dyski: 2x 1TB SSD w RAID 1
- Sieć: 4x 1GbE
- Zasilacz: redundantny"
                  value={requirements}
                  onChange={(e) => setRequirements(e.target.value)}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  select
                  label="Category (Optional)"
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  SelectProps={{ native: true }}
                >
                  <option value="">All Categories</option>
                  <option value="server">Server</option>
                  <option value="pc">PC</option>
                  <option value="laptop">Laptop</option>
                  <option value="network_switch">Network Switch</option>
                  <option value="storage_nas">Storage NAS</option>
                </TextField>
              </Grid>
              <Grid item xs={12} md={6}>
                <Button
                  fullWidth
                  variant="contained"
                  size="large"
                  startIcon={<SearchOutlined />}
                  onClick={handleSearch}
                  disabled={loading}
                  sx={{ height: '56px' }}
                >
                  {loading ? 'Searching...' : 'Search Products'}
                </Button>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {loading && <LinearProgress sx={{ mb: 3 }} />}

        {/* General Analysis */}
        {analysis && (
          <Alert severity="info" sx={{ mb: 3 }} icon={<InfoOutlined />}>
            <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
              General Analysis
            </Typography>
            <Typography variant="body2">{analysis}</Typography>
          </Alert>
        )}

        {/* Questions for Customer */}
        {questions.length > 0 && (
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <InfoOutlined color="primary" />
                Suggested Questions for Customer
              </Typography>
              <List dense>
                {questions.map((question, idx) => (
                  <ListItem key={idx}>
                    <ListItemText primary={`${idx + 1}. ${question}`} />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        )}

        {/* Search Results */}
        {searchResults.length > 0 && (
          <Box>
            <Typography variant="h6" gutterBottom>
              Matching Products ({searchResults.length})
            </Typography>
            {searchResults.map((product, idx) => (
              <Accordion key={idx} sx={{ mb: 2 }}>
                <AccordionSummary expandIcon={<ExpandMoreOutlined />}>
                  <Box sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                      {product.vendor} {product.name} {product.model && `(${product.model})`}
                    </Typography>
                    <Chip
                      label={`${(product.match_score * 100).toFixed(0)}% Match`}
                      color={getScoreColor(product.match_score)}
                      size="small"
                    />
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={3}>
                    {/* Exact Matches */}
                    {product.exact_matches.length > 0 && (
                      <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 2, bgcolor: 'success.50' }}>
                          <Typography variant="subtitle2" fontWeight="bold" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <CheckCircleOutlined color="success" fontSize="small" />
                            Exact Matches
                          </Typography>
                          <List dense>
                            {product.exact_matches.map((match, i) => (
                              <ListItem key={i}>
                                <ListItemText primary={`✓ ${match}`} />
                              </ListItem>
                            ))}
                          </List>
                        </Paper>
                      </Grid>
                    )}

                    {/* Close Matches */}
                    {product.close_matches.length > 0 && (
                      <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 2, bgcolor: 'warning.50' }}>
                          <Typography variant="subtitle2" fontWeight="bold" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <InfoOutlined color="warning" fontSize="small" />
                            Close Matches
                          </Typography>
                          <List dense>
                            {product.close_matches.map((match, i) => (
                              <ListItem key={i}>
                                <ListItemText primary={`≈ ${match}`} />
                              </ListItem>
                            ))}
                          </List>
                        </Paper>
                      </Grid>
                    )}

                    {/* Deviations */}
                    {product.deviations.length > 0 && (
                      <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 2, bgcolor: 'error.50' }}>
                          <Typography variant="subtitle2" fontWeight="bold" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <WarningOutlined color="error" fontSize="small" />
                            Deviations
                          </Typography>
                          <List dense>
                            {product.deviations.map((dev, i) => (
                              <ListItem key={i}>
                                <ListItemText
                                  primary={typeof dev === 'string' ? dev : (dev.requirement || dev.description || 'Specification mismatch')}
                                  secondary={typeof dev === 'object' && dev.details ? dev.details : null}
                                />
                              </ListItem>
                            ))}
                          </List>
                        </Paper>
                      </Grid>
                    )}

                    {/* Adjustable Requirements */}
                    {product.adjustable_requirements.length > 0 && (
                      <Grid item xs={12} md={6}>
                        <Paper sx={{ p: 2, bgcolor: 'info.50' }}>
                          <Typography variant="subtitle2" fontWeight="bold" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <InfoOutlined color="info" fontSize="small" />
                            Suggested Adjustments
                          </Typography>
                          <List dense>
                            {product.adjustable_requirements.map((req, i) => (
                              <ListItem key={i}>
                                <ListItemText
                                  primary={typeof req === 'string' ? req : (req.suggestion || req.requirement || 'Consider adjusting requirement')}
                                  secondary={typeof req === 'object' && req.reason ? `Reason: ${req.reason}` : null}
                                />
                              </ListItem>
                            ))}
                          </List>
                        </Paper>
                      </Grid>
                    )}

                    {/* Recommendation */}
                    {product.recommendation && (
                      <Grid item xs={12}>
                        <Divider sx={{ my: 2 }} />
                        <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                          Recommendation
                        </Typography>
                        <Typography variant="body2">{product.recommendation}</Typography>
                      </Grid>
                    )}
                  </Grid>
                </AccordionDetails>
              </Accordion>
            ))}
          </Box>
        )}

        {/* No Results */}
        {!loading && searchResults.length === 0 && requirements && (
          <Paper sx={{ p: 4, textAlign: 'center' }}>
            <SearchOutlined sx={{ fontSize: 60, color: 'text.disabled', mb: 2 }} />
            <Typography variant="h6" color="text.secondary">
              No matching products found
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Try adjusting your requirements or adding more products to the database
            </Typography>
          </Paper>
        )}
      </Box>
    </Layout>
  );
}
