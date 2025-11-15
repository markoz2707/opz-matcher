import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Grid,
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Box,
  Paper,
  Skeleton,
} from '@mui/material';
import {
  CloudUploadOutlined,
  SearchOutlined,
  DescriptionOutlined,
  TrendingUpOutlined,
  StorageOutlined,
  BusinessOutlined,
} from '@mui/icons-material';
import Layout from '../components/Layout';
import { useAuth } from '../contexts/AuthContext';
import { apiClient } from '../services/api';

export default function Dashboard() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [stats, setStats] = useState({
    totalProducts: 0,
    totalDocuments: 0,
    totalVendors: 0,
    totalOPZs: 0,
  });
  const [loadingStats, setLoadingStats] = useState(true);

  useEffect(() => {
    loadStatistics();
  }, []);

  const loadStatistics = async () => {
    setLoadingStats(true);
    try {
      const data = await apiClient.getStatistics();
      setStats(data);
    } catch (error) {
      // Stats already default to 0
    } finally {
      setLoadingStats(false);
    }
  };

  const modeCards = [
    {
      title: 'Data Import',
      description: 'Upload and process product datasheets, manuals, and technical documentation. Manage vendors, products, and benchmark data.',
      icon: <CloudUploadOutlined sx={{ fontSize: 60, color: 'primary.main' }} />,
      color: '#1976d2',
      path: '/data-import',
      features: ['Upload Documents', 'Manage Vendors', 'Import Benchmarks', 'Auto Extraction'],
    },
    {
      title: 'Product Search',
      description: 'Find matching products for OPZ requirements. Get intelligent recommendations with flexibility analysis and benchmark validation.',
      icon: <SearchOutlined sx={{ fontSize: 60, color: 'success.main' }} />,
      color: '#2e7d32',
      path: '/product-search',
      features: ['Smart Matching', 'Benchmark Analysis', 'Flexibility Suggestions', 'Multi-Criteria'],
    },
    {
      title: 'OPZ Creation',
      description: 'Generate professional OPZ documents for public tenders. Choose vendors, configure specifications, and export to DOCX.',
      icon: <DescriptionOutlined sx={{ fontSize: 60, color: 'warning.main' }} />,
      color: '#ed6c02',
      path: '/opz-creation',
      features: ['Template-Based', 'Multi-Vendor', 'DOCX Export', 'Refinement'],
    },
  ];

  return (
    <Layout>
      <Box>
        {/* Welcome Section */}
        <Paper
          elevation={0}
          sx={{
            p: 4,
            mb: 4,
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white',
            borderRadius: 2,
          }}
        >
          <Typography variant="h4" gutterBottom fontWeight="bold">
            Welcome back, {user?.full_name || user?.username}! 👋
          </Typography>
          <Typography variant="body1">
            Your AI-powered IT procurement assistant. Choose a mode below to get started.
          </Typography>
        </Paper>

        {/* Statistics Overview */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          {[
            { label: 'Products', value: stats.totalProducts, icon: <TrendingUpOutlined />, color: 'primary.main' },
            { label: 'Documents', value: stats.totalDocuments, icon: <StorageOutlined />, color: 'success.main' },
            { label: 'Vendors', value: stats.totalVendors, icon: <BusinessOutlined />, color: 'warning.main' },
            { label: 'OPZ Created', value: stats.totalOPZs, icon: <DescriptionOutlined />, color: 'info.main' },
          ].map((stat, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card>
                <CardContent>
                  {loadingStats ? (
                    <Box>
                      <Skeleton variant="text" width="60%" height={40} />
                      <Skeleton variant="text" width="80%" />
                    </Box>
                  ) : (
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <Box sx={{ color: stat.color }}>{stat.icon}</Box>
                      <Box>
                        <Typography variant="h5" fontWeight="bold">
                          {stat.value}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {stat.label}
                        </Typography>
                      </Box>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        {/* Mode Cards */}
        <Typography variant="h5" gutterBottom fontWeight="bold" sx={{ mb: 3 }}>
          Working Modes
        </Typography>
        <Grid container spacing={3}>
          {modeCards.map((mode, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Card
                sx={{
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: 'translateY(-8px)',
                    boxShadow: 6,
                  },
                }}
              >
                <CardContent sx={{ flexGrow: 1 }}>
                  <Box sx={{ textAlign: 'center', mb: 2 }}>
                    {mode.icon}
                  </Box>
                  <Typography variant="h5" gutterBottom fontWeight="bold" textAlign="center">
                    {mode.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {mode.description}
                  </Typography>
                  <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" fontWeight="bold" gutterBottom>
                      Key Features:
                    </Typography>
                    {mode.features.map((feature, idx) => (
                      <Typography
                        key={idx}
                        variant="body2"
                        sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}
                      >
                        • {feature}
                      </Typography>
                    ))}
                  </Box>
                </CardContent>
                <CardActions sx={{ p: 2, pt: 0 }}>
                  <Button
                    fullWidth
                    variant="contained"
                    onClick={() => navigate(mode.path)}
                    sx={{
                      backgroundColor: mode.color,
                      '&:hover': {
                        backgroundColor: mode.color,
                        filter: 'brightness(0.9)',
                      },
                    }}
                  >
                    Open {mode.title}
                  </Button>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>

        {/* Quick Start Guide */}
        <Paper sx={{ mt: 4, p: 3 }}>
          <Typography variant="h6" gutterBottom fontWeight="bold">
            Quick Start Guide
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                1. Import Data
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Start by adding vendors, products, and uploading datasheets. The AI will automatically extract specifications.
              </Typography>
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                2. Search Products
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Paste OPZ requirements to find matching products. Get intelligent suggestions for requirement adjustments.
              </Typography>
            </Grid>
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                3. Create OPZ
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Generate professional OPZ documents by configuring specifications and selecting vendors.
              </Typography>
            </Grid>
          </Grid>
        </Paper>
      </Box>
    </Layout>
  );
}
