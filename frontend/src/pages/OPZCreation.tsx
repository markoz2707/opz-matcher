import { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  Stepper,
  Step,
  StepLabel,
  Chip,
  MenuItem,
  Paper,
  List,
  ListItem,
  ListItemText,
  IconButton,
  LinearProgress,
} from '@mui/material';
import {
  AddOutlined,
  DeleteOutlined,
  DownloadOutlined,
  RefreshOutlined,
  DescriptionOutlined,
} from '@mui/icons-material';
import { useSnackbar } from 'notistack';
import Layout from '../components/Layout';
import { apiClient } from '../services/api';

const steps = ['Basic Information', 'Configuration', 'Select Vendors', 'Review & Generate'];

const categories = [
  { value: 'server', label: 'Server' },
  { value: 'pc', label: 'PC' },
  { value: 'laptop', label: 'Laptop' },
  { value: 'network_switch', label: 'Network Switch' },
  { value: 'storage_nas', label: 'Storage NAS' },
];

export default function OPZCreation() {
  const [activeStep, setActiveStep] = useState(0);
  const [vendors, setVendors] = useState<any[]>([]);
  const [myOPZs, setMyOPZs] = useState<any[]>([]);
  const { enqueueSnackbar } = useSnackbar();
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Form state
  const [title, setTitle] = useState('');
  const [category, setCategory] = useState('server');
  const [selectedVendors, setSelectedVendors] = useState<string[]>([]);
  const [configuration, setConfiguration] = useState<any>({
    processor: { family: '', min_cores: '', min_frequency: '' },
    memory: { capacity_gb: '', type: '' },
    storage: { type: '', capacity_gb: '', raid: '' },
    network: { ports: '', speed: '' },
  });

  // Generation state
  const [generating, setGenerating] = useState(false);
  const [generatedOPZ, setGeneratedOPZ] = useState<any>(null);

  useEffect(() => {
    loadVendors();
    loadMyOPZs();

    // Cleanup polling interval on unmount
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
    };
  }, []);

  const loadVendors = async () => {
    try {
      const data = await apiClient.getVendors();
      setVendors(data);
    } catch (error) {
      enqueueSnackbar('Failed to load vendors', { variant: 'error' });
    }
  };

  const loadMyOPZs = async () => {
    try {
      const data = await apiClient.listUserOPZs();
      setMyOPZs(data);
    } catch (error) {
      enqueueSnackbar('Failed to load OPZ documents', { variant: 'error' });
    }
  };

  const handleNext = () => {
    if (activeStep === 0 && !title.trim()) {
      enqueueSnackbar('Please enter a title', { variant: 'warning' });
      return;
    }
    if (activeStep === 2 && selectedVendors.length === 0) {
      enqueueSnackbar('Please select at least one vendor', { variant: 'warning' });
      return;
    }
    setActiveStep((prev) => prev + 1);
  };

  const handleBack = () => {
    setActiveStep((prev) => prev - 1);
  };

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      const result = await apiClient.createOPZ({
        title,
        category,
        configuration,
        selected_vendors: selectedVendors,
      });

      setGeneratedOPZ(result);
      enqueueSnackbar('OPZ generation started', { variant: 'success' });

      // Clear any existing polling interval
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }

      // Start polling for completion
      pollingIntervalRef.current = setInterval(async () => {
        try {
          const opzData = await apiClient.getOPZ(result.opz_id);
          if (opzData.status === 'generated') {
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current);
              pollingIntervalRef.current = null;
            }
            setGeneratedOPZ(opzData);
            setGenerating(false);
            enqueueSnackbar('OPZ generated successfully!', { variant: 'success' });
            loadMyOPZs();
          } else if (opzData.status === 'error') {
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current);
              pollingIntervalRef.current = null;
            }
            setGenerating(false);
            enqueueSnackbar('OPZ generation failed', { variant: 'error' });
          }
        } catch (error) {
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }
          setGenerating(false);
        }
      }, 3000);
    } catch (error: any) {
      setGenerating(false);
      enqueueSnackbar(error.response?.data?.detail || 'Failed to create OPZ', { variant: 'error' });
    }
  };

  const handleDownload = async (opzId: number, title: string) => {
    try {
      const blob = await apiClient.downloadOPZ(opzId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${title}.docx`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      enqueueSnackbar('OPZ downloaded successfully', { variant: 'success' });
    } catch (error) {
      enqueueSnackbar('Failed to download OPZ', { variant: 'error' });
    }
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Basic Information
            </Typography>
            <TextField
              fullWidth
              label="OPZ Title"
              margin="normal"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              required
              placeholder="e.g., Server for ERP System"
            />
            <TextField
              fullWidth
              select
              label="Device Category"
              margin="normal"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              {categories.map((cat) => (
                <MenuItem key={cat.value} value={cat.value}>
                  {cat.label}
                </MenuItem>
              ))}
            </TextField>
          </Box>
        );

      case 1:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Product Configuration
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Specify the technical requirements for your product
            </Typography>

            <Grid container spacing={2}>
              {/* Processor */}
              <Grid item xs={12}>
                <Typography variant="subtitle2" fontWeight="bold" sx={{ mt: 2, mb: 1 }}>
                  Processor
                </Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="Family"
                  placeholder="e.g., Intel Xeon"
                  value={configuration.processor.family}
                  onChange={(e) =>
                    setConfiguration({
                      ...configuration,
                      processor: { ...configuration.processor, family: e.target.value },
                    })
                  }
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="Minimum Cores"
                  type="number"
                  value={configuration.processor.min_cores}
                  onChange={(e) =>
                    setConfiguration({
                      ...configuration,
                      processor: { ...configuration.processor, min_cores: e.target.value },
                    })
                  }
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="Min Frequency (GHz)"
                  type="number"
                  value={configuration.processor.min_frequency}
                  onChange={(e) =>
                    setConfiguration({
                      ...configuration,
                      processor: { ...configuration.processor, min_frequency: e.target.value },
                    })
                  }
                />
              </Grid>

              {/* Memory */}
              <Grid item xs={12}>
                <Typography variant="subtitle2" fontWeight="bold" sx={{ mt: 2, mb: 1 }}>
                  Memory
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Capacity (GB)"
                  type="number"
                  value={configuration.memory.capacity_gb}
                  onChange={(e) =>
                    setConfiguration({
                      ...configuration,
                      memory: { ...configuration.memory, capacity_gb: e.target.value },
                    })
                  }
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Type"
                  placeholder="e.g., DDR4"
                  value={configuration.memory.type}
                  onChange={(e) =>
                    setConfiguration({
                      ...configuration,
                      memory: { ...configuration.memory, type: e.target.value },
                    })
                  }
                />
              </Grid>

              {/* Storage */}
              <Grid item xs={12}>
                <Typography variant="subtitle2" fontWeight="bold" sx={{ mt: 2, mb: 1 }}>
                  Storage
                </Typography>
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="Type"
                  placeholder="e.g., SSD"
                  value={configuration.storage.type}
                  onChange={(e) =>
                    setConfiguration({
                      ...configuration,
                      storage: { ...configuration.storage, type: e.target.value },
                    })
                  }
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="Capacity (GB)"
                  type="number"
                  value={configuration.storage.capacity_gb}
                  onChange={(e) =>
                    setConfiguration({
                      ...configuration,
                      storage: { ...configuration.storage, capacity_gb: e.target.value },
                    })
                  }
                />
              </Grid>
              <Grid item xs={12} md={4}>
                <TextField
                  fullWidth
                  label="RAID"
                  placeholder="e.g., RAID 1"
                  value={configuration.storage.raid}
                  onChange={(e) =>
                    setConfiguration({
                      ...configuration,
                      storage: { ...configuration.storage, raid: e.target.value },
                    })
                  }
                />
              </Grid>

              {/* Network */}
              <Grid item xs={12}>
                <Typography variant="subtitle2" fontWeight="bold" sx={{ mt: 2, mb: 1 }}>
                  Network
                </Typography>
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Ports"
                  type="number"
                  value={configuration.network.ports}
                  onChange={(e) =>
                    setConfiguration({
                      ...configuration,
                      network: { ...configuration.network, ports: e.target.value },
                    })
                  }
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Speed"
                  placeholder="e.g., 1GbE"
                  value={configuration.network.speed}
                  onChange={(e) =>
                    setConfiguration({
                      ...configuration,
                      network: { ...configuration.network, speed: e.target.value },
                    })
                  }
                />
              </Grid>
            </Grid>
          </Box>
        );

      case 2:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Select Vendors
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              Choose vendors that should be able to meet these requirements
            </Typography>
            <Paper sx={{ p: 2 }}>
              <Grid container spacing={1}>
                {vendors.map((vendor) => (
                  <Grid item key={vendor.id}>
                    <Chip
                      label={vendor.name}
                      onClick={() => {
                        if (selectedVendors.includes(vendor.name)) {
                          setSelectedVendors(selectedVendors.filter((v) => v !== vendor.name));
                        } else {
                          setSelectedVendors([...selectedVendors, vendor.name]);
                        }
                      }}
                      color={selectedVendors.includes(vendor.name) ? 'primary' : 'default'}
                      variant={selectedVendors.includes(vendor.name) ? 'filled' : 'outlined'}
                    />
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </Box>
        );

      case 3:
        return (
          <Box>
            <Typography variant="h6" gutterBottom>
              Review & Generate
            </Typography>
            <Paper sx={{ p: 2, mb: 2 }}>
              <Typography variant="subtitle2" fontWeight="bold">
                Title:
              </Typography>
              <Typography variant="body2" sx={{ mb: 2 }}>
                {title}
              </Typography>

              <Typography variant="subtitle2" fontWeight="bold">
                Category:
              </Typography>
              <Typography variant="body2" sx={{ mb: 2 }}>
                {categories.find((c) => c.value === category)?.label}
              </Typography>

              <Typography variant="subtitle2" fontWeight="bold">
                Selected Vendors:
              </Typography>
              <Box sx={{ mb: 2 }}>
                {selectedVendors.map((vendor) => (
                  <Chip key={vendor} label={vendor} size="small" sx={{ mr: 0.5 }} />
                ))}
              </Box>

              <Typography variant="subtitle2" fontWeight="bold">
                Configuration Summary:
              </Typography>
              <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
                {JSON.stringify(configuration, null, 2)}
              </Typography>
            </Paper>

            {generating && (
              <Box sx={{ mb: 2 }}>
                <LinearProgress />
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  Generating OPZ document... This may take up to 30 seconds.
                </Typography>
              </Box>
            )}

            {generatedOPZ && generatedOPZ.status === 'generated' && (
              <Paper sx={{ p: 2, bgcolor: 'success.50' }}>
                <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                  ✓ OPZ Generated Successfully!
                </Typography>
                <Button
                  variant="contained"
                  startIcon={<DownloadOutlined />}
                  onClick={() => handleDownload(generatedOPZ.opz_id, title)}
                >
                  Download DOCX
                </Button>
              </Paper>
            )}
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <Layout>
      <Box>
        <Typography variant="h5" fontWeight="bold" gutterBottom>
          OPZ Creation Mode
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          Create professional OPZ documents for public tenders with AI assistance
        </Typography>

        <Grid container spacing={3}>
          {/* Wizard */}
          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
                  {steps.map((label) => (
                    <Step key={label}>
                      <StepLabel>{label}</StepLabel>
                    </Step>
                  ))}
                </Stepper>

                {renderStepContent(activeStep)}

                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
                  <Button disabled={activeStep === 0} onClick={handleBack}>
                    Back
                  </Button>
                  <Box>
                    {activeStep === steps.length - 1 ? (
                      <Button
                        variant="contained"
                        onClick={handleGenerate}
                        disabled={generating || (generatedOPZ && generatedOPZ.status === 'generated')}
                      >
                        {generating ? 'Generating...' : 'Generate OPZ'}
                      </Button>
                    ) : (
                      <Button variant="contained" onClick={handleNext}>
                        Next
                      </Button>
                    )}
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* My OPZ Documents */}
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">My OPZ Documents</Typography>
                  <IconButton size="small" onClick={loadMyOPZs}>
                    <RefreshOutlined />
                  </IconButton>
                </Box>
                <List>
                  {myOPZs.map((opz) => (
                    <ListItem
                      key={opz.opz_id}
                      secondaryAction={
                        opz.status === 'generated' && (
                          <IconButton
                            edge="end"
                            onClick={() => handleDownload(opz.opz_id, opz.title)}
                          >
                            <DownloadOutlined />
                          </IconButton>
                        )
                      }
                    >
                      <ListItemText
                        primary={opz.title}
                        secondary={
                          <>
                            <Chip label={opz.status} size="small" sx={{ mr: 1 }} />
                            <Typography variant="caption" component="span">
                              {opz.category}
                            </Typography>
                          </>
                        }
                      />
                    </ListItem>
                  ))}
                  {myOPZs.length === 0 && (
                    <Typography variant="body2" color="text.secondary" textAlign="center">
                      No OPZ documents yet
                    </Typography>
                  )}
                </List>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Layout>
  );
}
