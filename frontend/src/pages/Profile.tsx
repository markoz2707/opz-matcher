import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Avatar,
  Grid,
  Divider,
} from '@mui/material';
import { PersonOutline } from '@mui/icons-material';
import Layout from '../components/Layout';
import { useAuth } from '../contexts/AuthContext';

export default function Profile() {
  const { user } = useAuth();

  return (
    <Layout>
      <Box>
        <Typography variant="h5" fontWeight="bold" gutterBottom>
          User Profile
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          View and manage your account information
        </Typography>

        <Grid container spacing={3}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent sx={{ textAlign: 'center' }}>
                <Avatar
                  sx={{
                    width: 120,
                    height: 120,
                    margin: '0 auto',
                    bgcolor: 'primary.main',
                    fontSize: 48,
                    mb: 2,
                  }}
                >
                  {user?.username?.charAt(0).toUpperCase()}
                </Avatar>
                <Typography variant="h5" gutterBottom>
                  {user?.full_name || user?.username}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {user?.email}
                </Typography>
                <Divider sx={{ my: 2 }} />
                <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1 }}>
                  {user?.is_superuser && (
                    <Typography
                      variant="caption"
                      sx={{
                        bgcolor: 'primary.main',
                        color: 'white',
                        px: 1.5,
                        py: 0.5,
                        borderRadius: 1,
                      }}
                    >
                      ADMIN
                    </Typography>
                  )}
                  <Typography
                    variant="caption"
                    sx={{
                      bgcolor: user?.is_active ? 'success.main' : 'error.main',
                      color: 'white',
                      px: 1.5,
                      py: 0.5,
                      borderRadius: 1,
                    }}
                  >
                    {user?.is_active ? 'ACTIVE' : 'INACTIVE'}
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={8}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <PersonOutline />
                  Account Information
                </Typography>
                <Divider sx={{ mb: 3 }} />

                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Username"
                      value={user?.username || ''}
                      InputProps={{ readOnly: true }}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Email"
                      value={user?.email || ''}
                      InputProps={{ readOnly: true }}
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Full Name"
                      value={user?.full_name || ''}
                      InputProps={{ readOnly: true }}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Account Type"
                      value={user?.is_superuser ? 'Administrator' : 'Standard User'}
                      InputProps={{ readOnly: true }}
                    />
                  </Grid>
                  <Grid item xs={12} md={6}>
                    <TextField
                      fullWidth
                      label="Status"
                      value={user?.is_active ? 'Active' : 'Inactive'}
                      InputProps={{ readOnly: true }}
                    />
                  </Grid>
                </Grid>

                <Box sx={{ mt: 4, p: 2, bgcolor: 'info.50', borderRadius: 1 }}>
                  <Typography variant="body2" color="text.secondary">
                    💡 <strong>Tip:</strong> Contact your administrator to update your account information.
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>
    </Layout>
  );
}
