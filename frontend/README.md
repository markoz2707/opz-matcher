# OPZ Product Matcher - Frontend

Modern React web dashboard for the OPZ Product Matcher application.

## Features

- **Modern UI**: Clean, professional interface built with Material-UI
- **Three Working Modes**:
  - 📤 **Data Import**: Upload documents, manage vendors and products
  - 🔍 **Product Search**: Find matching products for OPZ requirements
  - 📝 **OPZ Creation**: Generate professional OPZ documents
- **User Authentication**: Secure JWT-based authentication
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Updates**: Live status updates for document processing and OPZ generation

## Tech Stack

- **React 18** with TypeScript
- **Material-UI (MUI)** for components
- **React Router** for navigation
- **Axios** for API communication
- **React Dropzone** for file uploads
- **Notistack** for notifications
- **Vite** for fast development

## Quick Start

### Prerequisites

- Node.js 18+ and npm
- Backend API running on http://localhost:8000

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The app will open at http://localhost:3000

### Build for Production

```bash
# Create production build
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   └── Layout.tsx           # Main layout with navigation
│   ├── contexts/
│   │   └── AuthContext.tsx      # Authentication state management
│   ├── pages/
│   │   ├── Login.tsx            # Login page
│   │   ├── Register.tsx         # Registration page
│   │   ├── Dashboard.tsx        # Main dashboard
│   │   ├── DataImport.tsx       # Data import mode
│   │   ├── ProductSearch.tsx    # Product search mode
│   │   ├── OPZCreation.tsx      # OPZ creation mode
│   │   └── Profile.tsx          # User profile
│   ├── services/
│   │   └── api.ts               # API client
│   ├── App.tsx                  # Main app component
│   └── main.tsx                 # Entry point
├── package.json
└── vite.config.ts
```

## Configuration

Create a `.env` file in the frontend directory:

```env
VITE_API_URL=http://localhost:8000
```

## Features by Mode

### 1. Data Import Mode

- **Vendor Management**: Add and view vendors
- **Product Management**: Create products and assign to vendors
- **Document Upload**: Drag-and-drop file upload with support for:
  - PDF (with OCR support)
  - DOCX (Word documents)
  - XLSX (Excel spreadsheets)
  - TXT (Plain text)
  - Images (PNG, JPG, JPEG)
- **Automatic Processing**: AI extracts specifications from uploaded documents
- **Benchmark Import**: Import CPU, GPU, and storage benchmark data

### 2. Product Search Mode

- **Smart Search**: Paste OPZ requirements in Polish or English
- **Match Scoring**: View products ranked by match percentage
- **Detailed Analysis**:
  - ✓ Exact matches
  - ≈ Close matches
  - ✗ Deviations
  - 💡 Adjustable requirements
- **Customer Questions**: AI suggests questions to clarify requirements
- **Benchmark Validation**: Compare products using benchmark data

### 3. OPZ Creation Mode

- **Step-by-Step Wizard**:
  1. Basic Information (title, category)
  2. Configuration (processor, memory, storage, network)
  3. Vendor Selection
  4. Review & Generate
- **Multi-Vendor Support**: Select multiple vendors for requirements
- **Professional Output**: Generate DOCX documents
- **Document Management**: View and download your OPZ documents
- **Real-time Generation**: Live progress updates

## User Interface

### Dashboard
- Welcome message with user greeting
- Statistics overview (products, documents, searches, OPZ created)
- Quick access cards for each mode
- Quick start guide

### Navigation
- Sidebar navigation with mode icons
- User profile menu with logout
- Breadcrumb-style page titles

### Visual Design
- **Color Scheme**: Purple gradient theme
- **Typography**: Roboto font family
- **Components**: Material-UI components
- **Icons**: Material Design icons
- **Responsive**: Mobile-first design

## API Integration

The frontend communicates with the backend API using the `ApiClient` class:

```typescript
import { apiClient } from './services/api';

// Example: Login
const token = await apiClient.login(username, password);

// Example: Search products
const results = await apiClient.searchProducts({
  requirements_text: "Your OPZ requirements...",
  category: "server"
});

// Example: Create OPZ
const opz = await apiClient.createOPZ({
  title: "Server for ERP",
  category: "server",
  configuration: {...},
  selected_vendors: ["Dell", "HPE"]
});
```

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Adding New Features

1. Create component in `src/pages/` or `src/components/`
2. Add route in `App.tsx`
3. Update navigation in `Layout.tsx`
4. Add API methods in `api.ts` if needed

## Deployment

### Using Docker

The frontend can be served alongside the backend using nginx:

```dockerfile
FROM node:18 as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
```

### Using Vercel/Netlify

1. Push code to GitHub
2. Connect repository to Vercel/Netlify
3. Set environment variables
4. Deploy

## Troubleshooting

### API Connection Issues

- Verify backend is running on port 8000
- Check CORS settings in backend
- Verify API URL in `.env`

### Build Errors

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Authentication Issues

- Clear browser localStorage
- Check JWT token expiration
- Verify backend authentication endpoints

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome)

## Contributing

1. Follow TypeScript best practices
2. Use Material-UI components
3. Maintain responsive design
4. Add error handling
5. Update documentation

## License

MIT License - see LICENSE file for details
