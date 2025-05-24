import express, { Request, Response, Application } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';

import geocodeRoutes from './routes/geocode';
import simulationRoutes from './routes/simulations';
import { errorHandler, notFound } from './middleware/errorMiddleware';
import { validateEnvironment } from './utils/validation';

// Load environment variables
dotenv.config();

// Validate environment variables
validateEnvironment();

const app: Application = express();
const PORT = process.env.PORT || 3001;

// Security middleware
app.use(helmet({
  crossOriginEmbedderPolicy: false,
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "https://nominatim.openstreetmap.org"],
    },
  },
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  message: {
    error: 'Too many requests from this IP, please try again later.',
  },
  standardHeaders: true,
  legacyHeaders: false,
});

app.use(limiter);

// CORS configuration - simplified to avoid type issues
const corsOrigins: string[] = process.env.NODE_ENV === 'production' 
  ? (process.env.CORS_ORIGINS?.split(',') || [
      'https://pathfinder-visualizer.vercel.app',
      'https://pathfinder-visualizer-git-main.vercel.app'
    ])
  : [
      'http://localhost:3000',
      'http://127.0.0.1:3000'
    ];

// Use CORS with explicit typing
app.use(cors({
  origin: corsOrigins,
  credentials: true,
  optionsSuccessStatus: 200,
} as cors.CorsOptions));

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Compression middleware
app.use(compression());

// Logging middleware
if (process.env.NODE_ENV !== 'production') {
  app.use(morgan('dev'));
} else {
  app.use(morgan('combined'));
}

// Health check endpoint
app.get('/health', (req: Request, res: Response): void => {
  res.status(200).json({
    status: 'OK',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    environment: process.env.NODE_ENV || 'development',
  });
});

// API routes
app.use('/api/geocode', geocodeRoutes);
app.use('/api/simulations', simulationRoutes);

// API info endpoint
app.get('/api', (req: Request, res: Response): void => {
  res.json({
    name: 'PathFinder Visualizer API',
    version: '1.0.0',
    description: 'Backend API for pathfinding and sorting algorithm visualization',
    endpoints: {
      geocode: '/api/geocode',
      simulations: '/api/simulations',
      health: '/health',
    },
    author: 'Shreyas Kasture',
  });
});

// Error handling middleware
app.use(notFound);
app.use(errorHandler);

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('SIGTERM received. Shutting down gracefully...');
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('SIGINT received. Shutting down gracefully...');
  process.exit(0);
});

// Start server
const server = app.listen(PORT, () => {
  console.log(`
🚀 PathFinder API Server running!
📍 Environment: ${process.env.NODE_ENV || 'development'}
🌐 Port: ${PORT}
🔗 URL: http://localhost:${PORT}
💫 Developed with ❤️ by Shreyas Kasture
  `);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (err: Error) => {
  console.error('Unhandled Promise Rejection:', err.message);
  server.close(() => {
    process.exit(1);
  });
});

export default app;
