// api-server/src/middleware/errorMiddleware.ts
import { Request, Response, NextFunction } from 'express';

interface ErrorWithStatus extends Error {
  status?: number;
  statusCode?: number;
}

export const notFound = (req: Request, res: Response, next: NextFunction) => {
  const error = new Error(`Not Found - ${req.originalUrl}`) as ErrorWithStatus;
  error.status = 404;
  next(error);
};

export const errorHandler = (
  err: ErrorWithStatus,
  req: Request,
  res: Response,
  next: NextFunction
) => {
  const statusCode = err.status || err.statusCode || 500;
  
  // Log error details
  console.error(`Error ${statusCode}: ${err.message}`);
  if (process.env.NODE_ENV === 'development') {
    console.error(err.stack);
  }

  // Send error response
  res.status(statusCode).json({
    success: false,
    error: err.message,
    ...(process.env.NODE_ENV === 'development' && {
      stack: err.stack,
      details: {
        method: req.method,
        url: req.originalUrl,
        body: req.body,
        query: req.query,
        headers: req.headers,
      },
    }),
  });
};
