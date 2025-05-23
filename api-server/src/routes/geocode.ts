// api-server/src/routes/geocode.ts
import { Router, Request, Response } from 'express';
import axios from 'axios';
import rateLimit from 'express-rate-limit';

const router = Router();

// Rate limiting for geocoding endpoint
const geocodeLimit = rateLimit({
  windowMs: 60 * 1000, // 1 minute
  max: 20, // Limit each IP to 20 geocoding requests per minute
  message: {
    error: 'Too many geocoding requests, please try again later.',
  },
});

router.use(geocodeLimit);

interface GeocodeQuery {
  address?: string;
  lat?: string;
  lng?: string;
}

interface MapboxGeocodeResponse {
  features: Array<{
    center: [number, number];
    place_name: string;
    properties: {
      accuracy?: string;
    };
    bbox?: [number, number, number, number];
  }>;
}

interface MapboxReverseGeocodeResponse {
  features: Array<{
    place_name: string;
    center: [number, number];
    properties: {
      accuracy?: string;
    };
  }>;
}

// Forward geocoding: address -> coordinates
router.get('/', async (req: Request<{}, {}, {}, GeocodeQuery>, res: Response) => {
  try {
    const { address, lat, lng } = req.query;

    if (!address && (!lat || !lng)) {
      return res.status(400).json({
        success: false,
        error: 'Either address or lat/lng coordinates are required',
      });
    }

    const mapboxToken = process.env.MAPBOX_SECRET_KEY;
    if (!mapboxToken) {
      return res.status(500).json({
        success: false,
        error: 'Mapbox API key not configured',
      });
    }

    let apiUrl: string;
    let queryParam: string;

    if (address) {
      // Forward geocoding
      queryParam = encodeURIComponent(address);
      apiUrl = `https://api.mapbox.com/geocoding/v5/mapbox.places/${queryParam}.json`;
    } else {
      // Reverse geocoding
      queryParam = `${lng},${lat}`;
      apiUrl = `https://api.mapbox.com/geocoding/v5/mapbox.places/${queryParam}.json`;
    }

    const response = await axios.get<MapboxGeocodeResponse | MapboxReverseGeocodeResponse>(apiUrl, {
      params: {
        access_token: mapboxToken,
        limit: 5,
        types: 'place,locality,neighborhood,address',
      },
      timeout: 5000,
    });

    if (!response.data.features || response.data.features.length === 0) {
      return res.status(404).json({
        success: false,
        error: address ? 'No results found for the given address' : 'No results found for the given coordinates',
      });
    }

    const results = response.data.features.map(feature => ({
      place_name: feature.place_name,
      center: feature.center,
      accuracy: feature.properties?.accuracy || 'unknown',
      bbox: 'bbox' in feature ? feature.bbox : undefined,
    }));

    res.json({
      success: true,
      data: {
        results,
        query: address || `${lat},${lng}`,
        total: results.length,
      },
    });

  } catch (error) {
    console.error('Geocoding error:', error);

    if (axios.isAxiosError(error)) {
      if (error.response?.status === 401) {
        return res.status(500).json({
          success: false,
          error: 'Invalid Mapbox API key',
        });
      }
      
      if (error.response?.status === 422) {
        return res.status(400).json({
          success: false,
          error: 'Invalid query parameters',
        });
      }

      if (error.code === 'ECONNABORTED') {
        return res.status(408).json({
          success: false,
          error: 'Request timeout - please try again',
        });
      }
    }

    res.status(500).json({
      success: false,
      error: 'Failed to geocode address',
    });
  }
});

// Batch geocoding endpoint
router.post('/batch', async (req: Request, res: Response) => {
  try {
    const { addresses } = req.body;

    if (!addresses || !Array.isArray(addresses) || addresses.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'Array of addresses is required',
      });
    }

    if (addresses.length > 10) {
      return res.status(400).json({
        success: false,
        error: 'Maximum 10 addresses allowed per batch request',
      });
    }

    const mapboxToken = process.env.MAPBOX_SECRET_KEY;
    if (!mapboxToken) {
      return res.status(500).json({
        success: false,
        error: 'Mapbox API key not configured',
      });
    }

    const geocodePromises = addresses.map(async (address: string, index: number) => {
      try {
        const queryParam = encodeURIComponent(address);
        const apiUrl = `https://api.mapbox.com/geocoding/v5/mapbox.places/${queryParam}.json`;

        const response = await axios.get<MapboxGeocodeResponse>(apiUrl, {
          params: {
            access_token: mapboxToken,
            limit: 1,
            types: 'place,locality,neighborhood,address',
          },
          timeout: 5000,
        });

        if (response.data.features && response.data.features.length > 0) {
          const feature = response.data.features[0];
          return {
            index,
            address,
            success: true,
            result: {
              place_name: feature.place_name,
              center: feature.center,
              accuracy: feature.properties?.accuracy || 'unknown',
            },
          };
        } else {
          return {
            index,
            address,
            success: false,
            error: 'No results found',
          };
        }
      } catch (error) {
        return {
          index,
          address,
          success: false,
          error: 'Geocoding failed',
        };
      }
    });

    const results = await Promise.all(geocodePromises);

    res.json({
      success: true,
      data: {
        results,
        total: results.length,
        successful: results.filter(r => r.success).length,
        failed: results.filter(r => !r.success).length,
      },
    });

  } catch (error) {
    console.error('Batch geocoding error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to process batch geocoding request',
    });
  }
});

export default router;
