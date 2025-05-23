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

interface NominatimResponse {
  lat: string;
  lon: string;
  display_name: string;
  place_id: string;
  licence: string;
  osm_type: string;
  osm_id: string;
  boundingbox: string[];
  class: string;
  type: string;
  importance: number;
}

// Forward geocoding: address -> coordinates using Nominatim (OpenStreetMap)
router.get('/', async (req: Request<{}, {}, {}, GeocodeQuery>, res: Response) => {
  try {
    const { address, lat, lng } = req.query;

    if (!address && (!lat || !lng)) {
      return res.status(400).json({
        success: false,
        error: 'Either address or lat/lng coordinates are required',
      });
    }

    let apiUrl: string;

    if (address) {
      // Forward geocoding using Nominatim
      apiUrl = `https://nominatim.openstreetmap.org/search`;
      
      const response = await axios.get<NominatimResponse[]>(apiUrl, {
        params: {
          q: address,
          format: 'json',
          limit: 5,
          addressdetails: 1,
        },
        timeout: 5000,
        headers: {
          'User-Agent': 'PathFinder-Visualizer/1.0 (contact@pathfinder.com)', // Required by Nominatim
        },
      });

      if (!response.data || response.data.length === 0) {
        return res.status(404).json({
          success: false,
          error: 'No results found for the given address',
        });
      }

      const results = response.data.map(item => ({
        place_name: item.display_name,
        center: [parseFloat(item.lon), parseFloat(item.lat)],
        accuracy: 'approximate',
        bbox: item.boundingbox ? [
          parseFloat(item.boundingbox[2]), // west
          parseFloat(item.boundingbox[0]), // south
          parseFloat(item.boundingbox[3]), // east
          parseFloat(item.boundingbox[1])  // north
        ] : undefined,
      }));

      res.json({
        success: true,
        data: {
          results,
          query: address,
          total: results.length,
        },
      });

    } else {
      // Reverse geocoding
      apiUrl = `https://nominatim.openstreetmap.org/reverse`;
      
      const response = await axios.get<NominatimResponse>(apiUrl, {
        params: {
          lat: lat,
          lon: lng,
          format: 'json',
          addressdetails: 1,
        },
        timeout: 5000,
        headers: {
          'User-Agent': 'PathFinder-Visualizer/1.0 (contact@pathfinder.com)',
        },
      });

      if (!response.data || !response.data.display_name) {
        return res.status(404).json({
          success: false,
          error: 'No results found for the given coordinates',
        });
      }

      const result = {
        place_name: response.data.display_name,
        center: [parseFloat(response.data.lon), parseFloat(response.data.lat)],
        accuracy: 'approximate',
      };

      res.json({
        success: true,
        data: {
          results: [result],
          query: `${lat},${lng}`,
          total: 1,
        },
      });
    }

  } catch (error) {
    console.error('Geocoding error:', error);

    if (axios.isAxiosError(error)) {
      if (error.code === 'ECONNABORTED') {
        return res.status(408).json({
          success: false,
          error: 'Request timeout - please try again',
        });
      }

      if (error.response?.status === 429) {
        return res.status(429).json({
          success: false,
          error: 'Rate limit exceeded for geocoding service',
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

    const geocodePromises = addresses.map(async (address: string, index: number) => {
      try {
        const response = await axios.get<NominatimResponse[]>('https://nominatim.openstreetmap.org/search', {
          params: {
            q: address,
            format: 'json',
            limit: 1,
            addressdetails: 1,
          },
          timeout: 5000,
          headers: {
            'User-Agent': 'PathFinder-Visualizer/1.0 (contact@pathfinder.com)',
          },
        });

        // Add delay to respect Nominatim usage policy (max 1 request per second)
        await new Promise(resolve => setTimeout(resolve, 1000));

        if (response.data && response.data.length > 0) {
          const item = response.data[0];
          return {
            index,
            address,
            success: true,
            result: {
              place_name: item.display_name,
              center: [parseFloat(item.lon), parseFloat(item.lat)],
              accuracy: 'approximate',
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
