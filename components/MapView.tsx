// components/MapView.tsx
'use client';

import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import mapboxgl from 'mapbox-gl';
import { Point, PathNode } from '@/types';
import { usePathfindingStore } from '@/store';
import { validateMapboxToken } from '@/lib/utils';
import { MapPin, Navigation, Zap } from 'lucide-react';
import 'mapbox-gl/dist/mapbox-gl.css';

interface MapViewProps {
  className?: string;
}

export default function MapView({ className }: MapViewProps) {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);
  
  const {
    startPoint,
    endPoint,
    walls,
    visitedNodes,
    pathNodes,
    isRunning,
    setStartPoint,
    setEndPoint,
    addWall,
    removeWall,
  } = usePathfindingStore();

  const mapboxToken = process.env.NEXT_PUBLIC_MAPBOX_TOKEN;

  useEffect(() => {
    if (!validateMapboxToken(mapboxToken)) {
      setMapError('Invalid or missing Mapbox token. Please check your environment variables.');
      return;
    }

    if (map.current || !mapContainer.current) return;

    mapboxgl.accessToken = mapboxToken!;

    try {
      map.current = new mapboxgl.Map({
        container: mapContainer.current,
        style: 'mapbox://styles/mapbox/dark-v11',
        center: [-74.006, 40.7128], // New York City
        zoom: 13,
        attributionControl: false,
      });

      map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');

      map.current.on('load', () => {
        setMapLoaded(true);
        initializeMapLayers();
        setupMapInteractions();
      });

      map.current.on('error', (e) => {
        console.error('Mapbox error:', e);
        setMapError('Failed to load map. Please check your internet connection.');
      });

    } catch (error) {
      console.error('Map initialization error:', error);
      setMapError('Failed to initialize map.');
    }

    return () => {
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, [mapboxToken]);

  const initializeMapLayers = () => {
    if (!map.current) return;

    // Add sources for visualization
    map.current.addSource('walls', {
      type: 'geojson',
      data: {
        type: 'FeatureCollection',
        features: [],
      },
    });

    map.current.addSource('visited-nodes', {
      type: 'geojson',
      data: {
        type: 'FeatureCollection',
        features: [],
      },
    });

    map.current.addSource('path-nodes', {
      type: 'geojson',
      data: {
        type: 'FeatureCollection',
        features: [],
      },
    });

    map.current.addSource('start-end-points', {
      type: 'geojson',
      data: {
        type: 'FeatureCollection',
        features: [],
      },
    });

    // Add layers
    map.current.addLayer({
      id: 'walls',
      type: 'circle',
      source: 'walls',
      paint: {
        'circle-radius': 6,
        'circle-color': '#EF4444',
        'circle-opacity': 0.8,
        'circle-stroke-width': 2,
        'circle-stroke-color': '#DC2626',
      },
    });

    map.current.addLayer({
      id: 'visited-nodes',
      type: 'circle',
      source: 'visited-nodes',
      paint: {
        'circle-radius': 4,
        'circle-color': '#3B82F6',
        'circle-opacity': 0.6,
      },
    });

    map.current.addLayer({
      id: 'path-nodes',
      type: 'circle',
      source: 'path-nodes',
      paint: {
        'circle-radius': 5,
        'circle-color': '#10B981',
        'circle-opacity': 0.9,
        'circle-stroke-width': 2,
        'circle-stroke-color': '#059669',
      },
    });

    map.current.addLayer({
      id: 'start-end-points',
      type: 'circle',
      source: 'start-end-points',
      paint: {
        'circle-radius': 8,
        'circle-color': ['get', 'color'],
        'circle-opacity': 0.9,
        'circle-stroke-width': 3,
        'circle-stroke-color': '#FFFFFF',
      },
    });

    // Add path line layer
    map.current.addSource('path-line', {
      type: 'geojson',
      data: {
        type: 'Feature',
        properties: {},
        geometry: {
          type: 'LineString',
          coordinates: [],
        },
      },
    });

    map.current.addLayer({
      id: 'path-line',
      type: 'line',
      source: 'path-line',
      paint: {
        'line-color': '#10B981',
        'line-width': 4,
        'line-opacity': 0.8,
      },
    });
  };

  const setupMapInteractions = () => {
    if (!map.current) return;

    let isPlacingWalls = false;

    map.current.on('mousedown', () => {
      isPlacingWalls = true;
    });

    map.current.on('mouseup', () => {
      isPlacingWalls = false;
    });

    map.current.on('click', (e) => {
      if (isRunning) return;

      const point: Point = {
        lat: e.lngLat.lat,
        lng: e.lngLat.lng,
      };

      if (e.originalEvent.shiftKey) {
        // Shift + click to place end point
        setEndPoint(point);
      } else if (e.originalEvent.ctrlKey || e.originalEvent.metaKey) {
        // Ctrl/Cmd + click to toggle walls
        const existingWall = walls.find(
          w => Math.abs(w.lat - point.lat) < 0.001 && Math.abs(w.lng - point.lng) < 0.001
        );
        if (existingWall) {
          removeWall(existingWall);
        } else {
          addWall(point);
        }
      } else {
        // Regular click to place start point
        setStartPoint(point);
      }
    });

    map.current.on('mousemove', (e) => {
      if (isPlacingWalls && (e.originalEvent.ctrlKey || e.originalEvent.metaKey)) {
        const point: Point = {
          lat: e.lngLat.lat,
          lng: e.lngLat.lng,
        };
        addWall(point);
      }
    });

    // Change cursor based on modifier keys
    map.current.on('mousemove', (e) => {
      if (isRunning) {
        map.current!.getCanvas().style.cursor = 'not-allowed';
      } else if (e.originalEvent.shiftKey) {
        map.current!.getCanvas().style.cursor = 'crosshair';
      } else if (e.originalEvent.ctrlKey || e.originalEvent.metaKey) {
        map.current!.getCanvas().style.cursor = 'copy';
      } else {
        map.current!.getCanvas().style.cursor = 'pointer';
      }
    });
  };

  // Update map data when store changes
  useEffect(() => {
    if (!map.current || !mapLoaded) return;

    // Update walls
    const wallsFeatures = walls.map((wall, index) => ({
      type: 'Feature' as const,
      properties: { id: index },
      geometry: {
        type: 'Point' as const,
        coordinates: [wall.lng, wall.lat],
      },
    }));

    (map.current.getSource('walls') as mapboxgl.GeoJSONSource)?.setData({
      type: 'FeatureCollection',
      features: wallsFeatures,
    });
  }, [walls, mapLoaded]);

  useEffect(() => {
    if (!map.current || !mapLoaded) return;

    // Update visited nodes
    const visitedFeatures = visitedNodes.map((node, index) => ({
      type: 'Feature' as const,
      properties: { id: index },
      geometry: {
        type: 'Point' as const,
        coordinates: [node.lng, node.lat],
      },
    }));

    (map.current.getSource('visited-nodes') as mapboxgl.GeoJSONSource)?.setData({
      type: 'FeatureCollection',
      features: visitedFeatures,
    });
  }, [visitedNodes, mapLoaded]);

  useEffect(() => {
    if (!map.current || !mapLoaded) return;

    // Update path nodes
    const pathFeatures = pathNodes.map((node, index) => ({
      type: 'Feature' as const,
      properties: { id: index },
      geometry: {
        type: 'Point' as const,
        coordinates: [node.lng, node.lat],
      },
    }));

    (map.current.getSource('path-nodes') as mapboxgl.GeoJSONSource)?.setData({
      type: 'FeatureCollection',
      features: pathFeatures,
    });

    // Update path line
    if (pathNodes.length > 1) {
      const coordinates = pathNodes.map(node => [node.lng, node.lat]);
      (map.current.getSource('path-line') as mapboxgl.GeoJSONSource)?.setData({
        type: 'Feature',
        properties: {},
        geometry: {
          type: 'LineString',
          coordinates,
        },
      });
    }
  }, [pathNodes, mapLoaded]);

  useEffect(() => {
    if (!map.current || !mapLoaded) return;

    // Update start and end points
    const startEndFeatures = [];
    
    if (startPoint) {
      startEndFeatures.push({
        type: 'Feature' as const,
        properties: { 
          id: 'start',
          color: '#F59E0B',
          type: 'start'
        },
        geometry: {
          type: 'Point' as const,
          coordinates: [startPoint.lng, startPoint.lat],
        },
      });
    }

    if (endPoint) {
      startEndFeatures.push({
        type: 'Feature' as const,
        properties: { 
          id: 'end',
          color: '#EF4444',
          type: 'end'
        },
        geometry: {
          type: 'Point' as const,
          coordinates: [endPoint.lng, endPoint.lat],
        },
      });
    }

    (map.current.getSource('start-end-points') as mapboxgl.GeoJSONSource)?.setData({
      type: 'FeatureCollection',
      features: startEndFeatures,
    });
  }, [startPoint, endPoint, mapLoaded]);

  if (mapError) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className={`flex items-center justify-center bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg ${className}`}
      >
        <div className="text-center p-8">
          <div className="text-red-500 dark:text-red-400 mb-4">
            <Zap className="w-12 h-12 mx-auto" />
          </div>
          <h3 className="text-lg font-semibold text-red-800 dark:text-red-200 mb-2">
            Map Error
          </h3>
          <p className="text-red-600 dark:text-red-300 text-sm">
            {mapError}
          </p>
        </div>
      </motion.div>
    );
  }

  return (
    <div className={`relative ${className}`}>
      <div ref={mapContainer} className="w-full h-full rounded-lg overflow-hidden" />
      
      {!mapLoaded && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 flex items-center justify-center bg-gray-900/80 backdrop-blur-sm rounded-lg"
        >
          <div className="text-center">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              className="text-white mb-4"
            >
              <Navigation className="w-8 h-8 mx-auto" />
            </motion.div>
            <p className="text-white text-sm">Loading map...</p>
          </div>
        </motion.div>
      )}

      {mapLoaded && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="absolute bottom-4 left-4 bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm rounded-lg p-3 text-xs"
        >
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <MapPin className="w-3 h-3 text-amber-500" />
              <span>Click to place start point</span>
            </div>
            <div className="flex items-center gap-2">
              <MapPin className="w-3 h-3 text-red-500" />
              <span>Shift + Click to place end point</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-red-500 rounded-full" />
              <span>Ctrl/Cmd + Click to toggle walls</span>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
}
