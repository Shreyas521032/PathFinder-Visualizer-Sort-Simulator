// components/MapView.tsx
'use client';

import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import dynamic from 'next/dynamic';
import { MapPin, Navigation, Search, Zap } from 'lucide-react';
import { Point, PathNode } from '@/types';
import { usePathfindingStore } from '@/store';
import { geocodeAddress } from '@/lib/utils';

// Dynamically import Leaflet components to avoid SSR issues
const MapContainer = dynamic(
  () => import('react-leaflet').then((mod) => mod.MapContainer),
  { ssr: false }
);
const TileLayer = dynamic(
  () => import('react-leaflet').then((mod) => mod.TileLayer),
  { ssr: false }
);
const Marker = dynamic(
  () => import('react-leaflet').then((mod) => mod.Marker),
  { ssr: false }
);
const Popup = dynamic(
  () => import('react-leaflet').then((mod) => mod.Popup),
  { ssr: false }
);
const Polyline = dynamic(
  () => import('react-leaflet').then((mod) => mod.Polyline),
  { ssr: false }
);
const CircleMarker = dynamic(
  () => import('react-leaflet').then((mod) => mod.CircleMarker),
  { ssr: false }
);

interface MapViewProps {
  className?: string;
}

// Custom hook for Leaflet map events
function MapEvents() {
  const {
    walls,
    isRunning,
    setStartPoint,
    setEndPoint,
    addWall,
    removeWall,
  } = usePathfindingStore();

  const { useMapEvents } = require('react-leaflet');

  const map = useMapEvents({
    click(e) {
      if (isRunning) return;

      const point: Point = {
        lat: e.latlng.lat,
        lng: e.latlng.lng,
      };

      const originalEvent = (e as any).originalEvent;

      if (originalEvent?.shiftKey) {
        // Shift + click to place end point
        setEndPoint(point);
      } else if (originalEvent?.ctrlKey || originalEvent?.metaKey) {
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
    },
  });

  return null;
}

export default function MapView({ className }: MapViewProps) {
  const [mapLoaded, setMapLoaded] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const mapRef = useRef<any>(null);

  const {
    startPoint,
    endPoint,
    walls,
    visitedNodes,
    pathNodes,
    isRunning,
  } = usePathfindingStore();

  // Default center (New York City)
  const defaultCenter: [number, number] = [40.7128, -74.0060];

  useEffect(() => {
    // Import Leaflet CSS
    if (typeof window !== 'undefined') {
      import('leaflet/dist/leaflet.css');
      setMapLoaded(true);

      // Fix Leaflet default markers
      const L = require('leaflet');
      delete L.Icon.Default.prototype._getIconUrl;
      L.Icon.Default.mergeOptions({
        iconRetinaUrl: '/leaflet-icons/marker-icon-2x.png',
        iconUrl: '/leaflet-icons/marker-icon.png',
        shadowUrl: '/leaflet-icons/marker-shadow.png',
      });
    }
  }, []);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    setIsSearching(true);
    try {
      const result = await geocodeAddress(searchQuery);
      if (result && mapRef.current) {
        mapRef.current.setView([result.lat, result.lng], 13);
      }
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsSearching(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  if (!mapLoaded) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className={`flex items-center justify-center bg-gray-100 dark:bg-gray-800 rounded-lg ${className}`}
      >
        <div className="text-center">
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            className="text-blue-600 dark:text-blue-400 mb-4"
          >
            <Navigation className="w-8 h-8 mx-auto" />
          </motion.div>
          <p className="text-gray-600 dark:text-gray-300 text-sm">Loading map...</p>
        </div>
      </motion.div>
    );
  }

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

  // Create custom icons
  const createCustomIcon = (color: string, size: number = 25) => {
    if (typeof window === 'undefined') return null;
    
    const L = require('leaflet');
    return L.divIcon({
      className: 'custom-div-icon',
      html: `<div style="background-color: ${color}; width: ${size}px; height: ${size}px; border-radius: 50%; border: 3px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.3);"></div>`,
      iconSize: [size, size],
      iconAnchor: [size / 2, size / 2],
    });
  };

  return (
    <div className={`relative ${className}`}>
      {/* Search Bar */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="absolute top-4 left-4 right-4 z-10 bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm rounded-lg p-3 shadow-lg"
      >
        <div className="flex items-center gap-2">
          <Search className="w-5 h-5 text-gray-500" />
          <input
            type="text"
            placeholder="Search for a location..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            className="flex-1 bg-transparent outline-none text-gray-900 dark:text-white placeholder-gray-500"
          />
          <motion.button
            onClick={handleSearch}
            disabled={isSearching}
            className="px-3 py-1 bg-blue-600 text-white rounded text-sm disabled:opacity-50"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isSearching ? '...' : 'Go'}
          </motion.button>
        </div>
      </motion.div>

      {/* Map Container */}
      <div className="w-full h-full rounded-lg overflow-hidden">
        <MapContainer
          center={defaultCenter}
          zoom={13}
          style={{ height: '100%', width: '100%' }}
          ref={mapRef}
        >
          {/* OpenStreetMap Tiles */}
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            maxZoom={19}
          />

          {/* Map Events Handler */}
          <MapEvents />

          {/* Start Point Marker */}
          {startPoint && (
            <Marker
              position={[startPoint.lat, startPoint.lng]}
              icon={createCustomIcon('#F59E0B', 30)}
            >
              <Popup>
                <div className="text-center">
                  <strong>Start Point</strong>
                  <br />
                  Lat: {startPoint.lat.toFixed(4)}
                  <br />
                  Lng: {startPoint.lng.toFixed(4)}
                </div>
              </Popup>
            </Marker>
          )}

          {/* End Point Marker */}
          {endPoint && (
            <Marker
              position={[endPoint.lat, endPoint.lng]}
              icon={createCustomIcon('#EF4444', 30)}
            >
              <Popup>
                <div className="text-center">
                  <strong>End Point</strong>
                  <br />
                  Lat: {endPoint.lat.toFixed(4)}
                  <br />
                  Lng: {endPoint.lng.toFixed(4)}
                </div>
              </Popup>
            </Marker>
          )}

          {/* Wall Markers */}
          {walls.map((wall, index) => (
            <CircleMarker
              key={index}
              center={[wall.lat, wall.lng]}
              radius={8}
              pathOptions={{
                color: '#DC2626',
                fillColor: '#EF4444',
                fillOpacity: 0.8,
                weight: 2,
              }}
            >
              <Popup>Wall</Popup>
            </CircleMarker>
          ))}

          {/* Visited Nodes */}
          {visitedNodes.map((node, index) => (
            <CircleMarker
              key={`visited-${index}`}
              center={[node.lat, node.lng]}
              radius={4}
              pathOptions={{
                color: '#2563EB',
                fillColor: '#3B82F6',
                fillOpacity: 0.6,
                weight: 1,
              }}
            />
          ))}

          {/* Path Nodes */}
          {pathNodes.map((node, index) => (
            <CircleMarker
              key={`path-${index}`}
              center={[node.lat, node.lng]}
              radius={6}
              pathOptions={{
                color: '#059669',
                fillColor: '#10B981',
                fillOpacity: 0.9,
                weight: 2,
              }}
            />
          ))}

          {/* Path Line */}
          {pathNodes.length > 1 && (
            <Polyline
              positions={pathNodes.map(node => [node.lat, node.lng] as [number, number])}
              pathOptions={{
                color: '#10B981',
                weight: 4,
                opacity: 0.8,
              }}
            />
          )}
        </MapContainer>
      </div>

      {/* Instructions Panel */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="absolute bottom-4 left-4 bg-white/90 dark:bg-gray-900/90 backdrop-blur-sm rounded-lg p-3 text-xs shadow-lg"
      >
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-amber-500 rounded-full"></div>
            <span>Click to place start point</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full"></div>
            <span>Shift + Click to place end point</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded-full border-2 border-red-600"></div>
            <span>Ctrl/Cmd + Click to toggle walls</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span>Visited nodes</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span>Path nodes</span>
          </div>
        </div>
      </motion.div>

      {/* Running Status */}
      {isRunning && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="absolute top-20 left-1/2 transform -translate-x-1/2 bg-blue-600 text-white px-4 py-2 rounded-lg shadow-lg z-10"
        >
          <div className="flex items-center gap-2">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            >
              <Navigation className="w-4 h-4" />
            </motion.div>
            <span className="text-sm font-medium">Finding path...</span>
          </div>
        </motion.div>
      )}
    </div>
  );
}
