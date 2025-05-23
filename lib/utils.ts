// lib/utils.ts
import { type ClassValue, clsx } from 'clsx';
import { Point, PathNode, AlgorithmInfo } from '@/types';

export function cn(...inputs: ClassValue[]) {
  return clsx(inputs);
}

// Distance calculation utilities
export function calculateDistance(point1: Point, point2: Point): number {
  const R = 6371; // Earth's radius in kilometers
  const dLat = (point2.lat - point1.lat) * Math.PI / 180;
  const dLng = (point2.lng - point1.lng) * Math.PI / 180;
  const a = 
    Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(point1.lat * Math.PI / 180) * Math.cos(point2.lat * Math.PI / 180) *
    Math.sin(dLng/2) * Math.sin(dLng/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c;
}

export function calculateManhattanDistance(point1: Point, point2: Point): number {
  return Math.abs(point1.lat - point2.lat) + Math.abs(point1.lng - point2.lng);
}

export function calculateEuclideanDistance(point1: Point, point2: Point): number {
  return Math.sqrt(
    Math.pow(point2.lat - point1.lat, 2) + 
    Math.pow(point2.lng - point1.lng, 2)
  );
}

// Grid generation for pathfinding
export function generateGrid(
  center: Point, 
  radiusKm: number, 
  resolution: number = 20
): PathNode[][] {
  const grid: PathNode[][] = [];
  const latStep = (radiusKm * 2) / (resolution * 111); // Approximate degrees per km
  const lngStep = (radiusKm * 2) / (resolution * 111);
  
  for (let i = 0; i < resolution; i++) {
    const row: PathNode[] = [];
    for (let j = 0; j < resolution; j++) {
      const lat = center.lat - radiusKm/111 + (i * latStep);
      const lng = center.lng - radiusKm/111 + (j * lngStep);
      
      row.push({
        id: `${i}-${j}`,
        lat,
        lng,
        gCost: Infinity,
        hCost: 0,
        fCost: Infinity,
        isWall: Math.random() < 0.2, // 20% chance of wall
        isVisited: false,
        isPath: false,
      });
    }
    grid.push(row);
  }
  
  return grid;
}

// Pathfinding algorithm information
export const pathfindingAlgorithms: Record<string, AlgorithmInfo> = {
  astar: {
    name: 'A* Search',
    description: 'Optimal pathfinding using heuristic to guide search',
    timeComplexity: 'O(b^d)',
    spaceComplexity: 'O(b^d)',
    color: '#3B82F6',
    icon: '⭐',
  },
  dijkstra: {
    name: "Dijkstra's Algorithm",
    description: 'Finds shortest path by exploring all possibilities',
    timeComplexity: 'O((V + E) log V)',
    spaceComplexity: 'O(V)',
    color: '#EF4444',
    icon: '🎯',
  },
  bfs: {
    name: 'Breadth-First Search',
    description: 'Explores nodes level by level, guarantees shortest path',
    timeComplexity: 'O(V + E)',
    spaceComplexity: 'O(V)',
    color: '#10B981',
    icon: '📡',
  },
  dfs: {
    name: 'Depth-First Search',
    description: 'Explores as far as possible before backtracking',
    timeComplexity: 'O(V + E)',
    spaceComplexity: 'O(V)',
    color: '#F59E0B',
    icon: '🔍',
  },
  greedy: {
    name: 'Greedy Best-First',
    description: 'Uses heuristic to guide search, not always optimal',
    timeComplexity: 'O(b^m)',
    spaceComplexity: 'O(b^m)',
    color: '#8B5CF6',
    icon: '⚡',
  },
  bidirectional: {
    name: 'Bidirectional Search',
    description: 'Searches from both start and end simultaneously',
    timeComplexity: 'O(b^(d/2))',
    spaceComplexity: 'O(b^(d/2))',
    color: '#EC4899',
    icon: '↔️',
  },
};

// Sorting algorithm information
export const sortingAlgorithms: Record<string, AlgorithmInfo> = {
  bubble: {
    name: 'Bubble Sort',
    description: 'Repeatedly steps through list, compares adjacent elements',
    timeComplexity: 'O(n²)',
    spaceComplexity: 'O(1)',
    color: '#3B82F6',
    icon: '🫧',
  },
  merge: {
    name: 'Merge Sort',
    description: 'Divide and conquer algorithm that splits and merges',
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(n)',
    color: '#10B981',
    icon: '🔀',
  },
  quick: {
    name: 'Quick Sort',
    description: 'Picks pivot and partitions array around it',
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(log n)',
    color: '#EF4444',
    icon: '⚡',
  },
  heap: {
    name: 'Heap Sort',
    description: 'Uses heap data structure to sort elements',
    timeComplexity: 'O(n log n)',
    spaceComplexity: 'O(1)',
    color: '#F59E0B',
    icon: '🏔️',
  },
  insertion: {
    name: 'Insertion Sort',
    description: 'Builds sorted array one element at a time',
    timeComplexity: 'O(n²)',
    spaceComplexity: 'O(1)',
    color: '#8B5CF6',
    icon: '📥',
  },
  selection: {
    name: 'Selection Sort',
    description: 'Finds minimum element and places it at beginning',
    timeComplexity: 'O(n²)',
    spaceComplexity: 'O(1)',
    color: '#EC4899',
    icon: '👆',
  },
  radix: {
    name: 'Radix Sort',
    description: 'Non-comparative sorting using digit by digit sorting',
    timeComplexity: 'O(d × n)',
    spaceComplexity: 'O(n + k)',
    color: '#06B6D4',
    icon: '🔢',
  },
};

// Delay function for animations
export function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Generate random array for sorting
export function generateRandomArray(size: number, min: number = 10, max: number = 400): number[] {
  return Array.from({ length: size }, () => 
    Math.floor(Math.random() * (max - min + 1)) + min
  );
}

// Color utilities
export function getBarColor(
  index: number,
  isComparing: boolean,
  isSwapping: boolean,
  isSorted: boolean
): string {
  if (isSorted) return '#10B981'; // Green for sorted
  if (isSwapping) return '#EF4444'; // Red for swapping
  if (isComparing) return '#F59E0B'; // Yellow for comparing
  return '#6B7280'; // Gray for default
}

// Format time duration
export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
}

// Geocoding with Nominatim (OpenStreetMap)
export async function geocodeAddress(address: string): Promise<{
  lat: number;
  lng: number;
  display_name: string;
  bbox?: [number, number, number, number];
} | null> {
  try {
    const response = await fetch(
      `https://nominatim.openstreetmap.org/search?` +
      `format=json&q=${encodeURIComponent(address)}&limit=1&addressdetails=1`,
      {
        headers: {
          'User-Agent': 'PathFinder-Visualizer/1.0'
        }
      }
    );
    
    const data = await response.json();
    
    if (data && data.length > 0) {
      const result = data[0];
      return {
        lat: parseFloat(result.lat),
        lng: parseFloat(result.lon),
        display_name: result.display_name,
        bbox: result.boundingbox ? [
          parseFloat(result.boundingbox[2]), // west
          parseFloat(result.boundingbox[0]), // south
          parseFloat(result.boundingbox[3]), // east
          parseFloat(result.boundingbox[1])  // north
        ] : undefined,
      };
    }
    
    return null;
  } catch (error) {
    console.error('Geocoding error:', error);
    return null;
  }
}

// Reverse geocoding with Nominatim
export async function reverseGeocode(lat: number, lng: number): Promise<{
  display_name: string;
  address?: {
    house_number?: string;
    road?: string;
    city?: string;
    state?: string;
    country?: string;
    postcode?: string;
  };
} | null> {
  try {
    const response = await fetch(
      `https://nominatim.openstreetmap.org/reverse?` +
      `format=json&lat=${lat}&lon=${lng}&addressdetails=1`,
      {
        headers: {
          'User-Agent': 'PathFinder-Visualizer/1.0'
        }
      }
    );
    
    const data = await response.json();
    
    if (data && data.display_name) {
      return {
        display_name: data.display_name,
        address: data.address,
      };
    }
    
    return null;
  } catch (error) {
    console.error('Reverse geocoding error:', error);
    return null;
  }
}

// Batch geocoding with rate limiting
export async function batchGeocode(addresses: string[]): Promise<Array<{
  address: string;
  result: {
    lat: number;
    lng: number;
    display_name: string;
  } | null;
  success: boolean;
  error?: string;
}>> {
  const results = [];
  
  for (let i = 0; i < addresses.length; i++) {
    const address = addresses[i];
    
    try {
      const result = await geocodeAddress(address);
      
      results.push({
        address,
        result: result ? {
          lat: result.lat,
          lng: result.lng,
          display_name: result.display_name,
        } : null,
        success: !!result,
        error: result ? undefined : 'No results found',
      });
      
      // Rate limiting: 1 request per second for Nominatim
      if (i < addresses.length - 1) {
        await delay(1000);
      }
      
    } catch (error) {
      results.push({
        address,
        result: null,
        success: false,
        error: 'Geocoding failed',
      });
    }
  }
  
  return results;
}

// Search suggestions using Nominatim
export async function getSearchSuggestions(query: string, limit: number = 5): Promise<Array<{
  display_name: string;
  lat: number;
  lng: number;
  type: string;
  importance: number;
}>> {
  if (query.length < 3) return [];
  
  try {
    const response = await fetch(
      `https://nominatim.openstreetmap.org/search?` +
      `format=json&q=${encodeURIComponent(query)}&limit=${limit}&addressdetails=1`,
      {
        headers: {
          'User-Agent': 'PathFinder-Visualizer/1.0'
        }
      }
    );
    
    const data = await response.json();
    
    return data.map((item: any) => ({
      display_name: item.display_name,
      lat: parseFloat(item.lat),
      lng: parseFloat(item.lon),
      type: item.type || 'unknown',
      importance: item.importance || 0,
    }));
    
  } catch (error) {
    console.error('Search suggestions error:', error);
    return [];
  }
}

// Check if point is within bounds
export function isPointInBounds(point: Point, bounds: {
  north: number;
  south: number;
  east: number;
  west: number;
}): boolean {
  return (
    point.lat >= bounds.south &&
    point.lat <= bounds.north &&
    point.lng >= bounds.west &&
    point.lng <= bounds.east
  );
}

// Calculate map bounds from points
export function calculateBounds(points: Point[]): {
  north: number;
  south: number;
  east: number;
  west: number;
} | null {
  if (points.length === 0) return null;
  
  let north = points[0].lat;
  let south = points[0].lat;
  let east = points[0].lng;
  let west = points[0].lng;
  
  points.forEach(point => {
    north = Math.max(north, point.lat);
    south = Math.min(south, point.lat);
    east = Math.max(east, point.lng);
    west = Math.min(west, point.lng);
  });
  
  return { north, south, east, west };
}

// Generate random points within bounds
export function generateRandomPoints(
  bounds: { north: number; south: number; east: number; west: number },
  count: number
): Point[] {
  const points: Point[] = [];
  
  for (let i = 0; i < count; i++) {
    points.push({
      lat: bounds.south + Math.random() * (bounds.north - bounds.south),
      lng: bounds.west + Math.random() * (bounds.east - bounds.west),
    });
  }
  
  return points;
}

// API helpers
export async function fetchApi<T>(
  endpoint: string, 
  options: RequestInit = {}
): Promise<T> {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';
  const url = `${apiUrl}${endpoint}`;
  
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return response.json();
}

// Save simulation data to API
export async function saveSimulation(simulationData: {
  type: 'pathfinding' | 'sorting';
  algorithm: string;
  startTime: Date;
  endTime: Date;
  duration: number;
  steps: number;
  metadata?: any;
}): Promise<{ id: string } | null> {
  try {
    const response = await fetchApi<{ success: boolean; data: { id: string } }>('/api/simulations/save', {
      method: 'POST',
      body: JSON.stringify({
        ...simulationData,
        startTime: simulationData.startTime.toISOString(),
        endTime: simulationData.endTime.toISOString(),
      }),
    });
    
    return response.success ? response.data : null;
  } catch (error) {
    console.error('Error saving simulation:', error);
    return null;
  }
}

// Get simulation statistics from API
export async function getSimulationStats(type?: 'pathfinding' | 'sorting'): Promise<any> {
  try {
    const endpoint = type ? `/api/simulations/stats?type=${type}` : '/api/simulations/stats';
    const response = await fetchApi<{ success: boolean; data: any }>(endpoint);
    
    return response.success ? response.data : null;
  } catch (error) {
    console.error('Error fetching simulation stats:', error);
    return null;
  }
}

// Coordinate validation
export function validateCoordinates(lat: number, lng: number): boolean {
  return (
    typeof lat === 'number' &&
    typeof lng === 'number' &&
    lat >= -90 &&
    lat <= 90 &&
    lng >= -180 &&
    lng <= 180 &&
    !isNaN(lat) &&
    !isNaN(lng)
  );
}

// Format coordinates for display
export function formatCoordinates(lat: number, lng: number, precision: number = 4): string {
  return `${lat.toFixed(precision)}, ${lng.toFixed(precision)}`;
}

// Local storage helpers
export function saveToLocalStorage(key: string, value: any): void {
  if (typeof window !== 'undefined') {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error('Error saving to localStorage:', error);
    }
  }
}

export function loadFromLocalStorage<T>(key: string, defaultValue: T): T {
  if (typeof window !== 'undefined') {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.error('Error loading from localStorage:', error);
      return defaultValue;
    }
  }
  return defaultValue;
}

// Theme utilities
export function getSystemTheme(): 'light' | 'dark' {
  if (typeof window !== 'undefined') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  return 'light';
}

// Debounce function for search
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(null, args), wait);
  };
}

// Throttle function for performance
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  limit: number
): (...args: Parameters<T>) => void {
  let inThrottle: boolean;
  
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func.apply(null, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

// Random ID generator
export function generateId(length: number = 8): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

// Color manipulation utilities
export function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null;
}

export function rgbToHex(r: number, g: number, b: number): string {
  return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

// Performance monitoring
export function measurePerformance<T>(
  name: string,
  fn: () => T
): { result: T; duration: number } {
  const start = performance.now();
  const result = fn();
  const end = performance.now();
  const duration = end - start;
  
  console.log(`${name} took ${duration.toFixed(2)}ms`);
  
  return { result, duration };
}

// URL utilities
export function updateURLParams(params: Record<string, string>): void {
  if (typeof window !== 'undefined') {
    const url = new URL(window.location.href);
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.set(key, value);
    });
    window.history.replaceState({}, '', url.toString());
  }
}

export function getURLParam(key: string): string | null {
  if (typeof window !== 'undefined') {
    const params = new URLSearchParams(window.location.search);
    return params.get(key);
  }
  return null;
} with Nominatim (OpenStreetMap)
export async function geocodeAddress(address: string): Promise<{
  lat: number;
  lng: number;
  display_name: string;
} | null> {
  try {
    const response = await fetch(
      `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(address)}&limit=1`
    );
    const data = await response.json();
    
    if (data && data.length > 0) {
      return {
        lat: parseFloat(data[0].lat),
        lng: parseFloat(data[0].lon),
        display_name: data[0].display_name,
      };
    }
    
    return null;
  } catch (error) {
    console.error('Geocoding error:', error);
    return null;
  }
}

// Reverse geocoding with Nominatim
export async function reverseGeocode(lat: number, lng: number): Promise<{
  display_name: string;
} | null> {
  try {
    const response = await fetch(
      `https://nominatim.openstreetmap.org/reverse?format=json&lat=${lat}&lon=${lng}`
    );
    const data = await response.json();
    
    if (data && data.display_name) {
      return {
        display_name: data.display_name,
      };
    }
    
    return null;
  } catch (error) {
    console.error('Reverse geocoding error:', error);
    return null;
  }
}

// Local storage helpers
export function saveToLocalStorage(key: string, value: any): void {
  if (typeof window !== 'undefined') {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error('Error saving to localStorage:', error);
    }
  }
}

export function loadFromLocalStorage<T>(key: string, defaultValue: T): T {
  if (typeof window !== 'undefined') {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.error('Error loading from localStorage:', error);
      return defaultValue;
    }
  }
  return defaultValue;
}console.error('Error saving to localStorage:', error);
    }
  }
}

export function loadFromLocalStorage<T>(key: string, defaultValue: T): T {
  if (typeof window !== 'undefined') {
    try {
      const item = localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.error('Error loading from localStorage:', error);
      return defaultValue;
    }
  }
  return defaultValue;
}
