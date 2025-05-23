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

// Validate Mapbox token
export function validateMapboxToken(token: string | undefined): boolean {
  return token ? token.startsWith('pk.') && token.length > 50 : false;
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
