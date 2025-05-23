// types/index.ts
export interface Point {
  lat: number;
  lng: number;
}

export interface PathNode extends Point {
  id: string;
  gCost?: number;
  hCost?: number;
  fCost?: number;
  parent?: PathNode;
  isWall?: boolean;
  isVisited?: boolean;
  isPath?: boolean;
  isStart?: boolean;
  isEnd?: boolean;
}

export type PathfindingAlgorithm = 
  | 'astar' 
  | 'dijkstra' 
  | 'bfs' 
  | 'dfs' 
  | 'greedy' 
  | 'bidirectional';

export type SortingAlgorithm = 
  | 'bubble' 
  | 'merge' 
  | 'quick' 
  | 'heap' 
  | 'insertion' 
  | 'selection' 
  | 'radix';

export interface PathfindingState {
  algorithm: PathfindingAlgorithm;
  speed: number;
  isRunning: boolean;
  isComplete: boolean;
  startPoint: Point | null;
  endPoint: Point | null;
  visitedNodes: PathNode[];
  pathNodes: PathNode[];
  walls: Point[];
}

export interface SortingState {
  algorithm: SortingAlgorithm;
  arraySize: number;
  speed: number;
  isRunning: boolean;
  isComplete: boolean;
  array: number[];
  comparingIndices: number[];
  swappingIndices: number[];
  sortedIndices: number[];
}

export interface SimulationData {
  id: string;
  type: 'pathfinding' | 'sorting';
  algorithm: string;
  startTime: Date;
  endTime: Date;
  duration: number;
  steps: number;
  metadata?: any;
}

export interface AlgorithmInfo {
  name: string;
  description: string;
  timeComplexity: string;
  spaceComplexity: string;
  color: string;
  icon: string;
}

export interface TooltipStep {
  target: string;
  content: string;
  position: 'top' | 'bottom' | 'left' | 'right';
}

export interface GeocodeResponse {
  features: Array<{
    center: [number, number];
    place_name: string;
  }>;
}

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// Animation types for Framer Motion
export interface AnimationProps {
  initial?: any;
  animate?: any;
  exit?: any;
  transition?: any;
  variants?: any;
}

// Theme types
export type Theme = 'light' | 'dark';

export interface ThemeState {
  theme: Theme;
  toggleTheme: () => void;
}
