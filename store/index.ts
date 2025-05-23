// store/index.ts
import { create } from 'zustand';
import { 
  PathfindingState, 
  SortingState, 
  Point, 
  PathNode, 
  PathfindingAlgorithm, 
  SortingAlgorithm,
  Theme 
} from '@/types';

// Pathfinding Store
interface PathfindingStore extends PathfindingState {
  setAlgorithm: (algorithm: PathfindingAlgorithm) => void;
  setSpeed: (speed: number) => void;
  setStartPoint: (point: Point | null) => void;
  setEndPoint: (point: Point | null) => void;
  setIsRunning: (isRunning: boolean) => void;
  setIsComplete: (isComplete: boolean) => void;
  setVisitedNodes: (nodes: PathNode[]) => void;
  setPathNodes: (nodes: PathNode[]) => void;
  addWall: (point: Point) => void;
  removeWall: (point: Point) => void;
  clearWalls: () => void;
  reset: () => void;
}

export const usePathfindingStore = create<PathfindingStore>((set, get) => ({
  algorithm: 'astar',
  speed: 50,
  isRunning: false,
  isComplete: false,
  startPoint: null,
  endPoint: null,
  visitedNodes: [],
  pathNodes: [],
  walls: [],

  setAlgorithm: (algorithm) => set({ algorithm }),
  setSpeed: (speed) => set({ speed }),
  setStartPoint: (startPoint) => set({ startPoint }),
  setEndPoint: (endPoint) => set({ endPoint }),
  setIsRunning: (isRunning) => set({ isRunning }),
  setIsComplete: (isComplete) => set({ isComplete }),
  setVisitedNodes: (visitedNodes) => set({ visitedNodes }),
  setPathNodes: (pathNodes) => set({ pathNodes }),
  
  addWall: (point) => {
    const { walls } = get();
    const exists = walls.some(w => w.lat === point.lat && w.lng === point.lng);
    if (!exists) {
      set({ walls: [...walls, point] });
    }
  },
  
  removeWall: (point) => {
    const { walls } = get();
    set({ 
      walls: walls.filter(w => !(w.lat === point.lat && w.lng === point.lng))
    });
  },
  
  clearWalls: () => set({ walls: [] }),
  
  reset: () => set({
    isRunning: false,
    isComplete: false,
    visitedNodes: [],
    pathNodes: [],
    startPoint: null,
    endPoint: null,
  }),
}));

// Sorting Store
interface SortingStore extends SortingState {
  setAlgorithm: (algorithm: SortingAlgorithm) => void;
  setArraySize: (size: number) => void;
  setSpeed: (speed: number) => void;
  setIsRunning: (isRunning: boolean) => void;
  setIsComplete: (isComplete: boolean) => void;
  setArray: (array: number[]) => void;
  setComparingIndices: (indices: number[]) => void;
  setSwappingIndices: (indices: number[]) => void;
  setSortedIndices: (indices: number[]) => void;
  generateRandomArray: () => void;
  reset: () => void;
}

export const useSortingStore = create<SortingStore>((set, get) => ({
  algorithm: 'bubble',
  arraySize: 50,
  speed: 100,
  isRunning: false,
  isComplete: false,
  array: [],
  comparingIndices: [],
  swappingIndices: [],
  sortedIndices: [],

  setAlgorithm: (algorithm) => set({ algorithm }),
  setArraySize: (arraySize) => set({ arraySize }),
  setSpeed: (speed) => set({ speed }),
  setIsRunning: (isRunning) => set({ isRunning }),
  setIsComplete: (isComplete) => set({ isComplete }),
  setArray: (array) => set({ array }),
  setComparingIndices: (comparingIndices) => set({ comparingIndices }),
  setSwappingIndices: (swappingIndices) => set({ swappingIndices }),
  setSortedIndices: (sortedIndices) => set({ sortedIndices }),

  generateRandomArray: () => {
    const { arraySize } = get();
    const array = Array.from({ length: arraySize }, () => 
      Math.floor(Math.random() * 400) + 10
    );
    set({ 
      array, 
      comparingIndices: [], 
      swappingIndices: [], 
      sortedIndices: [],
      isComplete: false 
    });
  },

  reset: () => {
    const { generateRandomArray } = get();
    set({
      isRunning: false,
      isComplete: false,
      comparingIndices: [],
      swappingIndices: [],
      sortedIndices: [],
    });
    generateRandomArray();
  },
}));

// Theme Store
interface ThemeStore {
  theme: Theme;
  toggleTheme: () => void;
  setTheme: (theme: Theme) => void;
}

export const useThemeStore = create<ThemeStore>((set, get) => ({
  theme: 'dark',
  
  toggleTheme: () => {
    const { theme } = get();
    const newTheme = theme === 'light' ? 'dark' : 'light';
    set({ theme: newTheme });
    
    // Update DOM class
    if (typeof window !== 'undefined') {
      document.documentElement.classList.toggle('dark', newTheme === 'dark');
      localStorage.setItem('theme', newTheme);
    }
  },
  
  setTheme: (theme) => {
    set({ theme });
    if (typeof window !== 'undefined') {
      document.documentElement.classList.toggle('dark', theme === 'dark');
      localStorage.setItem('theme', theme);
    }
  },
}));

// App Store for global state
interface AppStore {
  showOnboarding: boolean;
  currentOnboardingStep: number;
  isLoading: boolean;
  setShowOnboarding: (show: boolean) => void;
  setCurrentOnboardingStep: (step: number) => void;
  setIsLoading: (loading: boolean) => void;
  nextOnboardingStep: () => void;
  previousOnboardingStep: () => void;
}

export const useAppStore = create<AppStore>((set, get) => ({
  showOnboarding: false,
  currentOnboardingStep: 0,
  isLoading: false,

  setShowOnboarding: (showOnboarding) => set({ showOnboarding }),
  setCurrentOnboardingStep: (currentOnboardingStep) => set({ currentOnboardingStep }),
  setIsLoading: (isLoading) => set({ isLoading }),

  nextOnboardingStep: () => {
    const { currentOnboardingStep } = get();
    set({ currentOnboardingStep: currentOnboardingStep + 1 });
  },

  previousOnboardingStep: () => {
    const { currentOnboardingStep } = get();
    if (currentOnboardingStep > 0) {
      set({ currentOnboardingStep: currentOnboardingStep - 1 });
    }
  },
}));
