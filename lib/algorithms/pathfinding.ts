// lib/algorithms/pathfinding.ts
import { Point, PathNode } from '@/types';
import { calculateEuclideanDistance, calculateManhattanDistance } from '@/lib/utils';

export interface PathfindingResult {
  visitedNodes: PathNode[];
  path: PathNode[];
  steps: number;
}

// A* Algorithm
export async function aStar(
  start: Point,
  end: Point,
  walls: Point[],
  onProgress?: (visitedNodes: PathNode[], currentPath: PathNode[]) => void
): Promise<PathfindingResult> {
  const grid = createGrid(start, end, walls);
  const openSet: PathNode[] = [];
  const closedSet: PathNode[] = [];
  const visitedNodes: PathNode[] = [];
  
  const startNode = grid.find(node => 
    Math.abs(node.lat - start.lat) < 0.001 && 
    Math.abs(node.lng - start.lng) < 0.001
  );
  
  const endNode = grid.find(node => 
    Math.abs(node.lat - end.lat) < 0.001 && 
    Math.abs(node.lng - end.lng) < 0.001
  );
  
  if (!startNode || !endNode) {
    return { visitedNodes: [], path: [], steps: 0 };
  }
  
  startNode.gCost = 0;
  startNode.hCost = calculateEuclideanDistance(startNode, endNode);
  startNode.fCost = startNode.gCost + startNode.hCost;
  
  openSet.push(startNode);
  let steps = 0;
  
  while (openSet.length > 0) {
    steps++;
    
    // Find node with lowest fCost
    let currentNode = openSet[0];
    for (let i = 1; i < openSet.length; i++) {
      if (openSet[i].fCost! < currentNode.fCost! || 
          (openSet[i].fCost === currentNode.fCost && openSet[i].hCost! < currentNode.hCost!)) {
        currentNode = openSet[i];
      }
    }
    
    openSet.splice(openSet.indexOf(currentNode), 1);
    closedSet.push(currentNode);
    visitedNodes.push({ ...currentNode });
    
    // Found the target
    if (currentNode === endNode) {
      const path = reconstructPath(currentNode);
      return { visitedNodes, path, steps };
    }
    
    // Check neighbors
    const neighbors = getNeighbors(currentNode, grid);
    for (const neighbor of neighbors) {
      if (closedSet.includes(neighbor) || neighbor.isWall) continue;
      
      const tentativeGCost = currentNode.gCost! + calculateEuclideanDistance(currentNode, neighbor);
      
      if (!openSet.includes(neighbor)) {
        openSet.push(neighbor);
      } else if (tentativeGCost >= neighbor.gCost!) {
        continue;
      }
      
      neighbor.parent = currentNode;
      neighbor.gCost = tentativeGCost;
      neighbor.hCost = calculateEuclideanDistance(neighbor, endNode);
      neighbor.fCost = neighbor.gCost + neighbor.hCost;
    }
    
    if (onProgress && steps % 10 === 0) {
      const currentPath = currentNode.parent ? reconstructPath(currentNode) : [];
      onProgress([...visitedNodes], currentPath);
      await new Promise(resolve => setTimeout(resolve, 50));
    }
  }
  
  return { visitedNodes, path: [], steps };
}

// Dijkstra's Algorithm
export async function dijkstra(
  start: Point,
  end: Point,
  walls: Point[],
  onProgress?: (visitedNodes: PathNode[], currentPath: PathNode[]) => void
): Promise<PathfindingResult> {
  const grid = createGrid(start, end, walls);
  const unvisited: PathNode[] = [...grid];
  const visitedNodes: PathNode[] = [];
  
  const startNode = grid.find(node => 
    Math.abs(node.lat - start.lat) < 0.001 && 
    Math.abs(node.lng - start.lng) < 0.001
  );
  
  const endNode = grid.find(node => 
    Math.abs(node.lat - end.lat) < 0.001 && 
    Math.abs(node.lng - end.lng) < 0.001
  );
  
  if (!startNode || !endNode) {
    return { visitedNodes: [], path: [], steps: 0 };
  }
  
  startNode.gCost = 0;
  let steps = 0;
  
  while (unvisited.length > 0) {
    steps++;
    
    // Sort by distance and get closest unvisited node
    unvisited.sort((a, b) => a.gCost! - b.gCost!);
    const currentNode = unvisited.shift()!;
    
    if (currentNode.gCost === Infinity) break;
    
    currentNode.isVisited = true;
    visitedNodes.push({ ...currentNode });
    
    if (currentNode === endNode) {
      const path = reconstructPath(currentNode);
      return { visitedNodes, path, steps };
    }
    
    const neighbors = getNeighbors(currentNode, grid);
    for (const neighbor of neighbors) {
      if (neighbor.isVisited || neighbor.isWall) continue;
      
      const distance = currentNode.gCost! + calculateEuclideanDistance(currentNode, neighbor);
      if (distance < neighbor.gCost!) {
        neighbor.gCost = distance;
        neighbor.parent = currentNode;
      }
    }
    
    if (onProgress && steps % 5 === 0) {
      const currentPath = currentNode.parent ? reconstructPath(currentNode) : [];
      onProgress([...visitedNodes], currentPath);
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
  
  return { visitedNodes, path: [], steps };
}

// Breadth-First Search
export async function bfs(
  start: Point,
  end: Point,
  walls: Point[],
  onProgress?: (visitedNodes: PathNode[], currentPath: PathNode[]) => void
): Promise<PathfindingResult> {
  const grid = createGrid(start, end, walls);
  const queue: PathNode[] = [];
  const visitedNodes: PathNode[] = [];
  
  const startNode = grid.find(node => 
    Math.abs(node.lat - start.lat) < 0.001 && 
    Math.abs(node.lng - start.lng) < 0.001
  );
  
  const endNode = grid.find(node => 
    Math.abs(node.lat - end.lat) < 0.001 && 
    Math.abs(node.lng - end.lng) < 0.001
  );
  
  if (!startNode || !endNode) {
    return { visitedNodes: [], path: [], steps: 0 };
  }
  
  queue.push(startNode);
  startNode.isVisited = true;
  let steps = 0;
  
  while (queue.length > 0) {
    steps++;
    const currentNode = queue.shift()!;
    visitedNodes.push({ ...currentNode });
    
    if (currentNode === endNode) {
      const path = reconstructPath(currentNode);
      return { visitedNodes, path, steps };
    }
    
    const neighbors = getNeighbors(currentNode, grid);
    for (const neighbor of neighbors) {
      if (neighbor.isVisited || neighbor.isWall) continue;
      
      neighbor.isVisited = true;
      neighbor.parent = currentNode;
      queue.push(neighbor);
    }
    
    if (onProgress && steps % 5 === 0) {
      onProgress([...visitedNodes], []);
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
  
  return { visitedNodes, path: [], steps };
}

// Depth-First Search
export async function dfs(
  start: Point,
  end: Point,
  walls: Point[],
  onProgress?: (visitedNodes: PathNode[], currentPath: PathNode[]) => void
): Promise<PathfindingResult> {
  const grid = createGrid(start, end, walls);
  const stack: PathNode[] = [];
  const visitedNodes: PathNode[] = [];
  
  const startNode = grid.find(node => 
    Math.abs(node.lat - start.lat) < 0.001 && 
    Math.abs(node.lng - start.lng) < 0.001
  );
  
  const endNode = grid.find(node => 
    Math.abs(node.lat - end.lat) < 0.001 && 
    Math.abs(node.lng - end.lng) < 0.001
  );
  
  if (!startNode || !endNode) {
    return { visitedNodes: [], path: [], steps: 0 };
  }
  
  stack.push(startNode);
  let steps = 0;
  
  while (stack.length > 0) {
    steps++;
    const currentNode = stack.pop()!;
    
    if (currentNode.isVisited) continue;
    
    currentNode.isVisited = true;
    visitedNodes.push({ ...currentNode });
    
    if (currentNode === endNode) {
      const path = reconstructPath(currentNode);
      return { visitedNodes, path, steps };
    }
    
    const neighbors = getNeighbors(currentNode, grid);
    for (const neighbor of neighbors) {
      if (neighbor.isVisited || neighbor.isWall) continue;
      neighbor.parent = currentNode;
      stack.push(neighbor);
    }
    
    if (onProgress && steps % 5 === 0) {
      onProgress([...visitedNodes], []);
      await new Promise(resolve => setTimeout(resolve, 150));
    }
  }
  
  return { visitedNodes, path: [], steps };
}

// Greedy Best-First Search
export async function greedyBestFirst(
  start: Point,
  end: Point,
  walls: Point[],
  onProgress?: (visitedNodes: PathNode[], currentPath: PathNode[]) => void
): Promise<PathfindingResult> {
  const grid = createGrid(start, end, walls);
  const openSet: PathNode[] = [];
  const visitedNodes: PathNode[] = [];
  
  const startNode = grid.find(node => 
    Math.abs(node.lat - start.lat) < 0.001 && 
    Math.abs(node.lng - start.lng) < 0.001
  );
  
  const endNode = grid.find(node => 
    Math.abs(node.lat - end.lat) < 0.001 && 
    Math.abs(node.lng - end.lng) < 0.001
  );
  
  if (!startNode || !endNode) {
    return { visitedNodes: [], path: [], steps: 0 };
  }
  
  startNode.hCost = calculateEuclideanDistance(startNode, endNode);
  openSet.push(startNode);
  let steps = 0;
  
  while (openSet.length > 0) {
    steps++;
    
    // Sort by heuristic cost
    openSet.sort((a, b) => a.hCost! - b.hCost!);
    const currentNode = openSet.shift()!;
    
    currentNode.isVisited = true;
    visitedNodes.push({ ...currentNode });
    
    if (currentNode === endNode) {
      const path = reconstructPath(currentNode);
      return { visitedNodes, path, steps };
    }
    
    const neighbors = getNeighbors(currentNode, grid);
    for (const neighbor of neighbors) {
      if (neighbor.isVisited || neighbor.isWall) continue;
      
      neighbor.hCost = calculateEuclideanDistance(neighbor, endNode);
      neighbor.parent = currentNode;
      openSet.push(neighbor);
    }
    
    if (onProgress && steps % 5 === 0) {
      const currentPath = currentNode.parent ? reconstructPath(currentNode) : [];
      onProgress([...visitedNodes], currentPath);
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
  
  return { visitedNodes, path: [], steps };
}

// Bidirectional Search
export async function bidirectionalSearch(
  start: Point,
  end: Point,
  walls: Point[],
  onProgress?: (visitedNodes: PathNode[], currentPath: PathNode[]) => void
): Promise<PathfindingResult> {
  const grid = createGrid(start, end, walls);
  const startQueue: PathNode[] = [];
  const endQueue: PathNode[] = [];
  const visitedFromStart = new Set<PathNode>();
  const visitedFromEnd = new Set<PathNode>();
  const visitedNodes: PathNode[] = [];
  
  const startNode = grid.find(node => 
    Math.abs(node.lat - start.lat) < 0.001 && 
    Math.abs(node.lng - start.lng) < 0.001
  );
  
  const endNode = grid.find(node => 
    Math.abs(node.lat - end.lat) < 0.001 && 
    Math.abs(node.lng - end.lng) < 0.001
  );
  
  if (!startNode || !endNode) {
    return { visitedNodes: [], path: [], steps: 0 };
  }
  
  startQueue.push(startNode);
  endQueue.push(endNode);
  visitedFromStart.add(startNode);
  visitedFromEnd.add(endNode);
  
  let steps = 0;
  
  while (startQueue.length > 0 && endQueue.length > 0) {
    steps++;
    
    // Expand from start
    if (startQueue.length > 0) {
      const currentStart = startQueue.shift()!;
      visitedNodes.push({ ...currentStart });
      
      const neighbors = getNeighbors(currentStart, grid);
      for (const neighbor of neighbors) {
        if (neighbor.isWall) continue;
        
        if (visitedFromEnd.has(neighbor)) {
          // Found intersection
          const pathFromStart = reconstructPath(currentStart);
          const pathFromEnd = reconstructPath(neighbor).reverse();
          const fullPath = [...pathFromStart, ...pathFromEnd];
          return { visitedNodes, path: fullPath, steps };
        }
        
        if (!visitedFromStart.has(neighbor)) {
          neighbor.parent = currentStart;
          visitedFromStart.add(neighbor);
          startQueue.push(neighbor);
        }
      }
    }
    
    // Expand from end
    if (endQueue.length > 0) {
      const currentEnd = endQueue.shift()!;
      visitedNodes.push({ ...currentEnd });
      
      const neighbors = getNeighbors(currentEnd, grid);
      for (const neighbor of neighbors) {
        if (neighbor.isWall) continue;
        
        if (visitedFromStart.has(neighbor)) {
          // Found intersection
          const pathFromStart = reconstructPath(neighbor);
          const pathFromEnd = reconstructPath(currentEnd).reverse();
          const fullPath = [...pathFromStart, ...pathFromEnd];
          return { visitedNodes, path: fullPath, steps };
        }
        
        if (!visitedFromEnd.has(neighbor)) {
          neighbor.parent = currentEnd;
          visitedFromEnd.add(neighbor);
          endQueue.push(neighbor);
        }
      }
    }
    
    if (onProgress && steps % 5 === 0) {
      onProgress([...visitedNodes], []);
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
  
  return { visitedNodes, path: [], steps };
}

// Helper functions
function createGrid(start: Point, end: Point, walls: Point[]): PathNode[] {
  const grid: PathNode[] = [];
  const resolution = 30;
  
  // Calculate bounds
  const minLat = Math.min(start.lat, end.lat) - 0.01;
  const maxLat = Math.max(start.lat, end.lat) + 0.01;
  const minLng = Math.min(start.lng, end.lng) - 0.01;
  const maxLng = Math.max(start.lng, end.lng) + 0.01;
  
  const latStep = (maxLat - minLat) / resolution;
  const lngStep = (maxLng - minLng) / resolution;
  
  for (let i = 0; i < resolution; i++) {
    for (let j = 0; j < resolution; j++) {
      const lat = minLat + i * latStep;
      const lng = minLng + j * lngStep;
      const isWall = walls.some(wall => 
        Math.abs(wall.lat - lat) < latStep && Math.abs(wall.lng - lng) < lngStep
      );
      
      grid.push({
        id: `${i}-${j}`,
        lat,
        lng,
        gCost: Infinity,
        hCost: 0,
        fCost: Infinity,
        isWall,
        isVisited: false,
        isPath: false,
      });
    }
  }
  
  return grid;
}

function getNeighbors(node: PathNode, grid: PathNode[]): PathNode[] {
  const neighbors: PathNode[] = [];
  const [row, col] = node.id.split('-').map(Number);
  const resolution = 30;
  
  // 8-directional movement
  const directions = [
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1],           [0, 1],
    [1, -1],  [1, 0],  [1, 1]
  ];
  
  for (const [dr, dc] of directions) {
    const newRow = row + dr;
    const newCol = col + dc;
    
    if (newRow >= 0 && newRow < resolution && newCol >= 0 && newCol < resolution) {
      const neighborId = `${newRow}-${newCol}`;
      const neighbor = grid.find(n => n.id === neighborId);
      if (neighbor) {
        neighbors.push(neighbor);
      }
    }
  }
  
  return neighbors;
}

function reconstructPath(node: PathNode): PathNode[] {
  const path: PathNode[] = [];
  let current: PathNode | undefined = node;
  
  while (current) {
    path.unshift({ ...current, isPath: true });
    current = current.parent;
  }
  
  return path;
}
