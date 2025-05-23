// lib/pathfinding-runner.ts
import { Point, PathNode, PathfindingAlgorithm } from '@/types';
import { 
  aStar, 
  dijkstra, 
  bfs, 
  dfs, 
  greedyBestFirst, 
  bidirectionalSearch,
  PathfindingResult 
} from '@/lib/algorithms/pathfinding';

export async function runPathfindingAlgorithm(
  algorithm: PathfindingAlgorithm,
  start: Point,
  end: Point,
  walls: Point[],
  onProgress?: (visitedNodes: PathNode[], pathNodes: PathNode[]) => void
): Promise<PathfindingResult> {
  switch (algorithm) {
    case 'astar':
      return aStar(start, end, walls, onProgress);
    case 'dijkstra':
      return dijkstra(start, end, walls, onProgress);
    case 'bfs':
      return bfs(start, end, walls, onProgress);
    case 'dfs':
      return dfs(start, end, walls, onProgress);
    case 'greedy':
      return greedyBestFirst(start, end, walls, onProgress);
    case 'bidirectional':
      return bidirectionalSearch(start, end, walls, onProgress);
    default:
      throw new Error(`Unknown algorithm: ${algorithm}`);
  }
}
