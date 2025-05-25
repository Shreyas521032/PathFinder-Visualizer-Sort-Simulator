import streamlit as st
import folium
from streamlit_folium import st_folium, folium_static
import requests
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import random
from collections import deque
import heapq
import math
from typing import List, Tuple, Dict, Optional

# Page configuration
st.set_page_config(
    page_title="Advanced PathFinder & Sort Visualizer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f8f9fa;
        border-radius: 10px 10px 0px 0px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    .algorithm-info {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .complexity-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        margin: 2px;
    }
    .complexity-best { background-color: #d4edda; color: #155724; }
    .complexity-average { background-color: #fff3cd; color: #856404; }
    .complexity-worst { background-color: #f8d7da; color: #721c24; }
    .grid-cell {
        width: 20px;
        height: 20px;
        border: 1px solid #ccc;
        display: inline-block;
        margin: 1px;
    }
    .path-stats {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    .footer-credit {
        text-align: center;
        padding: 10px;
        font-style: italic;
        font-weight: bold;
        margin-top: 20px;
        background: linear-gradient(90deg, #e3ffe7 0%, #d9e7ff 100%);
        border-radius: 10px;
    }
    .data-point-info {
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 3px solid #667eea;
    }
    .interactive-tip {
        font-size: 0.9rem;
        color: #6c757d;
        margin: 5px 0 15px 0;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🗺️ Advanced PathFinder & Sort Visualizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive pathfinding on real maps & grid systems + animated sorting algorithms</p>', unsafe_allow_html=True)

# Grid-based pathfinding for better visualization
class GridPathfinder:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.start = None
        self.goal = None
        self.obstacles = set()
    
    def set_obstacle(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.obstacles.add((x, y))
            self.grid[y][x] = 1
    
    def remove_obstacle(self, x: int, y: int):
        if (x, y) in self.obstacles:
            self.obstacles.remove((x, y))
            self.grid[y][x] = 0
    
    def is_valid(self, x: int, y: int) -> bool:
        return (0 <= x < self.width and 0 <= y < self.height and 
                (x, y) not in self.obstacles)
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        neighbors = []
        # 8-directional movement
        directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny))
        return neighbors
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        # Euclidean distance
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        # Diagonal movement cost
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        if dx == 1 and dy == 1:
            return math.sqrt(2)  # Diagonal
        return 1.0  # Horizontal/Vertical

# Enhanced Pathfinding algorithms for grid
class PathfindingAlgorithms:
    @staticmethod
    def a_star(grid: GridPathfinder, start: Tuple[int, int], goal: Tuple[int, int]):
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: grid.heuristic(start, goal)}
        visited = []
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            visited.append(current)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1], visited
            
            for neighbor in grid.get_neighbors(current[0], current[1]):
                tentative_g_score = g_score[current] + grid.get_distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + grid.heuristic(neighbor, goal)
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return [], visited
    
    @staticmethod
    def dijkstra(grid: GridPathfinder, start: Tuple[int, int], goal: Tuple[int, int]):
        distances = {start: 0}
        pq = [(0, start)]
        came_from = {}
        visited = []
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            visited.append(current)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1], visited
            
            if current_dist > distances.get(current, float('inf')):
                continue
                
            for neighbor in grid.get_neighbors(current[0], current[1]):
                distance = current_dist + grid.get_distance(current, neighbor)
                
                if neighbor not in distances or distance < distances[neighbor]:
                    distances[neighbor] = distance
                    came_from[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        return [], visited
    
    @staticmethod
    def bfs(grid: GridPathfinder, start: Tuple[int, int], goal: Tuple[int, int]):
        queue = deque([start])
        visited = [start]
        came_from = {}
        
        while queue:
            current = queue.popleft()
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1], visited
            
            for neighbor in grid.get_neighbors(current[0], current[1]):
                if neighbor not in visited:
                    visited.append(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)
        
        return [], visited
    
    @staticmethod
    def dfs(grid: GridPathfinder, start: Tuple[int, int], goal: Tuple[int, int]):
        stack = [start]
        visited = [start]
        came_from = {}
        
        while stack:
            current = stack.pop()
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1], visited
            
            for neighbor in grid.get_neighbors(current[0], current[1]):
                if neighbor not in visited:
                    visited.append(neighbor)
                    came_from[neighbor] = current
                    stack.append(neighbor)
        
        return [], visited
    
    @staticmethod
    def greedy_best_first(grid: GridPathfinder, start: Tuple[int, int], goal: Tuple[int, int]):
        open_set = [(grid.heuristic(start, goal), start)]
        visited = []
        came_from = {}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            visited.append(current)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1], visited
            
            for neighbor in grid.get_neighbors(current[0], current[1]):
                if neighbor not in visited and neighbor not in [item[1] for item in open_set]:
                    came_from[neighbor] = current
                    heuristic_cost = grid.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (heuristic_cost, neighbor))
        
        return [], visited
    
    @staticmethod
    def bidirectional_search(grid: GridPathfinder, start: Tuple[int, int], goal: Tuple[int, int]):
        if start == goal:
            return [start], [start]
        
        visited_forward = {start}
        queue_forward = deque([start])
        came_from_forward = {start: None}
        
        visited_backward = {goal}
        queue_backward = deque([goal])
        came_from_backward = {goal: None}
        
        visited = []
        
        while queue_forward and queue_backward:
            if queue_forward:
                current_forward = queue_forward.popleft()
                visited.append(current_forward)
                
                for neighbor in grid.get_neighbors(current_forward[0], current_forward[1]):
                    if neighbor in visited_backward:
                        path_forward = []
                        node = current_forward
                        while node is not None:
                            path_forward.append(node)
                            node = came_from_forward[node]
                        path_forward.reverse()
                        
                        path_backward = []
                        node = neighbor
                        while node is not None:
                            path_backward.append(node)
                            node = came_from_backward[node]
                        
                        return path_forward + path_backward, visited
                    
                    if neighbor not in visited_forward:
                        visited_forward.add(neighbor)
                        came_from_forward[neighbor] = current_forward
                        queue_forward.append(neighbor)
            
            if queue_backward:
                current_backward = queue_backward.popleft()
                visited.append(current_backward)
                
                for neighbor in grid.get_neighbors(current_backward[0], current_backward[1]):
                    if neighbor in visited_forward:
                        path_forward = []
                        node = neighbor
                        while node is not None:
                            path_forward.append(node)
                            node = came_from_forward[node]
                        path_forward.reverse()
                        
                        path_backward = []
                        node = current_backward
                        while node is not None:
                            path_backward.append(node)
                            node = came_from_backward[node]
                        
                        return path_forward + path_backward, visited
                    
                    if neighbor not in visited_backward:
                        visited_backward.add(neighbor)
                        came_from_backward[neighbor] = current_backward
                        queue_backward.append(neighbor)
        
        return [], visited

# Real Map API integration using OpenRouteService (better than OSMnx)
class RealMapPathfinder:
    def __init__(self):
        # You can get a free API key from https://openrouteservice.org/
        self.api_key = "YOUR_API_KEY_HERE"  # Replace with actual API key
        self.base_url = "https://api.openrouteservice.org/v2"
    
    def geocode(self, location: str) -> Optional[Tuple[float, float]]:
        """Geocode a location name to coordinates"""
        try:
            # Using Nominatim (free alternative)
            url = f"https://nominatim.openstreetmap.org/search"
            params = {
                'q': location,
                'format': 'json',
                'limit': 1
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data:
                    return float(data[0]['lat']), float(data[0]['lon'])
        except Exception as e:
            st.error(f"Geocoding error: {e}")
        return None
    
    def get_route(self, start_coords: Tuple[float, float], end_coords: Tuple[float, float], 
                  profile: str = "driving-car") -> Optional[Dict]:
        """Get route between two coordinates"""
        try:
            # For demo purposes, create a simple route
            # In production, use actual routing API
            route_coords = [
                [start_coords[1], start_coords[0]],  # [lon, lat]
                [end_coords[1], end_coords[0]]
            ]
            
            # Calculate simple distance
            distance = self._calculate_distance(start_coords, end_coords)
            
            return {
                'coordinates': route_coords,
                'distance': distance,
                'duration': distance / 50 * 3600  # Assume 50 km/h average speed
            }
        except Exception as e:
            st.error(f"Routing error: {e}")
        return None
    
    def _calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates using Haversine formula"""
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        R = 6371  # Earth's radius in kilometers
        
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance

# Enhanced Sorting algorithms (keeping the good parts)
class SortingAlgorithms:
    @staticmethod
    def bubble_sort(arr):
        arr = arr.copy()
        steps = []
        n = len(arr)
        
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                steps.append((arr.copy(), [j, j + 1], "comparing"))
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    steps.append((arr.copy(), [j, j + 1], "swapped"))
                    swapped = True
            if not swapped:
                break
        
        steps.append((arr.copy(), [], "completed"))
        return steps
    
    @staticmethod
    def selection_sort(arr):
        arr = arr.copy()
        steps = []
        n = len(arr)
        
        for i in range(n):
            min_idx = i
            steps.append((arr.copy(), [i], "current_min"))
            
            for j in range(i + 1, n):
                steps.append((arr.copy(), [min_idx, j], "comparing"))
                if arr[j] < arr[min_idx]:
                    min_idx = j
                    steps.append((arr.copy(), [min_idx], "new_min"))
            
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
                steps.append((arr.copy(), [i, min_idx], "swapped"))
        
        steps.append((arr.copy(), [], "completed"))
        return steps
    
    @staticmethod
    def insertion_sort(arr):
        arr = arr.copy()
        steps = []
        
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            steps.append((arr.copy(), [i], "current"))
            
            while j >= 0 and arr[j] > key:
                steps.append((arr.copy(), [j, j + 1], "comparing"))
                arr[j + 1] = arr[j]
                steps.append((arr.copy(), [j + 1], "shifted"))
                j -= 1
            
            arr[j + 1] = key
            steps.append((arr.copy(), [j + 1], "inserted"))
        
        steps.append((arr.copy(), [], "completed"))
        return steps
    
    @staticmethod
    def quick_sort(arr):
        steps = []
        
        def partition(arr, low, high):
            pivot = arr[high]
            i = low - 1
            steps.append((arr.copy(), [high], "pivot"))
            
            for j in range(low, high):
                steps.append((arr.copy(), [j, high], "comparing"))
                if arr[j] <= pivot:
                    i += 1
                    if i != j:
                        arr[i], arr[j] = arr[j], arr[i]
                        steps.append((arr.copy(), [i, j], "swapped"))
            
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            steps.append((arr.copy(), [i + 1, high], "pivot_placed"))
            return i + 1
        
        def quick_sort_helper(arr, low, high):
            if low < high:
                pi = partition(arr, low, high)
                quick_sort_helper(arr, low, pi - 1)
                quick_sort_helper(arr, pi + 1, high)
        
        arr = arr.copy()
        quick_sort_helper(arr, 0, len(arr) - 1)
        steps.append((arr.copy(), [], "completed"))
        return steps
    
    @staticmethod
    def merge_sort(arr):
        steps = []
        
        def merge(arr, left, mid, right):
            left_part = arr[left:mid + 1]
            right_part = arr[mid + 1:right + 1]
            
            i = j = 0
            k = left
            
            while i < len(left_part) and j < len(right_part):
                steps.append((arr.copy(), [left + i, mid + 1 + j], "comparing"))
                if left_part[i] <= right_part[j]:
                    arr[k] = left_part[i]
                    i += 1
                else:
                    arr[k] = right_part[j]
                    j += 1
                steps.append((arr.copy(), [k], "merged"))
                k += 1
            
            while i < len(left_part):
                arr[k] = left_part[i]
                steps.append((arr.copy(), [k], "merged"))
                i += 1
                k += 1
            
            while j < len(right_part):
                arr[k] = right_part[j]
                steps.append((arr.copy(), [k], "merged"))
                j += 1
                k += 1
        
        def merge_sort_helper(arr, left, right):
            if left < right:
                mid = (left + right) // 2
                merge_sort_helper(arr, left, mid)
                merge_sort_helper(arr, mid + 1, right)
                merge(arr, left, mid, right)
        
        arr = arr.copy()
        merge_sort_helper(arr, 0, len(arr) - 1)
        steps.append((arr.copy(), [], "completed"))
        return steps
    
    @staticmethod
    def heap_sort(arr):
        steps = []
        arr = arr.copy()
        n = len(arr)
        
        def heapify(arr, n, i):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            steps.append((arr.copy(), [i], "heapify_root"))
            
            if left < n and arr[left] > arr[largest]:
                largest = left
            
            if right < n and arr[right] > arr[largest]:
                largest = right
            
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                steps.append((arr.copy(), [i, largest], "heap_swap"))
                heapify(arr, n, largest)
        
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)
        
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            steps.append((arr.copy(), [0, i], "extract_max"))
            heapify(arr, i, 0)
        
        steps.append((arr.copy(), [], "completed"))
        return steps

# Algorithm information data
PATHFINDING_INFO = {
    "A* (A-Star)": {
        "description": "Optimal pathfinding algorithm that uses both actual distance and heuristic estimates to guide search efficiently.",
        "time_complexity": "O(b^d) where b is branching factor, d is depth",
        "space_complexity": "O(b^d)",
        "optimal": "Yes (with admissible heuristic)",
        "use_case": "Best for most pathfinding scenarios with good heuristics",
        "pros": ["Optimal path guaranteed", "Very efficient", "Widely applicable", "Good balance of speed and optimality"],
        "cons": ["Requires good heuristic function", "Can be memory intensive", "Slower than greedy approaches"]
    },
    "Dijkstra": {
        "description": "Finds shortest path by exploring nodes in order of their distance from start. Always finds optimal solution.",
        "time_complexity": "O((V + E) log V) with binary heap",
        "space_complexity": "O(V)",
        "optimal": "Yes",
        "use_case": "When guaranteed shortest path is needed without heuristics",
        "pros": ["Always finds optimal path", "No heuristic needed", "Well-established algorithm", "Handles negative weights"],
        "cons": ["Can be slow for large graphs", "Explores many unnecessary nodes", "High memory usage"]
    },
    "BFS (Breadth-First Search)": {
        "description": "Explores all nodes at current depth before moving deeper. Optimal for unweighted graphs.",
        "time_complexity": "O(V + E)",
        "space_complexity": "O(V)",
        "optimal": "Yes (for unweighted graphs)",
        "use_case": "Unweighted graphs where all edges have equal cost",
        "pros": ["Simple implementation", "Optimal for unweighted graphs", "Complete algorithm", "Finds shortest path"],
        "cons": ["Not optimal for weighted graphs", "High memory usage", "Can be slow for deep solutions"]
    },
    "DFS (Depth-First Search)": {
        "description": "Explores as far as possible along each branch before backtracking. Fast but not optimal.",
        "time_complexity": "O(V + E)",
        "space_complexity": "O(h) where h is max depth",
        "optimal": "No",
        "use_case": "When memory is limited or exploring all possibilities",
        "pros": ["Low memory usage", "Simple to implement", "Good for maze solving", "Fast execution"],
        "cons": ["Not optimal", "Can get stuck in infinite paths", "May not find shortest path", "Depth-dependent"]
    },
    "Greedy Best-First": {
        "description": "Uses extra memory", "Not in-place", "Slower than quicksort in practice", "Overhead for small arrays"]
    },
    "Heap Sort": {
        "description": "Uses binary heap data structure to repeatedly extract maximum element and build sorted array.",
        "best_case": "O(n log n)", "average_case": "O(n log n)", "worst_case": "O(n log n)",
        "space_complexity": "O(1)", "stable": "No",
        "use_case": "When guaranteed O(n log n) time and O(1) space needed",
        "pros": ["Guaranteed O(n log n)", "In-place", "No worst case degradation", "Memory efficient"],
        "cons": ["Not stable", "Slower than quicksort in practice", "Complex implementation", "Poor cache performance"]
    }
}

# Utility functions
def create_grid_visualization(grid: GridPathfinder, path: List[Tuple[int, int]] = None, 
                            visited: List[Tuple[int, int]] = None, start: Tuple[int, int] = None,
                            goal: Tuple[int, int] = None, click_mode: str = None) -> go.Figure:
    """Create a plotly heatmap visualization of the grid"""
    
    # Create visualization grid
    vis_grid = np.zeros((grid.height, grid.width))
    
    # Set obstacles
    for x, y in grid.obstacles:
        vis_grid[y][x] = -1
    
    # Set visited nodes
    if visited:
        for x, y in visited:
            if vis_grid[y][x] == 0:  # Don't override obstacles
                vis_grid[y][x] = 0.3
    
    # Set path
    if path:
        for x, y in path:
            if vis_grid[y][x] != -1:  # Don't override obstacles
                vis_grid[y][x] = 0.8
    
    # Set start and goal
    if start:
        vis_grid[start[1]][start[0]] = 1.0
    if goal:
        vis_grid[goal[1]][goal[0]] = 0.9
    
    # Create custom colorscale
    colorscale = [
        [0.0, 'black'],      # Obstacles
        [0.2, 'white'],      # Empty
        [0.4, 'lightblue'],  # Visited
        [0.6, 'yellow'],     # Path
        [0.8, 'orange'],     # Goal
        [1.0, 'green']       # Start
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=vis_grid,
        colorscale=colorscale,
        showscale=False,
        hovertemplate='X: %{x}<br>Y: %{y}<br>Type: %{z}<extra></extra>'
    ))
    
    # Add click mode annotation
    if click_mode:
        fig.add_annotation(
            text=f"Click Mode: {click_mode}",
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=14, color="black"),
            bgcolor="rgba(255, 255, 255, 0.7)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4
        )
    
    fig.update_layout(
        title="Grid Pathfinding Visualization",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        width=600,
        height=500,
        yaxis={'autorange': 'reversed'},  # Flip Y axis to match typical grid representation
        clickmode='event'  # Enable click events
    )
    
    return fig

def create_sample_maps():
    """Create predefined sample maps with obstacles"""
    maps = {
        "Empty Grid": lambda w, h: set(),
        "Maze": lambda w, h: create_maze_obstacles(w, h),
        "Random Obstacles": lambda w, h: create_random_obstacles(w, h, 0.2),
        "Diagonal Barriers": lambda w, h: create_diagonal_barriers(w, h),
        "Rooms": lambda w, h: create_rooms_obstacles(w, h)
    }
    return maps

def create_maze_obstacles(width: int, height: int) -> set:
    """Create a maze-like pattern of obstacles"""
    obstacles = set()
    
    # Create walls
    for y in range(2, height-2, 4):
        for x in range(1, width-1):
            if x % 4 != 2:
                obstacles.add((x, y))
    
    for x in range(2, width-2, 4):
        for y in range(1, height-1):
            if y % 4 != 2:
                obstacles.add((x, y))
    
    return obstacles

def create_random_obstacles(width: int, height: int, density: float) -> set:
    """Create random obstacles with given density"""
    obstacles = set()
    num_obstacles = int(width * height * density)
    
    for _ in range(num_obstacles):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        obstacles.add((x, y))
    
    return obstacles

def create_diagonal_barriers(width: int, height: int) -> set:
    """Create diagonal barrier patterns"""
    obstacles = set()
    
    # Main diagonal
    for i in range(min(width, height) // 2):
        if i < width and i < height:
            obstacles.add((i, i))
        if width - 1 - i >= 0 and i < height:
            obstacles.add((width - 1 - i, i))
    
    return obstacles

def create_rooms_obstacles(width: int, height: int) -> set:
    """Create room-like structure with doorways"""
    obstacles = set()
    
    # Horizontal walls
    mid_y = height // 2
    for x in range(width):
        if x != width // 4 and x != 3 * width // 4:  # Leave doorways
            obstacles.add((x, mid_y))
    
    # Vertical walls
    mid_x = width // 2
    for y in range(height):
        if y != height // 4 and y != 3 * height // 4:  # Leave doorways
            obstacles.add((mid_x, y))
    
    return obstacles

# Data for real-world sorting applications
REAL_WORLD_SORTING_EXAMPLES = {
    "Bubble Sort": {
        "applications": [
            {"name": "Education", "description": "Teaching basic sorting principles to beginners"},
            {"name": "Small Data Sets", "description": "Organizing small lists where simplicity matters more than efficiency"},
            {"name": "Nearly Sorted Data", "description": "Data that is already almost sorted with few out-of-place elements"}
        ],
        "visual": "education_sorting.png"
    },
    "Selection Sort": {
        "applications": [
            {"name": "Memory Constrained Systems", "description": "Embedded systems with limited memory where minimal swaps are needed"},
            {"name": "Flash Memory Devices", "description": "Where write operations are expensive and should be minimized"},
            {"name": "Small Files", "description": "Organizing small files or records where the overhead of more complex algorithms isn't justified"}
        ],
        "visual": "embedded_systems.png"
    },
    "Insertion Sort": {
        "applications": [
            {"name": "Online Sorting", "description": "Sorting data as it arrives in real-time (like card sorting in a card game)"},
            {"name": "Database Operations", "description": "Maintaining sorted lists as new records are inserted"},
            {"name": "Continuously Updated Lists", "description": "Applications that require maintaining a sorted order as new items arrive"}
        ],
        "visual": "card_sorting.png"
    },
    "Quick Sort": {
        "applications": [
            {"name": "Operating Systems", "description": "Used in various OS components including the Windows NT kernel"},
            {"name": "Programming Languages", "description": "Default sorting algorithm in many language libraries (Java, C++)"},
            {"name": "Database Systems", "description": "Used for efficient sorting of large datasets in memory"}
        ],
        "visual": "os_sorting.png"
    },
    "Merge Sort": {
        "applications": [
            {"name": "External Sorting", "description": "Sorting large files that don't fit in memory"},
            {"name": "Databases", "description": "Merging results from different database queries"},
            {"name": "Version Control", "description": "Merging changes in version control systems like Git"},
            {"name": "Network Traffic Analysis", "description": "Sorting and analyzing large network packet logs"}
        ],
        "visual": "database_merge.png"
    },
    "Heap Sort": {
        "applications": [
            {"name": "Priority Queues", "description": "Used in operating system job scheduling"},
            {"name": "Graph Algorithms", "description": "Dijkstra's algorithm for shortest paths uses heap structures"},
            {"name": "K-way Merging", "description": "Finding k smallest/largest elements in a large dataset"},
            {"name": "Memory Management", "description": "Efficient allocation of memory blocks in systems"}
        ],
        "visual": "priority_queue.png"
    }
}

# Main app tabs
tab1, tab2 = st.tabs(["🗺️ PathFinding Visualizer", "📊 Sorting Visualizer"])

# Tab 1: Enhanced PathFinding Visualizer
with tab1:
    st.header("🗺️ Advanced Pathfinding Visualization")
    
    # Create sub-tabs for different pathfinding modes
    pathfind_tabs = st.tabs(["🟩 Grid-Based", "🌍 Real Maps", "🆚 Algorithm Comparison"])
    
    # Grid-Based Pathfinding Tab
    with pathfind_tabs[0]:
        st.subheader("Interactive Grid Pathfinding")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### 🎛️ Grid Controls")
            
            # Grid settings
            grid_width = st.slider("Grid Width", 10, 50, 25)
            grid_height = st.slider("Grid Height", 10, 50, 20)
            
            # Algorithm selection
            algorithm = st.selectbox(
                "🧠 Algorithm",
                ["A* (A-Star)", "Dijkstra", "BFS (Breadth-First Search)", 
                 "DFS (Depth-First Search)", "Greedy Best-First", "Bidirectional Search"]
            )
            
            # Sample maps
            sample_maps = create_sample_maps()
            selected_map = st.selectbox("🗺️ Sample Maps", list(sample_maps.keys()))
            
            if st.button("🎲 Load Sample Map"):
                if 'grid' not in st.session_state:
                    st.session_state.grid = GridPathfinder(grid_width, grid_height)
                
                obstacles = sample_maps[selected_map](grid_width, grid_height)
                st.session_state.grid.obstacles = obstacles
                st.session_state.grid_updated = True
            
            # Interactive grid control
            st.markdown("### 🖱️ Interactive Mode")
            click_mode = st.radio(
                "Click Mode",
                ["Set Start", "Set Goal", "Add Obstacle", "Remove Obstacle"],
                key="click_mode"
            )
            
            # Store the click mode in session state
            st.session_state.current_click_mode = click_mode
            
            st.markdown('<p class="interactive-tip">Click directly on the grid to place points or obstacles!</p>', unsafe_allow_html=True)
            
            # Manual point setting (still available as an alternative)
            with st.expander("Manual Coordinates Input"):
                col_start, col_goal = st.columns(2)
                with col_start:
                    start_x = st.number_input("Start X", 0, grid_width-1, 0)
                    start_y = st.number_input("Start Y", 0, grid_height-1, 0)
                    if st.button("Set Start"):
                        st.session_state.start_point = (start_x, start_y)
                
                with col_goal:
                    goal_x = st.number_input("Goal X", 0, grid_width-1, grid_width-1)
                    goal_y = st.number_input("Goal Y", 0, grid_height-1, grid_height-1)
                    if st.button("Set Goal"):
                        st.session_state.goal_point = (goal_x, goal_y)
            
            # Action buttons
            if st.button("🎯 Find Path", type="primary"):
                if ('start_point' in st.session_state and 
                    'goal_point' in st.session_state and
                    'grid' in st.session_state):
                    
                    grid = st.session_state.grid
                    start = st.session_state.start_point
                    goal = st.session_state.goal_point
                    
                    # Run algorithm
                    start_time = time.time()
                    
                    if algorithm == "A* (A-Star)":
                        path, visited = PathfindingAlgorithms.a_star(grid, start, goal)
                    elif algorithm == "Dijkstra":
                        path, visited = PathfindingAlgorithms.dijkstra(grid, start, goal)
                    elif algorithm == "BFS (Breadth-First Search)":
                        path, visited = PathfindingAlgorithms.bfs(grid, start, goal)
                    elif algorithm == "DFS (Depth-First Search)":
                        path, visited = PathfindingAlgorithms.dfs(grid, start, goal)
                    elif algorithm == "Greedy Best-First":
                        path, visited = PathfindingAlgorithms.greedy_best_first(grid, start, goal)
                    else:  # Bidirectional Search
                        path, visited = PathfindingAlgorithms.bidirectional_search(grid, start, goal)
                    
                    end_time = time.time()
                    execution_time = (end_time - start_time) * 1000
                    
                    # Store results
                    st.session_state.grid_path = path
                    st.session_state.grid_visited = visited
                    st.session_state.grid_execution_time = execution_time
                    
                else:
                    st.error("Please set start and goal points first!")
            
            if st.button("🧹 Clear Grid"):
                st.session_state.grid = GridPathfinder(grid_width, grid_height)
                if 'start_point' in st.session_state:
                    del st.session_state.start_point
                if 'goal_point' in st.session_state:
                    del st.session_state.goal_point
                if 'grid_path' in st.session_state:
                    del st.session_state.grid_path
                if 'grid_visited' in st.session_state:
                    del st.session_state.grid_visited
                st.session_state.grid_updated = True
            
            # Results display
            if 'grid_execution_time' in st.session_state:
                st.markdown("### 📊 Results")
                st.metric("Execution Time", f"{st.session_state.grid_execution_time:.2f} ms")
                
                if 'grid_path' in st.session_state:
                    st.metric("Path Length", len(st.session_state.grid_path))
                
                if 'grid_visited' in st.session_state:
                    st.metric("Nodes Visited", len(st.session_state.grid_visited))
                    
                    if len(st.session_state.grid_visited) > 0 and len(st.session_state.grid_path) > 0:
                        efficiency = (len(st.session_state.grid_path) / len(st.session_state.grid_visited)) * 100
                        st.metric("Efficiency", f"{efficiency:.1f}%")
        
        with col1:
            # Initialize grid if not exists
            if 'grid' not in st.session_state:
                st.session_state.grid = GridPathfinder(grid_width, grid_height)
            
            # Update grid size if changed
            if (st.session_state.grid.width != grid_width or 
                st.session_state.grid.height != grid_height):
                st.session_state.grid = GridPathfinder(grid_width, grid_height)
            
            # Create visualization
            path = st.session_state.get('grid_path', [])
            visited = st.session_state.get('grid_visited', [])
            start = st.session_state.get('start_point', None)
            goal = st.session_state.get('goal_point', None)
            click_mode = st.session_state.get('current_click_mode', None)
            
            fig = create_grid_visualization(
                st.session_state.grid, path, visited, start, goal, click_mode
            )
            
            # Display the grid with click event handling
            grid_chart = st.plotly_chart(fig, use_container_width=True)
            
            # Handle grid clicks for interactive placing
            clicked_point = st.empty()
            
            # Get last click data if available
            last_click_data = st.session_state.get('last_click_data', None)
            
            # Check if we have new click data from Plotly
            if last_click_data:
                x, y = int(last_click_data['x']), int(last_click_data['y'])
                
                # Only process if within grid bounds
                if 0 <= x < grid_width and 0 <= y < grid_height:
                    # Handle click based on current mode
                    if click_mode == "Set Start":
                        st.session_state.start_point = (x, y)
                        st.session_state.grid.obstacles.discard((x, y))  # Remove obstacle if exists
                        clicked_point.info(f"Start point set to ({x}, {y})")
                    
                    elif click_mode == "Set Goal":
                        st.session_state.goal_point = (x, y)
                        st.session_state.grid.obstacles.discard((x, y))  # Remove obstacle if exists
                        clicked_point.info(f"Goal point set to ({x}, {y})")
                    
                    elif click_mode == "Add Obstacle":
                        # Don't add obstacle if it's the start or goal point
                        if (x, y) != st.session_state.get('start_point') and (x, y) != st.session_state.get('goal_point'):
                            st.session_state.grid.set_obstacle(x, y)
                            clicked_point.info(f"Obstacle added at ({x}, {y})")
                    
                    elif click_mode == "Remove Obstacle":
                        st.session_state.grid.remove_obstacle(x, y)
                        clicked_point.info(f"Obstacle removed at ({x}, {y})")
                
                # Clear click data after processing
                st.session_state.last_click_data = None
                st.rerun()  # Force rerun to update the visualization
            
            # Instructions
            st.info("""
            🖱️ **Instructions:**
            1. Select a click mode (Set Start, Set Goal, Add/Remove Obstacle)
            2. Click directly on the grid to place items
            3. Choose an algorithm and click 'Find Path'
            4. View the results and try different algorithms!
            
            **Legend:**
            - 🟩 Green: Start point
            - 🟧 Orange: Goal point  
            - 🟨 Yellow: Optimal path
            - 🔵 Light Blue: Visited nodes
            - ⬛ Black: Obstacles
            """)
            
            # Plotly click event handler (JavaScript)
            st.markdown("""
            <script>
                const gridClickHandler = () => {
                    const gridPlot = document.querySelector('[data-testid="stPlotlyChart"] .js-plotly-plot');
                    if (gridPlot) {
                        gridPlot.on('plotly_click', (data) => {
                            const clickData = {
                                x: data.points[0].x,
                                y: data.points[0].y
                            };
                            // Store click data to session state
                            window.parent.postMessage({
                                type: "streamlit:setComponentValue",
                                value: clickData,
                                key: "last_click_data"
                            }, "*");
                        });
                    }
                };
                
                // Run when document is ready and whenever Streamlit reruns
                if (document.readyState === 'complete') {
                    gridClickHandler();
                } else {
                    window.addEventListener('load', gridClickHandler);
                }
                window.addEventListener('streamlit:render', gridClickHandler);
            </script>
            """, unsafe_allow_html=True)
    
    # Real Maps Tab
    with pathfind_tabs[1]:
        st.subheader("🌍 Real-World Map Pathfinding")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### 🎛️ Map Controls")
            
            # Location inputs
            start_location = st.text_input("📍 Start Location", "New York, NY")
            end_location = st.text_input("🎯 End Location", "Boston, MA")
            
            # Transport mode
            transport_mode = st.selectbox(
                "🚗 Transport Mode",
                ["driving-car", "foot-walking", "cycling-regular"]
            )
            
            # Map style
            map_style = st.selectbox(
                "🗺️ Map Style",
                ["OpenStreetMap", "Stamen Terrain", "Stamen Toner", "CartoDB positron"]
            )
            
            # Interactive marker placement
            st.markdown("### 📍 Interactive Mode")
            st.markdown('<p class="interactive-tip">Click directly on the map to place start and end points!</p>', unsafe_allow_html=True)
            
            map_click_mode = st.radio(
                "Map Click Mode",
                ["Set Start Point", "Set End Point"],
                key="map_click_mode"
            )
            
            # Store map click mode in session state
            st.session_state.current_map_click_mode = map_click_mode
            
            if st.button("🗺️ Create Route", type="primary"):
                real_map_finder = RealMapPathfinder()
                
                with st.spinner("Geocoding locations..."):
                    start_coords = None
                    end_coords = None
                    
                    # Use clicked coordinates if available, otherwise geocode from text
                    if 'map_start_coords' in st.session_state:
                        start_coords = st.session_state.map_start_coords
                    else:
                        start_coords = real_map_finder.geocode(start_location)
                        
                    if 'map_end_coords' in st.session_state:
                        end_coords = st.session_state.map_end_coords
                    else:
                        end_coords = real_map_finder.geocode(end_location)
                
                if start_coords and end_coords:
                    with st.spinner("Calculating route..."):
                        route_data = real_map_finder.get_route(start_coords, end_coords, transport_mode)
                    
                    if route_data:
                        # Create folium map
                        center_lat = (start_coords[0] + end_coords[0]) / 2
                        center_lon = (start_coords[1] + end_coords[1]) / 2
                        
                        # Map style mapping
                        tile_mapping = {
                            "OpenStreetMap": "OpenStreetMap",
                            "Stamen Terrain": "Stamen Terrain",
                            "Stamen Toner": "Stamen Toner",
                            "CartoDB positron": "CartoDB positron"
                        }
                        
                        m = folium.Map(
                            location=[center_lat, center_lon],
                            zoom_start=8,
                            tiles=tile_mapping[map_style]
                        )
                        
                        # Add start marker
                        folium.Marker(
                            location=[start_coords[0], start_coords[1]],
                            popup=f"Start: {start_location}",
                            icon=folium.Icon(color='green', icon='play')
                        ).add_to(m)
                        
                        # Add end marker
                        folium.Marker(
                            location=[end_coords[0], end_coords[1]],
                            popup=f"End: {end_location}",
                            icon=folium.Icon(color='red', icon='stop')
                        ).add_to(m)
                        
                        # Add route line
                        route_coords = [[coord[1], coord[0]] for coord in route_data['coordinates']]
                        folium.PolyLine(
                            locations=route_coords,
                            color='blue',
                            weight=5,
                            opacity=0.8,
                            popup=f"Route: {route_data['distance']:.1f} km"
                        ).add_to(m)
                        
                        # Add a click handler for the map
                        m.add_child(folium.LatLngPopup())
                        
                        st.session_state.real_map = m
                        st.session_state.route_data = route_data
                        
                        # Display metrics
                        st.metric("Distance", f"{route_data['distance']:.1f} km")
                        st.metric("Est. Duration", f"{route_data['duration']/3600:.1f} hours")
                    else:
                        st.error("Could not calculate route")
                else:
                    st.error("Could not geocode one or both locations")
            
            # Sample routes
            st.markdown("### 🌟 Sample Routes")
            sample_routes = {
                "NYC to Boston": ("New York, NY", "Boston, MA"),
                "LA to San Francisco": ("Los Angeles, CA", "San Francisco, CA"),
                "London to Paris": ("London, UK", "Paris, France"),
                "Tokyo to Osaka": ("Tokyo, Japan", "Osaka, Japan")
            }
            
            selected_route = st.selectbox("Quick Routes", list(sample_routes.keys()))
            if st.button("Load Sample Route"):
                start_loc, end_loc = sample_routes[selected_route]
                st.session_state.temp_start = start_loc
                st.session_state.temp_end = end_loc
                st.rerun()
        
       with col1:
            if 'comparison_results' in st.session_state:
                # Display results table
                df = pd.DataFrame(st.session_state.comparison_results)
                st.dataframe(df, use_container_width=True)
                
                # Create performance charts
                valid_results = [r for r in st.session_state.comparison_results 
                               if r["Path Found"] == "Yes" and r["Execution Time (ms)"] != "Error"]
                
                if valid_results:
                    # Execution time chart
                    algorithms = [r["Algorithm"] for r in valid_results]
                    times = [float(r["Execution Time (ms)"]) for r in valid_results]
                    visited_counts = [int(r["Nodes Visited"]) for r in valid_results]
                    
                    fig_perf = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=("Execution Time", "Nodes Visited"),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    fig_perf.add_trace(
                        go.Bar(x=algorithms, y=times, name="Time (ms)", marker_color='lightblue'),
                        row=1, col=1
                    )
                    
                    fig_perf.add_trace(
                        go.Bar(x=algorithms, y=visited_counts, name="Nodes Visited", marker_color='lightcoral'),
                        row=1, col=2
                    )
                    
                    fig_perf.update_layout(
                        title="Algorithm Performance Comparison",
                        height=400,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig_perf, use_container_width=True)
                
                # Show test grid
                if 'comparison_grid' in st.session_state:
                    test_fig = create_grid_visualization(
                        st.session_state.comparison_grid,
                        start=st.session_state.comparison_start,
                        goal=st.session_state.comparison_goal
                    )
                    test_fig.update_layout(title="Test Grid Used for Comparison")
                    st.plotly_chart(test_fig, use_container_width=True)
            else:
                st.info("🏁 Run algorithm comparison to see detailed performance metrics and visualizations.")
    
    # Algorithm Information Section
    st.markdown("---")
    st.subheader("📚 Pathfinding Algorithm Reference")
    
    algo_info_tabs = st.tabs(list(PATHFINDING_INFO.keys()))
    
    for i, (algo_name, info) in enumerate(PATHFINDING_INFO.items()):
        with algo_info_tabs[i]:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Description:** {info['description']}")
                st.markdown(f"**Time Complexity:** `{info['time_complexity']}`")
                st.markdown(f"**Space Complexity:** `{info['space_complexity']}`")
                st.markdown(f"**Optimal:** {info['optimal']}")
                st.markdown(f"**Best Use Case:** {info['use_case']}")
            
            with col2:
                st.markdown("**Advantages:**")
                for pro in info['pros']:
                    st.markdown(f"✅ {pro}")
                
                st.markdown("**Disadvantages:**")
                for con in info['cons']:
                    st.markdown(f"❌ {con}")

# Tab 2: Sorting Visualizer (keeping the excellent implementation)
with tab2:
    st.header("📊 Advanced Sorting Algorithm Visualizer")
    
    # Create sub-tabs for sorting visualizer
    sort_viz_tabs = st.tabs(["📊 Basic Sorting", "🌍 Real-World Applications", "⚖️ Performance Comparison"])
    
    # Basic Sorting Tab
    with sort_viz_tabs[0]:
        # Sidebar for sorting
        with st.sidebar:
            st.subheader("🎛️ Sorting Controls")
            
            # Array configuration
            array_size = st.slider("📏 Array Size", 10, 100, 30)
            array_type = st.selectbox(
                "📊 Array Type",
                ["Random", "Nearly Sorted", "Reverse Sorted", "Few Unique", "Mostly Sorted"]
            )
            
            # Algorithm selection
            sort_algorithm = st.selectbox(
                "🔄 Sorting Algorithm",
                ["Bubble Sort", "Selection Sort", "Insertion Sort", "Quick Sort", 
                 "Merge Sort", "Heap Sort"]
            )
            
            # Animation speed
            animation_speed = st.slider("⚡ Animation Speed", 0.01, 1.0, 0.1, 0.01)
            
            # Visualization options
            show_comparisons = st.checkbox("👀 Show Comparisons", value=True)
            show_array_access = st.checkbox("📊 Count Array Accesses", value=True)
            
            # Generate array button
            generate_array = st.button("🎲 Generate New Array", type="primary")
            
            # Start sorting button
            start_sorting = st.button("▶️ Start Sorting")
            
            # Compare algorithms
            compare_algos = st.button("⚔️ Compare All Algorithms")
        
        # Generate array based on type
        if generate_array or 'sorting_array' not in st.session_state:
            if array_type == "Random":
                arr = [random.randint(1, 100) for _ in range(array_size)]
            elif array_type == "Nearly Sorted":
                arr = list(range(1, array_size + 1))
                # Shuffle a few elements
                for _ in range(array_size // 10):
                    i, j = random.randint(0, array_size - 1), random.randint(0, array_size - 1)
                    arr[i], arr[j] = arr[j], arr[i]
            elif array_type == "Reverse Sorted":
                arr = list(range(array_size, 0, -1))
            elif array_type == "Few Unique":
                unique_values = [random.randint(1, 20) for _ in range(5)]
                arr = [random.choice(unique_values) for _ in range(array_size)]
            else:  # Mostly Sorted
                arr = list(range(1, array_size + 1))
                # Shuffle only a few elements
                for _ in range(max(1, array_size // 20)):
                    i, j = random.randint(0, array_size - 1), random.randint(0, array_size - 1)
                    arr[i], arr[j] = arr[j], arr[i]
            
            st.session_state.sorting_array = arr
            st.session_state.original_array = arr.copy()
        
        # Display current array
        if 'sorting_array' in st.session_state:
            st.subheader(f"Current Array ({array_type})")
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(range(len(st.session_state.sorting_array))),
                    y=st.session_state.sorting_array,
                    marker_color='lightblue',
                    text=st.session_state.sorting_array,
                    textposition='outside' if len(st.session_state.sorting_array) <= 20 else 'none'
                )
            ])
            
            fig.update_layout(
                title=f"Array of size {len(st.session_state.sorting_array)}",
                xaxis_title="Index",
                yaxis_title="Value",
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Compare all algorithms
        if compare_algos and 'sorting_array' in st.session_state:
            st.subheader("🏆 Algorithm Comparison")
            
            algorithms = ["Bubble Sort", "Selection Sort", "Insertion Sort", "Quick Sort", "Merge Sort", "Heap Sort"]
            comparison_results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            arr = st.session_state.original_array.copy()
            
            for i, algo in enumerate(algorithms):
                status_text.text(f"Testing {algo}...")
                progress_bar.progress((i + 1) / len(algorithms))
                
                start_time = time.time()
                
                try:
                    if algo == "Bubble Sort":
                        steps = SortingAlgorithms.bubble_sort(arr)
                    elif algo == "Selection Sort":
                        steps = SortingAlgorithms.selection_sort(arr)
                    elif algo == "Insertion Sort":
                        steps = SortingAlgorithms.insertion_sort(arr)
                    elif algo == "Quick Sort":
                        steps = SortingAlgorithms.quick_sort(arr)
                    elif algo == "Merge Sort":
                        steps = SortingAlgorithms.merge_sort(arr)
                    else:  # Heap Sort
                        steps = SortingAlgorithms.heap_sort(arr)
                    
                    end_time = time.time()
                    execution_time = (end_time - start_time) * 1000
                    
                    comparison_results.append({
                        "Algorithm": algo,
                        "Steps": len(steps),
                        "Time (ms)": f"{execution_time:.2f}",
                        "Time_numeric": execution_time
                    })
                except Exception as e:
                    comparison_results.append({
                        "Algorithm": algo,
                        "Steps": "Error",
                        "Time (ms)": "Error",
                        "Time_numeric": float('inf')
                    })
            
            # Display results
            df = pd.DataFrame(comparison_results)
            df_display = df.drop('Time_numeric', axis=1)
            st.dataframe(df_display, use_container_width=True)
            
            # Create performance chart
            valid_results = [r for r in comparison_results if r["Time_numeric"] != float('inf')]
            if valid_results:
                fig = go.Figure()
                
                algorithms_list = [r["Algorithm"] for r in valid_results]
                times_list = [r["Time_numeric"] for r in valid_results]
                steps_list = [r["Steps"] for r in valid_results if isinstance(r["Steps"], int)]
                
                fig.add_trace(go.Bar(
                    name='Execution Time (ms)',
                    x=algorithms_list,
                    y=times_list,
                    yaxis='y',
                    offsetgroup=1,
                    marker_color='lightblue'
                ))
                
                if len(steps_list) == len(algorithms_list):
                    fig.add_trace(go.Bar(
                        name='Steps',
                        x=algorithms_list,
                        y=steps_list,
                        yaxis='y2',
                        offsetgroup=2,
                        marker_color='lightcoral'
                    ))
                
                fig.update_layout(
                    title='Algorithm Performance Comparison',
                    xaxis_title='Algorithm',
                    yaxis=dict(title='Execution Time (ms)', side='left'),
                    yaxis2=dict(title='Steps', side='right', overlaying='y'),
                    barmode='group',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            status_text.empty()
            progress_bar.empty()
        
        # Start sorting animation
        if start_sorting and 'sorting_array' in st.session_state:
            arr = st.session_state.original_array.copy()
            
            # Get sorting steps
            with st.spinner(f"Running {sort_algorithm}..."):
                start_time = time.time()
                
                if sort_algorithm == "Bubble Sort":
                    steps = SortingAlgorithms.bubble_sort(arr)
                elif sort_algorithm == "Selection Sort":
                    steps = SortingAlgorithms.selection_sort(arr)
                elif sort_algorithm == "Insertion Sort":
                    steps = SortingAlgorithms.insertion_sort(arr)
                elif sort_algorithm == "Quick Sort":
                    steps = SortingAlgorithms.quick_sort(arr)
                elif sort_algorithm == "Merge Sort":
                    steps = SortingAlgorithms.merge_sort(arr)
                else:  # Heap Sort
                    steps = SortingAlgorithms.heap_sort(arr)
                
                end_time = time.time()
                total_time = (end_time - start_time) * 1000
            
            # Create placeholders for animation
            progress_bar = st.progress(0)
            chart_placeholder = st.empty()
            status_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            # Animation counters
            comparisons = 0
            swaps = 0
            array_accesses = 0
            
            # Animate sorting
            for i, (current_array, highlighted, action) in enumerate(steps):
                # Update counters
                if "comparing" in action:
                    comparisons += 1
                    array_accesses += 2
                elif "swap" in action:
                    swaps += 1
                    array_accesses += 2
                elif action in ["shifted", "inserted", "merged"]:
                    array_accesses += 1
                
                # Update progress
                progress = (i + 1) / len(steps)
                progress_bar.progress(progress)
                
                # Create colors for bars
                colors = ['lightblue'] * len(current_array)
                for idx in highlighted:
                    if idx < len(colors):
                        if "comparing" in action:
                            colors[idx] = 'yellow'
                        elif "swap" in action:
                            colors[idx] = 'red'
                        elif "pivot" in action:
                            colors[idx] = 'purple'
                        elif action in ["merged", "inserted"]:
                            colors[idx] = 'green'
                        elif "current" in action or "min" in action:
                            colors[idx] = 'orange'
                        elif "gap" in action:
                            colors[idx] = 'cyan'
                
                # Create animated bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(range(len(current_array))),
                        y=current_array,
                        marker_color=colors,
                        text=current_array if len(current_array) <= 30 else None,
                        textposition='outside' if len(current_array) <= 30 else 'none'
                    )
                ])
                
                fig.update_layout(
                    title=f"{sort_algorithm} - Step {i + 1}/{len(steps)}",
                    xaxis_title="Index",
                    yaxis_title="Value",
                    showlegend=False,
                    height=400
                )
                
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
                # Update status and metrics
                if action == "completed":
                    status_placeholder.success("✅ Sorting completed!")
                else:
                    status_placeholder.info(f"Status: {action.replace('_', ' ').title()}")
                
                # Show metrics if enabled
                if show_array_access or show_comparisons:
                    col1, col2, col3, col4 = metrics_placeholder.columns(4)
                    if show_comparisons:
                        col1.metric("Comparisons", comparisons)
                        col2.metric("Swaps", swaps)
                    if show_array_access:
                        col3.metric("Array Accesses", array_accesses)
                    col4.metric("Progress", f"{progress*100:.1f}%")
                
                # Animation delay
                time.sleep(animation_speed)
            
            # Final success message with statistics
            st.balloons()
            
            # Final metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Steps", len(steps))
            col2.metric("Execution Time", f"{total_time:.2f} ms")
            col3.metric("Total Comparisons", comparisons)
            col4.metric("Total Swaps", swaps)
            
            st.success(f"🎉 {sort_algorithm} completed!")
            
            # Clear placeholders
            progress_bar.empty()
            status_placeholder.empty()
    
    # Real-World Applications Tab
    with sort_viz_tabs[1]:
        st.subheader("🌍 Real-World Sorting Algorithm Applications")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### 🎛️ Select Algorithm")
            
            # Algorithm selection for real-world examples
            real_world_algo = st.selectbox(
                "Algorithm",
                list(REAL_WORLD_SORTING_EXAMPLES.keys()),
                key="real_world_algo"
            )
            
            # Display algorithm info
            algo_info = SORTING_INFO[real_world_algo]
            
            st.markdown("### ⚙️ Algorithm Details")
            st.markdown(f"**{real_world_algo}**")
            st.markdown(f"{algo_info['description']}")
            
            # Complexity badges
            st.markdown("**Time Complexity:**")
            st.markdown(f'<span class="complexity-badge complexity-best">Best: {algo_info["best_case"]}</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="complexity-badge complexity-average">Average: {algo_info["average_case"]}</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="complexity-badge complexity-worst">Worst: {algo_info["worst_case"]}</span>', unsafe_allow_html=True)
            
            st.markdown(f"**Space: ** {algo_info['space_complexity']}")
            st.markdown(f"**Stable: ** {algo_info['stable']}")
            
        with col1:
            # Display real-world applications
            st.markdown("### 🏭 Real-World Use Cases")
            
            # Get applications for selected algorithm
            applications = REAL_WORLD_SORTING_EXAMPLES[real_world_algo]["applications"]
            
            # Display each application in a card-like format
            for app in applications:
                st.markdown(f"""
                <div class="data-point-info">
                    <h4>{app['name']}</h4>
                    <p>{app['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Interactive demo for real-world application
            st.markdown("### 🧪 Interactive Real-World Demo")
            
            # Create demo based on algorithm
            if real_world_algo == "Bubble Sort":
                st.markdown("#### 📚 Educational Tool: Teaching Sorting Concepts")
                
                # Create a small dataset for interactive demo
                demo_data = [45, 23, 67, 12, 89, 34]
                demo_labels = ["Alice", "Bob", "Charlie", "Dave", "Eve", "Frank"]
                
                st.markdown("Imagine we're sorting student scores:")
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=demo_labels,
                        y=demo_data,
                        marker_color='lightblue',
                        text=demo_data,
                        textposition='outside'
                    )
                ])
                
                fig.update_layout(
                    title="Student Test Scores",
                    xaxis_title="Student",
                    yaxis_title="Score",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("Sort Students by Score"):
                    # Create ordered pairs of (score, name)
                    pairs = list(zip(demo_data, demo_labels))
                    # Sort by score
                    sorted_pairs = sorted(pairs, key=lambda x: x[0])
                    # Unpack
                    sorted_scores, sorted_names = zip(*sorted_pairs)
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=sorted_names,
                            y=sorted_scores,
                            marker_color='lightgreen',
                            text=sorted_scores,
                            textposition='outside'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Student Test Scores (Sorted)",
                        xaxis_title="Student",
                        yaxis_title="Score",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("Bubble sort is perfect for this educational setting because:")
                    st.markdown("- It's intuitive and easy to understand")
                    st.markdown("- The step-by-step process is easy to visualize")
                    st.markdown("- It works well for small datasets like a classroom example")
            
            elif real_world_algo == "Insertion Sort":
                st.markdown("#### 🃏 Card Sorting Simulation")
                
                # Simulate a hand of cards
                card_values = {"A": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, 
                            "8": 8, "9": 9, "10": 10, "J": 11, "Q": 12, "K": 13}
                
                card_suits = ["♥", "♦", "♣", "♠"]
                
                # Generate random cards
                random_cards = []
                for _ in range(7):
                    value = random.choice(list(card_values.keys()))
                    suit = random.choice(card_suits)
                    random_cards.append(f"{value}{suit}")
                
                st.markdown("You're dealt this hand of cards:")
                
                # Display cards horizontally
                cols = st.columns(len(random_cards))
                for i, card in enumerate(random_cards):
                    cols[i].markdown(f"""
                    <div style="border: 2px solid black; border-radius: 10px; padding: 10px; text-align: center; background-color: white; color: {'red' if card[-1] in ['♥', '♦'] else 'black'}; font-size: 24px;">
                        {card}
                    </div>
                    """, unsafe_allow_html=True)
                
                if st.button("Sort Cards by Value"):
                    # Sort cards by value
                    def card_value(card):
                        return card_values[card[:-1]]
                    
                    sorted_cards = sorted(random_cards, key=card_value)
                    
                    st.markdown("As you receive each card, you insert it into the correct position:")
                    
                    # Display sorted cards horizontally
                    cols = st.columns(len(sorted_cards))
                    for i, card in enumerate(sorted_cards):
                        cols[i].markdown(f"""
                        <div style="border: 2px solid black; border-radius: 10px; padding: 10px; text-align: center; background-color: white; color: {'red' if card[-1] in ['♥', '♦'] else 'black'}; font-size: 24px;">
                            {card}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("Insertion sort mimics how humans naturally sort cards:")
                    st.markdown("- We take one card at a time")
                    st.markdown("- Insert it in the right position among already-sorted cards")
                    st.markdown("- Very efficient for small datasets or nearly sorted data")
            
            elif real_world_algo == "Quick Sort":
                st.markdown("#### 💻 Operating System File Sorting")
                
                # Simulate files with sizes
                file_types = [".txt", ".jpg", ".pdf", ".mp3", ".docx", ".xlsx", ".html", ".zip"]
                file_names = ["report", "image", "document", "project", "backup", "data", "profile", "notes"]
                
                # Generate random files with sizes
                files = []
                for _ in range(10):
                    name = random.choice(file_names) + random.choice(file_types)
                    size = random.randint(1, 1000)  # Size in KB
                    files.append({"name": name, "size": size})
                
                st.markdown("Your file explorer showing files by size:")
                
                # Display files as a table
                df_files = pd.DataFrame(files)
                st.dataframe(df_files, use_container_width=True)
                
                if st.button("Sort Files by Size"):
                    # Sort files by size
                    sorted_files = sorted(files, key=lambda x: x["size"], reverse=True)
                    
                    st.markdown("Files sorted by size (largest first):")
                    
                    # Display sorted files
                    df_sorted = pd.DataFrame(sorted_files)
                    st.dataframe(df_sorted, use_container_width=True)
                    
                    # Create visualization of file sizes
                    fig = go.Figure(data=[
                        go.Bar(
                            x=[f["name"] for f in sorted_files],
                            y=[f["size"] for f in sorted_files],
                            marker_color='lightblue',
                            text=[f"{f['size']} KB" for f in sorted_files],
                            textposition='outside'
                        )
                    ])
                    
                    fig.update_layout(
                        title="Files Sorted by Size",
                        xaxis_title="Filename",
                        yaxis_title="Size (KB)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("Operating systems like Windows use Quicksort because:")
                    st.markdown("- It's very efficient for large datasets")
                    st.markdown("- Has good average-case performance")
                    st.markdown("- Works well with virtual memory systems")
                    st.markdown("- Efficiently handles diverse file sizes")
            
            elif real_world_algo == "Merge Sort":
                st.markdown("#### 📊 Database Query Result Merging")
                
                # Simulate database queries from different tables
                st.markdown("Imagine we have results from two database tables:")
                
                # Create two sorted datasets
                query1_data = sorted([random.randint(1, 100) for _ in range(5)])
                query2_data = sorted([random.randint(1, 100) for _ in range(7)])
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Query 1 Results (Users Table):**")
                    st.write(f"User IDs: {query1_data}")
                
                with col2:
                    st.markdown("**Query 2 Results (Orders Table):**")
                    st.write(f"Order IDs: {query2_data}")
                
                if st.button("Merge Query Results"):
                    # Merge the two sorted lists
                    merged_results = sorted(query1_data + query2_data)
                    
                    st.markdown("**Merged Results (Combined User and Order IDs):**")
                    
                    # Visualize the merge process
                    fig = go.Figure()
                    
                    # Add query1 data
                    fig.add_trace(go.Scatter(
                        x=list(range(len(query1_data))),
                        y=query1_data,
                        mode='markers+lines',
                        name='Users Table',
                        marker=dict(size=10, color='blue')
                    ))
                    
                    # Add query2 data (offset x to show as separate dataset)
                    x_offset = len(query1_data) + 1
                    fig.add_trace(go.Scatter(
                        x=[x_offset + i for i in range(len(query2_data))],
                        y=query2_data,
                        mode='markers+lines',
                        name='Orders Table',
                        marker=dict(size=10, color='green')
                    ))
                    
                    # Add merged data (offset x again)
                    x_offset = len(query1_data) + len(query2_data) + 2
                    fig.add_trace(go.Scatter(
                        x=[x_offset + i for i in range(len(merged_results))],
                        y=merged_results,
                        mode='markers+lines',
                        name='Merged Results',
                        marker=dict(size=10, color='red')
                    ))
                    
                    fig.update_layout(
                        title="Database Query Merging",
                        xaxis_title="Position",
                        yaxis_title="ID Value",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("Merge sort is ideal for database operations because:")
                    st.markdown("- It efficiently combines already-sorted results")
                    st.markdown("- Stable sorting preserves record order within same values")
                    st.markdown("- Performs well on large datasets typical in databases")
                    st.markdown("- Predictable performance regardless of initial data order")
            
            elif real_world_algo == "Heap Sort":
                st.markdown("#### ⏱️ Priority Queue for Task Scheduling")
                
                # Simulate a task scheduler with priorities
                st.markdown("Imagine an operating system scheduling tasks by priority:")
                
                # Generate random tasks with priorities
                tasks = []
                for i in range(8):
                    name = f"Task-{i+1}"
                    priority = random.randint(1, 10)
                    tasks.append({"id": name, "priority": priority})
                
                # Display tasks
                df_tasks = pd.DataFrame(tasks)
                st.dataframe(df_tasks, use_container_width=True)
                
                if st.button("Process Tasks by Priority"):
                    # Sort tasks by priority (highest first)
                    sorted_tasks = sorted(tasks, key=lambda x: x["priority"], reverse=True)
                    
                    st.markdown("**Tasks Executed in Priority Order:**")
                    
                    # Process tasks one by one with animation
                    task_order = []
                    execution_log = st.empty()
                    
                    for i, task in enumerate(sorted_tasks):
                        task_order.append(task)
                        remaining = sorted_tasks[i+1:] if i < len(sorted_tasks)-1 else []
                        
                        # Show current state
                        execution_log.markdown(f"""
                        **Task Executed:** {task['id']} (Priority: {task['priority']})
                        
                        **Remaining in Queue:** {len(remaining)} tasks
                        """)
                        
                        # Create visualization
                        fig = go.Figure()
                        
                        # Add executed tasks
                        if task_order:
                            fig.add_trace(go.Bar(
                                x=[t["id"] for t in task_order],
                                y=[t["priority"] for t in task_order],
                                name="Executed",
                                marker_color='lightgreen'
                            ))
                        
                        # Add remaining tasks
                        if remaining:
                            fig.add_trace(go.Bar(
                                x=[t["id"] for t in remaining],
                                y=[t["priority"] for t in remaining],
                                name="Waiting",
                                marker_color='lightblue'
                            ))
                        
                        fig.update_layout(
                            title="Task Execution by Priority",
                            xaxis_title="Task ID",
                            yaxis_title="Priority",
                            height=400,
                            barmode='group'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Pause briefly to show animation
                        time.sleep(0.5)
                    
                    st.success("All tasks completed!")
                    
                    st.markdown("Heap sort is perfect for task scheduling because:")
                    st.markdown("- It efficiently maintains a priority queue")
                    st.markdown("- O(log n) time to extract highest priority item")
                    st.markdown("- Easily adjusts as new tasks arrive")
                    st.markdown("- Memory efficient with O(1) extra space")
            
            elif real_world_algo == "Selection Sort":
                st.markdown("#### 💾 Memory-Constrained Embedded Systems")
                
                # Simulate a small embedded system with limited memory
                st.markdown("Imagine a small IoT device sorting sensor readings:")
                
                # Generate random sensor data
                sensor_data = [random.randint(10, 40) for _ in range(6)]  # Temperature readings
                
                # Display memory constraints
                st.markdown("""
                **Device Specifications:**
                - 8-bit microcontroller
                - 2KB RAM available
                - Flash memory with limited write cycles
                """)
                
                # Show unsorted sensor readings
                fig = go.Figure(data=[
                    go.Scatter(
                        x=list(range(len(sensor_data))),
                        y=sensor_data,
                        mode='markers+lines',
                        marker=dict(size=12, color='orange')
                    )
                ])
                
                fig.update_layout(
                    title="Unsorted Temperature Readings",
                    xaxis_title="Reading Number",
                    yaxis_title="Temperature (°C)",
                    height=350
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("Sort with Minimal Memory Usage"):
                    # Count the number of writes
                    writes = 0
                    comparisons = 0
                    
                    # Copy the data for sorting
                    sorted_data = sensor_data.copy()
                    
                    # Perform selection sort while counting operations
                    for i in range(len(sorted_data)):
                        min_idx = i
                        for j in range(i+1, len(sorted_data)):
                            comparisons += 1
                            if sorted_data[j] < sorted_data[min_idx]:
                                min_idx = j
                        
                        if min_idx != i:
                            sorted_data[i], sorted_data[min_idx] = sorted_data[min_idx], sorted_data[i]
                            writes += 1
                    
                    # Show sorted data
                    fig = go.Figure(data=[
                        go.Scatter(
                            x=list(range(len(sorted_data))),
                            y=sorted_data,
                            mode='markers+lines',
                            marker=dict(size=12, color='green')
                        )
                    ])
                    
                    fig.update_layout(
                        title="Sorted Temperature Readings",
                        xaxis_title="Reading Number",
                        yaxis_title="Temperature (°C)",
                        height=350
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show memory efficiency
                    col1, col2 = st.columns(2)
                    col1.metric("Memory Writes", writes)
                    col2.metric("Comparisons", comparisons)
                    
                    st.markdown("Selection sort is ideal for embedded systems because:")
                    st.markdown("- Minimizes memory writes (critical for flash memory)")
                    st.markdown("- O(1) extra space requirement (works in-place)")
                    st.markdown("- Simple implementation for constrained devices")
                    st.markdown("- Predictable performance regardless of data order")
    
    # Algorithm information section
    st.markdown("---")
    st.subheader("🧠 Sorting Algorithm Reference")
    
    # Create tabs for different algorithms
    sort_algo_tabs = st.tabs(["Current Algorithm", "All Algorithms Comparison", "Complexity Analysis"])
    
    with sort_algo_tabs[0]:
        if sort_algorithm in SORTING_INFO:
            algo_info = SORTING_INFO[sort_algorithm]
            
            st.markdown(f"### {sort_algorithm}")
            st.markdown(f"**Description:** {algo_info['description']}")
            
            # Complexity badges
            st.markdown("**Time Complexity:**")
            col1, col2, col3 = st.columns(3)
            col1.markdown(f'<span class="complexity-badge complexity-best">Best: {algo_info["best_case"]}</span>', unsafe_allow_html=True)
            col2.markdown(f'<span class="complexity-badge complexity-average">Average: {algo_info["average_case"]}</span>', unsafe_allow_html=True)
            col3.markdown(f'<span class="complexity-badge complexity-worst">Worst: {algo_info["worst_case"]}</span>', unsafe_allow_html=True)
            
            st.markdown(f"**Space Complexity:** {algo_info['space_complexity']}")
            st.markdown(f"**Stable:** {algo_info['stable']}")
            st.markdown(f"**Best Use Case:** {algo_info['use_case']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Advantages:**")
                for pro in algo_info['pros']:
                    st.markdown(f"✅ {pro}")
            
            with col2:
                st.markdown("**Disadvantages:**")
                for con in algo_info['cons']:
                    st.markdown(f"❌ {con}")
    
    with sort_algo_tabs[1]:
        # Create comprehensive comparison table
        comparison_data = []
        for algo_name, info in SORTING_INFO.items():
            comparison_data.append({
                "Algorithm": algo_name,
                "Best Case": info["best_case"],
                "Average Case": info["average_case"],
                "Worst Case": info["worst_case"],
                "Space": info["space_complexity"],
                "Stable": info["stable"],
                "Use Case": info["use_case"]
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        # Visual comparison chart
        complexity_scores = {
            "O(1)": 1, "O(log n)": 2, "O(n)": 3, "O(n log n)": 4, "O(n²)": 5
        }
        
        chart_data = []
        for algo_name, info in SORTING_INFO.items():
            chart_data.append({
                "Algorithm": algo_name,
                "Best": complexity_scores.get(info["best_case"], 3),
                "Average": complexity_scores.get(info["average_case"], 3),
                "Worst": complexity_scores.get(info["worst_case"], 3)
            })
        
        df_chart = pd.DataFrame(chart_data)
        
        fig_complexity = go.Figure()
        
        fig_complexity.add_trace(go.Bar(
            name='Best Case',
            x=df_chart['Algorithm'],
            y=df_chart['Best'],
            marker_color='lightgreen'
        ))
        
        fig_complexity.add_trace(go.Bar(
            name='Average Case',
            x=df_chart['Algorithm'],
            y=df_chart['Average'],
            marker_color='lightblue'
        ))
        
        fig_complexity.add_trace(go.Bar(
            name='Worst Case',
            x=df_chart['Algorithm'],
            y=df_chart['Worst'],
            marker_color='lightcoral'
        ))
        
        fig_complexity.update_layout(
            title='Time Complexity Comparison (Lower is Better)',
            xaxis_title='Algorithm',
            yaxis_title='Complexity Score',
            barmode='group',
            height=400,
            yaxis=dict(
                tickvals=[1, 2, 3, 4, 5],
                ticktext=['O(1)', 'O(log n)', 'O(n)', 'O(n log n)', 'O(n²)']
            )
        )
        
        st.plotly_chart(fig_complexity, use_container_width=True)
    
    with sort_algo_tabs[2]:
        st.markdown("""
        ### Understanding Algorithm Complexity
        
        **Time Complexity** measures how running time increases with input size:
        - **O(1)**: Constant time - doesn't depend on input size
        - **O(log n)**: Logarithmic time - very efficient, divides problem in half
        - **O(n)**: Linear time - increases linearly with input size
        - **O(n log n)**: Linearithmic time - efficient for large datasets (optimal for comparison sorts)
        - **O(n²)**: Quadratic time - suitable only for small datasets
        
        **Space Complexity** measures extra memory needed:
        - **O(1)**: In-place algorithms (constant extra space)
        - **O(log n)**: Logarithmic space (usually for recursion stack)
        - **O(n)**: Linear extra space needed (like merge sort's temporary arrays)
        
        **Stability** means equal elements maintain their relative order after sorting.
        
        **When to Use Each Algorithm:**
        - **Small arrays (< 50 elements)**: Insertion Sort
        - **General purpose**: Quick Sort or Merge Sort
        - **Guaranteed O(n log n)**: Merge Sort or Heap Sort
        - **Memory constrained**: Heap Sort or Quick Sort
        - **Stable sorting needed**: Merge Sort or Insertion Sort
        - **Educational purposes**: Bubble Sort or Selection Sort
        """)
        
        # Performance tips
        st.markdown("""
        ### 💡 Performance Tips
        
        **Optimization Strategies:**
        1. **Hybrid approaches**: Use insertion sort for small subarrays in quick/merge sort
        2. **Pivot selection**: Use median-of-three for quick sort to avoid worst case
        3. **Early termination**: Stop bubble sort if no swaps occur in a pass
        4. **Adaptive algorithms**: Insertion sort performs well on nearly sorted data
        5. **Cache efficiency**: Quick sort has better cache performance than merge sort
        
        **Real-world Considerations:**
        - Modern languages often use hybrid algorithms (Timsort in Python, Introsort in C++)
        - Consider data characteristics: size, initial order, stability requirements
        - For very large datasets, consider external sorting algorithms
        - Parallel sorting algorithms can leverage multiple CPU cores
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h3>🚀 Advanced PathFinder & Sort Visualizer</h3>
    <div style='display: flex; justify-content: center; gap: 40px; margin: 20px 0;'>
        <div>
            <h4>🗺️ Pathfinding Features</h4>
            <p>✅ Interactive grid-based pathfinding</p>
            <p>✅ Real-world map integration</p>
            <p>✅ 6 different algorithms</p>
            <p>✅ Performance comparison</p>
            <p>✅ Multiple map types & obstacles</p>
        </div>
        <div>
            <h4>📊 Sorting Features</h4>
            <p>✅ Animated step-by-step visualization</p>
            <p>✅ 6 sorting algorithms</p>
            <p>✅ Multiple array types</p>
            <p>✅ Performance metrics</p>
            <p>✅ Algorithm comparison</p>
            <p>✅ Real-world applications</p>
        </div>
    </div>
    <p><strong>Built with ❤️ using:</strong> Streamlit • Plotly • Folium • NumPy • Pandas</p>
    <p class="footer-credit">Developed with love by Shreyas Kasture</p>
</div>
""", unsafe_allow_html=True)

# Additional features and improvements
if st.sidebar.button("🔧 Show Advanced Settings"):
    with st.sidebar.expander("Advanced Configuration", expanded=True):
        st.markdown("### 🎛️ Advanced Settings")
        
        # Performance settings
        st.markdown("**Performance Optimization:**")
        enable_caching = st.checkbox("Enable Result Caching", value=True)
        max_grid_size = st.slider("Max Grid Size", 20, 100, 50)
        animation_quality = st.selectbox("Animation Quality", ["High", "Medium", "Low"])
        
        # Debug settings
        st.markdown("**Debug Options:**")
        show_debug_info = st.checkbox("Show Debug Information")
        verbose_logging = st.checkbox("Verbose Logging")
        
        # Export settings
        st.markdown("**Export Options:**")
        if st.button("📁 Export Results"):
            st.info("Export functionality would save current results to file")
        
        if show_debug_info:
            st.markdown("**Debug Information:**")
            st.json({
                "session_state_keys": list(st.session_state.keys()),
                "current_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "streamlit_version": st.__version__
            })

# Session state cleanup
if st.sidebar.button("🧹 Clear All Data"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# JavaScript event handlers for interactive clicking
st.markdown("""
<script>
// Handle click events for the interactive grid and map
document.addEventListener('DOMContentLoaded', function() {
    // Set up observers to detect when new charts are added
    const observer = new MutationObserver(function(mutations) {
        setupClickHandlers();
    });
    
    // Start observing the document body for changes
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Initial setup
    setupClickHandlers();
});

function setupClickHandlers() {
    // Look for plotly charts and attach click handlers
    const gridPlots = document.querySelectorAll('[data-testid="stPlotlyChart"] .js-plotly-plot');
    
    gridPlots.forEach(plot => {
        // Only attach if not already attached
        if (!plot.getAttribute('data-handler-attached')) {
            plot.setAttribute('data-handler-attached', 'true');
            
            plot.on('plotly_click', function(data) {
                const clickData = {
                    x: data.points[0].x,
                    y: data.points[0].y
                };
                
                // Send to Streamlit
                window.parent.postMessage({
                    type: "streamlit:setComponentValue",
                    value: clickData,
                    key: "last_click_data"
                }, "*");
            });
        }
    });
}
</script>
""", unsafe_allow_html=True)
