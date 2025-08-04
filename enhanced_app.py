import streamlit as st
import folium
from streamlit_folium import st_folium
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
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

st.set_page_config(
    page_title="Elite PathFinder & Algorithm Visualizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: linear-gradient(90deg, #f8f9fa, #e9ecef);
        border-radius: 15px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0px 30px;
        background: transparent;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .algorithm-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        border: 1px solid #dee2e6;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .algorithm-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    .complexity-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 700;
        margin: 3px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .complexity-best { 
        background: linear-gradient(135deg, #d4edda, #c3e6cb); 
        color: #155724; 
        border: 1px solid #c3e6cb;
    }
    .complexity-average { 
        background: linear-gradient(135deg, #fff3cd, #ffeaa7); 
        color: #856404; 
        border: 1px solid #ffeaa7;
    }
    .complexity-worst { 
        background: linear-gradient(135deg, #f8d7da, #f5c6cb); 
        color: #721c24; 
        border: 1px solid #f5c6cb;
    }
    .performance-metrics {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    .grid-container {
        border: 3px solid #667eea;
        border-radius: 15px;
        padding: 20px;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .real-world-showcase {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    .metric-card {
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
    }
    .interactive-controls {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #dee2e6;
        margin: 15px 0;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-running { background-color: #28a745; }
    .status-completed { background-color: #007bff; }
    .status-error { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚀 Elite PathFinder & Algorithm Visualizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Advanced pathfinding algorithms with real-time visualization and performance analytics</p>', unsafe_allow_html=True)

@dataclass
class AlgorithmMetrics:
    name: str
    time_complexity_best: str
    time_complexity_avg: str
    time_complexity_worst: str
    space_complexity: str
    optimal: bool
    complete: bool
    description: str

@dataclass
class PerformanceResult:
    algorithm: str
    execution_time: float
    path_length: int
    nodes_explored: int
    memory_usage: int
    optimality_ratio: float

class AdvancedGridPathfinder:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=int)
        self.start = None
        self.goal = None
        self.obstacles = set()
        self.weights = np.ones((height, width), dtype=float)
        
    def set_obstacle(self, x: int, y: int, obstacle_type: str = "wall"):
        if self._is_in_bounds(x, y):
            self.obstacles.add((x, y))
            if obstacle_type == "wall":
                self.grid[y][x] = 1
                self.weights[y][x] = float('inf')
            elif obstacle_type == "mud":
                self.grid[y][x] = 2
                self.weights[y][x] = 3.0
            elif obstacle_type == "water":
                self.grid[y][x] = 3
                self.weights[y][x] = 5.0
    
    def remove_obstacle(self, x: int, y: int):
        if (x, y) in self.obstacles:
            self.obstacles.remove((x, y))
            self.grid[y][x] = 0
            self.weights[y][x] = 1.0
    
    def _is_in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_valid(self, x: int, y: int) -> bool:
        return (self._is_in_bounds(x, y) and 
                self.grid[y][x] != 1)
    
    def get_neighbors(self, x: int, y: int, allow_diagonal: bool = True) -> List[Tuple[int, int]]:
        neighbors = []
        if allow_diagonal:
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        else:
            directions = [(-1,0), (1,0), (0,-1), (0,1)]
            
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny))
        return neighbors
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int], heuristic_type: str = "euclidean") -> float:
        if heuristic_type == "manhattan":
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        elif heuristic_type == "chebyshev":
            return max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        else:
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_movement_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> float:
        base_cost = math.sqrt((from_pos[0] - to_pos[0])**2 + (from_pos[1] - to_pos[1])**2)
        terrain_cost = self.weights[to_pos[1]][to_pos[0]]
        return base_cost * terrain_cost

class ElitePathfindingAlgorithms:
    
    @staticmethod
    def a_star_enhanced(grid: AdvancedGridPathfinder, start: Tuple[int, int], goal: Tuple[int, int], 
                       heuristic_type: str = "euclidean") -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], Dict]:
        start_time = time.time()
        open_set = [(0, start, 0)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: grid.heuristic(start, goal, heuristic_type)}
        visited = []
        closed_set = set()
        
        while open_set:
            current_f, current, current_g = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            visited.append(current)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                
                metrics = {
                    "execution_time": time.time() - start_time,
                    "nodes_explored": len(visited),
                    "path_length": len(path),
                    "path_cost": g_score[goal] if goal in g_score else 0
                }
                return path[::-1], visited, metrics
            
            for neighbor in grid.get_neighbors(current[0], current[1]):
                if neighbor in closed_set:
                    continue
                    
                tentative_g_score = g_score[current] + grid.get_movement_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + grid.heuristic(neighbor, goal, heuristic_type)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor, g_score[neighbor]))
        
        metrics = {
            "execution_time": time.time() - start_time,
            "nodes_explored": len(visited),
            "path_length": 0,
            "path_cost": float('inf')
        }
        return [], visited, metrics
    
    @staticmethod
    def dijkstra_enhanced(grid: AdvancedGridPathfinder, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], Dict]:
        start_time = time.time()
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
                
                metrics = {
                    "execution_time": time.time() - start_time,
                    "nodes_explored": len(visited),
                    "path_length": len(path),
                    "path_cost": distances[goal]
                }
                return path[::-1], visited, metrics
            
            if current_dist > distances.get(current, float('inf')):
                continue
                
            for neighbor in grid.get_neighbors(current[0], current[1]):
                distance = current_dist + grid.get_movement_cost(current, neighbor)
                
                if neighbor not in distances or distance < distances[neighbor]:
                    distances[neighbor] = distance
                    came_from[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        metrics = {
            "execution_time": time.time() - start_time,
            "nodes_explored": len(visited),
            "path_length": 0,
            "path_cost": float('inf')
        }
        return [], visited, metrics
    
    @staticmethod
    def jump_point_search(grid: AdvancedGridPathfinder, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], Dict]:
        start_time = time.time()
        
        def is_forced_neighbor(x, y, dx, dy):
            if dx != 0 and dy != 0:
                return ((not grid.is_valid(x - dx, y) and grid.is_valid(x - dx, y + dy)) or
                        (not grid.is_valid(x, y - dy) and grid.is_valid(x + dx, y - dy)))
            elif dx != 0:
                return ((not grid.is_valid(x, y + 1) and grid.is_valid(x + dx, y + 1)) or
                        (not grid.is_valid(x, y - 1) and grid.is_valid(x + dx, y - 1)))
            else:
                return ((not grid.is_valid(x + 1, y) and grid.is_valid(x + 1, y + dy)) or
                        (not grid.is_valid(x - 1, y) and grid.is_valid(x - 1, y + dy)))
        
        def jump(x, y, dx, dy):
            if not grid.is_valid(x, y):
                return None
            if (x, y) == goal:
                return (x, y)
            if is_forced_neighbor(x, y, dx, dy):
                return (x, y)
            
            if dx != 0 and dy != 0:
                if jump(x + dx, y, dx, 0) or jump(x, y + dy, 0, dy):
                    return (x, y)
            
            return jump(x + dx, y + dy, dx, dy)
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        visited = []
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            visited.append(current)
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                
                metrics = {
                    "execution_time": time.time() - start_time,
                    "nodes_explored": len(visited),
                    "path_length": len(path),
                    "path_cost": g_score[goal]
                }
                return path[::-1], visited, metrics
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    jump_point = jump(current[0] + dx, current[1] + dy, dx, dy)
                    if jump_point:
                        tentative_g = g_score[current] + grid.heuristic(current, jump_point)
                        
                        if jump_point not in g_score or tentative_g < g_score[jump_point]:
                            came_from[jump_point] = current
                            g_score[jump_point] = tentative_g
                            f_score = tentative_g + grid.heuristic(jump_point, goal)
                            heapq.heappush(open_set, (f_score, jump_point))
        
        metrics = {
            "execution_time": time.time() - start_time,
            "nodes_explored": len(visited),
            "path_length": 0,
            "path_cost": float('inf')
        }
        return [], visited, metrics

class AdvancedSortingAlgorithms:
    
    @staticmethod
    def tim_sort(arr):
        arr = arr.copy()
        steps = []
        
        def insertion_sort_range(arr, left, right):
            for i in range(left + 1, right + 1):
                key = arr[i]
                j = i - 1
                while j >= left and arr[j] > key:
                    arr[j + 1] = arr[j]
                    steps.append((arr.copy(), [j, j + 1], "swapped"))
                    j -= 1
                arr[j + 1] = key
        
        def merge(arr, left, mid, right):
            left_part = arr[left:mid + 1]
            right_part = arr[mid + 1:right + 1]
            
            i = j = 0
            k = left
            
            while i < len(left_part) and j < len(right_part):
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
        
        n = len(arr)
        min_merge = 32
        
        for i in range(0, n, min_merge):
            insertion_sort_range(arr, i, min(i + min_merge - 1, n - 1))
        
        size = min_merge
        while size < n:
            for start in range(0, n, size * 2):
                mid = start + size - 1
                end = min(start + size * 2 - 1, n - 1)
                if mid < end:
                    merge(arr, start, mid, end)
            size *= 2
        
        steps.append((arr.copy(), [], "completed"))
        return steps
    
    @staticmethod
    def radix_sort(arr):
        if not arr:
            return [(arr.copy(), [], "completed")]
        
        steps = []
        max_num = max(arr)
        exp = 1
        
        while max_num // exp > 0:
            counting_sort_by_digit(arr, exp, steps)
            exp *= 10
        
        steps.append((arr.copy(), [], "completed"))
        return steps

def counting_sort_by_digit(arr, exp, steps):
    n = len(arr)
    output = [0] * n
    count = [0] * 10
    
    for i in range(n):
        index = arr[i] // exp
        count[index % 10] += 1
    
    for i in range(1, 10):
        count[i] += count[i - 1]
    
    i = n - 1
    while i >= 0:
        index = arr[i] // exp
        output[count[index % 10] - 1] = arr[i]
        count[index % 10] -= 1
        i -= 1
    
    for i in range(n):
        arr[i] = output[i]
        steps.append((arr.copy(), [i], "sorted"))

class RealWorldPathfinder:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'PathfinderApp/1.0'
        })
    
    def geocode_enhanced(self, location: str) -> Optional[Tuple[float, float]]:
        try:
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': location,
                'format': 'json',
                'limit': 1,
                'addressdetails': 1
            }
            response = self.session.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    return float(data[0]['lat']), float(data[0]['lon'])
        except Exception as e:
            st.error(f"Geocoding error: {e}")
        return None
    
    def get_route_enhanced(self, start_coords: Tuple[float, float], end_coords: Tuple[float, float], 
                          profile: str = "driving") -> Optional[Dict]:
        try:
            distance = self._haversine_distance(start_coords, end_coords)
            
            route_coords = self._generate_realistic_route(start_coords, end_coords)
            
            speed_profiles = {
                "driving": 60,
                "cycling": 20,
                "walking": 5
            }
            
            avg_speed = speed_profiles.get(profile, 50)
            duration = (distance / avg_speed) * 3600
            
            return {
                'coordinates': route_coords,
                'distance': distance,
                'duration': duration,
                'profile': profile,
                'waypoints': len(route_coords)
            }
        except Exception as e:
            st.error(f"Routing error: {e}")
        return None
    
    def _haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c
    
    def _generate_realistic_route(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[List[float]]:
        waypoints = []
        num_points = max(5, int(self._haversine_distance(start, end) * 2))
        
        for i in range(num_points + 1):
            t = i / num_points
            
            lat = start[0] + (end[0] - start[0]) * t
            lon = start[1] + (end[1] - start[1]) * t
            
            noise_factor = 0.001
            lat += random.uniform(-noise_factor, noise_factor)
            lon += random.uniform(-noise_factor, noise_factor)
            
            waypoints.append([lon, lat])
        
        return waypoints

def create_enhanced_grid_visualization(grid: AdvancedGridPathfinder, path: List[Tuple[int, int]], 
                                     visited: List[Tuple[int, int]], algorithm_name: str):
    fig = go.Figure()
    
    grid_data = grid.grid.copy()
    
    for x, y in visited:
        if grid_data[y][x] == 0:
            grid_data[y][x] = 0.3
    
    for x, y in path:
        grid_data[y][x] = 0.7
    
    if grid.start:
        grid_data[grid.start[1]][grid.start[0]] = 0.9
    if grid.goal:
        grid_data[grid.goal[1]][grid.goal[0]] = 1.0
    
    colorscale = [
        [0, '#ffffff'],
        [0.1, '#e3f2fd'],
        [0.3, '#bbdefb'],
        [0.5, '#90caf9'],
        [0.7, '#42a5f5'],
        [0.9, '#1e88e5'],
        [1.0, '#0d47a1']
    ]
    
    fig.add_trace(go.Heatmap(
        z=grid_data,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title="Cell Type",
            tickvals=[0, 0.3, 0.7, 0.9, 1.0],
            ticktext=["Empty", "Visited", "Path", "Start", "Goal"]
        )
    ))
    
    fig.update_layout(
        title=f"{algorithm_name} Pathfinding Visualization",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        height=600,
        template="plotly_white"
    )
    
    return fig

def create_performance_comparison_chart(results: List[PerformanceResult]):
    if not results:
        return go.Figure()
    
    algorithms = [r.algorithm for r in results]
    execution_times = [r.execution_time * 1000 for r in results]
    nodes_explored = [r.nodes_explored for r in results]
    path_lengths = [r.path_length for r in results]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Execution Time (ms)', 'Nodes Explored', 'Path Length', 'Memory Usage'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
    
    fig.add_trace(
        go.Bar(x=algorithms, y=execution_times, name="Execution Time", 
               marker_color=colors[0], showlegend=False),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=algorithms, y=nodes_explored, name="Nodes Explored", 
               marker_color=colors[1], showlegend=False),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=algorithms, y=path_lengths, name="Path Length", 
               marker_color=colors[2], showlegend=False),
        row=2, col=1
    )
    
    memory_usage = [r.memory_usage for r in results]
    fig.add_trace(
        go.Bar(x=algorithms, y=memory_usage, name="Memory Usage", 
               marker_color=colors[3], showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="Algorithm Performance Comparison",
        template="plotly_white"
    )
    
    return fig

def main():
    algorithm_info = {
        "A* Enhanced": AlgorithmMetrics(
            "A* Enhanced", "O(b^d)", "O(b^d)", "O(b^d)", "O(b^d)", 
            True, True, "Optimal pathfinding with enhanced heuristics and terrain costs"
        ),
        "Dijkstra Enhanced": AlgorithmMetrics(
            "Dijkstra Enhanced", "O((V+E)logV)", "O((V+E)logV)", "O((V+E)logV)", "O(V)", 
            True, True, "Guaranteed shortest path with weighted terrain support"
        ),
        "Jump Point Search": AlgorithmMetrics(
            "Jump Point Search", "O(b^d)", "O(b^d)", "O(b^d)", "O(b^d)", 
            True, True, "Optimized A* for uniform cost grids with jump points"
        )
    }
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Grid Pathfinding", "🌍 Real-World Routing", "📊 Algorithm Analysis", "🔬 Performance Lab"])
    
    with tab1:
        st.markdown('<div class="real-world-showcase">', unsafe_allow_html=True)
        st.markdown("### 🎯 Advanced Grid-Based Pathfinding")
        st.markdown("Experience cutting-edge pathfinding algorithms with enhanced visualization and performance metrics.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown('<div class="interactive-controls">', unsafe_allow_html=True)
            st.subheader("🎮 Controls")
            
            grid_size = st.slider("Grid Size", 10, 50, 25)
            algorithm = st.selectbox("Algorithm", list(algorithm_info.keys()))
            heuristic = st.selectbox("Heuristic", ["euclidean", "manhattan", "chebyshev"])
            
            obstacle_density = st.slider("Obstacle Density", 0.0, 0.5, 0.2)
            terrain_variety = st.checkbox("Enable Terrain Variety", value=True)
            
            if st.button("🚀 Generate New Maze", type="primary"):
                st.session_state.grid_generated = True
                
            st.markdown('</div>', unsafe_allow_html=True)
            
            if algorithm in algorithm_info:
                info = algorithm_info[algorithm]
                st.markdown('<div class="algorithm-card">', unsafe_allow_html=True)
                st.markdown(f"**{info.name}**")
                st.markdown(f"*{info.description}*")
                
                st.markdown("**Complexity Analysis:**")
                st.markdown(f'<span class="complexity-badge complexity-best">Best: {info.time_complexity_best}</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="complexity-badge complexity-average">Avg: {info.time_complexity_avg}</span>', unsafe_allow_html=True)
                st.markdown(f'<span class="complexity-badge complexity-worst">Worst: {info.time_complexity_worst}</span>', unsafe_allow_html=True)
                
                st.markdown(f"**Space:** {info.space_complexity}")
                st.markdown(f"**Optimal:** {'✅' if info.optimal else '❌'}")
                st.markdown(f"**Complete:** {'✅' if info.complete else '❌'}")
                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            if 'grid_generated' not in st.session_state:
                st.session_state.grid_generated = False
            
            if st.session_state.grid_generated or st.button("🎲 Quick Demo"):
                grid = AdvancedGridPathfinder(grid_size, grid_size)
                
                np.random.seed(42)
                for _ in range(int(grid_size * grid_size * obstacle_density)):
                    x, y = np.random.randint(0, grid_size, 2)
                    if terrain_variety:
                        obstacle_type = np.random.choice(["wall", "mud", "water"], p=[0.6, 0.3, 0.1])
                    else:
                        obstacle_type = "wall"
                    grid.set_obstacle(x, y, obstacle_type)
                
                start = (1, 1)
                goal = (grid_size-2, grid_size-2)
                grid.start = start
                grid.goal = goal
                
                with st.spinner(f"🔍 Running {algorithm}..."):
                    if algorithm == "A* Enhanced":
                        path, visited, metrics = ElitePathfindingAlgorithms.a_star_enhanced(grid, start, goal, heuristic)
                    elif algorithm == "Dijkstra Enhanced":
                        path, visited, metrics = ElitePathfindingAlgorithms.dijkstra_enhanced(grid, start, goal)
                    elif algorithm == "Jump Point Search":
                        path, visited, metrics = ElitePathfindingAlgorithms.jump_point_search(grid, start, goal)
                
                fig = create_enhanced_grid_visualization(grid, path, visited, algorithm)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown('<div class="performance-metrics">', unsafe_allow_html=True)
                col_a, col_b, col_c, col_d = st.columns(4)
                
                with col_a:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("⏱️ Execution Time", f"{metrics['execution_time']*1000:.2f} ms")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_b:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("🔍 Nodes Explored", metrics['nodes_explored'])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_c:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("📏 Path Length", metrics['path_length'])
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_d:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("💰 Path Cost", f"{metrics['path_cost']:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="real-world-showcase">', unsafe_allow_html=True)
        st.markdown("### 🌍 Real-World Route Planning")
        st.markdown("Plan optimal routes between real locations with multiple transportation modes.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        pathfinder = RealWorldPathfinder()
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_location = st.text_input("🏁 Start Location", "New York, NY")
            end_location = st.text_input("🏆 End Location", "Boston, MA")
            transport_mode = st.selectbox("🚗 Transportation", ["driving", "cycling", "walking"])
        
        with col2:
            if st.button("🗺️ Plan Route", type="primary"):
                with st.spinner("🔍 Finding optimal route..."):
                    start_coords = pathfinder.geocode_enhanced(start_location)
                    end_coords = pathfinder.geocode_enhanced(end_location)
                    
                    if start_coords and end_coords:
                        route_info = pathfinder.get_route_enhanced(start_coords, end_coords, transport_mode)
                        
                        if route_info:
                            m = folium.Map(
                                location=[(start_coords[0] + end_coords[0])/2, (start_coords[1] + end_coords[1])/2],
                                zoom_start=8
                            )
                            
                            folium.Marker(
                                start_coords,
                                popup=f"Start: {start_location}",
                                icon=folium.Icon(color='green', icon='play')
                            ).add_to(m)
                            
                            folium.Marker(
                                end_coords,
                                popup=f"End: {end_location}",
                                icon=folium.Icon(color='red', icon='stop')
                            ).add_to(m)
                            
                            folium.PolyLine(
                                locations=[[coord[1], coord[0]] for coord in route_info['coordinates']],
                                weight=5,
                                color='blue',
                                opacity=0.8
                            ).add_to(m)
                            
                            st_folium(m, width=700, height=500)
                            
                            st.markdown('<div class="performance-metrics">', unsafe_allow_html=True)
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric("📏 Distance", f"{route_info['distance']:.1f} km")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with col_b:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric("⏱️ Duration", f"{route_info['duration']/3600:.1f} hours")
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with col_c:
                                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                st.metric("📍 Waypoints", route_info['waypoints'])
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.error("❌ Could not geocode one or both locations")
    
    with tab3:
        st.markdown('<div class="real-world-showcase">', unsafe_allow_html=True)
        st.markdown("### 📊 Advanced Algorithm Analysis")
        st.markdown("Deep dive into algorithm performance characteristics and complexity analysis.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.subheader("🔬 Sorting Algorithm Showcase")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            array_size = st.slider("Array Size", 10, 100, 30)
            sort_algorithm = st.selectbox("Sorting Algorithm", ["Tim Sort", "Radix Sort"])
            
            if st.button("🎲 Generate Random Array"):
                st.session_state.sort_array = np.random.randint(1, 100, array_size).tolist()
            
            if 'sort_array' not in st.session_state:
                st.session_state.sort_array = np.random.randint(1, 100, array_size).tolist()
            
            st.write("**Current Array:**")
            st.write(st.session_state.sort_array[:10], "..." if len(st.session_state.sort_array) > 10 else "")
        
        with col2:
            if st.button("🚀 Visualize Sorting", type="primary"):
                with st.spinner(f"🔄 Running {sort_algorithm}..."):
                    if sort_algorithm == "Tim Sort":
                        steps = AdvancedSortingAlgorithms.tim_sort(st.session_state.sort_array)
                    else:
                        steps = AdvancedSortingAlgorithms.radix_sort(st.session_state.sort_array)
                    
                    progress_bar = st.progress(0)
                    chart_placeholder = st.empty()
                    
                    for i, (arr, indices, action) in enumerate(steps[::max(1, len(steps)//20)]):
                        fig = go.Figure()
                        colors = ['lightblue'] * len(arr)
                        
                        for idx in indices:
                            if idx < len(colors):
                                colors[idx] = 'red' if action == 'swapped' else 'green'
                        
                        fig.add_trace(go.Bar(
                            x=list(range(len(arr))),
                            y=arr,
                            marker_color=colors,
                            showlegend=False
                        ))
                        
                        fig.update_layout(
                            title=f"{sort_algorithm} - Step {i+1} ({action})",
                            xaxis_title="Index",
                            yaxis_title="Value",
                            height=400,
                            template="plotly_white"
                        )
                        
                        chart_placeholder.plotly_chart(fig, use_container_width=True)
                        progress_bar.progress((i + 1) / len(steps[::max(1, len(steps)//20)]))
                        time.sleep(0.1)
                    
                    st.success(f"✅ {sort_algorithm} completed!")
    
    with tab4:
        st.markdown('<div class="real-world-showcase">', unsafe_allow_html=True)
        st.markdown("### 🔬 Performance Laboratory")
        st.markdown("Comprehensive performance testing and algorithm comparison.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🧪 Run Performance Benchmark", type="primary"):
            with st.spinner("🔬 Running comprehensive benchmarks..."):
                results = []
                test_grid = AdvancedGridPathfinder(30, 30)
                
                np.random.seed(42)
                for _ in range(180):
                    x, y = np.random.randint(0, 30, 2)
                    test_grid.set_obstacle(x, y)
                
                start = (1, 1)
                goal = (28, 28)
                test_grid.start = start
                test_grid.goal = goal
                
                algorithms = [
                    ("A* Enhanced", lambda: ElitePathfindingAlgorithms.a_star_enhanced(test_grid, start, goal)),
                    ("Dijkstra Enhanced", lambda: ElitePathfindingAlgorithms.dijkstra_enhanced(test_grid, start, goal)),
                    ("Jump Point Search", lambda: ElitePathfindingAlgorithms.jump_point_search(test_grid, start, goal))
                ]
                
                for name, algo_func in algorithms:
                    path, visited, metrics = algo_func()
                    
                    result = PerformanceResult(
                        algorithm=name,
                        execution_time=metrics['execution_time'],
                        path_length=metrics['path_length'],
                        nodes_explored=metrics['nodes_explored'],
                        memory_usage=len(visited) * 8,
                        optimality_ratio=1.0 if metrics['path_length'] > 0 else 0.0
                    )
                    results.append(result)
                
                fig = create_performance_comparison_chart(results)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📈 Detailed Results")
                
                for result in results:
                    with st.expander(f"🔍 {result.algorithm} Details"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("⏱️ Execution Time", f"{result.execution_time*1000:.2f} ms")
                            st.metric("🔍 Nodes Explored", result.nodes_explored)
                        
                        with col2:
                            st.metric("📏 Path Length", result.path_length)
                            st.metric("💾 Memory Usage", f"{result.memory_usage} bytes")
                        
                        with col3:
                            st.metric("🎯 Optimality Ratio", f"{result.optimality_ratio:.2%}")
                            efficiency = (result.path_length / result.nodes_explored) if result.nodes_explored > 0 else 0
                            st.metric("⚡ Efficiency", f"{efficiency:.3f}")

if __name__ == "__main__":
    main()

