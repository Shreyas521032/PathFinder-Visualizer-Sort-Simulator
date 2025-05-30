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
    .interactive-grid {
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 10px;
        background: white;
    }
    .real-world-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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

# Real-world sorting applications
class RealWorldSortingApplications:
    @staticmethod
    def generate_student_data(n=50):
        """Generate sample student data for educational sorting"""
        names = ["Alex", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"] * (n//10 + 1)
        subjects = ["Math", "Science", "English", "History", "Art"]
        
        students = []
        for i in range(n):
            student = {
                "name": f"{names[i % len(names)]}{i//10 + 1}",
                "score": random.randint(60, 100),
                "age": random.randint(18, 25),
                "subject": random.choice(subjects),
                "grade": ""
            }
            # Assign grades based on score
            if student["score"] >= 90:
                student["grade"] = "A"
            elif student["score"] >= 80:
                student["grade"] = "B"
            elif student["score"] >= 70:
                student["grade"] = "C"
            else:
                student["grade"] = "D"
            
            students.append(student)
        
        return students
    
    @staticmethod
    def generate_employee_data(n=30):
        """Generate sample employee data for HR sorting"""
        departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]
        positions = ["Junior", "Senior", "Lead", "Manager", "Director"]
        
        employees = []
        for i in range(n):
            employee = {
                "id": f"EMP{1000 + i}",
                "name": f"Employee {i+1}",
                "salary": random.randint(40000, 150000),
                "department": random.choice(departments),
                "position": random.choice(positions),
                "experience": random.randint(1, 20),
                "performance": random.randint(1, 10)
            }
            employees.append(employee)
        
        return employees
    
    @staticmethod
    def generate_ecommerce_data(n=40):
        """Generate sample e-commerce data for business sorting"""
        categories = ["Electronics", "Clothing", "Books", "Home", "Sports"]
        
        products = []
        for i in range(n):
            product = {
                "id": f"PROD{2000 + i}",
                "name": f"Product {i+1}",
                "price": round(random.uniform(10, 500), 2),
                "rating": round(random.uniform(1, 5), 1),
                "sales": random.randint(10, 1000),
                "category": random.choice(categories),
                "stock": random.randint(0, 100)
            }
            products.append(product)
        
        return products
    
    @staticmethod
    def generate_financial_data(n=25):
        """Generate sample financial data for analysis sorting"""
        companies = ["TechCorp", "RetailGiant", "FinanceInc", "HealthcarePlus", "EnergyPower"] * (n//5 + 1)
        
        stocks = []
        for i in range(n):
            stock = {
                "symbol": f"{companies[i % len(companies)][:3].upper()}{i//5 + 1}",
                "company": companies[i % len(companies)],
                "price": round(random.uniform(20, 300), 2),
                "volume": random.randint(1000, 100000),
                "market_cap": random.randint(1000000, 10000000000),
                "pe_ratio": round(random.uniform(5, 50), 2),
                "dividend": round(random.uniform(0, 8), 2)
            }
            stocks.append(stock)
        
        return stocks

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
        "description": "Uses only heuristic function to guide search towards goal. Fast but not guaranteed optimal.",
        "time_complexity": "O(b^m) where m is max depth",
        "space_complexity": "O(b^m)",
        "optimal": "No",
        "use_case": "When speed is more important than optimality",
        "pros": ["Very fast", "Low memory usage", "Simple concept", "Good for approximate solutions"],
        "cons": ["Not optimal", "Can get trapped", "Heavily dependent on heuristic quality", "May fail completely"]
    },
    "Bidirectional Search": {
        "description": "Searches simultaneously from start and goal until they meet. Significantly reduces search space.",
        "time_complexity": "O(b^(d/2))",
        "space_complexity": "O(b^(d/2))",
        "optimal": "Yes (when both directions use optimal algorithms)",
        "use_case": "Large search spaces with known start and goal",
        "pros": ["Much faster than unidirectional", "Reduces search space exponentially", "Can be very efficient"],
        "cons": ["More complex implementation", "Requires both start and goal", "Higher memory usage", "Synchronization complexity"]
    }
}

SORTING_INFO = {
    "Bubble Sort": {
        "description": "Repeatedly compares adjacent elements and swaps them if they're in wrong order until no swaps needed.",
        "best_case": "O(n)", "average_case": "O(n²)", "worst_case": "O(n²)",
        "space_complexity": "O(1)", "stable": "Yes",
        "use_case": "Educational purposes, very small datasets",
        "pros": ["Simple to understand", "In-place sorting", "Stable", "Detects if list is sorted"],
        "cons": ["Very inefficient for large data", "Many comparisons", "Poor performance"]
    },
    "Selection Sort": {
        "description": "Finds minimum element and places it at beginning, then finds second minimum, and continues.",
        "best_case": "O(n²)", "average_case": "O(n²)", "worst_case": "O(n²)",
        "space_complexity": "O(1)", "stable": "No",
        "use_case": "When memory writes are costly",
        "pros": ["Simple implementation", "In-place sorting", "Minimum swaps", "Consistent performance"],
        "cons": ["Always O(n²)", "Not stable", "Inefficient for large data", "No early termination"]
    },
    "Insertion Sort": {
        "description": "Builds sorted array one element at a time by inserting each element in its correct position.",
        "best_case": "O(n)", "average_case": "O(n²)", "worst_case": "O(n²)",
        "space_complexity": "O(1)", "stable": "Yes",
        "use_case": "Small datasets, nearly sorted arrays, online algorithms",
        "pros": ["Efficient for small data", "Stable", "In-place", "Adaptive", "Simple implementation"],
        "cons": ["Inefficient for large data", "O(n²) average case", "More writes than selection sort"]
    },
    "Quick Sort": {
        "description": "Divides array around pivot element and recursively sorts partitions. Very efficient average case.",
        "best_case": "O(n log n)", "average_case": "O(n log n)", "worst_case": "O(n²)",
        "space_complexity": "O(log n)", "stable": "No",
        "use_case": "General purpose sorting when average performance matters",
        "pros": ["Fast average performance", "In-place sorting", "Cache efficient", "Widely used"],
        "cons": ["Worst case O(n²)", "Not stable", "Recursive overhead", "Pivot selection critical"]
    },
    "Merge Sort": {
        "description": "Divides array into halves, recursively sorts them, then merges sorted halves together.",
        "best_case": "O(n log n)", "average_case": "O(n log n)", "worst_case": "O(n log n)",
        "space_complexity": "O(n)", "stable": "Yes",
        "use_case": "When stable sorting and consistent performance needed",
        "pros": ["Guaranteed O(n log n)", "Stable", "Predictable performance", "Good for linked lists"],
        "cons": ["Uses extra memory", "Not in-place", "Slower than quicksort in practice", "Overhead for small arrays"]
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
                            goal: Tuple[int, int] = None) -> go.Figure:
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
    
    fig.update_layout(
        title="Grid Pathfinding Visualization",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        width=600,
        height=500,
        yaxis={'autorange': 'reversed'}  # Flip Y axis to match typical grid representation
    )
    
    return fig

def create_interactive_grid_visualization(grid: GridPathfinder, path: List[Tuple[int, int]] = None, 
                                        visited: List[Tuple[int, int]] = None, start: Tuple[int, int] = None,
                                        goal: Tuple[int, int] = None) -> go.Figure:
    """Create an interactive plotly visualization where users can click to set points"""
    
    # Create visualization grid
    vis_grid = np.zeros((grid.height, grid.width))
    hover_text = []
    
    for y in range(grid.height):
        hover_row = []
        for x in range(grid.width):
            if (x, y) in grid.obstacles:
                vis_grid[y][x] = -1
                hover_row.append(f"Obstacle at ({x},{y})")
            elif start and (x, y) == start:
                vis_grid[y][x] = 1.0
                hover_row.append(f"Start at ({x},{y})")
            elif goal and (x, y) == goal:
                vis_grid[y][x] = 0.9
                hover_row.append(f"Goal at ({x},{y})")
            elif path and (x, y) in path:
                vis_grid[y][x] = 0.8
                hover_row.append(f"Path at ({x},{y})")
            elif visited and (x, y) in visited:
                vis_grid[y][x] = 0.3
                hover_row.append(f"Visited ({x},{y})")
            else:
                vis_grid[y][x] = 0
                hover_row.append(f"Empty ({x},{y}) - Click to interact")
        hover_text.append(hover_row)
    
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
        text=hover_text,
        hovertemplate='%{text}<extra></extra>',
        name=""
    ))
    
    fig.update_layout(
        title="Interactive Grid - Click to Set Points",
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        width=600,
        height=500,
        yaxis={'autorange': 'reversed'}  # Flip Y axis to match typical grid representation
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

# Main app tabs
tab1, tab2 = st.tabs(["🗺️ PathFinding Visualizer", "📊 Sorting Visualizer"])

# Tab 1: Enhanced PathFinding Visualizer
with tab1:
    st.header("🗺️ Advanced Pathfinding Visualization")
    
    # Create sub-tabs for different pathfinding modes
    pathfind_tabs = st.tabs(["🟩 Interactive Grid", "🆚 Algorithm Comparison"])
    
    # Interactive Grid-Based Pathfinding Tab (Enhanced)
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
            
            # Interactive mode selection
            interaction_mode = st.radio(
                "🖱️ Interaction Mode",
                ["Set Start Point", "Set Goal Point", "Add Obstacles", "Remove Obstacles"]
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
            
            # Manual point setting
            st.markdown("### 📍 Manual Point Setting")
            
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
            
            # Create interactive visualization
            path = st.session_state.get('grid_path', [])
            visited = st.session_state.get('grid_visited', [])
            start = st.session_state.get('start_point', None)
            goal = st.session_state.get('goal_point', None)
            
            fig = create_interactive_grid_visualization(
                st.session_state.grid, path, visited, start, goal
            )
            
            # Display the interactive chart and capture click events
            clicked_data = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
            
            # Handle click events for interactive point placement
            if clicked_data and clicked_data.get('selection') and clicked_data['selection'].get('points'):
                for point in clicked_data['selection']['points']:
                    if 'x' in point and 'y' in point:
                        clicked_x = int(point['x'])
                        clicked_y = int(point['y'])
                        
                        if interaction_mode == "Set Start Point":
                            st.session_state.start_point = (clicked_x, clicked_y)
                            st.success(f"Start point set to ({clicked_x}, {clicked_y})")
                            
                        elif interaction_mode == "Set Goal Point":
                            st.session_state.goal_point = (clicked_x, clicked_y)
                            st.success(f"Goal point set to ({clicked_x}, {clicked_y})")
                            
                        elif interaction_mode == "Add Obstacles":
                            st.session_state.grid.set_obstacle(clicked_x, clicked_y)
                            st.success(f"Obstacle added at ({clicked_x}, {clicked_y})")
                            
                        elif interaction_mode == "Remove Obstacles":
                            st.session_state.grid.remove_obstacle(clicked_x, clicked_y)
                            st.success(f"Obstacle removed from ({clicked_x}, {clicked_y})")
                        
                        st.rerun()
            
            # Instructions
            st.info(f"""
            🖱️ **Interactive Instructions:**
            1. **Current Mode:** {interaction_mode}
            2. **Click on the grid** to {interaction_mode.lower()}
            3. Set start and goal points, then click 'Find Path'
            4. Switch interaction modes to add/remove obstacles
            
            **Legend:**
            - 🟩 Green: Start point
            - 🟧 Orange: Goal point  
            - 🟨 Yellow: Optimal path
            - 🔵 Light Blue: Visited nodes
            - ⬛ Black: Obstacles
            - ⬜ White: Empty cells
            """)
    
    # Real Maps Tab (Enhanced)
    # Algorithm Comparison Tab (unchanged but keeping for completeness)
    with pathfind_tabs[1]:
        st.subheader("🆚 Algorithm Performance Comparison")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### ⚙️ Comparison Settings")
            
            comp_width = st.slider("Grid Width", 10, 30, 20, key="comp_width")
            comp_height = st.slider("Grid Height", 10, 30, 15, key="comp_height")
            
            comp_map_type = st.selectbox(
                "Map Type", 
                ["Random Obstacles", "Maze", "Empty Grid", "Diagonal Barriers", "Rooms"],
                key="comp_map"
            )
            
            obstacle_density = st.slider("Obstacle Density", 0.0, 0.5, 0.2) if comp_map_type == "Random Obstacles" else None
            
            if st.button("🏁 Run Comparison", type="primary"):
                # Create test grid
                test_grid = GridPathfinder(comp_width, comp_height)
                sample_maps = create_sample_maps()
                
                if comp_map_type == "Random Obstacles":
                    obstacles = create_random_obstacles(comp_width, comp_height, obstacle_density)
                else:
                    obstacles = sample_maps[comp_map_type](comp_width, comp_height)
                
                test_grid.obstacles = obstacles
                
                # Set start and goal
                start = (0, 0)
                goal = (comp_width-1, comp_height-1)
                
                # Ensure start and goal are not obstacles
                test_grid.obstacles.discard(start)
                test_grid.obstacles.discard(goal)
                
                # Test all algorithms
                algorithms = [
                    "A* (A-Star)", "Dijkstra", "BFS (Breadth-First Search)",
                    "DFS (Depth-First Search)", "Greedy Best-First", "Bidirectional Search"
                ]
                
                results = []
                progress_bar = st.progress(0)
                
                for i, algo in enumerate(algorithms):
                    start_time = time.time()
                    
                    try:
                        if algo == "A* (A-Star)":
                            path, visited = PathfindingAlgorithms.a_star(test_grid, start, goal)
                        elif algo == "Dijkstra":
                            path, visited = PathfindingAlgorithms.dijkstra(test_grid, start, goal)
                        elif algo == "BFS (Breadth-First Search)":
                            path, visited = PathfindingAlgorithms.bfs(test_grid, start, goal)
                        elif algo == "DFS (Depth-First Search)":
                            path, visited = PathfindingAlgorithms.dfs(test_grid, start, goal)
                        elif algo == "Greedy Best-First":
                            path, visited = PathfindingAlgorithms.greedy_best_first(test_grid, start, goal)
                        else:  # Bidirectional Search
                            path, visited = PathfindingAlgorithms.bidirectional_search(test_grid, start, goal)
                        
                        end_time = time.time()
                        execution_time = (end_time - start_time) * 1000
                        
                        results.append({
                            "Algorithm": algo,
                            "Path Length": len(path) if path else 0,
                            "Nodes Visited": len(visited),
                            "Execution Time (ms)": f"{execution_time:.2f}",
                            "Path Found": "Yes" if path else "No",
                            "Efficiency (%)": f"{(len(path)/len(visited)*100):.1f}" if path and visited else "0"
                        })
                        
                    except Exception as e:
                        results.append({
                            "Algorithm": algo,
                            "Path Length": "Error",
                            "Nodes Visited": "Error", 
                            "Execution Time (ms)": "Error",
                            "Path Found": "Error",
                            "Efficiency (%)": "Error"
                        })
                    
                    progress_bar.progress((i + 1) / len(algorithms))
                
                st.session_state.comparison_results = results
                st.session_state.comparison_grid = test_grid
                st.session_state.comparison_start = start
                st.session_state.comparison_goal = goal
        
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

# Tab 2: Enhanced Sorting Visualizer with Real-World Applications
with tab2:
    st.header("📊 Advanced Sorting Algorithm Visualizer")
    
    # Create sub-tabs for sorting
    sort_tabs = st.tabs(["🎬 Algorithm Animation", "🌍 Real-World Applications", "🆚 Performance Analysis"])
    
    # Algorithm Animation Tab (Enhanced from original)
    with sort_tabs[0]:
        st.subheader("🎬 Interactive Sorting Animation")
        
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
    
    # Real-World Applications Tab (NEW)
    with sort_tabs[1]:
        st.subheader("🌍 Real-World Sorting Applications")
        
        # Application selector
        application_type = st.selectbox(
            "🎯 Choose Application Domain",
            ["🎓 Student Grade Management", "👥 Employee HR System", "🛒 E-commerce Product Sorting", "📈 Financial Stock Analysis"]
        )
        
        if application_type == "🎓 Student Grade Management":
            st.markdown('<div class="real-world-card"><h3>🎓 Student Grade Management System</h3><p>Sort students by various criteria for academic analysis and reporting</p></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("### 🎛️ Controls")
                num_students = st.slider("Number of Students", 10, 100, 50)
                sort_by = st.selectbox("Sort By", ["Score", "Age", "Name", "Grade"])
                sort_order = st.radio("Order", ["Ascending", "Descending"])
                
                if st.button("Generate Student Data", type="primary"):
                    students = RealWorldSortingApplications.generate_student_data(num_students)
                    st.session_state.students_data = students
                
                if st.button("Apply Sorting") and 'students_data' in st.session_state:
                    data = st.session_state.students_data.copy()
                    reverse = sort_order == "Descending"
                    
                    if sort_by == "Score":
                        data.sort(key=lambda x: x['score'], reverse=reverse)
                    elif sort_by == "Age":
                        data.sort(key=lambda x: x['age'], reverse=reverse)
                    elif sort_by == "Name":
                        data.sort(key=lambda x: x['name'], reverse=reverse)
                    else:  # Grade
                        grade_order = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
                        data.sort(key=lambda x: grade_order.get(x['grade'], 0), reverse=reverse)
                    
                    st.session_state.sorted_students = data
            
            with col1:
                if 'students_data' in st.session_state:
                    # Display original data
                    st.markdown("#### Original Data")
                    df_original = pd.DataFrame(st.session_state.students_data)
                    st.dataframe(df_original.head(10), use_container_width=True)
                    
                    # Display sorted data if available
                    if 'sorted_students' in st.session_state:
                        st.markdown("#### Sorted Data")
                        df_sorted = pd.DataFrame(st.session_state.sorted_students)
                        st.dataframe(df_sorted.head(10), use_container_width=True)
                        
                        # Create visualization
                        fig = px.scatter(df_sorted, x='age', y='score', color='grade', 
                                       title=f"Students Sorted by {sort_by} ({sort_order})",
                                       hover_data=['name', 'subject'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Performance metrics
                        top_performers = [s for s in st.session_state.sorted_students if s['score'] >= 90]
                        avg_score = sum(s['score'] for s in st.session_state.sorted_students) / len(st.session_state.sorted_students)
                        
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Top Performers (A grade)", len(top_performers))
                        col_b.metric("Average Score", f"{avg_score:.1f}")
                        col_c.metric("Total Students", len(st.session_state.sorted_students))
                else:
                    st.info("Generate student data to begin sorting analysis")
        
        elif application_type == "👥 Employee HR System":
            st.markdown('<div class="real-world-card"><h3>👥 Employee HR Management System</h3><p>Sort employees for payroll, performance reviews, and organizational analysis</p></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("### 🎛️ Controls")
                num_employees = st.slider("Number of Employees", 10, 100, 30)
                sort_by_emp = st.selectbox("Sort By", ["Salary", "Experience", "Performance", "Department"])
                sort_order_emp = st.radio("Order", ["Ascending", "Descending"], key="emp_order")
                
                if st.button("Generate Employee Data", type="primary"):
                    employees = RealWorldSortingApplications.generate_employee_data(num_employees)
                    st.session_state.employees_data = employees
                
                if st.button("Apply Employee Sorting") and 'employees_data' in st.session_state:
                    data = st.session_state.employees_data.copy()
                    reverse = sort_order_emp == "Descending"
                    
                    if sort_by_emp == "Salary":
                        data.sort(key=lambda x: x['salary'], reverse=reverse)
                    elif sort_by_emp == "Experience":
                        data.sort(key=lambda x: x['experience'], reverse=reverse)
                    elif sort_by_emp == "Performance":
                        data.sort(key=lambda x: x['performance'], reverse=reverse)
                    else:  # Department
                        data.sort(key=lambda x: x['department'], reverse=reverse)
                    
                    st.session_state.sorted_employees = data
            
            with col1:
                if 'employees_data' in st.session_state:
                    # Display data
                    st.markdown("#### Employee Overview")
                    df_emp = pd.DataFrame(st.session_state.employees_data)
                    
                    if 'sorted_employees' in st.session_state:
                        df_emp = pd.DataFrame(st.session_state.sorted_employees)
                    
                    st.dataframe(df_emp.head(10), use_container_width=True)
                    
                    # Create visualizations
                    if 'sorted_employees' in st.session_state:
                        fig1 = px.box(df_emp, x='department', y='salary', 
                                     title=f"Salary Distribution by Department (Sorted by {sort_by_emp})")
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        fig2 = px.scatter(df_emp, x='experience', y='performance', 
                                        color='department', size='salary',
                                        title="Experience vs Performance Analysis")
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # HR Insights
                        avg_salary = df_emp['salary'].mean()
                        top_performers = df_emp[df_emp['performance'] >= 8]
                        senior_employees = df_emp[df_emp['experience'] >= 10]
                        
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Average Salary", f"${avg_salary:,.0f}")
                        col_b.metric("Top Performers", len(top_performers))
                        col_c.metric("Senior Employees", len(senior_employees))
                else:
                    st.info("Generate employee data to begin HR analysis")
        
        elif application_type == "🛒 E-commerce Product Sorting":
            st.markdown('<div class="real-world-card"><h3>🛒 E-commerce Product Management</h3><p>Sort products for better customer experience and inventory management</p></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("### 🎛️ Controls")
                num_products = st.slider("Number of Products", 10, 100, 40)
                sort_by_prod = st.selectbox("Sort By", ["Price", "Rating", "Sales", "Stock"])
                sort_order_prod = st.radio("Order", ["Ascending", "Descending"], key="prod_order")
                filter_category = st.selectbox("Filter by Category", ["All", "Electronics", "Clothing", "Books", "Home", "Sports"])
                
                if st.button("Generate Product Data", type="primary"):
                    products = RealWorldSortingApplications.generate_ecommerce_data(num_products)
                    st.session_state.products_data = products
                
                if st.button("Apply Product Sorting") and 'products_data' in st.session_state:
                    data = st.session_state.products_data.copy()
                    
                    # Filter by category if specified
                    if filter_category != "All":
                        data = [p for p in data if p['category'] == filter_category]
                    
                    reverse = sort_order_prod == "Descending"
                    
                    if sort_by_prod == "Price":
                        data.sort(key=lambda x: x['price'], reverse=reverse)
                    elif sort_by_prod == "Rating":
                        data.sort(key=lambda x: x['rating'], reverse=reverse)
                    elif sort_by_prod == "Sales":
                        data.sort(key=lambda x: x['sales'], reverse=reverse)
                    else:  # Stock
                        data.sort(key=lambda x: x['stock'], reverse=reverse)
                    
                    st.session_state.sorted_products = data
            
            with col1:
                if 'products_data' in st.session_state:
                    # Display data
                    st.markdown("#### Product Catalog")
                    df_prod = pd.DataFrame(st.session_state.products_data)
                    
                    if 'sorted_products' in st.session_state:
                        df_prod = pd.DataFrame(st.session_state.sorted_products)
                    
                    st.dataframe(df_prod.head(10), use_container_width=True)
                    
                    # Create visualizations
                    if 'sorted_products' in st.session_state:
                        fig1 = px.scatter(df_prod, x='price', y='rating', 
                                        color='category', size='sales',
                                        title=f"Price vs Rating Analysis (Sorted by {sort_by_prod})")
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        fig2 = px.bar(df_prod.head(10), x='name', y='sales',
                                     title="Top 10 Products by Sales")
                        fig2.update_xaxes(tickangle=45)
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # E-commerce Insights
                        avg_price = df_prod['price'].mean()
                        low_stock = df_prod[df_prod['stock'] < 10]
                        top_rated = df_prod[df_prod['rating'] >= 4.5]
                        
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Average Price", f"${avg_price:.2f}")
                        col_b.metric("Low Stock Items", len(low_stock))
                        col_c.metric("Highly Rated", len(top_rated))
                else:
                    st.info("Generate product data to begin e-commerce analysis")
        
        else:  # Financial Stock Analysis
            st.markdown('<div class="real-world-card"><h3>📈 Financial Stock Analysis</h3><p>Sort stocks for investment decisions and portfolio management</p></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("### 🎛️ Controls")
                num_stocks = st.slider("Number of Stocks", 10, 50, 25)
                sort_by_stock = st.selectbox("Sort By", ["Price", "Volume", "Market Cap", "P/E Ratio", "Dividend"])
                sort_order_stock = st.radio("Order", ["Ascending", "Descending"], key="stock_order")
                
                if st.button("Generate Stock Data", type="primary"):
                    stocks = RealWorldSortingApplications.generate_financial_data(num_stocks)
                    st.session_state.stocks_data = stocks
                
                if st.button("Apply Stock Sorting") and 'stocks_data' in st.session_state:
                    data = st.session_state.stocks_data.copy()
                    reverse = sort_order_stock == "Descending"
                    
                    if sort_by_stock == "Price":
                        data.sort(key=lambda x: x['price'], reverse=reverse)
                    elif sort_by_stock == "Volume":
                        data.sort(key=lambda x: x['volume'], reverse=reverse)
                    elif sort_by_stock == "Market Cap":
                        data.sort(key=lambda x: x['market_cap'], reverse=reverse)
                    elif sort_by_stock == "P/E Ratio":
                        data.sort(key=lambda x: x['pe_ratio'], reverse=reverse)
                    else:  # Dividend
                        data.sort(key=lambda x: x['dividend'], reverse=reverse)
                    
                    st.session_state.sorted_stocks = data
            
            with col1:
                if 'stocks_data' in st.session_state:
                    # Display data
                    st.markdown("#### Stock Portfolio")
                    df_stock = pd.DataFrame(st.session_state.stocks_data)
                    
                    if 'sorted_stocks' in st.session_state:
                        df_stock = pd.DataFrame(st.session_state.sorted_stocks)
                    
                    st.dataframe(df_stock.head(10), use_container_width=True)
                    
                    # Create visualizations
                    if 'sorted_stocks' in st.session_state:
                        fig1 = px.scatter(df_stock, x='pe_ratio', y='dividend', 
                                        color='company', size='market_cap',
                                        title=f"P/E Ratio vs Dividend Yield (Sorted by {sort_by_stock})")
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        fig2 = px.treemap(df_stock.head(15), path=['company'], values='market_cap',
                                         title="Market Cap Distribution (Top 15)")
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Financial Insights
                        avg_pe = df_stock['pe_ratio'].mean()
                        high_dividend = df_stock[df_stock['dividend'] >= 5.0]
                        large_cap = df_stock[df_stock['market_cap'] >= 1000000000]
                        
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Avg P/E Ratio", f"{avg_pe:.1f}")
                        col_b.metric("High Dividend Stocks", len(high_dividend))
                        col_c.metric("Large Cap Stocks", len(large_cap))
                else:
                    st.info("Generate stock data to begin financial analysis")
        
        # Real-world sorting applications explanation
        st.markdown("---")
        st.markdown("### 💡 Why Sorting Matters in Real Applications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Performance Impact:**
            - **Database queries**: Sorted data enables binary search (O(log n) vs O(n))
            - **User experience**: Quick filtering and searching in web applications
            - **Memory efficiency**: Sorted data often compresses better
            - **Cache optimization**: Sequential access patterns improve performance
            """)
            
            st.markdown("""
            **Business Applications:**
            - **Recommendation systems**: Sort products by relevance score
            - **Financial analysis**: Rank investments by various metrics
            - **Logistics**: Optimize delivery routes and schedules
            - **Data analytics**: Identify trends and outliers efficiently
            """)
        
        with col2:
            st.markdown("""
            **Algorithm Selection Criteria:**
            - **Data size**: Small arrays → Insertion sort, Large → Quick/Merge sort
            - **Data characteristics**: Nearly sorted → Insertion sort
            - **Stability requirement**: Stable sorting → Merge sort
            - **Memory constraints**: In-place sorting → Quick sort, Heap sort
            """)
            
            st.markdown("""
            **Modern Implementations:**
            - **Hybrid algorithms**: Python's Timsort combines multiple strategies
            - **Parallel sorting**: Multi-core processors enable concurrent sorting
            - **External sorting**: Handle datasets larger than memory
            - **Specialized sorting**: Radix sort for integers, counting sort for small ranges
            """)
    
    # Performance Analysis Tab (Enhanced)
    with sort_tabs[2]:
        st.subheader("🆚 Comprehensive Performance Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### ⚙️ Analysis Settings")
            
            # Test configuration
            test_sizes = st.multiselect(
                "Array Sizes to Test",
                [10, 25, 50, 100, 200, 500],
                default=[10, 25, 50, 100]
            )
            
            test_types = st.multiselect(
                "Array Types to Test",
                ["Random", "Nearly Sorted", "Reverse Sorted", "Few Unique"],
                default=["Random", "Nearly Sorted"]
            )
            
            algorithms_to_test = st.multiselect(
                "Algorithms to Test",
                ["Bubble Sort", "Selection Sort", "Insertion Sort", "Quick Sort", "Merge Sort", "Heap Sort"],
                default=["Quick Sort", "Merge Sort", "Heap Sort"]
            )
            
            if st.button("🚀 Run Comprehensive Analysis", type="primary"):
                if test_sizes and test_types and algorithms_to_test:
                    comprehensive_results = []
                    total_tests = len(test_sizes) * len(test_types) * len(algorithms_to_test)
                    current_test = 0
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for size in test_sizes:
                        for array_type in test_types:
                            # Generate test array
                            if array_type == "Random":
                                test_arr = [random.randint(1, 100) for _ in range(size)]
                            elif array_type == "Nearly Sorted":
                                test_arr = list(range(1, size + 1))
                                for _ in range(size // 10):
                                    i, j = random.randint(0, size - 1), random.randint(0, size - 1)
                                    test_arr[i], test_arr[j] = test_arr[j], test_arr[i]
                            elif array_type == "Reverse Sorted":
                                test_arr = list(range(size, 0, -1))
                            else:  # Few Unique
                                unique_vals = [random.randint(1, 10) for _ in range(5)]
                                test_arr = [random.choice(unique_vals) for _ in range(size)]
                            
                            for algo in algorithms_to_test:
                                current_test += 1
                                status_text.text(f"Testing {algo} on {array_type} array of size {size}...")
                                progress_bar.progress(current_test / total_tests)
                                
                                start_time = time.time()
                                try:
                                    if algo == "Bubble Sort":
                                        steps = SortingAlgorithms.bubble_sort(test_arr)
                                    elif algo == "Selection Sort":
                                        steps = SortingAlgorithms.selection_sort(test_arr)
                                    elif algo == "Insertion Sort":
                                        steps = SortingAlgorithms.insertion_sort(test_arr)
                                    elif algo == "Quick Sort":
                                        steps = SortingAlgorithms.quick_sort(test_arr)
                                    elif algo == "Merge Sort":
                                        steps = SortingAlgorithms.merge_sort(test_arr)
                                    else:  # Heap Sort
                                        steps = SortingAlgorithms.heap_sort(test_arr)
                                    
                                    end_time = time.time()
                                    execution_time = (end_time - start_time) * 1000
                                    
                                    comprehensive_results.append({
                                        "Algorithm": algo,
                                        "Array Size": size,
                                        "Array Type": array_type,
                                        "Execution Time (ms)": execution_time,
                                        "Steps": len(steps)
                                    })
                                except Exception as e:
                                    comprehensive_results.append({
                                        "Algorithm": algo,
                                        "Array Size": size,
                                        "Array Type": array_type,
                                        "Execution Time (ms)": float('inf'),
                                        "Steps": 0
                                    })
                    
                    st.session_state.comprehensive_results = comprehensive_results
                    progress_bar.empty()
                    status_text.empty()
                else:
                    st.error("Please select at least one option from each category.")
        
        with col1:
            if 'comprehensive_results' in st.session_state:
                results_df = pd.DataFrame(st.session_state.comprehensive_results)
                
                # Display results table
                st.markdown("#### Detailed Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Performance visualization by algorithm
                fig1 = px.line(results_df, x='Array Size', y='Execution Time (ms)', 
                              color='Algorithm', facet_col='Array Type',
                              title="Algorithm Performance vs Array Size")
                st.plotly_chart(fig1, use_container_width=True)
                
                # Heatmap of performance
                pivot_table = results_df.pivot_table(
                    values='Execution Time (ms)', 
                    index='Algorithm', 
                    columns=['Array Size', 'Array Type'],
                    aggfunc='mean'
                )
                
                fig2 = px.imshow(pivot_table, aspect="auto", 
                               title="Performance Heatmap (Darker = Faster)")
                st.plotly_chart(fig2, use_container_width=True)
                
                # Algorithm efficiency comparison
                avg_performance = results_df.groupby('Algorithm')['Execution Time (ms)'].mean().sort_values()
                
                fig3 = px.bar(x=avg_performance.index, y=avg_performance.values,
                             title="Average Performance by Algorithm",
                             labels={'x': 'Algorithm', 'y': 'Avg Execution Time (ms)'})
                st.plotly_chart(fig3, use_container_width=True)
                
                # Best algorithm recommendation
                st.markdown("#### 🏆 Algorithm Recommendations")
                
                best_overall = avg_performance.index[0]
                st.success(f"**Best Overall Performance:** {best_overall}")
                
                # Best for different scenarios
                for arr_type in results_df['Array Type'].unique():
                    type_data = results_df[results_df['Array Type'] == arr_type]
                    best_for_type = type_data.groupby('Algorithm')['Execution Time (ms)'].mean().idxmin()
                    st.info(f"**Best for {arr_type} arrays:** {best_for_type}")
            else:
                st.info("🚀 Run comprehensive analysis to see detailed performance comparisons across different scenarios.")
    
    # Algorithm information section
    st.markdown("---")
    st.subheader("🧠 Sorting Algorithm Reference")
    
    # Create tabs for different algorithms
    sort_algo_tabs = st.tabs(["Current Algorithm", "All Algorithms Comparison", "Complexity Analysis"])
    
    with sort_algo_tabs[0]:
        if 'sort_algorithm' in locals() and sort_algorithm in SORTING_INFO:
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
            <h4>🗺️ Enhanced Pathfinding Features</h4>
            <p>✅ Interactive click-to-place grid points</p>
            <p>✅ Real-world map with pointer placement</p>
            <p>✅ 6 different algorithms with comparison</p>
            <p>✅ Multiple map types & obstacle patterns</p>
            <p>✅ Performance metrics & analysis</p>
        </div>
        <div>
            <h4>📊 Enhanced Sorting Features</h4>
            <p>✅ Animated step-by-step visualization</p>
            <p>✅ Real-world application scenarios</p>
            <p>✅ Comprehensive performance analysis</p>
            <p>✅ Interactive data management systems</p>
            <p>✅ Business intelligence applications</p>
        </div>
    </div>
    <p><strong>Built with ❤️ by Shreyas Kasture</strong></p>
</div>
""", unsafe_allow_html=True)

# Session state cleanup
if st.sidebar.button("🧹 Clear All Data"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
