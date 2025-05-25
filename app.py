import streamlit as st
import folium
from streamlit_folium import folium_static
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
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Advanced PathFinder & Sort Visualizer",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better styling and animations
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
        animation: gradientShift 3s ease-in-out infinite;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .developer-credit {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 30px;
        border-radius: 30px;
        text-align: center;
        margin: 20px auto;
        max-width: 500px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: linear-gradient(to right, #f8f9fa, #e9ecef);
        padding: 5px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f8f9fa;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3);
    }
    
    .algorithm-info {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .algorithm-info:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    
    .complexity-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 15px;
        font-size: 0.85em;
        font-weight: bold;
        margin: 3px;
        transition: all 0.3s ease;
    }
    
    .complexity-best { 
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
        color: #155724; 
    }
    .complexity-average { 
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); 
        color: #856404; 
    }
    .complexity-worst { 
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); 
        color: #721c24; 
    }
    
    .grid-cell {
        width: 30px;
        height: 30px;
        border: 1px solid #ccc;
        display: inline-block;
        margin: 1px;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: center;
        line-height: 30px;
    }
    
    .grid-cell:hover {
        transform: scale(1.1);
        border-color: #667eea;
    }
    
    .cell-empty {
        background-color: #ecf0f1;
    }
    
    .cell-start {
        background-color: #27ae60;
        color: white;
        font-weight: bold;
    }
    
    .cell-goal {
        background-color: #e74c3c;
        color: white;
        font-weight: bold;
    }
    
    .cell-obstacle {
        background-color: #2c3e50;
    }
    
    .cell-path {
        background-color: #f1c40f;
        color: black;
        font-weight: bold;
    }
    
    .cell-visited {
        background-color: #3498db;
        opacity: 0.7;
    }
    
    .grid-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin: 20px 0;
        user-select: none;
    }
    
    .grid-row {
        display: flex;
    }
    
    .path-stats {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    .step-control-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 25px;
        margin: 5px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .step-control-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .interactive-hint {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #2196f3;
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from { 
            transform: translateX(-20px); 
            opacity: 0; 
        }
        to { 
            transform: translateX(0); 
            opacity: 1; 
        }
    }
    
    .sort-bar {
        background: linear-gradient(to right, #667eea, #764ba2);
        margin: 0 1px;
        border-radius: 2px;
        transition: all 0.2s ease;
    }
    
    .sort-bar-comparing {
        background: linear-gradient(to right, #f39c12, #f1c40f);
    }
    
    .sort-bar-swapping {
        background: linear-gradient(to right, #e74c3c, #c0392b);
    }
    
    .sort-container {
        display: flex;
        align-items: flex-end;
        height: 300px;
        padding: 10px;
        margin: 20px 0;
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    
    .sort-stats {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    
    .footer-developer {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        margin-top: 40px;
        text-align: center;
        box-shadow: 0 -4px 20px rgba(0,0,0,0.1);
    }
    
    .footer-developer h3 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title with developer credit
st.markdown('<h1 class="main-header">🗺️ Advanced PathFinder & Sort Visualizer</h1>', unsafe_allow_html=True)
st.markdown('<div class="developer-credit">✨ Developed with ❤️ by Shreyas Kasture ✨</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive pathfinding on real maps & grid systems + animated sorting algorithms with real-world applications</p>', unsafe_allow_html=True)

# Enhanced Interactive Grid Pathfinder class
class InteractiveGridPathfinder:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.start = None
        self.goal = None
        self.obstacles = set()
        self.path = []
        self.visited = set()
    
    def clear(self):
        self.grid = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.obstacles = set()
        self.path = []
        self.visited = set()
    
    def set_obstacle(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height:
            if (x, y) != self.start and (x, y) != self.goal:
                self.obstacles.add((x, y))
                self.grid[y][x] = 1
    
    def remove_obstacle(self, x: int, y: int):
        if (x, y) in self.obstacles:
            self.obstacles.remove((x, y))
            self.grid[y][x] = 0
    
    def set_start(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.obstacles:
            self.start = (x, y)
    
    def set_goal(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.obstacles:
            self.goal = (x, y)
    
    def is_valid(self, x: int, y: int) -> bool:
        return (0 <= x < self.width and 0 <= y < self.height and 
                (x, y) not in self.obstacles)
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        neighbors = []
        # 8-directional movement
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny))
        return neighbors
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        # Manhattan distance with tie-breaking
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy) + 0.001 * (dx + dy)
    
    def get_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        if dx == 1 and dy == 1:
            return math.sqrt(2)
        return 1.0
    
    def a_star(self):
        """Run A* algorithm and store the path and visited nodes"""
        if not self.start or not self.goal:
            return False
        
        # Clear previous results
        self.path = []
        self.visited = set()
        
        open_set = []
        heapq.heappush(open_set, (0, self.start))
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self.heuristic(self.start, self.goal)}
        open_set_hash = {self.start}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)
            
            self.visited.add(current)
            
            if current == self.goal:
                # Reconstruct path
                while current in came_from:
                    self.path.append(current)
                    current = came_from[current]
                self.path.append(self.start)
                self.path.reverse()
                return True
            
            for neighbor in self.get_neighbors(current[0], current[1]):
                tentative_g_score = g_score[current] + self.get_distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, self.goal)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        return False
    
    def dijkstra(self):
        """Run Dijkstra's algorithm and store the path and visited nodes"""
        if not self.start or not self.goal:
            return False
        
        # Clear previous results
        self.path = []
        self.visited = set()
        
        # Priority queue for Dijkstra's
        pq = [(0, self.start)]
        dist = {self.start: 0}
        prev = {}
        
        while pq:
            current_dist, current = heapq.heappop(pq)
            
            self.visited.add(current)
            
            if current == self.goal:
                # Reconstruct path
                while current in prev:
                    self.path.append(current)
                    current = prev[current]
                self.path.append(self.start)
                self.path.reverse()
                return True
            
            if current_dist > dist.get(current, float('inf')):
                continue
            
            for neighbor in self.get_neighbors(current[0], current[1]):
                distance = current_dist + self.get_distance(current, neighbor)
                
                if distance < dist.get(neighbor, float('inf')):
                    dist[neighbor] = distance
                    prev[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        return False
    
    def bfs(self):
        """Run BFS algorithm and store the path and visited nodes"""
        if not self.start or not self.goal:
            return False
        
        # Clear previous results
        self.path = []
        self.visited = set()
        
        # Queue for BFS
        queue = deque([self.start])
        visited = {self.start}
        prev = {}
        
        while queue:
            current = queue.popleft()
            self.visited.add(current)
            
            if current == self.goal:
                # Reconstruct path
                while current in prev:
                    self.path.append(current)
                    current = prev[current]
                self.path.append(self.start)
                self.path.reverse()
                return True
            
            for neighbor in self.get_neighbors(current[0], current[1]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    prev[neighbor] = current
        
        return False
    
    def dfs(self):
        """Run DFS algorithm and store the path and visited nodes"""
        if not self.start or not self.goal:
            return False
        
        # Clear previous results
        self.path = []
        self.visited = set()
        
        # Stack for DFS
        stack = [self.start]
        visited = {self.start}
        prev = {}
        
        while stack:
            current = stack.pop()
            self.visited.add(current)
            
            if current == self.goal:
                # Reconstruct path
                while current in prev:
                    self.path.append(current)
                    current = prev[current]
                self.path.append(self.start)
                self.path.reverse()
                return True
            
            for neighbor in self.get_neighbors(current[0], current[1]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
                    prev[neighbor] = current
        
        return False
    
    def run_algorithm(self, algorithm: str):
        """Run selected pathfinding algorithm"""
        if algorithm == "A* (A-Star)":
            return self.a_star()
        elif algorithm == "Dijkstra":
            return self.dijkstra()
        elif algorithm == "BFS":
            return self.bfs()
        elif algorithm == "DFS":
            return self.dfs()
        else:
            return self.a_star()  # Default to A* if unknown algorithm

# Sorting Algorithm implementations
class SortingAlgorithms:
    @staticmethod
    def bubble_sort(arr, callback=None):
        """
        Bubble Sort implementation with callback for visualization
        callback: function(arr, comparing, swapping, stats)
        """
        n = len(arr)
        stats = {"comparisons": 0, "swaps": 0, "accesses": 0}
        
        for i in range(n):
            for j in range(0, n-i-1):
                stats["comparisons"] += 1
                stats["accesses"] += 2
                
                if callback:
                    callback(arr.copy(), [j, j+1], [], stats)
                
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
                    stats["swaps"] += 1
                    stats["accesses"] += 2
                    
                    if callback:
                        callback(arr.copy(), [], [j, j+1], stats)
        
        return arr, stats
    
    @staticmethod
    def selection_sort(arr, callback=None):
        """Selection Sort implementation with callback for visualization"""
        n = len(arr)
        stats = {"comparisons": 0, "swaps": 0, "accesses": 0}
        
        for i in range(n):
            min_idx = i
            for j in range(i+1, n):
                stats["comparisons"] += 1
                stats["accesses"] += 2
                
                if callback:
                    callback(arr.copy(), [min_idx, j], [], stats)
                
                if arr[j] < arr[min_idx]:
                    min_idx = j
            
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
                stats["swaps"] += 1
                stats["accesses"] += 2
                
                if callback:
                    callback(arr.copy(), [], [i, min_idx], stats)
        
        return arr, stats
    
    @staticmethod
    def insertion_sort(arr, callback=None):
        """Insertion Sort implementation with callback for visualization"""
        n = len(arr)
        stats = {"comparisons": 0, "swaps": 0, "accesses": 0}
        
        for i in range(1, n):
            key = arr[i]
            j = i-1
            stats["accesses"] += 1
            
            if callback:
                callback(arr.copy(), [i], [], stats)
            
            while j >= 0 and key < arr[j]:
                stats["comparisons"] += 1
                stats["accesses"] += 1
                
                if callback:
                    callback(arr.copy(), [j, j+1], [], stats)
                
                arr[j+1] = arr[j]
                stats["swaps"] += 1
                stats["accesses"] += 1
                j -= 1
            
            arr[j+1] = key
            stats["accesses"] += 1
            
            if callback:
                callback(arr.copy(), [], [j+1], stats)
        
        return arr, stats
    
    @staticmethod
    def quick_sort(arr, callback=None):
        """Quick Sort implementation with callback for visualization"""
        stats = {"comparisons": 0, "swaps": 0, "accesses": 0}
        
        def partition(arr, low, high):
            pivot = arr[high]
            stats["accesses"] += 1
            i = low - 1
            
            for j in range(low, high):
                stats["comparisons"] += 1
                stats["accesses"] += 1
                
                if callback:
                    callback(arr.copy(), [j, high], [], stats)
                
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
                    stats["swaps"] += 1
                    stats["accesses"] += 2
                    
                    if callback:
                        callback(arr.copy(), [], [i, j], stats)
            
            arr[i+1], arr[high] = arr[high], arr[i+1]
            stats["swaps"] += 1
            stats["accesses"] += 2
            
            if callback:
                callback(arr.copy(), [], [i+1, high], stats)
            
            return i+1
        
        def quick_sort_impl(arr, low, high):
            if low < high:
                pi = partition(arr, low, high)
                quick_sort_impl(arr, low, pi-1)
                quick_sort_impl(arr, pi+1, high)
        
        quick_sort_impl(arr, 0, len(arr)-1)
        return arr, stats
    
    @staticmethod
    def merge_sort(arr, callback=None):
        """Merge Sort implementation with callback for visualization"""
        stats = {"comparisons": 0, "swaps": 0, "accesses": 0}
        
        def merge(arr, left, mid, right):
            L = arr[left:mid+1]
            R = arr[mid+1:right+1]
            stats["accesses"] += right - left + 1
            
            i = j = 0
            k = left
            
            while i < len(L) and j < len(R):
                stats["comparisons"] += 1
                stats["accesses"] += 2
                
                if callback:
                    temp_arr = arr.copy()
                    callback(temp_arr, [left+i, mid+1+j], [], stats)
                
                if L[i] <= R[j]:
                    arr[k] = L[i]
                    i += 1
                else:
                    arr[k] = R[j]
                    j += 1
                
                stats["swaps"] += 1
                stats["accesses"] += 1
                
                if callback:
                    temp_arr = arr.copy()
                    callback(temp_arr, [], [k], stats)
                
                k += 1
            
            while i < len(L):
                arr[k] = L[i]
                stats["accesses"] += 1
                if callback:
                    temp_arr = arr.copy()
                    callback(temp_arr, [], [k], stats)
                i += 1
                k += 1
            
            while j < len(R):
                arr[k] = R[j]
                stats["accesses"] += 1
                if callback:
                    temp_arr = arr.copy()
                    callback(temp_arr, [], [k], stats)
                j += 1
                k += 1
        
        def merge_sort_impl(arr, left, right):
            if left < right:
                mid = (left + right) // 2
                merge_sort_impl(arr, left, mid)
                merge_sort_impl(arr, mid+1, right)
                merge(arr, left, mid, right)
        
        merge_sort_impl(arr, 0, len(arr)-1)
        return arr, stats
    
    @staticmethod
    def heap_sort(arr, callback=None):
        """Heap Sort implementation with callback for visualization"""
        stats = {"comparisons": 0, "swaps": 0, "accesses": 0}
        n = len(arr)
        
        def heapify(arr, n, i):
            largest = i
            left = 2 * i + 1
            right = 2 * i + 2
            
            if left < n:
                stats["comparisons"] += 1
                stats["accesses"] += 2
                
                if callback:
                    callback(arr.copy(), [largest, left], [], stats)
                
                if arr[left] > arr[largest]:
                    largest = left
            
            if right < n:
                stats["comparisons"] += 1
                stats["accesses"] += 2
                
                if callback:
                    callback(arr.copy(), [largest, right], [], stats)
                
                if arr[right] > arr[largest]:
                    largest = right
            
            if largest != i:
                arr[i], arr[largest] = arr[largest], arr[i]
                stats["swaps"] += 1
                stats["accesses"] += 2
                
                if callback:
                    callback(arr.copy(), [], [i, largest], stats)
                
                heapify(arr, n, largest)
        
        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)
        
        # Extract elements one by one
        for i in range(n-1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            stats["swaps"] += 1
            stats["accesses"] += 2
            
            if callback:
                callback(arr.copy(), [], [0, i], stats)
            
            heapify(arr, i, 0)
        
        return arr, stats
    
    @staticmethod
    def run_algorithm(algorithm, arr, callback=None):
        """Run selected sorting algorithm"""
        if algorithm == "Bubble Sort":
            return SortingAlgorithms.bubble_sort(arr, callback)
        elif algorithm == "Selection Sort":
            return SortingAlgorithms.selection_sort(arr, callback)
        elif algorithm == "Insertion Sort":
            return SortingAlgorithms.insertion_sort(arr, callback)
        elif algorithm == "Quick Sort":
            return SortingAlgorithms.quick_sort(arr, callback)
        elif algorithm == "Merge Sort":
            return SortingAlgorithms.merge_sort(arr, callback)
        elif algorithm == "Heap Sort":
            return SortingAlgorithms.heap_sort(arr, callback)
        else:
            return SortingAlgorithms.bubble_sort(arr, callback)  # Default to bubble sort

# Main app tabs
tab1, tab2 = st.tabs(["🗺️ PathFinding Visualizer", "📊 Sorting Visualizer"])

# Tab 1: Enhanced PathFinding Visualizer
with tab1:
    st.header("🗺️ Advanced Interactive Pathfinding Visualization")
    
    # Create sub-tabs for different pathfinding modes
    pathfind_tabs = st.tabs(["🟩 Interactive Grid", "🌍 Real Maps", "🆚 Algorithm Comparison"])
    
    # Interactive Grid-Based Pathfinding Tab
    with pathfind_tabs[0]:
        st.subheader("Interactive Grid Pathfinding")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### 🎛️ Grid Controls")
            
            # Grid settings
            grid_width = st.slider("Grid Width", 5, 25, 15)
            grid_height = st.slider("Grid Height", 5, 25, 15)
            
            # Algorithm selection
            algorithm = st.selectbox(
                "🧠 Algorithm",
                ["A* (A-Star)", "Dijkstra", "BFS", "DFS"]
            )
            
            # Interactive mode selection
            st.markdown("### 🖱️ Interactive Mode")
            interaction_mode = st.radio(
                "Select Mode",
                ["Set Start", "Set Goal", "Add Obstacles", "Remove Obstacles"]
            )
            
            # Create grid button
            if st.button("🆕 Create New Grid", type="primary"):
                st.session_state.grid = InteractiveGridPathfinder(grid_width, grid_height)
            
            # Clear grid button
            if st.button("🧹 Clear Grid"):
                if 'grid' in st.session_state:
                    st.session_state.grid.clear()
            
            # Find path button
            if st.button("🔍 Find Path", type="primary"):
                if 'grid' in st.session_state and st.session_state.grid.start and st.session_state.grid.goal:
                    path_found = st.session_state.grid.run_algorithm(algorithm)
                    if path_found:
                        st.success(f"✅ Path found! Length: {len(st.session_state.grid.path)}")
                    else:
                        st.error("❌ No path found between start and goal!")
                else:
                    st.error("Please set both start and goal points!")
            
            # Preset patterns
            st.markdown("### 🎨 Preset Patterns")
            if st.button("🏰 Load Maze"):
                if 'grid' not in st.session_state:
                    st.session_state.grid = InteractiveGridPathfinder(grid_width, grid_height)
                # Create maze pattern
                for y in range(2, grid_height-2, 2):
                    for x in range(1, grid_width-1):
                        if x % 4 != 2:
                            st.session_state.grid.set_obstacle(x, y)
            
            if st.button("🎲 Random Obstacles"):
                if 'grid' not in st.session_state:
                    st.session_state.grid = InteractiveGridPathfinder(grid_width, grid_height)
                # Random obstacles
                num_obstacles = int(grid_width * grid_height * 0.2)
                for _ in range(num_obstacles):
                    x, y = random.randint(0, grid_width-1), random.randint(0, grid_height-1)
                    st.session_state.grid.set_obstacle(x, y)
            
            # Results display
            if 'grid' in st.session_state and st.session_state.grid.path:
                st.markdown("### 📊 Path Information")
                st.metric("Path Length", len(st.session_state.grid.path))
                st.metric("Nodes Explored", len(st.session_state.grid.visited))
                
                if len(st.session_state.grid.visited) > 0:
                    efficiency = (len(st.session_state.grid.path) / len(st.session_state.grid.visited)) * 100
                    st.metric("Efficiency", f"{efficiency:.1f}%")
        
        with col1:
            # Initialize grid if not exists
            if 'grid' not in st.session_state:
                st.session_state.grid = InteractiveGridPathfinder(grid_width, grid_height)
            
            # Create interactive hint
            st.markdown("""
            <div class="interactive-hint">
                💡 <strong>Click on the grid cells below to interact!</strong>
                <br>• 🟢 Set start point • 🔴 Set goal point • ⬛ Add/remove obstacles
                <br>• Click "Find Path" button after setting start and goal points
            </div>
            """, unsafe_allow_html=True)
            
            # Render the interactive grid
            st.markdown('<div class="grid-container">', unsafe_allow_html=True)
            
            for y in range(st.session_state.grid.height):
                st.markdown('<div class="grid-row">', unsafe_allow_html=True)
                
                for x in range(st.session_state.grid.width):
                    # Determine cell class and content
                    cell_class = "grid-cell cell-empty"
                    cell_content = ""
                    
                    if (x, y) == st.session_state.grid.start:
                        cell_class = "grid-cell cell-start"
                        cell_content = "S"
                    elif (x, y) == st.session_state.grid.goal:
                        cell_class = "grid-cell cell-goal"
                        cell_content = "G"
                    elif (x, y) in st.session_state.grid.obstacles:
                        cell_class = "grid-cell cell-obstacle"
                    elif (x, y) in st.session_state.grid.path:
                        cell_class = "grid-cell cell-path"
                        cell_content = "•"
                    elif (x, y) in st.session_state.grid.visited:
                        cell_class = "grid-cell cell-visited"
                    
                    # Create the cell with a unique key for click handling
                    cell_key = f"cell_{x}_{y}"
                    st.markdown(
                        f'<div class="{cell_class}" id="{cell_key}" onclick="handleCellClick(\'{cell_key}\', {x}, {y})">{cell_content}</div>',
                        unsafe_allow_html=True
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # JavaScript for handling cell clicks
            st.markdown("""
            <script>
            function handleCellClick(cellId, x, y) {
                // Use Streamlit's setQueryParam to trigger a rerun
                const searchParams = new URLSearchParams(window.location.search);
                searchParams.set('cell_clicked', cellId);
                searchParams.set('cell_x', x);
                searchParams.set('cell_y', y);
                
                // Update URL without refreshing page
                window.history.replaceState(null, null, '?' + searchParams.toString());
                
                // Trigger Streamlit rerun
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: {clicked: cellId, x: x, y: y}
                }, '*');
            }
            </script>
            """, unsafe_allow_html=True)
            
            # Handle cell clicks
            query_params = st.experimental_get_query_params()
            if 'cell_clicked' in query_params and 'cell_x' in query_params and 'cell_y' in query_params:
                x = int(query_params['cell_x'][0])
                y = int(query_params['cell_y'][0])
                
                if interaction_mode == "Set Start":
                    st.session_state.grid.set_start(x, y)
                elif interaction_mode == "Set Goal":
                    st.session_state.grid.set_goal(x, y)
                elif interaction_mode == "Add Obstacles":
                    st.session_state.grid.set_obstacle(x, y)
                elif interaction_mode == "Remove Obstacles":
                    st.session_state.grid.remove_obstacle(x, y)
                
                # Clear the query parameters
                st.experimental_set_query_params()
                st.rerun()
            
            # Alternative click handling using Streamlit components
            cell_clicked = st.selectbox(
                "Click on cells above or select position here:",
                [(x, y) for y in range(grid_height) for x in range(grid_width)],
                label_visibility="collapsed"
            )
            
            col_action1, col_action2 = st.columns(2)
            with col_action1:
                if st.button(f"Apply {interaction_mode}"):
                    x, y = cell_clicked
                    if interaction_mode == "Set Start":
                        st.session_state.grid.set_start(x, y)
                    elif interaction_mode == "Set Goal":
                        st.session_state.grid.set_goal(x, y)
                    elif interaction_mode == "Add Obstacles":
                        st.session_state.grid.set_obstacle(x, y)
                    elif interaction_mode == "Remove Obstacles":
                        st.session_state.grid.remove_obstacle(x, y)
                    st.rerun()
            
            with col_action2:
                if st.button("Clear Cell"):
                    x, y = cell_clicked
                    st.session_state.grid.remove_obstacle(x, y)
                    st.rerun()
    
    # Real Maps Tab with Enhanced Functionality
    with pathfind_tabs[1]:
        st.subheader("🌍 Real-World Map Pathfinding")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### 🎛️ Map Controls")
            
            # Interactive mode for maps
            map_interaction_mode = st.radio(
                "🖱️ Click Mode",
                ["Set Start Point", "Set End Point", "Add Waypoint", "Remove Waypoint"],
                key="map_interaction_mode"
            )
            
            # Transport mode
            transport_mode = st.selectbox(
                "🚗 Transport Mode",
                ["driving", "walking", "cycling", "transit"]
            )
            
            # Map style
            map_style = st.selectbox(
                "🗺️ Map Style",
                ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark", "Stamen Terrain"]
            )
            
            if st.button("🗺️ Calculate Route", type="primary"):
                if 'map_start_coords' in st.session_state and 'map_end_coords' in st.session_state:
                    # Create route visualization
                    start_coords = st.session_state.map_start_coords
                    end_coords = st.session_state.map_end_coords
                    
                    # Simulate a route calculation
                    # In a real app, you would use a routing API like OpenRouteService, OSRM, or Google Maps
                    waypoints = st.session_state.get('waypoints', [])
                    
                    # Calculate distances
                    points = [start_coords] + waypoints + [end_coords]
                    total_distance = 0
                    for i in range(len(points) - 1):
                        # Calculate haversine distance
                        lat1, lon1 = points[i]
                        lat2, lon2 = points[i+1]
                        
                        R = 6371  # Earth radius in km
                        dLat = math.radians(lat2 - lat1)
                        dLon = math.radians(lon2 - lon1)
                        a = (math.sin(dLat/2) * math.sin(dLat/2) + 
                             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                             math.sin(dLon/2) * math.sin(dLon/2))
                        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                        distance = R * c
                        
                        total_distance += distance
                    
                    # Store route info
                    st.session_state.route_info = {
                        'distance': total_distance,
                        'duration': total_distance / (40 if transport_mode == "driving" else 
                                                     15 if transport_mode == "cycling" else 
                                                     5 if transport_mode == "walking" else 25) * 60,
                        'start': start_coords,
                        'end': end_coords,
                        'waypoints': waypoints,
                        'transport_mode': transport_mode
                    }
                else:
                    st.error("Please set both start and end points by clicking on the map!")
            
            # Quick locations
            st.markdown("### 🌟 Quick Locations")
            quick_locations = {
                "New York City": (40.7128, -74.0060),
                "London": (51.5074, -0.1278),
                "Tokyo": (35.6762, 139.6503),
                "Paris": (48.8566, 2.3522),
                "Sydney": (-33.8688, 151.2093),
                "Mumbai": (19.0760, 72.8777),
                "Dubai": (25.2048, 55.2708),
                "Singapore": (1.3521, 103.8198)
            }
            
            selected_location = st.selectbox("Select City", list(quick_locations.keys()))
            if st.button("🏙️ Go to City"):
                st.session_state.map_center = quick_locations[selected_location]
                st.rerun()
            
            if st.button("🧹 Clear All Points"):
                for key in ['map_start_coords', 'map_end_coords', 'waypoints', 'route_info']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            
            # Route information
            if 'route_info' in st.session_state:
                st.markdown("### 📊 Route Information")
                info = st.session_state.route_info
                st.metric("Total Distance", f"{info['distance']:.2f} km")
                st.metric("Est. Duration", f"{info['duration']:.0f} minutes")
                st.metric("Transport Mode", info['transport_mode'].title())
                if info.get('waypoints'):
                    st.metric("Waypoints", len(info['waypoints']))
        
        with col1:
            # Initialize map center
            if 'map_center' not in st.session_state:
                st.session_state.map_center = (40.7128, -74.0060)  # NYC default
            
            # Initialize waypoints
            if 'waypoints' not in st.session_state:
                st.session_state.waypoints = []
            
            # Create map with the selected style
            tile_mapping = {
                "OpenStreetMap": "OpenStreetMap",
                "CartoDB Positron": "CartoDB positron",
                "CartoDB Dark": "CartoDB dark_matter",
                "Stamen Terrain": "Stamen Terrain"
            }
            
            m = folium.Map(
                location=st.session_state.map_center,
                zoom_start=12,
                tiles=tile_mapping[map_style]
            )
            
            # Add markers if points are set
            if 'map_start_coords' in st.session_state:
                folium.Marker(
                    location=st.session_state.map_start_coords,
                    popup="Start Point",
                    icon=folium.Icon(color='green', icon='play', prefix='fa'),
                    draggable=True
                ).add_to(m)
            
            if 'map_end_coords' in st.session_state:
                folium.Marker(
                    location=st.session_state.map_end_coords,
                    popup="End Point",
                    icon=folium.Icon(color='red', icon='stop', prefix='fa'),
                    draggable=True
                ).add_to(m)
            
            # Add waypoints
            for i, waypoint in enumerate(st.session_state.waypoints):
                folium.Marker(
                    location=waypoint,
                    popup=f"Waypoint {i+1}",
                    icon=folium.Icon(color='blue', icon='info', prefix='fa')
                ).add_to(m)
            
            # Add route if calculated
            if 'route_info' in st.session_state and 'map_start_coords' in st.session_state and 'map_end_coords' in st.session_state:
                # Create route with waypoints
                route_coords = [st.session_state.map_start_coords]
                route_coords.extend(st.session_state.route_info.get('waypoints', []))
                route_coords.append(st.session_state.map_end_coords)
                
                # Draw route
                folium.PolyLine(
                    locations=route_coords,
                    color='blue',
                    weight=5,
                    opacity=0.8,
                    popup=f"Route: {st.session_state.route_info['distance']:.2f} km"
                ).add_to(m)
                
                # Add distance markers between points
                for i in range(len(route_coords) - 1):
                    mid_lat = (route_coords[i][0] + route_coords[i+1][0]) / 2
                    mid_lon = (route_coords[i][1] + route_coords[i+1][1]) / 2
                    
                    # Calculate segment distance
                    lat1, lon1 = route_coords[i]
                    lat2, lon2 = route_coords[i+1]
                    
                    R = 6371  # Earth radius in km
                    dLat = math.radians(lat2 - lat1)
                    dLon = math.radians(lon2 - lon1)
                    a = (math.sin(dLat/2) * math.sin(dLat/2) + 
                         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
                         math.sin(dLon/2) * math.sin(dLon/2))
                    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                    segment_dist = R * c
                    
                    folium.Marker(
                        location=[mid_lat, mid_lon],
                        popup=f"Segment {i+1}: {segment_dist:.2f} km",
                        icon=folium.DivIcon(html=f"""
                            <div style="background-color: white; border: 2px solid blue; 
                                        border-radius: 50%; width: 30px; height: 30px; 
                                        text-align: center; line-height: 30px; font-weight: bold;">
                                {i+1}
                            </div>
                        """)
                    ).add_to(m)
            
            # Display map
            folium_static(m)
            
            # Handle map clicks
            if st.button("✅ Confirm Map Click"):
                map_lat = st.number_input("Latitude", value=st.session_state.map_center[0], 
                                         format="%.6f", key="map_click_lat")
                map_lng = st.number_input("Longitude", value=st.session_state.map_center[1], 
                                         format="%.6f", key="map_click_lng")
                
                clicked_coords = (map_lat, map_lng)
                
                if map_interaction_mode == "Set Start Point":
                    st.session_state.map_start_coords = clicked_coords
                    st.success(f"Start point set at: {clicked_coords}")
                    st.rerun()
                elif map_interaction_mode == "Set End Point":
                    st.session_state.map_end_coords = clicked_coords
                    st.success(f"End point set at: {clicked_coords}")
                    st.rerun()
                elif map_interaction_mode == "Add Waypoint":
                    if 'waypoints' not in st.session_state:
                        st.session_state.waypoints = []
                    st.session_state.waypoints.append(clicked_coords)
                    st.success(f"Waypoint added at: {clicked_coords}")
                    st.rerun()
                elif map_interaction_mode == "Remove Waypoint":
                    if 'waypoints' in st.session_state and st.session_state.waypoints:
                        # Find closest waypoint to clicked position
                        closest_idx = 0
                        closest_dist = float('inf')
                        for i, waypoint in enumerate(st.session_state.waypoints):
                            dist = ((waypoint[0] - clicked_coords[0])**2 + 
                                    (waypoint[1] - clicked_coords[1])**2)**0.5
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_idx = i
                        
                        st.session_state.waypoints.pop(closest_idx)
                        st.success("Closest waypoint removed")
                        st.rerun()
    
    # Algorithm Comparison Tab
    with pathfind_tabs[2]:
        st.subheader("🆚 Algorithm Performance Comparison")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### ⚙️ Comparison Settings")
            
            comp_width = st.slider("Grid Width", 10, 30, 20, key="comp_width")
            comp_height = st.slider("Grid Height", 10, 30, 15, key="comp_height")
            
            obstacle_density = st.slider("Obstacle Density", 0.0, 0.4, 0.2)
            num_runs = st.slider("Number of Test Runs", 1, 5, 3)
            
            test_scenarios = st.multiselect(
                "Test Scenarios",
                ["Empty Grid", "Sparse Obstacles", "Dense Obstacles", "Maze", "Diagonal Barriers"],
                default=["Sparse Obstacles", "Dense Obstacles"]
            )
            
            if st.button("🏁 Run Comparison", type="primary"):
                algorithms = ["A* (A-Star)", "Dijkstra", "BFS", "DFS"]
                all_results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_tests = len(test_scenarios) * len(algorithms) * num_runs
                test_count = 0
                
                for scenario in test_scenarios:
                    scenario_results = {algo: {'times': [], 'paths': [], 'visited': []} for algo in algorithms}
                    
                    for run in range(num_runs):
                        # Create test grid
                        test_grid = InteractiveGridPathfinder(comp_width, comp_height)
                        
                        # Generate obstacles based on scenario
                        if scenario == "Sparse Obstacles":
                            num_obstacles = int(comp_width * comp_height * 0.1)
                        elif scenario == "Dense Obstacles":
                            num_obstacles = int(comp_width * comp_height * obstacle_density)
                        elif scenario == "Maze":
                            num_obstacles = 0
                            for y in range(2, comp_height-2, 4):
                                for x in range(1, comp_width-1):
                                    if x % 4 != 2:
                                        test_grid.set_obstacle(x, y)
                        elif scenario == "Diagonal Barriers":
                            for i in range(min(comp_width, comp_height) // 2):
                                if i < comp_width and i < comp_height:
                                    test_grid.set_obstacle(i, i)
                        else:  # Empty Grid
                            num_obstacles = 0
                        
                        # Add random obstacles if needed
                        if scenario in ["Sparse Obstacles", "Dense Obstacles"]:
                            for _ in range(num_obstacles):
                                x = random.randint(1, comp_width-2)
                                y = random.randint(1, comp_height-2)
                                test_grid.set_obstacle(x, y)
                        
                        # Set start and goal
                        test_grid.set_start(0, 0)
                        test_grid.set_goal(comp_width-1, comp_height-1)
                        
                        # Test each algorithm
                        for algo in algorithms:
                            test_count += 1
                            progress_bar.progress(test_count / total_tests)
                            status_text.text(f"Testing {algo} on {scenario} (Run {run+1}/{num_runs})")
                            
                            start_time = time.time()
                            
                            # Run algorithm
                            path_found = test_grid.run_algorithm(algo)
                            
                            end_time = time.time()
                            execution_time = (end_time - start_time) * 1000
                            
                            # Extract results
                            path_length = len(test_grid.path) if test_grid.path else 0
                            nodes_visited = len(test_grid.visited) if test_grid.visited else 0
                            
                            scenario_results[algo]['times'].append(execution_time)
                            scenario_results[algo]['paths'].append(path_length)
                            scenario_results[algo]['visited'].append(nodes_visited)
                    
                    # Calculate averages
                    for algo in algorithms:
                        all_results.append({
                            'Scenario': scenario,
                            'Algorithm': algo,
                            'Avg Time (ms)': np.mean(scenario_results[algo]['times']),
                            'Avg Path Length': np.mean(scenario_results[algo]['paths']),
                            'Avg Nodes Visited': np.mean(scenario_results[algo]['visited']),
                            'Efficiency %': (np.mean(scenario_results[algo]['paths']) / np.mean(scenario_results[algo]['visited']) * 100) 
                                if np.mean(scenario_results[algo]['visited']) > 0 else 0
                        })
                
                st.session_state.comparison_results = pd.DataFrame(all_results)
                progress_bar.empty()
                status_text.empty()
        
        with col1:
            if 'comparison_results' in st.session_state:
                df = st.session_state.comparison_results
                
                # Display results table
                st.markdown("### 📊 Comparison Results")
                st.dataframe(
                    df.style.highlight_min(subset=['Avg Time (ms)', 'Avg Nodes Visited'], color='lightgreen')
                            .highlight_max(subset=['Efficiency %'], color='lightgreen'),
                    use_container_width=True
                )
                
                # Create visualization
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Execution Time by Scenario", "Nodes Visited", 
                                  "Path Length Comparison", "Algorithm Efficiency"),
                    specs=[[{"type": "bar"}, {"type": "bar"}],
                          [{"type": "bar"}, {"type": "scatter"}]]
                )
                
                # Time comparison
                for scenario in df['Scenario'].unique():
                    scenario_data = df[df['Scenario'] == scenario]
                    fig.add_trace(
                        go.Bar(
                            x=scenario_data['Algorithm'],
                            y=scenario_data['Avg Time (ms)'],
                            name=scenario,
                            legendgroup=scenario
                        ),
                        row=1, col=1
                    )
                
                # Nodes visited
                for scenario in df['Scenario'].unique():
                    scenario_data = df[df['Scenario'] == scenario]
                    fig.add_trace(
                        go.Bar(
                            x=scenario_data['Algorithm'],
                            y=scenario_data['Avg Nodes Visited'],
                            name=scenario,
                            legendgroup=scenario,
                            showlegend=False
                        ),
                        row=1, col=2
                    )
                
                # Path length
                for algorithm in df['Algorithm'].unique():
                    algo_data = df[df['Algorithm'] == algorithm]
                    fig.add_trace(
                        go.Bar(
                            x=algo_data['Scenario'],
                            y=algo_data['Avg Path Length'],
                            name=algorithm
                        ),
                        row=2, col=1
                    )
                
                # Efficiency scatter
                fig.add_trace(
                    go.Scatter(
                        x=df['Avg Time (ms)'],
                        y=df['Efficiency %'],
                        mode='markers+text',
                        text=df['Algorithm'],
                        textposition="top center",
                        marker=dict(
                            size=15,
                            color=df['Efficiency %'],
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Efficiency %")
                        ),
                        showlegend=False
                    ),
                    row=2, col=2
                )
                
                fig.update_layout(
                    title="Algorithm Performance Analysis",
                    height=800,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary insights
                st.markdown("### 🔍 Performance Insights")
                
                best_time = df.loc[df['Avg Time (ms)'].idxmin()]
                best_efficiency = df.loc[df['Efficiency %'].idxmax()]
                
                col_insight1, col_insight2 = st.columns(2)
                with col_insight1:
                    st.info(f"""
                    **⚡ Fastest Algorithm:** {best_time['Algorithm']}
                    - Scenario: {best_time['Scenario']}
                    - Avg Time: {best_time['Avg Time (ms)']:.2f} ms
                    """)
                
                with col_insight2:
                    st.success(f"""
                    **🎯 Most Efficient:** {best_efficiency['Algorithm']}
                    - Scenario: {best_efficiency['Scenario']}
                    - Efficiency: {best_efficiency['Efficiency %']:.1f}%
                    """)
            else:
                st.info("🏁 Run the comparison to see detailed performance metrics across different scenarios and algorithms.")

# Tab 2: Enhanced Sorting Visualizer
with tab2:
    st.header("📊 Advanced Sorting Algorithm Visualizer")
    
    # Create sub-tabs for sorting
    sort_tabs = st.tabs(["📊 Classic Sorting", "🌍 Real-World Applications", "🆚 Algorithm Race"])
    
    # Classic Sorting Tab with Step-by-Step Animation
    with sort_tabs[0]:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### 🎛️ Sorting Controls")
            
            # Array configuration
            array_size = st.slider("📏 Array Size", 5, 50, 20)
            array_type = st.selectbox(
                "📊 Array Type",
                ["Random", "Nearly Sorted", "Reverse Sorted", "Few Unique"]
            )
            
            # Algorithm selection
            sort_algorithm = st.selectbox(
                "🔄 Sorting Algorithm",
                ["Bubble Sort", "Selection Sort", "Insertion Sort", "Quick Sort", "Merge Sort", "Heap Sort"]
            )
            
            # Animation settings
            animation_speed = st.slider("⚡ Animation Speed", 0.01, 1.0, 0.3)
            show_comparisons = st.checkbox("👀 Show Comparisons", value=True)
            show_swaps = st.checkbox("🔄 Show Swaps", value=True)
            
            # Generate array button
            if st.button("🎲 Generate New Array", type="primary"):
                if array_type == "Random":
                    arr = [random.randint(1, 100) for _ in range(array_size)]
                elif array_type == "Nearly Sorted":
                    arr = list(range(1, array_size + 1))
                    for _ in range(array_size // 10):
                        i, j = random.randint(0, array_size - 1), random.randint(0, array_size - 1)
                        arr[i], arr[j] = arr[j], arr[i]
                elif array_type == "Reverse Sorted":
                    arr = list(range(array_size, 0, -1))
                elif array_type == "Few Unique":
                    unique_values = [random.randint(1, 20) for _ in range(5)]
                    arr = [random.choice(unique_values) for _ in range(array_size)]
                
                st.session_state.sorting_array = arr.copy()
                st.session_state.sorting_steps = []
                st.session_state.current_sorting_step = 0
                st.session_state.sorting_stats = {"comparisons": 0, "swaps": 0, "accesses": 0}
            
            # Step-by-step controls
            if st.button("▶️ Start Sorting"):
                if 'sorting_array' in st.session_state:
                    st.session_state.sorting_in_progress = True
                    st.session_state.sorting_steps = []
                    st.session_state.current_sorting_step = 0
                    
                    # Capture steps during sorting
                    def step_callback(arr, comparing, swapping, stats):
                        st.session_state.sorting_steps.append({
                            'array': arr.copy(),
                            'comparing': comparing.copy() if comparing else [],
                            'swapping': swapping.copy() if swapping else [],
                            'stats': stats.copy()
                        })
                    
                    # Run sorting algorithm with step callback
                    sorted_arr, final_stats = SortingAlgorithms.run_algorithm(
                        sort_algorithm, 
                        st.session_state.sorting_array.copy(),
                        step_callback
                    )
                    
                    # Ensure we have the final state
                    if not st.session_state.sorting_steps or st.session_state.sorting_steps[-1]['array'] != sorted_arr:
                        st.session_state.sorting_steps.append({
                            'array': sorted_arr.copy(),
                            'comparing': [],
                            'swapping': [],
                            'stats': final_stats
                        })
                    
                    st.session_state.sorting_stats = final_stats
                    st.rerun()
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("⏮️ Previous Step"):
                    if ('current_sorting_step' in st.session_state and 
                        st.session_state.current_sorting_step > 0):
                        st.session_state.current_sorting_step -= 1
                        st.rerun()
            
            with col_btn2:
                if st.button("⏭️ Next Step"):
                    if ('current_sorting_step' in st.session_state and 
                        'sorting_steps' in st.session_state and
                        st.session_state.current_sorting_step < len(st.session_state.sorting_steps) - 1):
                        st.session_state.current_sorting_step += 1
                        st.rerun()
            
            if st.button("⏯️ Auto-Play"):
                st.session_state.auto_play_sorting = True
                st.rerun()
            
            if st.button("⏹️ Stop"):
                st.session_state.auto_play_sorting = False
                st.rerun()
            
            # Sorting statistics
            if 'sorting_stats' in st.session_state:
                st.markdown("### 📊 Sorting Statistics")
                
                # Get current stats
                if ('sorting_steps' in st.session_state and 
                    'current_sorting_step' in st.session_state and 
                    st.session_state.sorting_steps):
                    
                    current_step = st.session_state.sorting_steps[st.session_state.current_sorting_step]
                    stats = current_step['stats']
                else:
                    stats = st.session_state.sorting_stats
                
                st.metric("Comparisons", stats['comparisons'])
                st.metric("Swaps", stats['swaps'])
                st.metric("Array Accesses", stats['accesses'])
                
                if 'sorting_steps' in st.session_state:
                    current_step = st.session_state.current_sorting_step + 1
                    total_steps = len(st.session_state.sorting_steps)
                    st.metric("Step", f"{current_step}/{total_steps}")
                    
                    # Calculate progress percentage
                    progress = current_step / total_steps
                    st.progress(progress)
        
        with col1:
            st.markdown("### 📊 Sorting Visualization")
            
            # Display the current state of the sorting array
            if 'sorting_array' in st.session_state:
                # Get current array state
                if ('sorting_steps' in st.session_state and 
                    'current_sorting_step' in st.session_state and 
                    st.session_state.sorting_steps):
                    
                    current_step = st.session_state.sorting_steps[st.session_state.current_sorting_step]
                    arr = current_step['array']
                    comparing = current_step.get('comparing', [])
                    swapping = current_step.get('swapping', [])
                else:
                    arr = st.session_state.sorting_array
                    comparing = []
                    swapping = []
                
                # Create bar chart visualization
                fig = go.Figure()
                
                for i, val in enumerate(arr):
                    color = 'lightblue'
                    if i in swapping and show_swaps:
                        color = 'red'
                    elif i in comparing and show_comparisons:
                        color = 'orange'
                    
                    fig.add_trace(go.Bar(
                        x=[i],
                        y=[val],
                        marker_color=color,
                        showlegend=False,
                        text=val,
                        textposition='outside'
                    ))
                
                fig.update_layout(
                    title=f"{sort_algorithm} - Step {st.session_state.get('current_sorting_step', 0) + 1}",
                    xaxis_title="Index",
                    yaxis_title="Value",
                    height=400,
                    margin=dict(l=20, r=20, t=60, b=40),
                    bargap=0.15,
                    xaxis=dict(showticklabels=False)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create step description
                if ('sorting_steps' in st.session_state and 
                    'current_sorting_step' in st.session_state and 
                    len(comparing) > 0):
                    st.info(f"Comparing elements at indices {comparing}")
                elif ('sorting_steps' in st.session_state and 
                      'current_sorting_step' in st.session_state and 
                      len(swapping) > 0):
                    st.success(f"Swapping elements at indices {swapping}")
                
                # Auto-play logic
                if st.session_state.get('auto_play_sorting', False):
                    if ('current_sorting_step' in st.session_state and 
                        'sorting_steps' in st.session_state and
                        st.session_state.current_sorting_step < len(st.session_state.sorting_steps) - 1):
                        time.sleep(animation_speed)
                        st.session_state.current_sorting_step += 1
                        st.rerun()
                    else:
                        st.session_state.auto_play_sorting = False
            else:
                st.info("Generate an array to begin visualization")
                
                # Show sample array placeholder
                sample_arr = [random.randint(1, 100) for _ in range(20)]
                
                fig = go.Figure()
                for i, val in enumerate(sample_arr):
                    fig.add_trace(go.Bar(
                        x=[i],
                        y=[val],
                        marker_color='lightgrey',
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title="Sample Array (Click 'Generate New Array' to begin)",
                    xaxis_title="Index",
                    yaxis_title="Value",
                    height=400,
                    margin=dict(l=20, r=20, t=60, b=40),
                    bargap=0.15,
                    xaxis=dict(showticklabels=False)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            # Algorithm complexity information
            st.markdown("### 🧮 Algorithm Information")
            
            complexity_info = {
                "Bubble Sort": {
                    "Time (Best)": "O(n)",
                    "Time (Average)": "O(n²)",
                    "Time (Worst)": "O(n²)",
                    "Space": "O(1)",
                    "Stable": "Yes",
                    "Description": "Simple comparison-based algorithm that repeatedly steps through the list, compares adjacent elements, and swaps them if they are in the wrong order."
                },
                "Selection Sort": {
                    "Time (Best)": "O(n²)",
                    "Time (Average)": "O(n²)",
                    "Time (Worst)": "O(n²)",
                    "Space": "O(1)",
                    "Stable": "No",
                    "Description": "Divides the input list into a sorted and an unsorted region, repeatedly finding the minimum element from the unsorted region and moving it to the sorted region."
                },
                "Insertion Sort": {
                    "Time (Best)": "O(n)",
                    "Time (Average)": "O(n²)",
                    "Time (Worst)": "O(n²)",
                    "Space": "O(1)",
                    "Stable": "Yes",
                    "Description": "Builds the sorted array one item at a time by comparing each new element with the already sorted elements and inserting it at the correct position."
                },
                "Quick Sort": {
                    "Time (Best)": "O(n log n)",
                    "Time (Average)": "O(n log n)",
                    "Time (Worst)": "O(n²)",
                    "Space": "O(log n)",
                    "Stable": "No",
                    "Description": "Divides the array into smaller subarrays using a pivot element, then recursively sorts the subarrays. Highly efficient for large datasets."
                },
                "Merge Sort": {
                    "Time (Best)": "O(n log n)",
                    "Time (Average)": "O(n log n)",
                    "Time (Worst)": "O(n log n)",
                    "Space": "O(n)",
                    "Stable": "Yes",
                    "Description": "Divides the array into halves, recursively sorts them, and then merges the sorted halves. Guarantees O(n log n) performance but requires extra space."
                },
                "Heap Sort": {
                    "Time (Best)": "O(n log n)",
                    "Time (Average)": "O(n log n)",
                    "Time (Worst)": "O(n log n)",
                    "Space": "O(1)",
                    "Stable": "No",
                    "Description": "Builds a max heap from the array and repeatedly extracts the maximum element. Combines the benefits of good performance with minimal space usage."
                }
            }
            
            if sort_algorithm in complexity_info:
                info = complexity_info[sort_algorithm]
                
                with st.expander("📚 Algorithm Details", expanded=True):
                    st.markdown(f"**{sort_algorithm}**")
                    st.markdown(info["Description"])
                    
                    col_c1, col_c2, col_c3 = st.columns(3)
                    
                    with col_c1:
                        st.markdown("**Time Complexity:**")
                        st.markdown(f"Best: {info['Time (Best)']}")
                        st.markdown(f"Average: {info['Time (Average)']}")
                        st.markdown(f"Worst: {info['Time (Worst)']}")
                    
                    with col_c2:
                        st.markdown("**Space Complexity:**")
                        st.markdown(info["Space"])
                    
                    with col_c3:
                        st.markdown("**Stability:**")
                        st.markdown(info["Stable"])
    
    # Real-World Applications Tab
    with sort_tabs[1]:
        st.subheader("🌍 Real-World Sorting Applications")
        
        real_world_app = st.selectbox(
            "Select Application",
            ["📅 Task Scheduler", "📁 File Organizer", "🎵 Music Playlist Optimizer", 
             "📦 Inventory Manager", "🎓 Student Ranking System"]
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if real_world_app == "📅 Task Scheduler":
                st.markdown("### Task Priority Scheduler")
                st.info("Sort tasks by priority and deadline for optimal productivity")
                
                # Generate sample tasks
                if st.button("Generate Sample Tasks", key="gen_tasks"):
                    tasks = []
                    task_names = ["Email Client", "Report Writing", "Team Meeting", "Code Review", 
                                 "Documentation", "Unit Testing", "Sprint Planning", "Customer Call"]
                    priorities = ["Low", "Medium", "High", "Critical"]
                    
                    for i in range(8):
                        tasks.append({
                            'id': i + 1,
                            'name': random.choice(task_names),
                            'priority': random.choice(priorities),
                            'priority_value': ["Low", "Medium", "High", "Critical"].index(random.choice(priorities)),
                            'deadline': datetime.now() + timedelta(hours=random.randint(1, 48)),
                            'duration': random.randint(15, 120)
                        })
                    st.session_state.tasks = tasks
                
                if 'tasks' in st.session_state:
                    # Display unsorted tasks
                    st.markdown("#### 📋 Unsorted Tasks")
                    df_tasks = pd.DataFrame(st.session_state.tasks)
                    st.dataframe(df_tasks[['name', 'priority', 'deadline', 'duration']], use_container_width=True)
                    
                    # Sort options
                    sort_by = st.selectbox("Sort By", ["Priority then Deadline", "Deadline only", "Duration"])
                    
                    if st.button("🔄 Sort Tasks"):
                        if sort_by == "Priority then Deadline":
                            sorted_tasks = sorted(st.session_state.tasks, 
                                                key=lambda x: (-x['priority_value'], x['deadline']))
                        elif sort_by == "Deadline only":
                            sorted_tasks = sorted(st.session_state.tasks, key=lambda x: x['deadline'])
                        else:
                            sorted_tasks = sorted(st.session_state.tasks, key=lambda x: x['duration'])
                        
                        st.session_state.sorted_tasks = sorted_tasks
                    
                    if 'sorted_tasks' in st.session_state:
                        st.markdown("#### ✅ Sorted Tasks")
                        df_sorted = pd.DataFrame(st.session_state.sorted_tasks)
                        st.dataframe(df_sorted[['name', 'priority', 'deadline', 'duration']], use_container_width=True)
                        
                        # Gantt chart visualization
                        fig = px.timeline(
                            df_sorted,
                            x_start=[row['deadline'] - timedelta(minutes=row['duration']) for _, row in df_sorted.iterrows()],
                            x_end='deadline',
                            y='name',
                            color='priority',
                            title="Task Timeline (Gantt Chart)"
                        )
                        fig.update_yaxes(categoryorder="total ascending")
                        st.plotly_chart(fig, use_container_width=True)
            
            elif real_world_app == "📁 File Organizer":
                st.markdown("### Smart File Organizer")
                st.info("Organize files by type, date, and size for better file management")
                
                if st.button("Generate Sample Files", key="gen_files"):
                    files = []
                    extensions = ['pdf', 'docx', 'xlsx', 'png', 'jpg', 'mp4', 'txt']
                    file_prefixes = ['Report', 'Document', 'Image', 'Video', 'Data']
                    
                    for i in range(20):
                        ext = random.choice(extensions)
                        files.append({
                            'name': f"{random.choice(file_prefixes)}_{i+1}.{ext}",
                            'size': random.randint(100, 10000),  # KB
                            'modified': datetime.now() - timedelta(days=random.randint(0, 365)),
                            'type': ext
                        })
                    st.session_state.files = files
                
                if 'files' in st.session_state:
                    st.markdown("#### 📂 File List")
                    df_files = pd.DataFrame(st.session_state.files)
                    st.dataframe(df_files, use_container_width=True)
                    
                    # Sorting options
                    sort_method = st.selectbox(
                        "Sort Method",
                        ["By Type", "By Size (Largest First)", "By Date (Newest First)", "By Name"]
                    )
                    
                    if st.button("🔄 Organize Files"):
                        if sort_method == "By Type":
                            sorted_files = sorted(st.session_state.files, key=lambda x: x['type'])
                        elif sort_method == "By Size (Largest First)":
                            sorted_files = sorted(st.session_state.files, key=lambda x: x['size'], reverse=True)
                        elif sort_method == "By Date (Newest First)":
                            sorted_files = sorted(st.session_state.files, key=lambda x: x['modified'], reverse=True)
                        else:
                            sorted_files = sorted(st.session_state.files, key=lambda x: x['name'])
                        
                        # Group by type
                        file_groups = {}
                        for file in sorted_files:
                            file_type = file['type']
                            if file_type not in file_groups:
                                file_groups[file_type] = []
                            file_groups[file_type].append(file)
                        
                        st.session_state.sorted_files = sorted_files
                        st.session_state.file_groups = file_groups
                    
                    # Display results
                    if 'file_groups' in st.session_state:
                        st.markdown("#### 📁 Organized by Type")
                        for file_type, files in st.session_state.file_groups.items():
                            with st.expander(f"{file_type.upper()} Files ({len(files)})"):
                                df = pd.DataFrame(files)
                                st.dataframe(df[['name', 'size', 'modified']], use_container_width=True)
                    
                    elif 'sorted_files' in st.session_state:
                        st.markdown("#### 📁 Sorted Files")
                        df_sorted = pd.DataFrame(st.session_state.sorted_files)
                        st.dataframe(df_sorted, use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Real-World Benefits")
            
            if real_world_app == "📅 Task Scheduler":
                st.info("""
                **Benefits of Task Sorting:**
                - 📈 Increased productivity
                - ⏰ Better time management
                - 🎯 Focus on priorities
                - 📊 Clear workflow visualization
                """)
                
                st.markdown("### 🧠 Algorithm Used")
                st.success("""
                **Priority Sorting:** Quick Sort or Merge Sort
                
                These algorithms are ideal for task scheduling because:
                - Efficient for medium-sized task lists
                - Stable sorting preserves original order for same-priority tasks
                - O(n log n) time complexity ensures quick organization even for busy schedules
                """)
                
            elif real_world_app == "📁 File Organizer":
                st.info("""
                **Benefits of File Sorting:**
                - 🔍 Faster file retrieval
                - 💾 Better storage management
                - 📊 Clear file overview
                - 🧹 Cleaner directories
                """)
                
                st.markdown("### 🧠 Algorithm Used")
                st.success("""
                **File Type Grouping:** Bucket Sort + Insertion Sort
                
                This combination works well because:
                - Bucket sort groups files by type efficiently
                - Insertion sort works well for sorting each small bucket
                - Natural ordering within file types is preserved
                - Very fast for common file organization tasks
                """)
    
    # Algorithm Race Tab
    with sort_tabs[2]:
        st.subheader("🆚 Sorting Algorithm Race")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### 🏁 Race Settings")
            
            race_array_size = st.slider("Array Size", 10, 1000, 100)
            race_array_type = st.selectbox(
                "Array Type",
                ["Random", "Nearly Sorted", "Reverse Sorted"],
                key="race_array_type"
            )
            
            selected_algorithms = st.multiselect(
                "Select Algorithms to Race",
                ["Bubble Sort", "Selection Sort", "Insertion Sort", "Quick Sort", "Merge Sort", "Heap Sort"],
                default=["Bubble Sort", "Quick Sort", "Merge Sort"]
            )
            
            if st.button("🏁 Start Race!", type="primary"):
                # Generate array
                if race_array_type == "Random":
                    race_array = [random.randint(1, 1000) for _ in range(race_array_size)]
                elif race_array_type == "Nearly Sorted":
                    race_array = list(range(1, race_array_size + 1))
                    for _ in range(race_array_size // 10):
                        i, j = random.randint(0, race_array_size - 1), random.randint(0, race_array_size - 1)
                        race_array[i], race_array[j] = race_array[j], race_array[i]
                else:
                    race_array = list(range(race_array_size, 0, -1))
                
                st.session_state.race_array = race_array
                st.session_state.race_algorithms = selected_algorithms
                st.session_state.race_results = {}
                
                # Run race
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, algo in enumerate(selected_algorithms):
                    status_text.text(f"Running {algo}...")
                    progress_bar.progress((i) / len(selected_algorithms))
                    
                    # Time the algorithm
                    start_time = time.time()
                    arr_copy = race_array.copy()
                    sorted_arr, stats = SortingAlgorithms.run_algorithm(algo, arr_copy)
                    end_time = time.time()
                    
                    # Store results
                    st.session_state.race_results[algo] = {
                        'time': end_time - start_time,
                        'comparisons': stats['comparisons'],
                        'swaps': stats['swaps'],
                        'accesses': stats['accesses']
                    }
                
                progress_bar.progress(1.0)
                status_text.text("Race completed!")
        
        with col1:
            if 'race_results' in st.session_state:
                st.markdown("### 🏁 Race Results")
                
                # Create results dataframe
                results = []
                for algo, stats in st.session_state.race_results.items():
                    results.append({
                        'Algorithm': algo,
                        'Time (seconds)': stats['time'],
                        'Comparisons': stats['comparisons'],
                        'Swaps': stats['swaps'],
                        'Array Accesses': stats['accesses']
                    })
                
                results_df = pd.DataFrame(results)
                results_df['Rank'] = results_df['Time (seconds)'].rank().astype(int)
                results_df = results_df.sort_values('Rank')
                
                # Winner announcement
                winner = results_df.iloc[0]['Algorithm']
                winner_time = results_df.iloc[0]['Time (seconds)']
                st.success(f"🥇 **Winner: {winner}** with {winner_time:.6f} seconds!")
                
                # Results table
                st.dataframe(results_df, use_container_width=True)
                
                # Performance chart
                fig = px.bar(
                    results_df,
                    x='Algorithm',
                    y='Time (seconds)',
                    color='Algorithm',
                    title=f"Algorithm Performance Comparison (Array Size: {race_array_size})"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics
                fig2 = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=("Comparisons", "Swaps", "Array Accesses")
                )
                
                fig2.add_trace(
                    go.Bar(x=results_df['Algorithm'], y=results_df['Comparisons'], name="Comparisons"),
                    row=1, col=1
                )
                
                fig2.add_trace(
                    go.Bar(x=results_df['Algorithm'], y=results_df['Swaps'], name="Swaps"),
                    row=1, col=2
                )
                
                fig2.add_trace(
                    go.Bar(x=results_df['Algorithm'], y=results_df['Array Accesses'], name="Array Accesses"),
                    row=1, col=3
                )
                
                fig2.update_layout(
                    title="Detailed Algorithm Metrics",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Select algorithms and click 'Start Race!' to compare their performance.")

# Footer with developer credits
st.markdown("---")
st.markdown("""
<div class="footer-developer">
    <h3>✨ Developed with ❤️ by Shreyas Kasture ✨</h3>
    <p>Advanced PathFinder & Sort Visualizer - Bringing algorithms to life!</p>
    <p style="font-size: 0.9em; opacity: 0.8;">
        Made with Streamlit, Plotly, and Folium | Interactive Learning Experience
    </p>
</div>
""", unsafe_allow_html=True)
