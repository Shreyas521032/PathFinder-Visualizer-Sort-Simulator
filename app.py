import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
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

# Page configuration
st.set_page_config(
    page_title="PathFinder & Sort Visualizer",
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
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🗺️ PathFinder & Sort Visualizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive pathfinding on real maps & animated sorting algorithms</p>', unsafe_allow_html=True)

# Enhanced Pathfinding algorithms implementation
class PathfindingAlgorithms:
    @staticmethod
    def heuristic(node1, node2, graph):
        """Calculate Euclidean distance between two nodes"""
        try:
            x1, y1 = graph.nodes[node1]['x'], graph.nodes[node1]['y']
            x2, y2 = graph.nodes[node2]['x'], graph.nodes[node2]['y']
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        except:
            return 0
    
    @staticmethod
    def a_star(graph, start, goal):
        """A* pathfinding algorithm with detailed tracking"""
        open_set = [(0, start)]
        came_from = {}
        g_score = {node: float('inf') for node in graph.nodes()}
        g_score[start] = 0
        f_score = {node: float('inf') for node in graph.nodes()}
        f_score[start] = PathfindingAlgorithms.heuristic(start, goal, graph)
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
            
            for neighbor in graph.neighbors(current):
                tentative_g_score = g_score[current] + graph.edges[current, neighbor, 0].get('length', 1)
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + PathfindingAlgorithms.heuristic(neighbor, goal, graph)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return [], visited
    
    @staticmethod
    def dijkstra(graph, start, goal):
        """Dijkstra's algorithm with detailed tracking"""
        distances = {node: float('inf') for node in graph.nodes()}
        distances[start] = 0
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
            
            if current_dist > distances[current]:
                continue
                
            for neighbor in graph.neighbors(current):
                weight = graph.edges[current, neighbor, 0].get('length', 1)
                distance = current_dist + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    came_from[neighbor] = current
                    heapq.heappush(pq, (distance, neighbor))
        
        return [], visited
    
    @staticmethod
    def bfs(graph, start, goal):
        """Breadth-First Search with detailed tracking"""
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
            
            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    visited.append(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)
        
        return [], visited
    
    @staticmethod
    def dfs(graph, start, goal):
        """Depth-First Search with detailed tracking"""
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
            
            for neighbor in graph.neighbors(current):
                if neighbor not in visited:
                    visited.append(neighbor)
                    came_from[neighbor] = current
                    stack.append(neighbor)
        
        return [], visited
    
    @staticmethod
    def greedy_best_first(graph, start, goal):
        """Greedy Best-First Search using heuristic only"""
        open_set = [(PathfindingAlgorithms.heuristic(start, goal, graph), start)]
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
            
            for neighbor in graph.neighbors(current):
                if neighbor not in visited and neighbor not in [item[1] for item in open_set]:
                    came_from[neighbor] = current
                    heuristic_cost = PathfindingAlgorithms.heuristic(neighbor, goal, graph)
                    heapq.heappush(open_set, (heuristic_cost, neighbor))
        
        return [], visited
    
    @staticmethod
    def bidirectional_search(graph, start, goal):
        """Bidirectional search from both start and goal"""
        if start == goal:
            return [start], [start]
        
        # Forward search from start
        visited_forward = {start}
        queue_forward = deque([start])
        came_from_forward = {start: None}
        
        # Backward search from goal
        visited_backward = {goal}
        queue_backward = deque([goal])
        came_from_backward = {goal: None}
        
        visited = []
        
        while queue_forward and queue_backward:
            # Forward step
            if queue_forward:
                current_forward = queue_forward.popleft()
                visited.append(current_forward)
                
                for neighbor in graph.neighbors(current_forward):
                    if neighbor in visited_backward:
                        # Found intersection
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
            
            # Backward step
            if queue_backward:
                current_backward = queue_backward.popleft()
                visited.append(current_backward)
                
                for neighbor in graph.neighbors(current_backward):
                    if neighbor in visited_forward:
                        # Found intersection
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

# Enhanced Sorting algorithms implementation
class SortingAlgorithms:
    @staticmethod
    def bubble_sort(arr):
        """Bubble Sort with step tracking"""
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
        """Selection Sort with step tracking"""
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
        """Insertion Sort with step tracking"""
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
        """Quick Sort with step tracking"""
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
        """Merge Sort with step tracking"""
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
        """Heap Sort with step tracking"""
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
        
        # Build max heap
        for i in range(n // 2 - 1, -1, -1):
            heapify(arr, n, i)
        
        # Extract elements one by one
        for i in range(n - 1, 0, -1):
            arr[0], arr[i] = arr[i], arr[0]
            steps.append((arr.copy(), [0, i], "extract_max"))
            heapify(arr, i, 0)
        
        steps.append((arr.copy(), [], "completed"))
        return steps
    
    @staticmethod
    def shell_sort(arr):
        """Shell Sort with step tracking"""
        arr = arr.copy()
        steps = []
        n = len(arr)
        gap = n // 2
        
        while gap > 0:
            steps.append((arr.copy(), [], f"gap_{gap}"))
            
            for i in range(gap, n):
                temp = arr[i]
                j = i
                
                while j >= gap and arr[j - gap] > temp:
                    steps.append((arr.copy(), [j, j - gap], "comparing"))
                    arr[j] = arr[j - gap]
                    steps.append((arr.copy(), [j], "shifted"))
                    j -= gap
                
                arr[j] = temp
                steps.append((arr.copy(), [j], "inserted"))
            
            gap //= 2
        
        steps.append((arr.copy(), [], "completed"))
        return steps

# Algorithm information data
PATHFINDING_INFO = {
    "A* (A-Star)": {
        "description": "Combines the benefits of Dijkstra's algorithm and Greedy Best-First Search. Uses both actual distance and heuristic.",
        "time_complexity": "O(b^d) where b is branching factor, d is depth",
        "space_complexity": "O(b^d)",
        "optimal": "Yes (with admissible heuristic)",
        "use_case": "Best for most pathfinding scenarios, especially when you have a good heuristic",
        "pros": ["Optimal path", "Efficient", "Widely applicable"],
        "cons": ["Requires good heuristic", "Can be memory intensive"]
    },
    "Dijkstra": {
        "description": "Finds shortest path by exploring nodes in order of their distance from start. Guarantees optimal solution.",
        "time_complexity": "O((V + E) log V) with binary heap",
        "space_complexity": "O(V)",
        "optimal": "Yes",
        "use_case": "When you need guaranteed shortest path and don't have a good heuristic",
        "pros": ["Always optimal", "No heuristic needed", "Well-established"],
        "cons": ["Can be slow", "Explores many unnecessary nodes"]
    },
    "BFS (Breadth-First Search)": {
        "description": "Explores all nodes at current depth before moving to next depth level. Guarantees shortest path in unweighted graphs.",
        "time_complexity": "O(V + E)",
        "space_complexity": "O(V)",
        "optimal": "Yes (for unweighted graphs)",
        "use_case": "Unweighted graphs, when all edges have same cost",
        "pros": ["Simple to implement", "Optimal for unweighted graphs", "Complete"],
        "cons": ["Not optimal for weighted graphs", "High memory usage"]
    },
    "DFS (Depth-First Search)": {
        "description": "Explores as far as possible along each branch before backtracking. Does not guarantee optimal path.",
        "time_complexity": "O(V + E)",
        "space_complexity": "O(h) where h is max depth",
        "optimal": "No",
        "use_case": "When you need to explore all possibilities or memory is limited",
        "pros": ["Low memory usage", "Simple to implement", "Good for maze solving"],
        "cons": ["Not optimal", "Can get stuck in infinite loops", "May not find shortest path"]
    },
    "Greedy Best-First": {
        "description": "Uses only heuristic to guide search towards goal. Fast but not guaranteed to find optimal path.",
        "time_complexity": "O(b^m) where m is max depth",
        "space_complexity": "O(b^m)",
        "optimal": "No",
        "use_case": "When speed is more important than optimality",
        "pros": ["Very fast", "Low memory usage", "Simple concept"],
        "cons": ["Not optimal", "Can get stuck", "Heavily dependent on heuristic quality"]
    },
    "Bidirectional Search": {
        "description": "Searches simultaneously from start and goal until they meet. Can be much faster than unidirectional search.",
        "time_complexity": "O(b^(d/2))",
        "space_complexity": "O(b^(d/2))",
        "optimal": "Yes (when both directions use optimal algorithms)",
        "use_case": "When start and goal are both known and search space is large",
        "pros": ["Much faster than unidirectional", "Reduces search space significantly"],
        "cons": ["More complex to implement", "Requires both start and goal", "Higher memory usage"]
    }
}

SORTING_INFO = {
    "Bubble Sort": {
        "description": "Repeatedly steps through list, compares adjacent elements and swaps them if they're in wrong order.",
        "best_case": "O(n)",
        "average_case": "O(n²)",
        "worst_case": "O(n²)",
        "space_complexity": "O(1)",
        "stable": "Yes",
        "use_case": "Educational purposes, very small datasets",
        "pros": ["Simple to understand", "In-place sorting", "Stable"],
        "cons": ["Very inefficient for large datasets", "Many comparisons and swaps"]
    },
    "Selection Sort": {
        "description": "Finds minimum element and places it at beginning, then finds second minimum, and so on.",
        "best_case": "O(n²)",
        "average_case": "O(n²)",
        "worst_case": "O(n²)",
        "space_complexity": "O(1)",
        "stable": "No",
        "use_case": "When memory writes are costly",
        "pros": ["Simple to understand", "In-place sorting", "Minimum number of swaps"],
        "cons": ["Inefficient for large datasets", "Not stable", "Always O(n²)"]
    },
    "Insertion Sort": {
        "description": "Builds final sorted array one item at a time. Efficient for small datasets and nearly sorted arrays.",
        "best_case": "O(n)",
        "average_case": "O(n²)",
        "worst_case": "O(n²)",
        "space_complexity": "O(1)",
        "stable": "Yes",
        "use_case": "Small datasets, nearly sorted arrays, online algorithms",
        "pros": ["Simple implementation", "Efficient for small data", "Stable", "In-place"],
        "cons": ["Inefficient for large datasets", "More writes than selection sort"]
    },
    "Quick Sort": {
        "description": "Divides array into partitions around a pivot element, then recursively sorts partitions.",
        "best_case": "O(n log n)",
        "average_case": "O(n log n)",
        "worst_case": "O(n²)",
        "space_complexity": "O(log n)",
        "stable": "No",
        "use_case": "General purpose sorting, when average performance matters",
        "pros": ["Fast average performance", "In-place sorting", "Cache efficient"],
        "cons": ["Worst case O(n²)", "Not stable", "Recursive overhead"]
    },
    "Merge Sort": {
        "description": "Divides array into halves, recursively sorts them, then merges sorted halves back together.",
        "best_case": "O(n log n)",
        "average_case": "O(n log n)",
        "worst_case": "O(n log n)",
        "space_complexity": "O(n)",
        "stable": "Yes",
        "use_case": "When stable sorting is needed, guaranteed O(n log n) performance",
        "pros": ["Guaranteed O(n log n)", "Stable", "Predictable performance"],
        "cons": ["Uses extra memory", "Slower than quicksort in practice", "Not in-place"]
    },
    "Heap Sort": {
        "description": "Uses binary heap data structure. Builds max heap, then repeatedly extracts maximum element.",
        "best_case": "O(n log n)",
        "average_case": "O(n log n)",
        "worst_case": "O(n log n)",
        "space_complexity": "O(1)",
        "stable": "No",
        "use_case": "When guaranteed O(n log n) and O(1) space is needed",
        "pros": ["Guaranteed O(n log n)", "In-place sorting", "No worst case degradation"],
        "cons": ["Not stable", "Slower than quicksort in practice", "Complex implementation"]
    },
    "Shell Sort": {
        "description": "Generalization of insertion sort. Sorts elements at specific intervals, gradually reducing the interval.",
        "best_case": "O(n log n)",
        "average_case": "Depends on gap sequence",
        "worst_case": "O(n²)",
        "space_complexity": "O(1)",
        "stable": "No",
        "use_case": "Medium-sized datasets, when simple implementation is preferred",
        "pros": ["Better than O(n²) algorithms", "In-place sorting", "Simple to implement"],
        "cons": ["Not stable", "Performance depends on gap sequence", "Complex analysis"]
    }
}

# Utility function to find nearest node
def find_nearest_node(graph, lat, lon):
    """Find the nearest node in the graph to given coordinates"""
    min_dist = float('inf')
    nearest_node = None
    
    for node, data in graph.nodes(data=True):
        node_lat, node_lon = data['y'], data['x']
        dist = math.sqrt((lat - node_lat)**2 + (lon - node_lon)**2)
        if dist < min_dist:
            min_dist = dist
            nearest_node = node
    
    return nearest_node

# Main app tabs
tab1, tab2 = st.tabs(["🗺️ PathFinding Visualizer", "📊 Sorting Visualizer"])

# Tab 1: Enhanced PathFinding Visualizer
with tab1:
    st.header("🗺️ Real-World Pathfinding with Interactive Location Selection")
    
    # Sidebar for pathfinding
    with st.sidebar:
        st.subheader("🎛️ Pathfinding Controls")
        
        # Location input
        location = st.text_input("📍 Enter Location", value="Times Square, New York")
        
        # Algorithm selection
        algorithm = st.selectbox(
            "🧠 Choose Algorithm",
            ["A* (A-Star)", "Dijkstra", "BFS (Breadth-First Search)", 
             "DFS (Depth-First Search)", "Greedy Best-First", "Bidirectional Search"]
        )
        
        # Distance parameter
        distance = st.slider("🌐 Network Distance (meters)", 500, 3000, 1000, 100)
        
        # Network type
        network_type = st.selectbox(
            "🚗 Network Type",
            ["drive", "walk", "bike"]
        )
        
        # Action buttons
        load_map = st.button("🗺️ Load Map", type="primary")
        clear_points = st.button("🧹 Clear Selected Points")
        find_path = st.button("🎯 Find Path", disabled=True)
        
        # Instructions
        st.info("🖱️ **Instructions:**\n1. Load a map\n2. Click on the map to select start and end points\n3. Click 'Find Path' to run the algorithm")
    
    # Main pathfinding area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Initialize session state
        if 'selected_points' not in st.session_state:
            st.session_state.selected_points = []
        
        if load_map or 'graph' not in st.session_state:
            try:
                with st.spinner(f"Loading map for {location}..."):
                    # Get graph from OpenStreetMap
                    G = ox.graph_from_address(location, dist=distance, network_type=network_type)
                    st.session_state.graph = G
                    st.session_state.location = location
                    st.session_state.selected_points = []  # Clear points when loading new map
                    
                    # Create folium map
                    center_lat = np.mean([data['y'] for node, data in G.nodes(data=True)])
                    center_lon = np.mean([data['x'] for node, data in G.nodes(data=True)])
                    
                    m = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=15,
                        tiles='OpenStreetMap'
                    )
                    
                    # Add graph to map
                    for edge in G.edges():
                        node1, node2 = edge[0], edge[1]
                        lat1, lon1 = G.nodes[node1]['y'], G.nodes[node1]['x']
                        lat2, lon2 = G.nodes[node2]['y'], G.nodes[node2]['x']
                        
                        folium.PolyLine(
                            locations=[(lat1, lon1), (lat2, lon2)],
                            color='blue',
                            weight=2,
                            opacity=0.6
                        ).add_to(m)
                    
                    st.session_state.map = m
                    st.session_state.nodes_list = list(G.nodes())
                    
                st.success(f"✅ Map loaded successfully! Graph has {len(G.nodes())} nodes and {len(G.edges())} edges.")
                
            except Exception as e:
                st.error(f"❌ Error loading map: {str(e)}")
                st.info("💡 Try a different location or check your internet connection.")
        
        # Clear selected points
        if clear_points:
            st.session_state.selected_points = []
            st.rerun()
        
        # Display map and handle clicks
        if 'map' in st.session_state:
            # Create a new map with selected points
            current_map = st.session_state.map._repr_html_()
            G = st.session_state.graph
            
            # Create new map for display
            center_lat = np.mean([data['y'] for node, data in G.nodes(data=True)])
            center_lon = np.mean([data['x'] for node, data in G.nodes(data=True)])
            
            display_map = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=15,
                tiles='OpenStreetMap'
            )
            
            # Add graph to display map
            for edge in G.edges():
                node1, node2 = edge[0], edge[1]
                lat1, lon1 = G.nodes[node1]['y'], G.nodes[node1]['x']
                lat2, lon2 = G.nodes[node2]['y'], G.nodes[node2]['x']
                
                folium.PolyLine(
                    locations=[(lat1, lon1), (lat2, lon2)],
                    color='lightblue',
                    weight=1,
                    opacity=0.4
                ).add_to(display_map)
            
            # Add selected points to map
            if len(st.session_state.selected_points) >= 1:
                start_point = st.session_state.selected_points[0]
                folium.Marker(
                    location=[start_point['lat'], start_point['lng']],
                    popup="Start Point",
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(display_map)
            
            if len(st.session_state.selected_points) >= 2:
                end_point = st.session_state.selected_points[1]
                folium.Marker(
                    location=[end_point['lat'], end_point['lng']],
                    popup="End Point",
                    icon=folium.Icon(color='red', icon='stop')
                ).add_to(display_map)
            
            # Display map and get click data
            map_data = st_folium(display_map, width=700, height=500, returned_objects=["last_clicked"])
            
            # Handle map clicks
            if map_data['last_clicked'] is not None:
                click_lat = map_data['last_clicked']['lat']
                click_lng = map_data['last_clicked']['lng']
                
                # Add point if we have less than 2 points
                if len(st.session_state.selected_points) < 2:
                    st.session_state.selected_points.append({'lat': click_lat, 'lng': click_lng})
                    st.rerun()
                elif len(st.session_state.selected_points) >= 2:
                    # Replace second point if clicking again
                    st.session_state.selected_points[1] = {'lat': click_lat, 'lng': click_lng}
                    st.rerun()
        
        # Enable find path button if we have 2 points
        if len(st.session_state.selected_points) >= 2:
            st.sidebar.button("🎯 Find Path", key="find_path_enabled", type="primary")
            
            if st.sidebar.button("🎯 Find Path", key="find_path_enabled") or find_path:
                try:
                    with st.spinner("Finding optimal path..."):
                        G = st.session_state.graph
                        
                        # Find nearest nodes to clicked points
                        start_point = st.session_state.selected_points[0]
                        end_point = st.session_state.selected_points[1]
                        
                        start_node = find_nearest_node(G, start_point['lat'], start_point['lng'])
                        end_node = find_nearest_node(G, end_point['lat'], end_point['lng'])
                        
                        if start_node is None or end_node is None:
                            st.error("❌ Could not find valid nodes near selected points")
                            continue
                        
                        # Run selected algorithm
                        start_time = time.time()
                        if algorithm == "A* (A-Star)":
                            path, visited = PathfindingAlgorithms.a_star(G, start_node, end_node)
                        elif algorithm == "Dijkstra":
                            path, visited = PathfindingAlgorithms.dijkstra(G, start_node, end_node)
                        elif algorithm == "BFS (Breadth-First Search)":
                            path, visited = PathfindingAlgorithms.bfs(G, start_node, end_node)
                        elif algorithm == "DFS (Depth-First Search)":
                            path, visited = PathfindingAlgorithms.dfs(G, start_node, end_node)
                        elif algorithm == "Greedy Best-First":
                            path, visited = PathfindingAlgorithms.greedy_best_first(G, start_node, end_node)
                        else:  # Bidirectional Search
                            path, visited = PathfindingAlgorithms.bidirectional_search(G, start_node, end_node)
                        
                        end_time = time.time()
                        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
                        
                        # Calculate path distance
                        path_distance = 0
                        if len(path) > 1:
                            for i in range(len(path) - 1):
                                if G.has_edge(path[i], path[i+1]):
                                    path_distance += G.edges[path[i], path[i+1], 0].get('length', 0)
                        
                        # Create result map
                        result_map = folium.Map(
                            location=[center_lat, center_lon],
                            zoom_start=15,
                            tiles='OpenStreetMap'
                        )
                        
                        # Add base graph (lighter)
                        for edge in G.edges():
                            node1, node2 = edge[0], edge[1]
                            lat1, lon1 = G.nodes[node1]['y'], G.nodes[node1]['x']
                            lat2, lon2 = G.nodes[node2]['y'], G.nodes[node2]['x']
                            
                            folium.PolyLine(
                                locations=[(lat1, lon1), (lat2, lon2)],
                                color='lightgray',
                                weight=1,
                                opacity=0.3
                            ).add_to(result_map)
                        
                        # Add visited nodes
                        for node in visited:
                            if node in G.nodes:
                                lat, lon = G.nodes[node]['y'], G.nodes[node]['x']
                                folium.CircleMarker(
                                    location=[lat, lon],
                                    radius=2,
                                    color='orange',
                                    fillColor='orange',
                                    fillOpacity=0.6,
                                    popup=f"Visited: {node}"
                                ).add_to(result_map)
                        
                        # Add path
                        if path and len(path) > 1:
                            path_coords = []
                            for node in path:
                                if node in G.nodes:
                                    lat, lon = G.nodes[node]['y'], G.nodes[node]['x']
                                    path_coords.append([lat, lon])
                            
                            if path_coords:
                                folium.PolyLine(
                                    locations=path_coords,
                                    color='red',
                                    weight=4,
                                    opacity=0.8,
                                    popup=f"Optimal Path ({len(path)} nodes)"
                                ).add_to(result_map)
                        
                        # Add start and end markers
                        folium.Marker(
                            location=[start_point['lat'], start_point['lng']],
                            popup="Start Point",
                            icon=folium.Icon(color='green', icon='play')
                        ).add_to(result_map)
                        
                        folium.Marker(
                            location=[end_point['lat'], end_point['lng']],
                            popup="End Point",
                            icon=folium.Icon(color='red', icon='stop')
                        ).add_to(result_map)
                        
                        # Store results
                        st.session_state.result_map = result_map
                        st.session_state.path_info = {
                            'algorithm': algorithm,
                            'path_length': len(path),
                            'visited_nodes': len(visited),
                            'path_found': len(path) > 0,
                            'execution_time': execution_time,
                            'path_distance': path_distance / 1000 if path_distance > 0 else 0  # Convert to km
                        }
                        
                        # Display result map
                        st_folium(result_map, width=700, height=500)
                        
                        if len(path) > 0:
                            st.success(f"✅ Path found! {len(path)} nodes, {path_distance/1000:.2f} km")
                        else:
                            st.error("❌ No path found between selected points")
                
                except Exception as e:
                    st.error(f"❌ Error finding path: {str(e)}")
    
    with col2:
        st.subheader("📊 Algorithm Performance")
        
        # Display selection status
        points_count = len(st.session_state.get('selected_points', []))
        if points_count == 0:
            st.info("🖱️ Click on map to select start point")
        elif points_count == 1:
            st.info("🖱️ Click on map to select end point")
        else:
            st.success("✅ Both points selected!")
        
        # Display results if available
        if 'path_info' in st.session_state:
            info = st.session_state.path_info
            st.metric("Algorithm", info['algorithm'])
            st.metric("Execution Time", f"{info['execution_time']:.2f} ms")
            st.metric("Path Length", f"{info['path_length']} nodes")
            st.metric("Path Distance", f"{info['path_distance']:.2f} km")
            st.metric("Visited Nodes", info['visited_nodes'])
            
            # Efficiency ratio
            if info['visited_nodes'] > 0:
                efficiency = (info['path_length'] / info['visited_nodes']) * 100
                st.metric("Efficiency", f"{efficiency:.1f}%")
        
        # Algorithm information
        st.subheader("🧠 Algorithm Details")
        
        if algorithm in PATHFINDING_INFO:
            algo_info = PATHFINDING_INFO[algorithm]
            
            with st.expander(f"ℹ️ {algorithm} Information"):
                st.markdown(f"**Description:** {algo_info['description']}")
                st.markdown(f"**Time Complexity:** {algo_info['time_complexity']}")
                st.markdown(f"**Space Complexity:** {algo_info['space_complexity']}")
                st.markdown(f"**Optimal:** {algo_info['optimal']}")
                st.markdown(f"**Best Use Case:** {algo_info['use_case']}")
                
                st.markdown("**Pros:**")
                for pro in algo_info['pros']:
                    st.markdown(f"• {pro}")
                
                st.markdown("**Cons:**")
                for con in algo_info['cons']:
                    st.markdown(f"• {con}")

# Tab 2: Enhanced Sorting Visualizer
with tab2:
    st.header("📊 Advanced Sorting Algorithm Visualizer")
    
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
             "Merge Sort", "Heap Sort", "Shell Sort"]
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
        
        algorithms = ["Bubble Sort", "Selection Sort", "Insertion Sort", "Quick Sort", "Merge Sort", "Heap Sort", "Shell Sort"]
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
                elif algo == "Heap Sort":
                    steps = SortingAlgorithms.heap_sort(arr)
                else:  # Shell Sort
                    steps = SortingAlgorithms.shell_sort(arr)
                
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
                offsetgroup=1
            ))
            
            if len(steps_list) == len(algorithms_list):
                fig.add_trace(go.Bar(
                    name='Steps',
                    x=algorithms_list,
                    y=steps_list,
                    yaxis='y2',
                    offsetgroup=2
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
            elif sort_algorithm == "Heap Sort":
                steps = SortingAlgorithms.heap_sort(arr)
            else:  # Shell Sort
                steps = SortingAlgorithms.shell_sort(arr)
            
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
    
    # Algorithm information section
    st.subheader("🧠 Algorithm Complexity & Details")
    
    # Create tabs for different algorithms
    algo_tabs = st.tabs(["Current Algorithm", "All Algorithms Comparison", "Complexity Analysis"])
    
    with algo_tabs[0]:
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
    
    with algo_tabs[1]:
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
    
    with algo_tabs[2]:
        st.markdown("""
        ### Understanding Algorithm Complexity
        
        **Time Complexity** measures how the running time increases with input size:
        - **O(1)**: Constant time - doesn't depend on input size
        - **O(n)**: Linear time - increases linearly with input size
        - **O(n log n)**: Linearithmic time - efficient for large datasets
        - **O(n²)**: Quadratic time - suitable only for small datasets
        
        **Space Complexity** measures extra memory needed:
        - **O(1)**: In-place algorithms (constant extra space)
        - **O(n)**: Linear extra space needed
        - **O(log n)**: Logarithmic space (usually for recursion)
        
        **Stability** means equal elements maintain their relative order after sorting.
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 Enhanced Pathfinding & Sorting Visualizer</p>
    <p>🗺️ Interactive location selection with 6 pathfinding algorithms</p>
    <p>📊 Advanced sorting visualization with 7 algorithms and detailed analysis</p>
    <p>Built with ❤️ using Streamlit, OSMnx, NetworkX, Folium & Plotly</p>
</div>
""", unsafe_allow_html=True)
