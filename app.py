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
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🗺️ PathFinder & Sort Visualizer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive pathfinding on real maps & animated sorting algorithms</p>', unsafe_allow_html=True)

# Pathfinding algorithms implementation
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
        """A* pathfinding algorithm"""
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
        """Dijkstra's algorithm"""
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
        """Breadth-First Search"""
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

# Sorting algorithms implementation
class SortingAlgorithms:
    @staticmethod
    def bubble_sort(arr):
        """Bubble Sort with step tracking"""
        arr = arr.copy()
        steps = []
        n = len(arr)
        
        for i in range(n):
            for j in range(0, n - i - 1):
                steps.append((arr.copy(), [j, j + 1], "comparing"))
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
                    steps.append((arr.copy(), [j, j + 1], "swapped"))
        
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
                    arr[i], arr[j] = arr[j], arr[i]
                    if i != j:
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

# Main app tabs
tab1, tab2 = st.tabs(["🗺️ PathFinding Visualizer", "📊 Sorting Visualizer"])

# Tab 1: PathFinding Visualizer
with tab1:
    st.header("🗺️ Real-World Pathfinding")
    
    # Sidebar for pathfinding
    with st.sidebar:
        st.subheader("🎛️ Pathfinding Controls")
        
        # Location input
        location = st.text_input("📍 Enter Location", value="Times Square, New York")
        
        # Algorithm selection
        algorithm = st.selectbox(
            "🧠 Choose Algorithm",
            ["A* (A-Star)", "Dijkstra", "BFS (Breadth-First Search)"]
        )
        
        # Distance parameter
        distance = st.slider("🌐 Network Distance (meters)", 500, 2000, 1000, 100)
        
        # Network type
        network_type = st.selectbox(
            "🚗 Network Type",
            ["drive", "walk", "bike"]
        )
        
        # Action buttons
        load_map = st.button("🗺️ Load Map", type="primary")
        find_path = st.button("🎯 Find Path")
    
    # Main pathfinding area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if load_map or 'graph' not in st.session_state:
            try:
                with st.spinner(f"Loading map for {location}..."):
                    # Get graph from OpenStreetMap
                    G = ox.graph_from_address(location, dist=distance, network_type=network_type)
                    st.session_state.graph = G
                    st.session_state.location = location
                    
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
        
        # Display map
        if 'map' in st.session_state:
            if find_path and len(st.session_state.nodes_list) >= 2:
                try:
                    with st.spinner("Finding optimal path..."):
                        # Select random start and end nodes
                        start_node = random.choice(st.session_state.nodes_list)
                        end_node = random.choice(st.session_state.nodes_list)
                        
                        # Ensure start and end are different
                        while end_node == start_node:
                            end_node = random.choice(st.session_state.nodes_list)
                        
                        # Run selected algorithm
                        if algorithm == "A* (A-Star)":
                            path, visited = PathfindingAlgorithms.a_star(st.session_state.graph, start_node, end_node)
                        elif algorithm == "Dijkstra":
                            path, visited = PathfindingAlgorithms.dijkstra(st.session_state.graph, start_node, end_node)
                        else:  # BFS
                            path, visited = PathfindingAlgorithms.bfs(st.session_state.graph, start_node, end_node)
                        
                        # Create new map with path
                        G = st.session_state.graph
                        center_lat = np.mean([data['y'] for node, data in G.nodes(data=True)])
                        center_lon = np.mean([data['x'] for node, data in G.nodes(data=True)])
                        
                        m = folium.Map(
                            location=[center_lat, center_lon],
                            zoom_start=15,
                            tiles='OpenStreetMap'
                        )
                        
                        # Add base graph
                        for edge in G.edges():
                            node1, node2 = edge[0], edge[1]
                            lat1, lon1 = G.nodes[node1]['y'], G.nodes[node1]['x']
                            lat2, lon2 = G.nodes[node2]['y'], G.nodes[node2]['x']
                            
                            folium.PolyLine(
                                locations=[(lat1, lon1), (lat2, lon2)],
                                color='lightblue',
                                weight=1,
                                opacity=0.4
                            ).add_to(m)
                        
                        # Add visited nodes
                        for node in visited:
                            lat, lon = G.nodes[node]['y'], G.nodes[node]['x']
                            folium.CircleMarker(
                                location=[lat, lon],
                                radius=3,
                                color='orange',
                                fillColor='orange',
                                fillOpacity=0.7
                            ).add_to(m)
                        
                        # Add path
                        if path:
                            path_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in path]
                            folium.PolyLine(
                                locations=path_coords,
                                color='red',
                                weight=4,
                                opacity=0.8
                            ).add_to(m)
                        
                        # Add start and end markers
                        start_lat, start_lon = G.nodes[start_node]['y'], G.nodes[start_node]['x']
                        end_lat, end_lon = G.nodes[end_node]['y'], G.nodes[end_node]['x']
                        
                        folium.Marker(
                            location=[start_lat, start_lon],
                            popup="Start",
                            icon=folium.Icon(color='green', icon='play')
                        ).add_to(m)
                        
                        folium.Marker(
                            location=[end_lat, end_lon],
                            popup="Goal",
                            icon=folium.Icon(color='red', icon='stop')
                        ).add_to(m)
                        
                        st.session_state.path_map = m
                        st.session_state.path_info = {
                            'algorithm': algorithm,
                            'path_length': len(path),
                            'visited_nodes': len(visited),
                            'path_found': len(path) > 0
                        }
                
                except Exception as e:
                    st.error(f"❌ Error finding path: {str(e)}")
            
            # Display the map
            map_to_show = st.session_state.get('path_map', st.session_state.map)
            st_folium(map_to_show, width=700, height=500)
    
    with col2:
        st.subheader("📊 Algorithm Stats")
        
        if 'path_info' in st.session_state:
            info = st.session_state.path_info
            st.metric("Algorithm Used", info['algorithm'])
            st.metric("Path Length", f"{info['path_length']} nodes")
            st.metric("Visited Nodes", info['visited_nodes'])
            
            if info['path_found']:
                st.success("✅ Path Found!")
            else:
                st.error("❌ No Path Found")
        else:
            st.info("👆 Load a map and find a path to see statistics")
        
        # Algorithm descriptions
        st.subheader("🧠 Algorithm Info")
        
        if algorithm == "A* (A-Star)":
            st.markdown("""
            **A* Algorithm:**
            - Uses heuristic function
            - Guaranteed optimal path
            - Efficient for pathfinding
            - Best for most scenarios
            """)
        elif algorithm == "Dijkstra":
            st.markdown("""
            **Dijkstra's Algorithm:**
            - No heuristic function
            - Guaranteed optimal path
            - Explores more nodes
            - Good for weighted graphs
            """)
        else:
            st.markdown("""
            **BFS Algorithm:**
            - Simple breadth-first search
            - Optimal for unweighted graphs
            - Explores level by level
            - Easy to understand
            """)

# Tab 2: Sorting Visualizer
with tab2:
    st.header("📊 Sorting Algorithm Visualizer")
    
    # Sidebar for sorting
    with st.sidebar:
        st.subheader("🎛️ Sorting Controls")
        
        # Array configuration
        array_size = st.slider("📏 Array Size", 10, 100, 30)
        array_type = st.selectbox(
            "📊 Array Type",
            ["Random", "Nearly Sorted", "Reverse Sorted", "Few Unique"]
        )
        
        # Algorithm selection
        sort_algorithm = st.selectbox(
            "🔄 Sorting Algorithm",
            ["Bubble Sort", "Quick Sort", "Merge Sort", "Insertion Sort"]
        )
        
        # Animation speed
        animation_speed = st.slider("⚡ Animation Speed", 0.01, 1.0, 0.1, 0.01)
        
        # Generate array button
        generate_array = st.button("🎲 Generate New Array", type="primary")
        
        # Start sorting button
        start_sorting = st.button("▶️ Start Sorting")
    
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
        else:  # Few Unique
            unique_values = [random.randint(1, 20) for _ in range(5)]
            arr = [random.choice(unique_values) for _ in range(array_size)]
        
        st.session_state.sorting_array = arr
        st.session_state.original_array = arr.copy()
    
    # Display current array
    if 'sorting_array' in st.session_state:
        st.subheader("Current Array")
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(range(len(st.session_state.sorting_array))),
                y=st.session_state.sorting_array,
                marker_color='lightblue',
                text=st.session_state.sorting_array,
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=f"Array of size {len(st.session_state.sorting_array)} ({array_type})",
            xaxis_title="Index",
            yaxis_title="Value",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Start sorting animation
    if start_sorting and 'sorting_array' in st.session_state:
        arr = st.session_state.original_array.copy()
        
        # Get sorting steps
        with st.spinner(f"Running {sort_algorithm}..."):
            if sort_algorithm == "Bubble Sort":
                steps = SortingAlgorithms.bubble_sort(arr)
            elif sort_algorithm == "Quick Sort":
                steps = SortingAlgorithms.quick_sort(arr)
            elif sort_algorithm == "Merge Sort":
                steps = SortingAlgorithms.merge_sort(arr)
            else:  # Insertion Sort
                steps = SortingAlgorithms.insertion_sort(arr)
        
        # Create placeholders for animation
        progress_bar = st.progress(0)
        chart_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # Animate sorting
        for i, (current_array, highlighted, action) in enumerate(steps):
            # Update progress
            progress = (i + 1) / len(steps)
            progress_bar.progress(progress)
            
            # Create colors for bars
            colors = ['lightblue'] * len(current_array)
            for idx in highlighted:
                if idx < len(colors):
                    if action == "comparing":
                        colors[idx] = 'yellow'
                    elif action == "swapped":
                        colors[idx] = 'red'
                    elif action == "pivot":
                        colors[idx] = 'purple'
                    elif action == "merged":
                        colors[idx] = 'green'
                    elif action == "current":
                        colors[idx] = 'orange'
                    elif action == "inserted":
                        colors[idx] = 'green'
            
            # Create animated bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(range(len(current_array))),
                    y=current_array,
                    marker_color=colors,
                    text=current_array,
                    textposition='outside'
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
            
            # Update status
            if action == "completed":
                status_placeholder.success("✅ Sorting completed!")
            else:
                status_placeholder.info(f"Status: {action.replace('_', ' ').title()}")
            
            # Animation delay
            time.sleep(animation_speed)
        
        # Final success message
        st.balloons()
        st.success(f"🎉 {sort_algorithm} completed in {len(steps)} steps!")
    
    # Algorithm complexity information
    st.subheader("🧠 Algorithm Complexity")
    
    complexity_data = {
        "Bubble Sort": {"Best": "O(n)", "Average": "O(n²)", "Worst": "O(n²)", "Space": "O(1)"},
        "Quick Sort": {"Best": "O(n log n)", "Average": "O(n log n)", "Worst": "O(n²)", "Space": "O(log n)"},
        "Merge Sort": {"Best": "O(n log n)", "Average": "O(n log n)", "Worst": "O(n log n)", "Space": "O(n)"},
        "Insertion Sort": {"Best": "O(n)", "Average": "O(n²)", "Worst": "O(n²)", "Space": "O(1)"}
    }
    
    df = pd.DataFrame(complexity_data).T
    st.dataframe(df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 Built with Streamlit | 🗺️ Powered by OpenStreetMap & OSMnx | 📊 Visualized with Plotly</p>
    <p>Made with ❤️ for learning algorithms and data structures</p>
</div>
""", unsafe_allow_html=True)
