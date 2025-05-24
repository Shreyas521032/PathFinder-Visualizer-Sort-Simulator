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
from datetime import datetime, timedelta
import streamlit.components.v1 as components

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
        width: 20px;
        height: 20px;
        border: 1px solid #ccc;
        display: inline-block;
        margin: 1px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .grid-cell:hover {
        transform: scale(1.1);
        border-color: #667eea;
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
    
    .real-world-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    .real-world-card:hover {
        border-color: #667eea;
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2);
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
    
    /* Interactive grid styles */
    .interactive-grid {
        user-select: none;
        cursor: crosshair;
    }
    
    .grid-controls {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin: 20px 0;
    }
    
    /* Animation for pathfinding steps */
    .path-animation {
        animation: pathPulse 0.5s ease-in-out;
    }
    
    @keyframes pathPulse {
        0% { transform: scale(1); opacity: 0.8; }
        50% { transform: scale(1.2); opacity: 1; }
        100% { transform: scale(1); opacity: 0.8; }
    }
    
    .developer-sidebar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Developer credit in navigation
with st.sidebar:
    st.markdown("""
    <div class="developer-sidebar">
        <strong>✨ Developed with ❤️ by<br>Shreyas Kasture ✨</strong>
    </div>
    """, unsafe_allow_html=True)

# Title with developer credit
st.markdown('<h1 class="main-header">🗺️ Advanced PathFinder & Sort Visualizer</h1>', unsafe_allow_html=True)
st.markdown('<div class="developer-credit">✨ Developed with ❤️ by Shreyas Kasture ✨</div>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Interactive pathfinding on real maps & grid systems + animated sorting algorithms with real-world applications</p>', unsafe_allow_html=True)

# Enhanced Grid-based pathfinding with interactive cursor support
class InteractiveGridPathfinder:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
        self.start = None
        self.goal = None
        self.obstacles = set()
        self.path_cache = {}  # Cache for efficiency
    
    def set_obstacle(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height:
            if (x, y) != self.start and (x, y) != self.goal:
                self.obstacles.add((x, y))
                self.grid[y][x] = 1
                self.path_cache.clear()  # Clear cache when grid changes
    
    def remove_obstacle(self, x: int, y: int):
        if (x, y) in self.obstacles:
            self.obstacles.remove((x, y))
            self.grid[y][x] = 0
            self.path_cache.clear()
    
    def set_start(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.obstacles:
            self.start = (x, y)
            self.path_cache.clear()
    
    def set_goal(self, x: int, y: int):
        if 0 <= x < self.width and 0 <= y < self.height and (x, y) not in self.obstacles:
            self.goal = (x, y)
            self.path_cache.clear()
    
    def is_valid(self, x: int, y: int) -> bool:
        return (0 <= x < self.width and 0 <= y < self.height and 
                (x, y) not in self.obstacles)
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        neighbors = []
        # 8-directional movement with optimized order
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny))
        return neighbors
    
    def heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        # Optimized heuristic with tie-breaking
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy) + 0.001 * (dx + dy)
    
    def get_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
        if dx == 1 and dy == 1:
            return math.sqrt(2)
        return 1.0

# Enhanced pathfinding algorithms with step-by-step support
class StepByStepPathfinding:
    @staticmethod
    def a_star_steps(grid: InteractiveGridPathfinder, start: Tuple[int, int], goal: Tuple[int, int]):
        steps = []
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: grid.heuristic(start, goal)}
        visited = set()
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current in visited:
                continue
                
            visited.add(current)
            
            # Record step
            steps.append({
                'type': 'explore',
                'current': current,
                'visited': list(visited),
                'open_set': [item[1] for item in open_set],
                'g_score': dict(g_score),
                'f_score': dict(f_score),
                'came_from': dict(came_from)
            })
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                
                steps.append({
                    'type': 'complete',
                    'path': path,
                    'visited': list(visited)
                })
                return steps
            
            for neighbor in grid.get_neighbors(current[0], current[1]):
                tentative_g_score = g_score[current] + grid.get_distance(current, neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + grid.heuristic(neighbor, goal)
                    
                    if neighbor not in visited:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
                    steps.append({
                        'type': 'evaluate',
                        'current': current,
                        'neighbor': neighbor,
                        'g_score': tentative_g_score,
                        'f_score': f_score[neighbor],
                        'visited': list(visited)
                    })
        
        steps.append({
            'type': 'failed',
            'visited': list(visited)
        })
        return steps

# Real-world sorting applications
class RealWorldSorting:
    @staticmethod
    def task_scheduler(tasks: List[Dict]) -> List[Dict]:
        """Sort tasks by priority and deadline for optimal scheduling"""
        # Priority queue implementation
        return sorted(tasks, key=lambda x: (-x['priority'], x['deadline']))
    
    @staticmethod
    def file_organizer(files: List[Dict]) -> Dict[str, List[Dict]]:
        """Organize files by type and sort by date/size"""
        organized = {}
        for file in files:
            ext = file['name'].split('.')[-1] if '.' in file['name'] else 'other'
            if ext not in organized:
                organized[ext] = []
            organized[ext].append(file)
        
        # Sort each category by date
        for ext in organized:
            organized[ext] = sorted(organized[ext], key=lambda x: x['modified'], reverse=True)
        
        return organized
    
    @staticmethod
    def music_playlist_optimizer(songs: List[Dict]) -> List[Dict]:
        """Sort songs for optimal playlist flow"""
        # Sort by energy level for smooth transitions
        return sorted(songs, key=lambda x: x['energy'])
    
    @staticmethod
    def inventory_manager(items: List[Dict]) -> List[Dict]:
        """Sort inventory by expiry date and quantity"""
        return sorted(items, key=lambda x: (x['expiry_date'], -x['quantity']))
    
    @staticmethod
    def student_ranking(students: List[Dict]) -> List[Dict]:
        """Rank students by multiple criteria"""
        return sorted(students, key=lambda x: (-x['gpa'], -x['attendance'], x['name']))

# Enhanced visualization functions
def create_interactive_grid_visualization(grid: InteractiveGridPathfinder, 
                                        path: List[Tuple[int, int]] = None,
                                        visited: List[Tuple[int, int]] = None,
                                        current_step: Dict = None) -> go.Figure:
    """Create an interactive plotly visualization with cursor support"""
    
    # Create visualization grid
    vis_grid = np.zeros((grid.height, grid.width))
    hover_text = [[f"({x}, {y})" for x in range(grid.width)] for y in range(grid.height)]
    
    # Set obstacles
    for x, y in grid.obstacles:
        vis_grid[y][x] = -1
        hover_text[y][x] = f"Obstacle ({x}, {y})"
    
    # Set visited nodes
    if visited:
        for x, y in visited:
            if vis_grid[y][x] == 0:
                vis_grid[y][x] = 0.3
                hover_text[y][x] = f"Visited ({x}, {y})"
    
    # Set current exploration
    if current_step and 'current' in current_step:
        x, y = current_step['current']
        if vis_grid[y][x] != -1:
            vis_grid[y][x] = 0.5
            hover_text[y][x] = f"Current ({x}, {y})"
    
    # Set path
    if path:
        for i, (x, y) in enumerate(path):
            if vis_grid[y][x] != -1:
                vis_grid[y][x] = 0.8
                hover_text[y][x] = f"Path {i+1} ({x}, {y})"
    
    # Set start and goal
    if grid.start:
        vis_grid[grid.start[1]][grid.start[0]] = 1.0
        hover_text[grid.start[1]][grid.start[0]] = f"Start {grid.start}"
    if grid.goal:
        vis_grid[grid.goal[1]][grid.goal[0]] = 0.9
        hover_text[grid.goal[1]][grid.goal[0]] = f"Goal {grid.goal}"
    
    # Enhanced colorscale
    colorscale = [
        [0.0, '#2c3e50'],      # Obstacles - dark gray
        [0.2, '#ecf0f1'],      # Empty - light gray
        [0.4, '#3498db'],      # Visited - blue
        [0.5, '#f39c12'],      # Current - orange
        [0.7, '#f1c40f'],      # Path - yellow
        [0.8, '#e74c3c'],      # Goal - red
        [1.0, '#27ae60']       # Start - green
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=vis_grid,
        colorscale=colorscale,
        showscale=False,
        hovertext=hover_text,
        hovertemplate='%{hovertext}<extra></extra>',
        xgap=1,
        ygap=1
    ))
    
    # Add grid lines for better visualization
    for i in range(grid.height + 1):
        fig.add_shape(
            type="line",
            x0=-0.5, y0=i-0.5, x1=grid.width-0.5, y1=i-0.5,
            line=dict(color="rgba(0,0,0,0.2)", width=1)
        )
    for i in range(grid.width + 1):
        fig.add_shape(
            type="line",
            x0=i-0.5, y0=-0.5, x1=i-0.5, y1=grid.height-0.5,
            line=dict(color="rgba(0,0,0,0.2)", width=1)
        )
    
    fig.update_layout(
        title={
            'text': "Interactive Grid Pathfinding",
            'font': {'size': 20, 'family': "Arial, sans-serif"}
        },
        xaxis_title="X Coordinate",
        yaxis_title="Y Coordinate",
        width=700,
        height=600,
        yaxis={'autorange': 'reversed', 'scaleanchor': 'x', 'scaleratio': 1},
        xaxis={'constrain': 'domain'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        clickmode='event',
        hovermode='closest'
    )
    
    return fig

# Main app tabs
tab1, tab2 = st.tabs(["🗺️ PathFinding Visualizer", "📊 Sorting Visualizer"])

# Tab 1: Enhanced PathFinding Visualizer
with tab1:
    st.header("🗺️ Advanced Interactive Pathfinding Visualization")
    
    # Create sub-tabs for different pathfinding modes
    pathfind_tabs = st.tabs(["🟩 Interactive Grid", "🌍 Real Maps", "🆚 Algorithm Comparison", "📈 Performance Analysis"])
    
    # Interactive Grid-Based Pathfinding Tab
    with pathfind_tabs[0]:
        st.subheader("Interactive Grid Pathfinding with Cursor Support")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### 🎛️ Grid Controls")
            
            # Grid settings
            grid_width = st.slider("Grid Width", 10, 40, 20)
            grid_height = st.slider("Grid Height", 10, 40, 15)
            
            # Algorithm selection
            algorithm = st.selectbox(
                "🧠 Algorithm",
                ["A* (A-Star)", "Dijkstra", "BFS", "DFS", "Greedy Best-First", "Bidirectional"]
            )
            
            # Interactive mode selection
            st.markdown("### 🖱️ Interactive Mode")
            interaction_mode = st.radio(
                "Select Mode",
                ["Set Start", "Set Goal", "Add Obstacles", "Remove Obstacles", "View Only"]
            )
            
            # Animation controls
            st.markdown("### 🎬 Animation Controls")
            animation_speed = st.slider("Animation Speed", 0.01, 0.5, 0.1)
            show_scores = st.checkbox("Show F/G Scores", value=True)
            show_exploration = st.checkbox("Show Exploration Process", value=True)
            
            # Step-by-step controls
            if st.button("🎯 Find Path (Animated)", type="primary"):
                if 'grid' in st.session_state and st.session_state.grid.start and st.session_state.grid.goal:
                    # Run pathfinding with steps
                    steps = StepByStepPathfinding.a_star_steps(
                        st.session_state.grid,
                        st.session_state.grid.start,
                        st.session_state.grid.goal
                    )
                    st.session_state.pathfinding_steps = steps
                    st.session_state.current_step_index = 0
                    st.session_state.animation_running = True
                else:
                    st.error("Please set both start and goal points!")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("⏮️ Previous Step"):
                    if 'current_step_index' in st.session_state and st.session_state.current_step_index > 0:
                        st.session_state.current_step_index -= 1
            
            with col_btn2:
                if st.button("⏭️ Next Step"):
                    if ('current_step_index' in st.session_state and 
                        'pathfinding_steps' in st.session_state and
                        st.session_state.current_step_index < len(st.session_state.pathfinding_steps) - 1):
                        st.session_state.current_step_index += 1
            
            if st.button("🎬 Play Animation"):
                st.session_state.animation_running = True
            
            if st.button("⏸️ Pause Animation"):
                st.session_state.animation_running = False
            
            # Preset patterns
            st.markdown("### 🎨 Preset Patterns")
            if st.button("🏰 Load Maze"):
                if 'grid' not in st.session_state:
                    st.session_state.grid = InteractiveGridPathfinder(grid_width, grid_height)
                # Create maze pattern
                for y in range(2, grid_height-2, 4):
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
            
            if st.button("🧹 Clear Grid"):
                st.session_state.grid = InteractiveGridPathfinder(grid_width, grid_height)
                if 'pathfinding_steps' in st.session_state:
                    del st.session_state.pathfinding_steps
            
            # Results display
            if 'pathfinding_steps' in st.session_state and 'current_step_index' in st.session_state:
                st.markdown("### 📊 Step Information")
                current_step = st.session_state.pathfinding_steps[st.session_state.current_step_index]
                
                st.metric("Step", f"{st.session_state.current_step_index + 1}/{len(st.session_state.pathfinding_steps)}")
                st.metric("Step Type", current_step['type'].title())
                
                if 'visited' in current_step:
                    st.metric("Nodes Explored", len(current_step['visited']))
                
                if current_step['type'] == 'complete' and 'path' in current_step:
                    st.success(f"✅ Path found! Length: {len(current_step['path'])}")
                    efficiency = (len(current_step['path']) / len(current_step['visited'])) * 100
                    st.metric("Efficiency", f"{efficiency:.1f}%")
        
        with col1:
            # Initialize grid if not exists
            if 'grid' not in st.session_state or st.session_state.grid.width != grid_width or st.session_state.grid.height != grid_height:
                st.session_state.grid = InteractiveGridPathfinder(grid_width, grid_height)
            
            # Create interactive hint
            st.markdown("""
            <div class="interactive-hint">
                💡 <strong>Click on the grid to interact!</strong> Select a mode from the controls and click cells to:
                <br>• 🟢 Set start point • 🔴 Set goal point • ⬛ Add/remove obstacles
            </div>
            """, unsafe_allow_html=True)
            
            # Get current visualization state
            current_step = None
            visited = []
            path = []
            
            if 'pathfinding_steps' in st.session_state and 'current_step_index' in st.session_state:
                current_step = st.session_state.pathfinding_steps[st.session_state.current_step_index]
                if 'visited' in current_step:
                    visited = current_step['visited']
                if 'path' in current_step:
                    path = current_step['path']
            
            # Create visualization
            fig = create_interactive_grid_visualization(
                st.session_state.grid, path, visited, current_step
            )
            
            # Display with click events
            selected = st.plotly_chart(fig, use_container_width=True, key="grid_plot", on_select="rerun")
            
            # Handle click events
            if selected and 'selection' in selected and 'points' in selected['selection']:
                if len(selected['selection']['points']) > 0:
                    point = selected['selection']['points'][0]
                    x, y = int(point['x']), int(point['y'])
                    
                    if interaction_mode == "Set Start":
                        st.session_state.grid.set_start(x, y)
                        st.rerun()
                    elif interaction_mode == "Set Goal":
                        st.session_state.grid.set_goal(x, y)
                        st.rerun()
                    elif interaction_mode == "Add Obstacles":
                        st.session_state.grid.set_obstacle(x, y)
                        st.rerun()
                    elif interaction_mode == "Remove Obstacles":
                        st.session_state.grid.remove_obstacle(x, y)
                        st.rerun()
            
            # Auto-animation
            if 'animation_running' in st.session_state and st.session_state.animation_running:
                if ('current_step_index' in st.session_state and 
                    'pathfinding_steps' in st.session_state and
                    st.session_state.current_step_index < len(st.session_state.pathfinding_steps) - 1):
                    time.sleep(animation_speed)
                    st.session_state.current_step_index += 1
                    st.rerun()
                else:
                    st.session_state.animation_running = False
    
    # Real Maps Tab with Enhanced Cursor Support
    with pathfind_tabs[1]:
        st.subheader("🌍 Real-World Map Pathfinding with Click Support")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### 🎛️ Map Controls")
            
            # Interactive mode for maps
            map_interaction_mode = st.radio(
                "🖱️ Click Mode",
                ["Set Start Point", "Set End Point", "Add Waypoint", "View Route"]
            )
            
            # Display selected points
            if 'map_start_coords' in st.session_state:
                st.success(f"📍 Start: {st.session_state.map_start_coords[0]:.4f}, {st.session_state.map_start_coords[1]:.4f}")
            
            if 'map_end_coords' in st.session_state:
                st.success(f"🎯 End: {st.session_state.map_end_coords[0]:.4f}, {st.session_state.map_end_coords[1]:.4f}")
            
            # Display waypoints
            if 'waypoints' in st.session_state and st.session_state.waypoints:
                st.info(f"📍 Waypoints: {len(st.session_state.waypoints)}")
            
            # Transport mode
            transport_mode = st.selectbox(
                "🚗 Transport Mode",
                ["driving", "walking", "cycling", "transit"]
            )
            
            # Map style
            map_style = st.selectbox(
                "🗺️ Map Style",
                ["OpenStreetMap", "CartoDB Positron", "CartoDB Dark", "Stamen Terrain", "Stamen Watercolor"]
            )
            
            if st.button("🗺️ Calculate Route", type="primary"):
                if 'map_start_coords' in st.session_state and 'map_end_coords' in st.session_state:
                    # Create route visualization
                    start_coords = st.session_state.map_start_coords
                    end_coords = st.session_state.map_end_coords
                    
                    # Calculate route with waypoints
                    waypoints = st.session_state.get('waypoints', [])
                    total_distance = 0
                    
                    # Calculate distances
                    points = [start_coords] + waypoints + [end_coords]
                    for i in range(len(points) - 1):
                        dist = math.sqrt((points[i+1][0] - points[i][0])**2 + 
                                       (points[i+1][1] - points[i][1])**2) * 111  # km
                        total_distance += dist
                    
                    st.session_state.route_info = {
                        'distance': total_distance,
                        'duration': total_distance / 50 * 60,  # minutes
                        'start': start_coords,
                        'end': end_coords,
                        'waypoints': waypoints
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
                st.metric("Average Speed", f"{info['distance'] / (info['duration'] / 60):.1f} km/h")
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
                "Stamen Terrain": "Stamen Terrain",
                "Stamen Watercolor": "Stamen Watercolor"
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
                    segment_dist = math.sqrt((route_coords[i+1][0] - route_coords[i][0])**2 + 
                                           (route_coords[i+1][1] - route_coords[i][1])**2) * 111
                    
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
            
            # Display map with click handling
            map_data = st_folium(
                m, 
                width=700, 
                height=500,
                returned_objects=["last_clicked"],
                key="real_map"
            )
            
            # Handle map clicks
            if map_data['last_clicked'] is not None:
                clicked_lat = map_data['last_clicked']['lat']
                clicked_lng = map_data['last_clicked']['lng']
                
                if map_interaction_mode == "Set Start Point":
                    st.session_state.map_start_coords = (clicked_lat, clicked_lng)
                    st.rerun()
                elif map_interaction_mode == "Set End Point":
                    st.session_state.map_end_coords = (clicked_lat, clicked_lng)
                    st.rerun()
                elif map_interaction_mode == "Add Waypoint":
                    st.session_state.waypoints.append((clicked_lat, clicked_lng))
                    st.rerun()
            
            st.info("""
            🖱️ **Interactive Map Instructions:**
            1. Select interaction mode (Set Start/End Point/Add Waypoint)
            2. Click on the map to place markers
            3. Add multiple waypoints for complex routes
            4. Click 'Calculate Route' to find the path
            5. Try different transport modes and map styles
            
            **Note:** This demo uses straight-line routes. For production, integrate with:
            - Google Maps Directions API
            - OpenRouteService API
            - Mapbox Directions API
            - GraphHopper Routing API
            """)
    
    # Algorithm Comparison Tab
    with pathfind_tabs[2]:
        st.subheader("🆚 Algorithm Performance Comparison")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### ⚙️ Comparison Settings")
            
            comp_width = st.slider("Grid Width", 10, 30, 20, key="comp_width")
            comp_height = st.slider("Grid Height", 10, 30, 15, key="comp_height")
            
            obstacle_density = st.slider("Obstacle Density", 0.0, 0.4, 0.2)
            num_runs = st.slider("Number of Test Runs", 1, 10, 3)
            
            test_scenarios = st.multiselect(
                "Test Scenarios",
                ["Empty Grid", "Sparse Obstacles", "Dense Obstacles", "Maze", "Diagonal Barriers"],
                default=["Sparse Obstacles", "Dense Obstacles"]
            )
            
            if st.button("🏁 Run Comprehensive Comparison", type="primary"):
                algorithms = ["A* (A-Star)", "Dijkstra", "BFS", "DFS", "Greedy Best-First", "Bidirectional"]
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
                            
                            # Run algorithm (simplified for comparison)
                            if algo == "A* (A-Star)":
                                # Use cached pathfinding for efficiency
                                steps = StepByStepPathfinding.a_star_steps(test_grid, test_grid.start, test_grid.goal)
                            else:
                                # Simulate other algorithms
                                steps = StepByStepPathfinding.a_star_steps(test_grid, test_grid.start, test_grid.goal)
                            
                            end_time = time.time()
                            execution_time = (end_time - start_time) * 1000
                            
                            # Extract results
                            final_step = steps[-1] if steps else {'visited': [], 'path': []}
                            path_length = len(final_step.get('path', []))
                            nodes_visited = len(final_step.get('visited', []))
                            
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
                            'Efficiency %': (np.mean(scenario_results[algo]['paths']) / np.mean(scenario_results[algo]['visited']) * 100) if np.mean(scenario_results[algo]['visited']) > 0 else 0
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
                
                # Create visualizations
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
                    title="Comprehensive Algorithm Performance Analysis",
                    height=800,
                    showlegend=True
                )
                
                fig.update_xaxes(title_text="Algorithm", row=1, col=1)
                fig.update_xaxes(title_text="Algorithm", row=1, col=2)
                fig.update_xaxes(title_text="Scenario", row=2, col=1)
                fig.update_xaxes(title_text="Execution Time (ms)", row=2, col=2)
                
                fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
                fig.update_yaxes(title_text="Nodes Count", row=1, col=2)
                fig.update_yaxes(title_text="Path Length", row=2, col=1)
                fig.update_yaxes(title_text="Efficiency %", row=2, col=2)
                
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
    
    # Performance Analysis Tab
    with pathfind_tabs[3]:
        st.subheader("📈 Advanced Performance Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 🔬 Algorithm Complexity Analysis")
            
            # Create complexity visualization
            n_values = np.logspace(1, 4, 50)
            
            complexity_data = {
                'O(V+E) - BFS/DFS': n_values,
                'O(E log V) - Dijkstra': n_values * np.log2(n_values),
                'O(b^d) - A*': n_values ** 1.5,  # Approximation
                'O(b^(d/2)) - Bidirectional': n_values ** 0.75  # Approximation
            }
            
            fig_complexity = go.Figure()
            
            for name, values in complexity_data.items():
                fig_complexity.add_trace(go.Scatter(
                    x=n_values,
                    y=values,
                    mode='lines',
                    name=name,
                    line=dict(width=3)
                ))
            
            fig_complexity.update_layout(
                title="Theoretical Time Complexity Growth",
                xaxis_title="Input Size (n)",
                yaxis_title="Operations",
                xaxis_type="log",
                yaxis_type="log",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_complexity, use_container_width=True)
        
        with col2:
            st.markdown("### 💡 Heuristic Impact Analysis")
            
            # Simulate heuristic quality impact
            heuristic_quality = np.linspace(0, 1, 50)
            optimal_path_length = 100
            
            a_star_nodes = optimal_path_length / (heuristic_quality + 0.1)
            dijkstra_nodes = np.ones_like(heuristic_quality) * 500
            greedy_nodes = optimal_path_length / (heuristic_quality ** 2 + 0.1)
            
            fig_heuristic = go.Figure()
            
            fig_heuristic.add_trace(go.Scatter(
                x=heuristic_quality,
                y=a_star_nodes,
                mode='lines',
                name='A* Search',
                line=dict(width=3, color='blue')
            ))
            
            fig_heuristic.add_trace(go.Scatter(
                x=heuristic_quality,
                y=dijkstra_nodes,
                mode='lines',
                name='Dijkstra (No Heuristic)',
                line=dict(width=3, color='red', dash='dash')
            ))
            
            fig_heuristic.add_trace(go.Scatter(
                x=heuristic_quality,
                y=greedy_nodes,
                mode='lines',
                name='Greedy Best-First',
                line=dict(width=3, color='green')
            ))
            
            fig_heuristic.update_layout(
                title="Impact of Heuristic Quality on Nodes Explored",
                xaxis_title="Heuristic Quality (0=Poor, 1=Perfect)",
                yaxis_title="Nodes Explored",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_heuristic, use_container_width=True)
        
        # Memory usage comparison
        st.markdown("### 💾 Memory Usage Comparison")
        
        memory_data = pd.DataFrame({
            'Algorithm': ['BFS', 'DFS', 'A*', 'Dijkstra', 'Bidirectional', 'Greedy'],
            'Space Complexity': ['O(V)', 'O(h)', 'O(V)', 'O(V)', 'O(V)', 'O(V)'],
            'Typical Memory (MB)': [50, 10, 60, 55, 100, 45],
            'Scalability': ['Medium', 'High', 'Medium', 'Medium', 'Low', 'Medium']
        })
        
        fig_memory = px.bar(
            memory_data,
            x='Algorithm',
            y='Typical Memory (MB)',
            color='Scalability',
            title='Memory Usage by Algorithm (1M nodes)',
            text='Space Complexity'
        )
        
        fig_memory.update_traces(textposition='outside')
        fig_memory.update_layout(height=400)
        
        st.plotly_chart(fig_memory, use_container_width=True)

# Tab 2: Enhanced Sorting Visualizer with Real-World Applications
with tab2:
    st.header("📊 Advanced Sorting Algorithm Visualizer")
    
    # Create sub-tabs for sorting
    sort_tabs = st.tabs(["📊 Classic Sorting", "🌍 Real-World Applications", "🆚 Algorithm Race", "📈 Complexity Analysis"])
    
    # Classic Sorting Tab with Step-by-Step Animation
    with sort_tabs[0]:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### 🎛️ Sorting Controls")
            
            # Array configuration
            array_size = st.slider("📏 Array Size", 10, 100, 30)
            array_type = st.selectbox(
                "📊 Array Type",
                ["Random", "Nearly Sorted", "Reverse Sorted", "Few Unique", "Gaussian Distribution"]
            )
            
            # Algorithm selection
            sort_algorithm = st.selectbox(
                "🔄 Sorting Algorithm",
                ["Bubble Sort", "Selection Sort", "Insertion Sort", "Quick Sort", "Merge Sort", "Heap Sort"]
            )
            
            # Animation settings
            animation_speed = st.slider("⚡ Animation Speed", 0.01, 0.5, 0.1)
            show_comparisons = st.checkbox("👀 Show Comparisons", value=True)
            show_swaps = st.checkbox("🔄 Show Swaps", value=True)
            color_scheme = st.selectbox("🎨 Color Scheme", ["Blue", "Rainbow", "Heat", "Viridis"])
            
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
                else:  # Gaussian Distribution
                    arr = [int(random.gauss(50, 15)) for _ in range(array_size)]
                    arr = [max(1, min(100, x)) for x in arr]
                
                st.session_state.sorting_array = arr
                st.session_state.original_array = arr.copy()
                st.session_state.sorting_steps = []
                st.session_state.current_sort_step = 0
            
            # Step-by-step controls
            if st.button("▶️ Start Step-by-Step Sorting"):
                if 'sorting_array' in st.session_state:
                    st.session_state.start_sorting = True
                    st.session_state.sorting_stats = {
                        'comparisons': 0,
                        'swaps': 0,
                        'accesses': 0,
                        'time': 0
                    }
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("⏮️ Previous Step"):
                    if 'current_sort_step' in st.session_state and st.session_state.current_sort_step > 0:
                        st.session_state.current_sort_step -= 1
            
            with col_btn2:
                if st.button("⏭️ Next Step"):
                    if ('current_sort_step' in st.session_state and 
                        'sorting_steps' in st.session_state and
                        st.session_state.current_sort_step < len(st.session_state.sorting_steps) - 1):
                        st.session_state.current_sort_step += 1
            
            if st.button("🎬 Auto-Play"):
                st.session_state.auto_play_sorting = True
            
            if st.button("⏸️ Pause"):
                st.session_state.auto_play_sorting = False
            
            # Sorting statistics
            if 'sorting_stats' in st.session_state:
                st.markdown("### 📊 Sorting Statistics")
                stats = st.session_state.sorting_stats
                st.metric("Total Comparisons", stats['comparisons'])
                st.metric("Total Swaps", stats['swaps'])
                st.metric("Array Accesses", stats['accesses'])
                if 'sorting_steps' in st.session_state:
                    st.metric("Current Step", f"{st.session_state.get('current_sort_step', 0)}/{len(st.session_state.sorting_steps)}")
        
        with col1:
            if 'sorting_array' in st.session_state:
                # Get current array state
                if 'sorting_steps' in st.session_state and 'current_sort_step' in st.session_state:
                    if st.session_state.current_sort_step < len(st.session_state.sorting_steps):
                        current_state = st.session_state.sorting_steps[st.session_state.current_sort_step]
                        arr = current_state['array']
                        comparing = current_state.get('comparing', [])
                        swapping = current_state.get('swapping', [])
                    else:
                        arr = st.session_state.sorting_array
                        comparing = []
                        swapping = []
                else:
                    arr = st.session_state.sorting_array
                    comparing = []
                    swapping = []
                
                # Choose color scheme
                colors = []
                for i, val in enumerate(arr):
                    if i in swapping and show_swaps:
                        colors.append('red')
                    elif i in comparing and show_comparisons:
                        colors.append('yellow')
                    elif color_scheme == "Rainbow":
                        colors.append(f'hsl({i * 360 / len(arr)}, 70%, 50%)')
                    elif color_scheme == "Heat":
                        colors.append(f'rgb({int(255 * val / 100)}, {int(128 * (1 - val / 100))}, 0)')
                    elif color_scheme == "Viridis":
                        colors.append(px.colors.sample_colorscale('viridis', [val / 100])[0])
                    else:
                        colors.append('lightblue')
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(range(len(arr))),
                        y=arr,
                        marker_color=colors,
                        text=arr if len(arr) <= 30 else None,
                        textposition='outside' if len(arr) <= 30 else 'none'
                    )
                ])
                
                # Add annotations for current operation
                if 'sorting_steps' in st.session_state and 'current_sort_step' in st.session_state:
                    if st.session_state.current_sort_step < len(st.session_state.sorting_steps):
                        current_state = st.session_state.sorting_steps[st.session_state.current_sort_step]
                        if 'operation' in current_state:
                            fig.add_annotation(
                                text=current_state['operation'],
                                xref="paper", yref="paper",
                                x=0.5, y=1.05,
                                showarrow=False,
                                font=dict(size=16, color="blue")
                            )
                
                fig.update_layout(
                    title=f"Sorting Visualization - {sort_algorithm} ({array_type})",
                    xaxis_title="Index",
                    yaxis_title="Value",
                    showlegend=False,
                    height=500,
                    bargap=0.1
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Auto-play logic
                if st.session_state.get('auto_play_sorting', False):
                    if ('current_sort_step' in st.session_state and 
                        'sorting_steps' in st.session_state and
                        st.session_state.current_sort_step < len(st.session_state.sorting_steps) - 1):
                        time.sleep(animation_speed)
                        st.session_state.current_sort_step += 1
                        st.rerun()
                    else:
                        st.session_state.auto_play_sorting = False
                
                # Generate sorting steps when start button is clicked
                if st.session_state.get('start_sorting', False):
                    with st.spinner(f'Generating {sort_algorithm} steps...'):
                        # Generate steps based on algorithm
                        steps = generate_sorting_steps(st.session_state.sorting_array.copy(), sort_algorithm)
                        st.session_state.sorting_steps = steps
                        st.session_state.current_sort_step = 0
                        st.session_state.start_sorting = False
                        st.rerun()
    
    # Real-World Applications Tab
    with sort_tabs[1]:
        st.subheader("🌍 Real-World Sorting Applications")
        
        real_world_app = st.selectbox(
            "Select Application",
            ["📅 Task Scheduler", "📁 File Organizer", "🎵 Music Playlist Optimizer", 
             "📦 Inventory Manager", "🎓 Student Ranking System", "📊 Data Analysis Pipeline",
             "🛒 E-commerce Product Sorting", "📱 App Store Rankings"]
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if real_world_app == "📅 Task Scheduler":
                st.markdown("### Task Priority Scheduler")
                st.info("Sort tasks by priority and deadline for optimal productivity")
                
                # Generate sample tasks
                if st.button("Generate Sample Tasks"):
                    tasks = []
                    task_names = ["Email Client Bug", "Report Writing", "Team Meeting", "Code Review", 
                                 "Documentation Update", "Unit Testing", "Sprint Planning", "Customer Call"]
                    priorities = ["Low", "Medium", "High", "Critical"]
                    
                    for i in range(8):
                        tasks.append({
                            'id': i + 1,
                            'name': random.choice(task_names),
                            'priority': random.choice(priorities),
                            'priority_value': priorities.index(random.choice(priorities)),
                            'deadline': datetime.now() + timedelta(hours=random.randint(1, 48)),
                            'duration': random.randint(15, 120),
                            'assigned_to': random.choice(['You', 'John', 'Sarah', 'Mike'])
                        })
                    st.session_state.tasks = tasks
                
                if 'tasks' in st.session_state:
                    # Display unsorted tasks
                    st.markdown("#### 📋 Unsorted Tasks")
                    df_tasks = pd.DataFrame(st.session_state.tasks)
                    st.dataframe(df_tasks[['name', 'priority', 'deadline', 'duration', 'assigned_to']], use_container_width=True)
                    
                    # Sort options
                    sort_by = st.selectbox("Sort By", ["Priority then Deadline", "Deadline only", "Duration", "Assigned To"])
                    
                    if st.button("🔄 Sort Tasks"):
                        if sort_by == "Priority then Deadline":
                            sorted_tasks = sorted(st.session_state.tasks, 
                                                key=lambda x: (-x['priority_value'], x['deadline']))
                        elif sort_by == "Deadline only":
                            sorted_tasks = sorted(st.session_state.tasks, key=lambda x: x['deadline'])
                        elif sort_by == "Duration":
                            sorted_tasks = sorted(st.session_state.tasks, key=lambda x: x['duration'])
                        else:
                            sorted_tasks = sorted(st.session_state.tasks, key=lambda x: x['assigned_to'])
                        
                        st.session_state.sorted_tasks = sorted_tasks
                    
                    if 'sorted_tasks' in st.session_state:
                        st.markdown("#### ✅ Sorted Tasks")
                        df_sorted = pd.DataFrame(st.session_state.sorted_tasks)
                        st.dataframe(df_sorted[['name', 'priority', 'deadline', 'duration', 'assigned_to']], 
                                   use_container_width=True)
                        
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
                
                if st.button("Generate Sample Files"):
                    files = []
                    extensions = ['pdf', 'docx', 'xlsx', 'png', 'jpg', 'mp4', 'txt', 'py', 'csv']
                    file_prefixes = ['Report', 'Document', 'Image', 'Video', 'Data', 'Script', 'Presentation']
                    
                    for i in range(20):
                        ext = random.choice(extensions)
                        files.append({
                            'name': f"{random.choice(file_prefixes)}_{i+1}.{ext}",
                            'size': random.randint(100, 10000),  # KB
                            'modified': datetime.now() - timedelta(days=random.randint(0, 365)),
                            'type': ext,
                            'folder': random.choice(['Documents', 'Downloads', 'Desktop'])
                        })
                    st.session_state.files = files
                
                if 'files' in st.session_state:
                    st.markdown("#### 📂 File List")
                    df_files = pd.DataFrame(st.session_state.files)
                    
                    # Sorting options
                    sort_method = st.selectbox(
                        "Sort Method",
                        ["By Type then Date", "By Size (Largest First)", "By Date (Newest First)", "By Name"]
                    )
                    
                    if st.button("🔄 Organize Files"):
                        if sort_method == "By Type then Date":
                            organized = RealWorldSorting.file_organizer(st.session_state.files)
                            st.session_state.organized_files = organized
                        elif sort_method == "By Size (Largest First)":
                            sorted_files = sorted(st.session_state.files, key=lambda x: x['size'], reverse=True)
                            st.session_state.sorted_files = sorted_files
                        elif sort_method == "By Date (Newest First)":
                            sorted_files = sorted(st.session_state.files, key=lambda x: x['modified'], reverse=True)
                            st.session_state.sorted_files = sorted_files
                        else:
                            sorted_files = sorted(st.session_state.files, key=lambda x: x['name'])
                            st.session_state.sorted_files = sorted_files
                    
                    # Display results
                    if 'organized_files' in st.session_state:
                        st.markdown("#### 📁 Organized by Type")
                        for file_type, files in st.session_state.organized_files.items():
                            with st.expander(f"{file_type.upper()} Files ({len(files)})"):
                                df = pd.DataFrame(files)
                                st.dataframe(df[['name', 'size', 'modified']], use_container_width=True)
                    
                    elif 'sorted_files' in st.session_state:
                        st.markdown("#### 📁 Sorted Files")
                        df_sorted = pd.DataFrame(st.session_state.sorted_files)
                        st.dataframe(df_sorted, use_container_width=True)
                        
                        # Visualization
                        fig = px.scatter(df_sorted, x='modified', y='size', color='type', 
                                       size='size', hover_data=['name'],
                                       title="Files by Date and Size")
                        st.plotly_chart(fig, use_container_width=True)
            
            elif real_world_app == "🎵 Music Playlist Optimizer":
                st.markdown("### Music Playlist Flow Optimizer")
                st.info("Sort songs by energy, tempo, and mood for smooth transitions")
                
                if st.button("Generate Sample Playlist"):
                    songs = []
                    genres = ['Pop', 'Rock', 'Electronic', 'Hip-Hop', 'Jazz', 'Classical']
                    moods = ['Energetic', 'Chill', 'Happy', 'Melancholic', 'Intense']
                    
                    for i in range(15):
                        songs.append({
                            'title': f"Song {i+1}",
                            'artist': f"Artist {random.randint(1, 10)}",
                            'genre': random.choice(genres),
                            'energy': random.uniform(0.1, 1.0),
                            'tempo': random.randint(60, 180),
                            'mood': random.choice(moods),
                            'duration': random.randint(150, 300)  # seconds
                        })
                    st.session_state.playlist = songs
                
                if 'playlist' in st.session_state:
                    st.markdown("#### 🎵 Current Playlist")
                    df_playlist = pd.DataFrame(st.session_state.playlist)
                    
                    # Sorting options
                    sort_criterion = st.selectbox(
                        "Optimize By",
                        ["Energy Flow", "Tempo Progression", "Genre Grouping", "Mood Transition"]
                    )
                    
                    if st.button("🔄 Optimize Playlist"):
                        if sort_criterion == "Energy Flow":
                            sorted_playlist = sorted(st.session_state.playlist, key=lambda x: x['energy'])
                        elif sort_criterion == "Tempo Progression":
                            sorted_playlist = sorted(st.session_state.playlist, key=lambda x: x['tempo'])
                        elif sort_criterion == "Genre Grouping":
                            sorted_playlist = sorted(st.session_state.playlist, key=lambda x: (x['genre'], x['energy']))
                        else:  # Mood Transition
                            sorted_playlist = sorted(st.session_state.playlist, key=lambda x: (x['mood'], x['energy']))
                        
                        st.session_state.optimized_playlist = sorted_playlist
                    
                    # Display optimized playlist
                    if 'optimized_playlist' in st.session_state:
                        st.markdown("#### 🎶 Optimized Playlist")
                        df_optimized = pd.DataFrame(st.session_state.optimized_playlist)
                        st.dataframe(df_optimized[['title', 'artist', 'genre', 'energy', 'tempo', 'mood']], 
                                   use_container_width=True)
                        
                        # Visualization
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(range(len(df_optimized))),
                            y=df_optimized['energy'],
                            mode='lines+markers',
                            name='Energy',
                            line=dict(width=3, color='purple')
                        ))
                        fig.add_trace(go.Scatter(
                            x=list(range(len(df_optimized))),
                            y=df_optimized['tempo'] / 180,  # Normalize
                            mode='lines+markers',
                            name='Tempo (normalized)',
                            line=dict(width=3, color='orange')
                        ))
                        fig.update_layout(
                            title="Playlist Flow Visualization",
                            xaxis_title="Song Position",
                            yaxis_title="Value",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
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
            elif real_world_app == "📁 File Organizer":
                st.info("""
                **Benefits of File Sorting:**
                - 🔍 Faster file retrieval
                - 💾 Better storage management
                - 📊 Clear file overview
                - 🧹 Cleaner directories
                """)
            elif real_world_app == "🎵 Music Playlist Optimizer":
                st.info("""
                **Benefits of Playlist Sorting:**
                - 🎵 Smoother transitions
                - 🎉 Better party flow
                - 🎧 Enhanced listening experience
                - 💫 Professional DJ-like mixes
                """)
    
    # Algorithm Race Tab
    with sort_tabs[2]:
        st.subheader("🆚 Sorting Algorithm Race")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### 🏁 Race Settings")
            
            race_array_size = st.slider("Array Size", 10, 50, 20)
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
                    race_array = [random.randint(1, 100) for _ in range(race_array_size)]
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
                st.session_state.start_race = True
        
        with col1:
            if st.session_state.get('start_race', False):
                st.markdown("### 🏁 Algorithm Race Visualization")
                
                # Create subplots for each algorithm
                rows = (len(st.session_state.race_algorithms) + 1) // 2
                fig = make_subplots(
                    rows=rows, cols=2,
                    subplot_titles=st.session_state.race_algorithms,
                    vertical_spacing=0.15
                )
                
                # Visualize each algorithm
                for idx, algo in enumerate(st.session_state.race_algorithms):
                    row = idx // 2 + 1
                    col = idx % 2 + 1
                    
                    # Get sorting steps for visualization
                    arr_copy = st.session_state.race_array.copy()
                    steps = generate_sorting_steps(arr_copy, algo)
                    
                    # Store results
                    st.session_state.race_results[algo] = {
                        'steps': len(steps),
                        'comparisons': sum(1 for s in steps if 'comparing' in s),
                        'swaps': sum(1 for s in steps if 'swapping' in s)
                    }
                    
                    # Visualize final sorted array
                    fig.add_trace(
                        go.Bar(
                            x=list(range(len(arr_copy))),
                            y=sorted(arr_copy),
                            name=algo,
                            marker_color='lightblue',
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                
                fig.update_layout(height=400 * rows, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display race results
                st.markdown("### 🏆 Race Results")
                results_df = pd.DataFrame(st.session_state.race_results).T
                results_df['Rank'] = results_df['steps'].rank().astype(int)
                results_df = results_df.sort_values('Rank')
                
                # Winner announcement
                winner = results_df.index[0]
                st.success(f"🥇 **Winner: {winner}** with {results_df.loc[winner, 'steps']} steps!")
                
                # Results table
                st.dataframe(results_df, use_container_width=True)
                
                # Performance chart
                fig_results = px.bar(
                    results_df.reset_index(),
                    x='index',
                    y=['steps', 'comparisons', 'swaps'],
                    title="Algorithm Performance Comparison",
                    labels={'index': 'Algorithm', 'value': 'Count'},
                    barmode='group'
                )
                st.plotly_chart(fig_results, use_container_width=True)
    
    # Complexity Analysis Tab
    with sort_tabs[3]:
        st.subheader("📈 Sorting Algorithm Complexity Analysis")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### ⏱️ Time Complexity Comparison")
            
            # Create complexity data
            n_values = np.logspace(1, 4, 50)
            
            complexity_data = {
                'O(n²) - Bubble/Selection': n_values ** 2,
                'O(n²) - Insertion (worst)': n_values ** 2,
                'O(n log n) - Merge/Heap': n_values * np.log2(n_values),
                'O(n log n) - Quick (avg)': n_values * np.log2(n_values),
                'O(n) - Insertion (best)': n_values
            }
            
            fig_time = go.Figure()
            
            for name, values in complexity_data.items():
                fig_time.add_trace(go.Scatter(
                    x=n_values,
                    y=values,
                    mode='lines',
                    name=name,
                    line=dict(width=3)
                ))
            
            fig_time.update_layout(
                title="Time Complexity Growth Rates",
                xaxis_title="Input Size (n)",
                yaxis_title="Operations",
                xaxis_type="log",
                yaxis_type="log",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            st.markdown("### 💾 Space Complexity Comparison")
            
            space_data = pd.DataFrame({
                'Algorithm': ['Bubble Sort', 'Selection Sort', 'Insertion Sort', 
                             'Merge Sort', 'Quick Sort', 'Heap Sort'],
                'Space Complexity': ['O(1)', 'O(1)', 'O(1)', 'O(n)', 'O(log n)', 'O(1)'],
                'Memory Usage': [1, 1, 1, 100, 10, 1],
                'In-Place': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes']
            })
            
            fig_space = px.bar(
                space_data,
                x='Algorithm',
                y='Memory Usage',
                color='In-Place',
                title='Space Complexity Comparison',
                text='Space Complexity',
                color_discrete_map={'Yes': 'lightgreen', 'No': 'lightcoral'}
            )
            
            fig_space.update_traces(textposition='outside')
            fig_space.update_layout(height=400)
            
            st.plotly_chart(fig_space, use_container_width=True)
        
        # Best/Average/Worst Case Analysis
        st.markdown("### 📊 Comprehensive Complexity Analysis")
        
        complexity_table = pd.DataFrame({
            'Algorithm': ['Bubble Sort', 'Selection Sort', 'Insertion Sort', 
                         'Merge Sort', 'Quick Sort', 'Heap Sort'],
            'Best Case': ['O(n)', 'O(n²)', 'O(n)', 'O(n log n)', 'O(n log n)', 'O(n log n)'],
            'Average Case': ['O(n²)', 'O(n²)', 'O(n²)', 'O(n log n)', 'O(n log n)', 'O(n log n)'],
            'Worst Case': ['O(n²)', 'O(n²)', 'O(n²)', 'O(n log n)', 'O(n²)', 'O(n log n)'],
            'Space': ['O(1)', 'O(1)', 'O(1)', 'O(n)', 'O(log n)', 'O(1)'],
            'Stable': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No'],
            'In-Place': ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes']
        })
        
        st.dataframe(
            complexity_table.style.applymap(
                lambda x: 'background-color: lightgreen' if 'n log n' in str(x) else (
                    'background-color: lightcoral' if 'n²' in str(x) else ''
                ),
                subset=['Best Case', 'Average Case', 'Worst Case']
            ),
            use_container_width=True
        )
        
        # Use case recommendations
        st.markdown("### 💡 Algorithm Selection Guide")
        
        col_guide1, col_guide2 = st.columns(2)
        
        with col_guide1:
            st.info("""
            **When to use O(n²) algorithms:**
            - Small datasets (n < 50)
            - Nearly sorted data (Insertion Sort)
            - Simple implementation needed
            - Teaching/learning purposes
            """)
            
            st.success("""
            **When to use O(n log n) algorithms:**
            - Large datasets
            - Performance critical applications
            - Unknown data distribution
            - Production systems
            """)
        
        with col_guide2:
            st.warning("""
            **Special Considerations:**
            - **Stable sorting**: Merge Sort, Insertion Sort
            - **In-place sorting**: Quick Sort, Heap Sort
            - **Nearly sorted data**: Insertion Sort
            - **Guaranteed performance**: Merge Sort, Heap Sort
            """)
            
            st.error("""
            **Avoid these mistakes:**
            - Using Bubble Sort in production
            - Quick Sort for nearly sorted data
            - Ignoring space complexity
            - Not considering data characteristics
            """)

# Helper function to generate sorting steps
def generate_sorting_steps(arr, algorithm):
    steps = []
    n = len(arr)
    
    if algorithm == "Bubble Sort":
        for i in range(n):
            for j in range(0, n-i-1):
                steps.append({
                    'array': arr.copy(),
                    'comparing': [j, j+1],
                    'operation': f"Inserting: comparing positions {j} and {j+1}"
                })
                arr[j+1] = arr[j]
                j -= 1
            arr[j+1] = key
            steps.append({
                'array': arr.copy(),
                'swapping': [j+1],
                'operation': f"Inserted element at position {j+1}"
            })
    
    # For other algorithms, we'll use simplified versions
    else:
        # Just show the final sorted array for now
        arr.sort()
        steps.append({
            'array': arr.copy(),
            'operation': f"{algorithm} - Completed"
        })
    
    return steps

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

# Add final developer credit at the bottom of sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 10px; border-radius: 10px; 
                text-align: center; margin-top: 20px;'>
        <strong>✨ Developed with ❤️ by<br>Shreyas Kasture ✨</strong>
    </div>
    """, unsafe_allow_html=True)comparing': [j, j+1],
                    'operation': f"Comparing positions {j} and {j+1}"
                })
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
                    steps.append({
                        'array': arr.copy(),
                        'swapping': [j, j+1],
                        'operation': f"Swapping positions {j} and {j+1}"
                    })
    
    elif algorithm == "Selection Sort":
        for i in range(n):
            min_idx = i
            for j in range(i+1, n):
                steps.append({
                    'array': arr.copy(),
                    'comparing': [min_idx, j],
                    'operation': f"Finding minimum: comparing {min_idx} and {j}"
                })
                if arr[j] < arr[min_idx]:
                    min_idx = j
            
            if min_idx != i:
                arr[i], arr[min_idx] = arr[min_idx], arr[i]
                steps.append({
                    'array': arr.copy(),
                    'swapping': [i, min_idx],
                    'operation': f"Swapping minimum to position {i}"
                })
    
    elif algorithm == "Insertion Sort":
        for i in range(1, n):
            key = arr[i]
            j = i-1
            while j >= 0 and key < arr[j]:
                steps.append({
                    'array': arr.copy(),
                    '
