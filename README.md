# 🗺️ Advanced PathFinder & Sort Visualizer

🌍 **Live Project**: [https://shreyas-pathfinder-visualizer-sort-simulator.streamlit.app](https://shreyas-pathfinder-visualizer-sort-simulator.streamlit.app/)

An interactive web application built with Streamlit that provides comprehensive visualization and analysis of pathfinding algorithms and sorting algorithms. Perfect for educational purposes, algorithm comparison, and real-world application demonstrations.

## 🚀 Features Overview

### 🗺️ Pathfinding Visualizer
- **Interactive Grid-Based Pathfinding** with click-to-place functionality
- **6 Different Pathfinding Algorithms** with performance comparison
- **Multiple Map Types** and obstacle patterns
- **Real-time Algorithm Visualization** with step-by-step execution
- **Performance Metrics** and efficiency analysis

### 📊 Sorting Visualizer
- **Animated Sorting Algorithms** with step-by-step visualization
- **6 Different Sorting Algorithms** with comprehensive analysis
- **Real-World Applications** demonstrating practical use cases
- **Performance Comparison** across different scenarios
- **Interactive Data Management** systems

## 🛠️ Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Dependencies
```bash
pip install streamlit
pip install plotly
pip install pandas
pip install numpy
pip install folium
pip install streamlit-folium
pip install requests
```

### Quick Install
```bash
# Clone the repository
git clone https://github.com/yourusername/pathfinder-sort-visualizer.git
cd pathfinder-sort-visualizer

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 🚀 Quick Start

1. **Launch the Application:**
   ```bash
   streamlit run app.py
   ```

2. **Access the Interface:**
   - Open your browser to `http://localhost:8501`
   - Choose between PathFinding or Sorting tabs

3. **Start Exploring:**
   - Use interactive grids for pathfinding
   - Generate arrays and watch sorting animations
   - Compare algorithm performances

## 🗺️ Pathfinding Features

### Interactive Grid Pathfinding

#### **🖱️ Click-to-Place Functionality**
- **Set Start Point:** Click on grid to place starting position
- **Set Goal Point:** Click on grid to place destination
- **Add Obstacles:** Click to create barriers and walls
- **Remove Obstacles:** Click to clear existing barriers

#### **🎛️ Interaction Modes**
- Switch between different placement modes
- Real-time visual feedback
- Instant grid updates

#### **🗺️ Sample Maps**
- **Empty Grid:** Clean slate for custom designs
- **Maze:** Complex maze patterns with corridors
- **Random Obstacles:** Randomly distributed barriers
- **Diagonal Barriers:** Diagonal wall patterns
- **Rooms:** Room-like structures with doorways

#### **📊 Visual Elements**
- 🟩 **Green:** Start point
- 🟧 **Orange:** Goal point
- 🟨 **Yellow:** Optimal path
- 🔵 **Light Blue:** Visited nodes
- ⬛ **Black:** Obstacles
- ⬜ **White:** Empty cells

### Algorithm Comparison

#### **Performance Metrics**
- **Execution Time:** Algorithm speed in milliseconds
- **Path Length:** Number of steps in optimal path
- **Nodes Visited:** Search space exploration
- **Efficiency Percentage:** Path optimality ratio

#### **Visualization Features**
- Real-time pathfinding animation
- Step-by-step algorithm execution
- Performance comparison charts
- Efficiency analysis graphs

## 📊 Sorting Features

### Algorithm Animation

#### **🎬 Interactive Sorting Visualization**
- **Step-by-step Animation:** Watch algorithms in action
- **Variable Speed Control:** Adjust animation speed (0.01s - 1.0s)
- **Visual Highlighting:** See comparisons and swaps
- **Progress Tracking:** Real-time progress indicators

#### **📏 Array Configuration**
- **Size Options:** 10 to 100 elements
- **Array Types:**
  - **Random:** Completely randomized data
  - **Nearly Sorted:** Mostly ordered with few disruptions
  - **Reverse Sorted:** Completely reversed order
  - **Few Unique:** Limited unique values
  - **Mostly Sorted:** Minor disruptions in order

#### **📊 Metrics Tracking**
- **Comparisons:** Number of element comparisons
- **Swaps:** Number of element exchanges
- **Array Accesses:** Total memory accesses
- **Execution Time:** Real-time performance measurement

### Real-World Applications

#### **🎓 Student Grade Management System**
```
Features:
- Sort students by score, age, name, or grade
- Grade distribution visualization
- Academic performance metrics
- Statistical analysis and insights

Use Cases:
- Class ranking and performance analysis
- Academic reporting and statistics
- Student performance tracking
- Educational data management
```

#### **👥 Employee HR Management System**
```
Features:
- Sort employees by salary, experience, performance, department
- Salary distribution analysis
- Performance correlation charts
- Workforce analytics

Use Cases:
- Payroll management and analysis
- Performance review preparation
- Organizational structure analysis
- HR reporting and insights
```

#### **🛒 E-commerce Product Management**
```
Features:
- Sort products by price, rating, sales, stock
- Category-based filtering
- Sales performance analysis
- Inventory management insights

Use Cases:
- Product catalog organization
- Inventory optimization
- Sales trend analysis
- Customer preference insights
```

#### **📈 Financial Stock Analysis**
```
Features:
- Sort stocks by price, volume, market cap, P/E ratio, dividend
- Investment analysis visualizations
- Portfolio performance metrics
- Financial trend analysis

Use Cases:
- Investment decision support
- Portfolio optimization
- Market trend analysis
- Financial reporting
```

## 🧠 Algorithms Included

### Pathfinding Algorithms

| Algorithm | Time Complexity | Space Complexity | Optimal | Best Use Case |
|-----------|----------------|------------------|---------|---------------|
| **A* (A-Star)** | O(b^d) | O(b^d) | ✅ Yes | General pathfinding with heuristics |
| **Dijkstra** | O((V+E) log V) | O(V) | ✅ Yes | Guaranteed shortest path |
| **BFS** | O(V+E) | O(V) | ✅ Yes (unweighted) | Unweighted graphs |
| **DFS** | O(V+E) | O(h) | ❌ No | Memory-limited scenarios |
| **Greedy Best-First** | O(b^m) | O(b^m) | ❌ No | Speed over optimality |
| **Bidirectional** | O(b^(d/2)) | O(b^(d/2)) | ✅ Yes | Large search spaces |

### Sorting Algorithms

| Algorithm | Best Case | Average Case | Worst Case | Space | Stable | Best Use Case |
|-----------|-----------|--------------|------------|-------|--------|---------------|
| **Bubble Sort** | O(n) | O(n²) | O(n²) | O(1) | ✅ Yes | Educational, very small data |
| **Selection Sort** | O(n²) | O(n²) | O(n²) | O(1) | ❌ No | Memory write constraints |
| **Insertion Sort** | O(n) | O(n²) | O(n²) | O(1) | ✅ Yes | Small/nearly sorted data |
| **Quick Sort** | O(n log n) | O(n log n) | O(n²) | O(log n) | ❌ No | General purpose |
| **Merge Sort** | O(n log n) | O(n log n) | O(n log n) | O(n) | ✅ Yes | Stable sorting needed |
| **Heap Sort** | O(n log n) | O(n log n) | O(n log n) | O(1) | ❌ No | Guaranteed performance |

## 📖 Usage Guide

### Pathfinding Workflow

1. **Setup Grid:**
   ```
   - Choose grid dimensions (10x10 to 50x50)
   - Select interaction mode (Start/Goal/Obstacles)
   - Load sample map or create custom layout
   ```

2. **Place Points:**
   ```
   - Click on grid to set start point (green)
   - Click on grid to set goal point (orange)
   - Add obstacles by clicking in obstacle mode
   ```

3. **Run Algorithm:**
   ```
   - Select pathfinding algorithm
   - Click "Find Path" button
   - Watch real-time visualization
   - Analyze performance metrics
   ```

4. **Compare Results:**
   ```
   - Run multiple algorithms on same grid
   - Compare execution times and efficiency
   - Analyze path optimality
   ```

### Sorting Workflow

1. **Configure Array:**
   ```
   - Set array size (10-100 elements)
   - Choose array type (Random, Nearly Sorted, etc.)
   - Generate new array data
   ```

2. **Select Algorithm:**
   ```
   - Choose sorting algorithm
   - Set animation speed
   - Enable/disable metrics tracking
   ```

3. **Run Visualization:**
   ```
   - Click "Start Sorting" button
   - Watch step-by-step animation
   - Monitor real-time metrics
   ```

4. **Analyze Performance:**
   ```
   - Compare algorithm speeds
   - Analyze complexity patterns
   - Review efficiency metrics
   ```

### Real-World Applications Workflow

1. **Choose Domain:**
   ```
   - Student Management
   - Employee HR System
   - E-commerce Products
   - Financial Stocks
   ```

2. **Generate Data:**
   ```
   - Set data size parameters
   - Generate realistic sample data
   - Review data characteristics
   ```

3. **Apply Sorting:**
   ```
   - Select sorting criteria
   - Choose sort order (Asc/Desc)
   - Apply filters if available
   ```

4. **Analyze Results:**
   ```
   - Review sorted data tables
   - Examine visualization charts
   - Extract business insights
   ```

## 📊 Performance Analysis

### Comprehensive Testing Features

#### **Multi-Parameter Analysis**
- **Array Sizes:** Test from 10 to 500 elements
- **Array Types:** Multiple data distributions
- **Algorithm Comparison:** Side-by-side performance
- **Statistical Analysis:** Average, best, worst case scenarios

#### **Visualization Tools**
- **Performance Charts:** Line graphs showing scaling behavior
- **Heatmaps:** Algorithm efficiency across scenarios
- **Bar Charts:** Direct speed comparisons
- **Complexity Analysis:** Visual complexity demonstrations

#### **Export Capabilities**
- **CSV Export:** Download performance analysis results
- **Chart Export:** Save visualizations
- **Report Generation:** Comprehensive analysis reports

### Performance Insights

#### **Algorithm Selection Guide**
```
Small Arrays (< 50):     Insertion Sort
General Purpose:         Quick Sort, Merge Sort
Guaranteed O(n log n):   Merge Sort, Heap Sort
Memory Constrained:      Heap Sort, Quick Sort
Stability Required:      Merge Sort, Insertion Sort
Educational:             Bubble Sort, Selection Sort
```

#### **Pathfinding Selection Guide**
```
With Good Heuristic:     A* Algorithm
No Heuristic Available:  Dijkstra's Algorithm
Unweighted Graphs:       BFS
Memory Limited:          DFS
Speed Priority:          Greedy Best-First
Large Search Space:      Bidirectional Search
```
---

## 🚀 Get Started Today!

Built with ❤️ by Shreyas Kasture

Ready to explore algorithms? Clone the repository and start visualizing!

```bash
git clone https://github.com/yourusername/pathfinder-sort-visualizer.git
cd pathfinder-sort-visualizer
pip install -r requirements.txt
streamlit run app.py
```

**Happy Algorithm Exploring! 🎉**
