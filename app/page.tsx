// app/page.tsx
'use client';

import { useEffect } from 'react';
import { motion } from 'framer-motion';
import MapView from '@/components/MapView';
import PathfindingControls from '@/components/PathfindingControls';
import { usePathfindingStore } from '@/store';

export default function PathfindingPage() {
  const { generateRandomArray } = usePathfindingStore();

  useEffect(() => {
    // Initialize random array for sorting store when app loads
    generateRandomArray();
  }, [generateRandomArray]);

  return (
    <div className="min-h-[calc(100vh-4rem)] p-4 sm:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl sm:text-5xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent mb-4">
            Pathfinding Visualizer
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Explore pathfinding algorithms on real-world maps. Visualize how different algorithms 
            like A*, Dijkstra, BFS, and DFS find the optimal path between two points.
          </p>
        </motion.div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-4 gap-6 h-[calc(100vh-16rem)]">
          {/* Controls Sidebar */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="lg:col-span-1"
          >
            <PathfindingControls />
          </motion.div>

          {/* Map Container */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="lg:col-span-3"
          >
            <MapView className="w-full h-full min-h-[500px] lg:min-h-0" />
          </motion.div>
        </div>

        {/* Algorithm Info Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="mt-8 grid md:grid-cols-2 lg:grid-cols-3 gap-4"
        >
          {[
            {
              name: "A* Search",
              description: "Optimal pathfinding using heuristic guidance",
              color: "from-blue-500 to-blue-600",
              complexity: "O(b^d)",
            },
            {
              name: "Dijkstra's Algorithm",
              description: "Guaranteed shortest path by exploring all possibilities",
              color: "from-red-500 to-red-600",
              complexity: "O((V + E) log V)",
            },
            {
              name: "Breadth-First Search",
              description: "Level-by-level exploration guaranteeing shortest path",
              color: "from-green-500 to-green-600",
              complexity: "O(V + E)",
            },
          ].map((algorithm, index) => (
            <motion.div
              key={algorithm.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.8 + index * 0.1 }}
              className="bg-white dark:bg-gray-900 rounded-xl p-6 border border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 transition-colors"
            >
              <div className={`w-12 h-12 bg-gradient-to-r ${algorithm.color} rounded-lg mb-4 flex items-center justify-center`}>
                <span className="text-white font-bold text-lg">
                  {algorithm.name.charAt(0)}
                </span>
              </div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
                {algorithm.name}
              </h3>
              <p className="text-sm text-gray-600 dark:text-gray-300 mb-3">
                {algorithm.description}
              </p>
              <div className="text-xs text-gray-500 dark:text-gray-400">
                Time Complexity: <code className="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded">{algorithm.complexity}</code>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Instructions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 1 }}
          className="mt-8 bg-blue-50 dark:bg-blue-900/20 rounded-xl p-6 border border-blue-200 dark:border-blue-800"
        >
          <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-200 mb-4">
            How to Use
          </h3>
          <div className="grid md:grid-cols-2 gap-4 text-sm text-blue-800 dark:text-blue-300">
            <div>
              <h4 className="font-medium mb-2">Getting Started:</h4>
              <ul className="space-y-1">
                <li>• Click on the map to place your start point</li>
                <li>• Shift + Click to place your destination</li>
                <li>• Ctrl/Cmd + Click to add or remove walls</li>
                <li>• Choose your algorithm and adjust speed</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">Tips:</h4>
              <ul className="space-y-1">
                <li>• Try different algorithms to see performance differences</li>
                <li>• Add walls to create more complex pathfinding scenarios</li>
                <li>• Adjust speed to better understand algorithm behavior</li>
                <li>• Reset and try new locations to experiment</li>
              </ul>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
