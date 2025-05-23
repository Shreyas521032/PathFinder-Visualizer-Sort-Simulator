// app/sort/page.tsx
'use client';

import { useEffect } from 'react';
import { motion } from 'framer-motion';
import SortingVisualizer from '@/components/SortingVisualizer';
import SortControls from '@/components/SortControls';
import { useSortingStore } from '@/store';

export default function SortingPage() {
  const { generateRandomArray } = useSortingStore();

  useEffect(() => {
    // Initialize random array when component mounts
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
          <h1 className="text-4xl sm:text-5xl font-bold bg-gradient-to-r from-purple-600 via-pink-600 to-indigo-600 bg-clip-text text-transparent mb-4">
            Sorting Algorithm Visualizer
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            Watch sorting algorithms in action! Compare different approaches like Bubble Sort, 
            Merge Sort, Quick Sort, and more to understand their performance characteristics.
          </p>
        </motion.div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-4 gap-6">
          {/* Controls Sidebar */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="lg:col-span-1 lg:order-2"
          >
            <SortControls />
          </motion.div>

          {/* Visualizer Container */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="lg:col-span-3 lg:order-1"
          >
            <SortingVisualizer className="w-full" />
          </motion.div>
        </div>

        {/* Algorithm Comparison Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="mt-8 grid md:grid-cols-2 lg:grid-cols-4 gap-4"
        >
          {[
            {
              name: "Bubble Sort",
              description: "Simple comparison-based algorithm",
              timeComplexity: "O(n²)",
              spaceComplexity: "O(1)",
              color: "from-blue-500 to-blue-600",
              stability: "Stable",
            },
            {
              name: "Merge Sort",
              description: "Efficient divide-and-conquer approach",
              timeComplexity: "O(n log n)",
              spaceComplexity: "O(n)",
              color: "from-green-500 to-green-600",
              stability: "Stable",
            },
            {
              name: "Quick Sort",
              description: "Fast average-case performance",
              timeComplexity: "O(n log n)",
              spaceComplexity: "O(log n)",
              color: "from-red-500 to-red-600",
              stability: "Unstable",
            },
            {
              name: "Heap Sort",
              description: "Consistent performance guarantee",
              timeComplexity: "O(n log n)",
              spaceComplexity: "O(1)",
              color: "from-yellow-500 to-yellow-600",
              stability: "Unstable",
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
              <div className="space-y-1 text-xs text-gray-500 dark:text-gray-400">
                <div>
                  Time: <code className="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded">{algorithm.timeComplexity}</code>
                </div>
                <div>
                  Space: <code className="bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded">{algorithm.spaceComplexity}</code>
                </div>
                <div>
                  <span className={`inline-block px-2 py-0.5 rounded text-xs ${
                    algorithm.stability === 'Stable' 
                      ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300'
                      : 'bg-orange-100 text-orange-800 dark:bg-orange-900/20 dark:text-orange-300'
                  }`}>
                    {algorithm.stability}
                  </span>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>

        {/* Performance Comparison */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 1 }}
          className="mt-8 bg-purple-50 dark:bg-purple-900/20 rounded-xl p-6 border border-purple-200 dark:border-purple-800"
        >
          <h3 className="text-lg font-semibold text-purple-900 dark:text-purple-200 mb-4">
            Algorithm Performance Guide
          </h3>
          <div className="grid md:grid-cols-3 gap-6 text-sm text-purple-800 dark:text-purple-300">
            <div>
              <h4 className="font-medium mb-2">Best for Learning:</h4>
              <ul className="space-y-1">
                <li>• <strong>Bubble Sort</strong> - Easy to understand</li>
                <li>• <strong>Insertion Sort</strong> - Good for small arrays</li>
                <li>• <strong>Selection Sort</strong> - Simple selection process</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">Best Performance:</h4>
              <ul className="space-y-1">
                <li>• <strong>Merge Sort</strong> - Consistent O(n log n)</li>
                <li>• <strong>Quick Sort</strong> - Fast average case</li>
                <li>• <strong>Heap Sort</strong> - Guaranteed O(n log n)</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">Special Cases:</h4>
              <ul className="space-y-1">
                <li>• <strong>Radix Sort</strong> - Great for integers</li>
                <li>• <strong>Insertion Sort</strong> - Nearly sorted data</li>
                <li>• <strong>Merge Sort</strong> - Stable sorting needed</li>
              </ul>
            </div>
          </div>
        </motion.div>

        {/* Instructions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 1.2 }}
          className="mt-8 bg-gray-50 dark:bg-gray-900/50 rounded-xl p-6 border border-gray-200 dark:border-gray-700"
        >
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
            How to Use the Visualizer
          </h3>
          <div className="grid md:grid-cols-2 gap-4 text-sm text-gray-600 dark:text-gray-300">
            <div>
              <h4 className="font-medium mb-2">Controls:</h4>
              <ul className="space-y-1">
                <li>• Select any sorting algorithm from the dropdown</li>
                <li>• Adjust array size (10-100 elements)</li>
                <li>• Control animation speed for better observation</li>
                <li>• Generate new random arrays to test</li>
              </ul>
            </div>
            <div>
              <h4 className="font-medium mb-2">Understanding the Colors:</h4>
              <ul className="space-y-1">
                <li>• <span className="text-gray-500">Gray</span> - Unsorted elements</li>
                <li>• <span className="text-yellow-500">Yellow</span> - Elements being compared</li>
                <li>• <span className="text-red-500">Red</span> - Elements being swapped</li>
                <li>• <span className="text-green-500">Green</span> - Sorted elements</li>
              </ul>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
