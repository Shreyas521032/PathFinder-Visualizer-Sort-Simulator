// components/SortingVisualizer.tsx
'use client';

import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { useSortingStore } from '@/store';
import { getBarColor } from '@/lib/utils';

interface SortingVisualizerProps {
  className?: string;
}

export default function SortingVisualizer({ className }: SortingVisualizerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  
  const {
    array,
    comparingIndices,
    swappingIndices,
    sortedIndices,
    isRunning,
    isComplete,
  } = useSortingStore();

  const maxHeight = 400;
  const maxValue = Math.max(...array);

  const getBarHeight = (value: number) => {
    return (value / maxValue) * maxHeight;
  };

  const isComparing = (index: number) => comparingIndices.includes(index);
  const isSwapping = (index: number) => swappingIndices.includes(index);
  const isSorted = (index: number) => sortedIndices.includes(index);

  return (
    <div 
      ref={containerRef}
      className={`bg-white dark:bg-gray-900 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6 overflow-hidden ${className}`}
    >
      <div className="mb-4">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
          Sorting Visualization
        </h2>
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-gray-400 rounded"></div>
            <span className="text-gray-600 dark:text-gray-400">Unsorted</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-yellow-500 rounded"></div>
            <span className="text-gray-600 dark:text-gray-400">Comparing</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-red-500 rounded"></div>
            <span className="text-gray-600 dark:text-gray-400">Swapping</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-500 rounded"></div>
            <span className="text-gray-600 dark:text-gray-400">Sorted</span>
          </div>
        </div>
      </div>

      <div 
        className="flex items-end justify-center gap-1 mx-auto"
        style={{ height: `${maxHeight + 20}px` }}
      >
        {array.map((value, index) => {
          const barHeight = getBarHeight(value);
          const barColor = getBarColor(
            index,
            isComparing(index),
            isSwapping(index),
            isSorted(index)
          );

          return (
            <motion.div
              key={index}
              className="relative flex flex-col items-center"
              style={{ 
                width: `${Math.max(800 / array.length - 2, 2)}px`,
                minWidth: '2px'
              }}
              initial={{ height: 0 }}
              animate={{ 
                height: barHeight,
                backgroundColor: barColor,
              }}
              transition={{ 
                duration: isRunning ? 0.3 : 0.5,
                ease: "easeInOut"
              }}
            >
              {/* Bar */}
              <motion.div
                className="w-full rounded-t-sm"
                style={{ 
                  height: `${barHeight}px`,
                  backgroundColor: barColor,
                }}
                animate={{
                  scale: isSwapping(index) ? [1, 1.1, 1] : 1,
                  boxShadow: isComparing(index) 
                    ? '0 0 10px rgba(245, 158, 11, 0.5)' 
                    : isSwapping(index)
                    ? '0 0 10px rgba(239, 68, 68, 0.5)'
                    : 'none'
                }}
                transition={{ 
                  duration: 0.3,
                  ease: "easeInOut"
                }}
              />
              
              {/* Value label for smaller arrays */}
              {array.length <= 20 && (
                <motion.div
                  className="absolute -top-6 text-xs font-medium text-gray-600 dark:text-gray-400"
                  animate={{
                    color: isComparing(index) || isSwapping(index) 
                      ? '#FFFFFF' 
                      : isSorted(index)
                      ? '#10B981'
                      : '#6B7280'
                  }}
                >
                  {value}
                </motion.div>
              )}
            </motion.div>
          );
        })}
      </div>

      {/* Array size and completion status */}
      <div className="mt-4 text-center text-sm text-gray-600 dark:text-gray-400">
        <div className="flex items-center justify-center gap-4">
          <span>Array Size: {array.length}</span>
          {isComplete && (
            <motion.span
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-green-600 dark:text-green-400 font-medium"
            >
              ✓ Sorting Complete!
            </motion.span>
          )}
          {isRunning && (
            <motion.span
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 1, repeat: Infinity }}
              className="text-blue-600 dark:text-blue-400 font-medium"
            >
              Sorting in progress...
            </motion.span>
          )}
        </div>
      </div>

      {/* Progress indicator */}
      {(isRunning || isComplete) && (
        <div className="mt-3">
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <motion.div
              className="bg-blue-600 h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ 
                width: `${(sortedIndices.length / array.length) * 100}%`
              }}
              transition={{ duration: 0.5 }}
            />
          </div>
          <div className="text-xs text-center mt-1 text-gray-500 dark:text-gray-400">
            {Math.round((sortedIndices.length / array.length) * 100)}% Complete
          </div>
        </div>
      )}
    </div>
  );
}
