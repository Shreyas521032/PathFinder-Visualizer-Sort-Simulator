// components/SortControls.tsx
'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Play, 
  Square, 
  RotateCcw, 
  Shuffle,
  Info,
  Clock,
  TrendingUp,
  Zap,
  BarChart3
} from 'lucide-react';
import { useSortingStore } from '@/store';
import { SortingAlgorithm } from '@/types';
import { sortingAlgorithms } from '@/lib/utils';
import { runSortingAlgorithm } from '@/lib/sorting-runner';

export default function SortControls() {
  const [showInfo, setShowInfo] = useState(false);
  const [stats, setStats] = useState<{
    duration: number;
    comparisons: number;
    swaps: number;
  } | null>(null);

  const {
    algorithm,
    arraySize,
    speed,
    isRunning,
    isComplete,
    array,
    setAlgorithm,
    setArraySize,
    setSpeed,
    setIsRunning,
    setIsComplete,
    setArray,
    setComparingIndices,
    setSwappingIndices,
    setSortedIndices,
    generateRandomArray,
    reset,
  } = useSortingStore();

  const canRun = array.length > 0 && !isRunning;

  const handleStart = async () => {
    if (!canRun) return;

    setIsRunning(true);
    setStats(null);
    const startTime = Date.now();

    try {
      const result = await runSortingAlgorithm(
        algorithm,
        array,
        speed,
        (step) => {
          setArray(step.array);
          setComparingIndices(step.comparingIndices);
          setSwappingIndices(step.swappingIndices);
          setSortedIndices(step.sortedIndices);
        }
      );

      const duration = Date.now() - startTime;
      setStats({
        duration,
        comparisons: result.comparisons,
        swaps: result.swaps,
      });

      setIsComplete(true);
    } catch (error) {
      console.error('Sorting error:', error);
    } finally {
      setIsRunning(false);
    }
  };

  const handleStop = () => {
    setIsRunning(false);
  };

  const handleReset = () => {
    reset();
    setStats(null);
  };

  const handleGenerateArray = () => {
    generateRandomArray();
    setStats(null);
  };

  const handleArraySizeChange = (newSize: number) => {
    setArraySize(newSize);
    generateRandomArray();
    setStats(null);
  };

  const algorithmInfo = sortingAlgorithms[algorithm];

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="bg-white dark:bg-gray-900 rounded-xl shadow-lg border border-gray-200 dark:border-gray-700 p-6 space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">
          Sorting Controls
        </h2>
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => setShowInfo(!showInfo)}
          className="p-2 text-gray-500 hover:text-blue-500 transition-colors"
        >
          <Info className="w-5 h-5" />
        </motion.button>
      </div>

      {/* Algorithm Selection */}
      <div className="space-y-3">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Algorithm
        </label>
        <select
          value={algorithm}
          onChange={(e) => setAlgorithm(e.target.value as SortingAlgorithm)}
          disabled={isRunning}
          className="w-full px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:opacity-50"
        >
          {Object.entries(sortingAlgorithms).map(([key, info]) => (
            <option key={key} value={key}>
              {info.icon} {info.name}
            </option>
          ))}
        </select>
      </div>

      {/* Algorithm Info */}
      {showInfo && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-4 border border-purple-200 dark:border-purple-800"
        >
          <h3 className="font-semibold text-purple-900 dark:text-purple-200 mb-2">
            {algorithmInfo.name}
          </h3>
          <p className="text-sm text-purple-700 dark:text-purple-300 mb-3">
            {algorithmInfo.description}
          </p>
          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <span className="font-medium text-purple-800 dark:text-purple-200">Time:</span>
              <span className="ml-1 text-purple-600 dark:text-purple-400">
                {algorithmInfo.timeComplexity}
              </span>
            </div>
            <div>
              <span className="font-medium text-purple-800 dark:text-purple-200">Space:</span>
              <span className="ml-1 text-purple-600 dark:text-purple-400">
                {algorithmInfo.spaceComplexity}
              </span>
            </div>
          </div>
        </motion.div>
      )}

      {/* Array Size Control */}
      <div className="space-y-3">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center gap-2">
          <BarChart3 className="w-4 h-4" />
          Array Size: {arraySize}
        </label>
        <input
          type="range"
          min="10"
          max="100"
          step="5"
          value={arraySize}
          onChange={(e) => handleArraySizeChange(Number(e.target.value))}
          disabled={isRunning}
          className="w-full accent-purple-500"
        />
        <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
          <span>10</span>
          <span>100</span>
        </div>
      </div>

      {/* Speed Control */}
      <div className="space-y-3">
        <label className="text-sm font-medium text-gray-700 dark:text-gray-300 flex items-center gap-2">
          <Zap className="w-4 h-4" />
          Speed: {speed}ms
        </label>
        <input
          type="range"
          min="10"
          max="500"
          step="10"
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          disabled={isRunning}
          className="w-full accent-blue-500"
        />
        <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
          <span>Fast</span>
          <span>Slow</span>
        </div>
      </div>

      {/* Statistics */}
      {stats && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4 border border-green-200 dark:border-green-800"
        >
          <h3 className="font-semibold text-green-900 dark:text-green-200 mb-3 flex items-center gap-2">
            <TrendingUp className="w-4 h-4" />
            Results
          </h3>
          <div className="grid grid-cols-3 gap-3 text-sm">
            <div className="text-center">
              <div className="font-semibold text-green-800 dark:text-green-200">
                {stats.duration}ms
              </div>
              <div className="text-green-600 dark:text-green-400 text-xs">Duration</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-green-800 dark:text-green-200">
                {stats.comparisons}
              </div>
              <div className="text-green-600 dark:text-green-400 text-xs">Comparisons</div>
            </div>
            <div className="text-center">
              <div className="font-semibold text-green-800 dark:text-green-200">
                {stats.swaps}
              </div>
              <div className="text-green-600 dark:text-green-400 text-xs">Swaps</div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Control Buttons */}
      <div className="space-y-3">
        {!isRunning ? (
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleStart}
            disabled={!canRun}
            className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-400 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Play className="w-5 h-5" />
            Start Sorting
          </motion.button>
        ) : (
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleStop}
            className="w-full bg-red-600 hover:bg-red-700 text-white font-medium py-3 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Square className="w-5 h-5" />
            Stop
          </motion.button>
        )}

        <div className="grid grid-cols-2 gap-3">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleReset}
            disabled={isRunning}
            className="bg-gray-600 hover:bg-gray-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </motion.button>
          
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleGenerateArray}
            disabled={isRunning}
            className="bg-orange-600 hover:bg-orange-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Shuffle className="w-4 h-4" />
            Shuffle
          </motion.button>
        </div>
      </div>

      {/* Running Status */}
      {isRunning && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center justify-center gap-2 text-purple-600 dark:text-purple-400"
        >
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          >
            <Clock className="w-4 h-4" />
          </motion.div>
          <span className="text-sm font-medium">Running {algorithmInfo.name}...</span>
        </motion.div>
      )}
    </motion.div>
  );
}
