// lib/sorting-runner.ts
import { SortingAlgorithm } from '@/types';
import { 
  bubbleSort,
  mergeSort,
  quickSort,
  heapSort,
  insertionSort,
  selectionSort,
  radixSort,
  SortingStep,
  SortingResult
} from '@/lib/algorithms/sorting';

export async function runSortingAlgorithm(
  algorithm: SortingAlgorithm,
  array: number[],
  speed: number,
  onStep?: (step: SortingStep) => void
): Promise<SortingResult> {
  switch (algorithm) {
    case 'bubble':
      return bubbleSort(array, speed, onStep);
    case 'merge':
      return mergeSort(array, speed, onStep);
    case 'quick':
      return quickSort(array, speed, onStep);
    case 'heap':
      return heapSort(array, speed, onStep);
    case 'insertion':
      return insertionSort(array, speed, onStep);
    case 'selection':
      return selectionSort(array, speed, onStep);
    case 'radix':
      return radixSort(array, speed, onStep);
    default:
      throw new Error(`Unknown algorithm: ${algorithm}`);
  }
}
