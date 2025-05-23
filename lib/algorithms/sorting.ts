// lib/algorithms/sorting.ts
import { delay } from '@/lib/utils';

export interface SortingStep {
  array: number[];
  comparingIndices: number[];
  swappingIndices: number[];
  sortedIndices: number[];
}

export interface SortingResult {
  steps: SortingStep[];
  comparisons: number;
  swaps: number;
}

// Bubble Sort
export async function bubbleSort(
  array: number[],
  speed: number,
  onStep?: (step: SortingStep) => void
): Promise<SortingResult> {
  const arr = [...array];
  const steps: SortingStep[] = [];
  const sortedIndices: number[] = [];
  let comparisons = 0;
  let swaps = 0;
  
  for (let i = 0; i < arr.length; i++) {
    for (let j = 0; j < arr.length - i - 1; j++) {
      comparisons++;
      
      const step: SortingStep = {
        array: [...arr],
        comparingIndices: [j, j + 1],
        swappingIndices: [],
        sortedIndices: [...sortedIndices],
      };
      
      steps.push(step);
      onStep?.(step);
      await delay(speed);
      
      if (arr[j] > arr[j + 1]) {
        swaps++;
        [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
        
        const swapStep: SortingStep = {
          array: [...arr],
          comparingIndices: [],
          swappingIndices: [j, j + 1],
          sortedIndices: [...sortedIndices],
        };
        
        steps.push(swapStep);
        onStep?.(swapStep);
        await delay(speed);
      }
    }
    sortedIndices.unshift(arr.length - 1 - i);
  }
  
  return { steps, comparisons, swaps };
}

// Merge Sort
export async function mergeSort(
  array: number[],
  speed: number,
  onStep?: (step: SortingStep) => void
): Promise<SortingResult> {
  const arr = [...array];
  const steps: SortingStep[] = [];
  let comparisons = 0;
  let swaps = 0;
  
  async function merge(left: number, mid: number, right: number) {
    const leftArr = arr.slice(left, mid + 1);
    const rightArr = arr.slice(mid + 1, right + 1);
    
    let i = 0, j = 0, k = left;
    
    while (i < leftArr.length && j < rightArr.length) {
      comparisons++;
      
      const step: SortingStep = {
        array: [...arr],
        comparingIndices: [left + i, mid + 1 + j],
        swappingIndices: [],
        sortedIndices: [],
      };
      
      steps.push(step);
      onStep?.(step);
      await delay(speed);
      
      if (leftArr[i] <= rightArr[j]) {
        arr[k] = leftArr[i];
        i++;
      } else {
        arr[k] = rightArr[j];
        j++;
        swaps++;
      }
      k++;
    }
    
    while (i < leftArr.length) {
      arr[k] = leftArr[i];
      i++;
      k++;
    }
    
    while (j < rightArr.length) {
      arr[k] = rightArr[j];
      j++;
      k++;
    }
  }
  
  async function mergeSortHelper(left: number, right: number) {
    if (left < right) {
      const mid = Math.floor((left + right) / 2);
      await mergeSortHelper(left, mid);
      await mergeSortHelper(mid + 1, right);
      await merge(left, mid, right);
    }
  }
  
  await mergeSortHelper(0, arr.length - 1);
  
  const finalStep: SortingStep = {
    array: [...arr],
    comparingIndices: [],
    swappingIndices: [],
    sortedIndices: Array.from({ length: arr.length }, (_, i) => i),
  };
  
  steps.push(finalStep);
  onStep?.(finalStep);
  
  return { steps, comparisons, swaps };
}

// Quick Sort
export async function quickSort(
  array: number[],
  speed: number,
  onStep?: (step: SortingStep) => void
): Promise<SortingResult> {
  const arr = [...array];
  const steps: SortingStep[] = [];
  let comparisons = 0;
  let swaps = 0;
  
  async function partition(low: number, high: number): Promise<number> {
    const pivot = arr[high];
    let i = low - 1;
    
    for (let j = low; j < high; j++) {
      comparisons++;
      
      const step: SortingStep = {
        array: [...arr],
        comparingIndices: [j, high],
        swappingIndices: [],
        sortedIndices: [],
      };
      
      steps.push(step);
      onStep?.(step);
      await delay(speed);
      
      if (arr[j] < pivot) {
        i++;
        if (i !== j) {
          swaps++;
          [arr[i], arr[j]] = [arr[j], arr[i]];
          
          const swapStep: SortingStep = {
            array: [...arr],
            comparingIndices: [],
            swappingIndices: [i, j],
            sortedIndices: [],
          };
          
          steps.push(swapStep);
          onStep?.(swapStep);
          await delay(speed);
        }
      }
    }
    
    if (i + 1 !== high) {
      swaps++;
      [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];
      
      const swapStep: SortingStep = {
        array: [...arr],
        comparingIndices: [],
        swappingIndices: [i + 1, high],
        sortedIndices: [],
      };
      
      steps.push(swapStep);
      onStep?.(swapStep);
      await delay(speed);
    }
    
    return i + 1;
  }
  
  async function quickSortHelper(low: number, high: number) {
    if (low < high) {
      const pi = await partition(low, high);
      await quickSortHelper(low, pi - 1);
      await quickSortHelper(pi + 1, high);
    }
  }
  
  await quickSortHelper(0, arr.length - 1);
  
  const finalStep: SortingStep = {
    array: [...arr],
    comparingIndices: [],
    swappingIndices: [],
    sortedIndices: Array.from({ length: arr.length }, (_, i) => i),
  };
  
  steps.push(finalStep);
  onStep?.(finalStep);
  
  return { steps, comparisons, swaps };
}

// Heap Sort
export async function heapSort(
  array: number[],
  speed: number,
  onStep?: (step: SortingStep) => void
): Promise<SortingResult> {
  const arr = [...array];
  const steps: SortingStep[] = [];
  let comparisons = 0;
  let swaps = 0;
  
  async function heapify(n: number, i: number) {
    let largest = i;
    const left = 2 * i + 1;
    const right = 2 * i + 2;
    
    if (left < n) {
      comparisons++;
      if (arr[left] > arr[largest]) {
        largest = left;
      }
    }
    
    if (right < n) {
      comparisons++;
      if (arr[right] > arr[largest]) {
        largest = right;
      }
    }
    
    if (largest !== i) {
      swaps++;
      [arr[i], arr[largest]] = [arr[largest], arr[i]];
      
      const step: SortingStep = {
        array: [...arr],
        comparingIndices: [],
        swappingIndices: [i, largest],
        sortedIndices: [],
      };
      
      steps.push(step);
      onStep?.(step);
      await delay(speed);
      
      await heapify(n, largest);
    }
  }
  
  // Build max heap
  for (let i = Math.floor(arr.length / 2) - 1; i >= 0; i--) {
    await heapify(arr.length, i);
  }
  
  // Extract elements from heap
  const sortedIndices: number[] = [];
  for (let i = arr.length - 1; i > 0; i--) {
    swaps++;
    [arr[0], arr[i]] = [arr[i], arr[0]];
    
    const step: SortingStep = {
      array: [...arr],
      comparingIndices: [],
      swappingIndices: [0, i],
      sortedIndices: [...sortedIndices],
    };
    
    steps.push(step);
    onStep?.(step);
    await delay(speed);
    
    sortedIndices.unshift(i);
    await heapify(i, 0);
  }
  
  sortedIndices.unshift(0);
  
  const finalStep: SortingStep = {
    array: [...arr],
    comparingIndices: [],
    swappingIndices: [],
    sortedIndices,
  };
  
  steps.push(finalStep);
  onStep?.(finalStep);
  
  return { steps, comparisons, swaps };
}

// Insertion Sort
export async function insertionSort(
  array: number[],
  speed: number,
  onStep?: (step: SortingStep) => void
): Promise<SortingResult> {
  const arr = [...array];
  const steps: SortingStep[] = [];
  const sortedIndices: number[] = [0];
  let comparisons = 0;
  let swaps = 0;
  
  for (let i = 1; i < arr.length; i++) {
    const key = arr[i];
    let j = i - 1;
    
    while (j >= 0) {
      comparisons++;
      
      const step: SortingStep = {
        array: [...arr],
        comparingIndices: [j, j + 1],
        swappingIndices: [],
        sortedIndices: [...sortedIndices],
      };
      
      steps.push(step);
      onStep?.(step);
      await delay(speed);
      
      if (arr[j] <= key) break;
      
      swaps++;
      arr[j + 1] = arr[j];
      
      const swapStep: SortingStep = {
        array: [...arr],
        comparingIndices: [],
        swappingIndices: [j, j + 1],
        sortedIndices: [...sortedIndices],
      };
      
      steps.push(swapStep);
      onStep?.(swapStep);
      await delay(speed);
      
      j--;
    }
    
    arr[j + 1] = key;
    sortedIndices.push(i);
  }
  
  return { steps, comparisons, swaps };
}

// Selection Sort
export async function selectionSort(
  array: number[],
  speed: number,
  onStep?: (step: SortingStep) => void
): Promise<SortingResult> {
  const arr = [...array];
  const steps: SortingStep[] = [];
  const sortedIndices: number[] = [];
  let comparisons = 0;
  let swaps = 0;
  
  for (let i = 0; i < arr.length; i++) {
    let minIndex = i;
    
    for (let j = i + 1; j < arr.length; j++) {
      comparisons++;
      
      const step: SortingStep = {
        array: [...arr],
        comparingIndices: [minIndex, j],
        swappingIndices: [],
        sortedIndices: [...sortedIndices],
      };
      
      steps.push(step);
      onStep?.(step);
      await delay(speed);
      
      if (arr[j] < arr[minIndex]) {
        minIndex = j;
      }
    }
    
    if (minIndex !== i) {
      swaps++;
      [arr[i], arr[minIndex]] = [arr[minIndex], arr[i]];
      
      const swapStep: SortingStep = {
        array: [...arr],
        comparingIndices: [],
        swappingIndices: [i, minIndex],
        sortedIndices: [...sortedIndices],
      };
      
      steps.push(swapStep);
      onStep?.(swapStep);
      await delay(speed);
    }
    
    sortedIndices.push(i);
  }
  
  return { steps, comparisons, swaps };
}

// Radix Sort
export async function radixSort(
  array: number[],
  speed: number,
  onStep?: (step: SortingStep) => void
): Promise<SortingResult> {
  const arr = [...array];
  const steps: SortingStep[] = [];
  let comparisons = 0;
  let swaps = 0;
  
  const max = Math.max(...arr);
  
  for (let exp = 1; Math.floor(max / exp) > 0; exp *= 10) {
    await countingSort(arr, exp, steps, onStep, speed);
    swaps += arr.length; // Each position change counts as a swap
  }
  
  const finalStep: SortingStep = {
    array: [...arr],
    comparingIndices: [],
    swappingIndices: [],
    sortedIndices: Array.from({ length: arr.length }, (_, i) => i),
  };
  
  steps.push(finalStep);
  onStep?.(finalStep);
  
  return { steps, comparisons, swaps };
  
  async function countingSort(
    arr: number[],
    exp: number,
    steps: SortingStep[],
    onStep?: (step: SortingStep) => void,
    speed: number = 100
  ) {
    const output = new Array(arr.length);
    const count = new Array(10).fill(0);
    
    // Count occurrences of each digit
    for (let i = 0; i < arr.length; i++) {
      count[Math.floor(arr[i] / exp) % 10]++;
    }
    
    // Change count[i] to actual position
    for (let i = 1; i < 10; i++) {
      count[i] += count[i - 1];
    }
    
    // Build output array
    for (let i = arr.length - 1; i >= 0; i--) {
      const digit = Math.floor(arr[i] / exp) % 10;
      output[count[digit] - 1] = arr[i];
      count[digit]--;
      
      const step: SortingStep = {
        array: [...arr],
        comparingIndices: [i],
        swappingIndices: [],
        sortedIndices: [],
      };
      
      steps.push(step);
      onStep?.(step);
      await delay(speed);
    }
    
    // Copy output array to arr
    for (let i = 0; i < arr.length; i++) {
      arr[i] = output[i];
    }
  }
}
