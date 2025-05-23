// api-server/src/routes/simulations.ts
import { Router, Request, Response } from 'express';
import { v4 as uuidv4 } from 'uuid';

const router = Router();

// In-memory storage for simulations (in production, use a database)
interface Simulation {
  id: string;
  type: 'pathfinding' | 'sorting';
  algorithm: string;
  startTime: Date;
  endTime: Date;
  duration: number;
  steps: number;
  metadata: any;
  userAgent?: string;
  ip?: string;
}

const simulations: Simulation[] = [];

// Save simulation data
router.post('/save', (req: Request, res: Response) => {
  try {
    const {
      type,
      algorithm,
      startTime,
      endTime,
      duration,
      steps,
      metadata,
    } = req.body;

    // Validation
    if (!type || !algorithm || !startTime || !endTime || typeof duration !== 'number' || typeof steps !== 'number') {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: type, algorithm, startTime, endTime, duration, steps',
      });
    }

    if (!['pathfinding', 'sorting'].includes(type)) {
      return res.status(400).json({
        success: false,
        error: 'Type must be either "pathfinding" or "sorting"',
      });
    }

    if (duration < 0 || steps < 0) {
      return res.status(400).json({
        success: false,
        error: 'Duration and steps must be non-negative numbers',
      });
    }

    // Create simulation record
    const simulation: Simulation = {
      id: uuidv4(),
      type,
      algorithm,
      startTime: new Date(startTime),
      endTime: new Date(endTime),
      duration,
      steps,
      metadata: metadata || {},
      userAgent: req.get('User-Agent'),
      ip: req.ip || req.connection.remoteAddress,
    };

    // Add to storage
    simulations.push(simulation);

    // Keep only last 1000 simulations to prevent memory issues
    if (simulations.length > 1000) {
      simulations.splice(0, simulations.length - 1000);
    }

    res.status(201).json({
      success: true,
      data: {
        id: simulation.id,
        saved: true,
      },
    });

  } catch (error) {
    console.error('Save simulation error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to save simulation data',
    });
  }
});

// Get simulations with filtering and pagination
router.get('/', (req: Request, res: Response) => {
  try {
    const {
      type,
      algorithm,
      limit = '50',
      offset = '0',
      sortBy = 'endTime',
      sortOrder = 'desc',
    } = req.query;

    let filteredSimulations = [...simulations];

    // Apply filters
    if (type && ['pathfinding', 'sorting'].includes(type as string)) {
      filteredSimulations = filteredSimulations.filter(sim => sim.type === type);
    }

    if (algorithm) {
      filteredSimulations = filteredSimulations.filter(sim => 
        sim.algorithm.toLowerCase().includes((algorithm as string).toLowerCase())
      );
    }

    // Apply sorting
    const validSortFields = ['endTime', 'duration', 'steps', 'algorithm'];
    const sortField = validSortFields.includes(sortBy as string) ? sortBy as keyof Simulation : 'endTime';
    const order = sortOrder === 'asc' ? 1 : -1;

    filteredSimulations.sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];
      
      if (aVal instanceof Date && bVal instanceof Date) {
        return (aVal.getTime() - bVal.getTime()) * order;
      }
      
      if (typeof aVal === 'number' && typeof bVal === 'number') {
        return (aVal - bVal) * order;
      }
      
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return aVal.localeCompare(bVal) * order;
      }
      
      return 0;
    });

    // Apply pagination
    const limitNum = Math.min(parseInt(limit as string) || 50, 100);
    const offsetNum = Math.max(parseInt(offset as string) || 0, 0);
    const paginatedSimulations = filteredSimulations.slice(offsetNum, offsetNum + limitNum);

    // Remove sensitive data
    const publicSimulations = paginatedSimulations.map(sim => ({
      id: sim.id,
      type: sim.type,
      algorithm: sim.algorithm,
      startTime: sim.startTime,
      endTime: sim.endTime,
      duration: sim.duration,
      steps: sim.steps,
      metadata: sim.metadata,
    }));

    res.json({
      success: true,
      data: {
        simulations: publicSimulations,
        pagination: {
          total: filteredSimulations.length,
          limit: limitNum,
          offset: offsetNum,
          hasMore: offsetNum + limitNum < filteredSimulations.length,
        },
        filters: {
          type: type || null,
          algorithm: algorithm || null,
        },
        sorting: {
          sortBy: sortField,
          sortOrder,
        },
      },
    });

  } catch (error) {
    console.error('Get simulations error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve simulations',
    });
  }
});

// Get simulation statistics
router.get('/stats', (req: Request, res: Response) => {
  try {
    const { type } = req.query;

    let targetSimulations = simulations;
    if (type && ['pathfinding', 'sorting'].includes(type as string)) {
      targetSimulations = simulations.filter(sim => sim.type === type);
    }

    if (targetSimulations.length === 0) {
      return res.json({
        success: true,
        data: {
          totalSimulations: 0,
          averageDuration: 0,
          averageSteps: 0,
          algorithmDistribution: {},
          typeDistribution: {},
        },
      });
    }

    // Calculate statistics
    const totalSimulations = targetSimulations.length;
    const totalDuration = targetSimulations.reduce((sum, sim) => sum + sim.duration, 0);
    const totalSteps = targetSimulations.reduce((sum, sim) => sum + sim.steps, 0);
    
    const averageDuration = totalDuration / totalSimulations;
    const averageSteps = totalSteps / totalSimulations;

    // Algorithm distribution
    const algorithmDistribution: Record<string, number> = {};
    targetSimulations.forEach(sim => {
      algorithmDistribution[sim.algorithm] = (algorithmDistribution[sim.algorithm] || 0) + 1;
    });

    // Type distribution
    const typeDistribution: Record<string, number> = {};
    simulations.forEach(sim => {
      typeDistribution[sim.type] = (typeDistribution[sim.type] || 0) + 1;
    });

    // Most used algorithms
    const mostUsedAlgorithms = Object.entries(algorithmDistribution)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([algorithm, count]) => ({ algorithm, count }));

    // Performance insights
    const algorithmPerformance: Record<string, {
      averageDuration: number;
      averageSteps: number;
      count: number;
    }> = {};

    Object.keys(algorithmDistribution).forEach(algorithm => {
      const algorithmSims = targetSimulations.filter(sim => sim.algorithm === algorithm);
      const avgDuration = algorithmSims.reduce((sum, sim) => sum + sim.duration, 0) / algorithmSims.length;
      const avgSteps = algorithmSims.reduce((sum, sim) => sum + sim.steps, 0) / algorithmSims.length;
      
      algorithmPerformance[algorithm] = {
        averageDuration: Math.round(avgDuration),
        averageSteps: Math.round(avgSteps),
        count: algorithmSims.length,
      };
    });

    res.json({
      success: true,
      data: {
        totalSimulations,
        averageDuration: Math.round(averageDuration),
        averageSteps: Math.round(averageSteps),
        algorithmDistribution,
        typeDistribution,
        mostUsedAlgorithms,
        algorithmPerformance,
        timeRange: {
          earliest: targetSimulations.length > 0 
            ? Math.min(...targetSimulations.map(sim => sim.startTime.getTime()))
            : null,
          latest: targetSimulations.length > 0 
            ? Math.max(...targetSimulations.map(sim => sim.endTime.getTime()))
            : null,
        },
      },
    });

  } catch (error) {
    console.error('Get stats error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve statistics',
    });
  }
});

// Delete simulation by ID
router.delete('/:id', (req: Request, res: Response) => {
  try {
    const { id } = req.params;

    const index = simulations.findIndex(sim => sim.id === id);
    if (index === -1) {
      return res.status(404).json({
        success: false,
        error: 'Simulation not found',
      });
    }

    simulations.splice(index, 1);

    res.json({
      success: true,
      data: {
        deleted: true,
        id,
      },
    });

  } catch (error) {
    console.error('Delete simulation error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to delete simulation',
    });
  }
});

// Clear all simulations (admin endpoint)
router.delete('/', (req: Request, res: Response) => {
  try {
    const deletedCount = simulations.length;
    simulations.length = 0;

    res.json({
      success: true,
      data: {
        deletedCount,
        message: 'All simulations cleared',
      },
    });

  } catch (error) {
    console.error('Clear simulations error:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to clear simulations',
    });
  }
});

export default router;
