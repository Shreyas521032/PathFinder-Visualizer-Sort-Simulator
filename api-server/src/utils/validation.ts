// api// api-server/src/utils/validation.ts

export function validateEnvironment(): void {
  const optionalEnvVars = [
    'NODE_ENV',
    'PORT',
    'CORS_ORIGINS'
  ];

  console.log('✅ Environment variables validated successfully');
  console.log('ℹ️  Using OpenStreetMap/Nominatim for geocoding (no API key required)');
  
  // Log optional environment variables if set
  optionalEnvVars.forEach(varName => {
    if (process.env[varName]) {
      console.log(`✓ ${varName}: ${process.env[varName]}`);
    }
  });
}

export function validateCoordinates(lat: number, lng: number): boolean {
  return (
    typeof lat === 'number' &&
    typeof lng === 'number' &&
    lat >= -90 &&
    lat <= 90 &&
    lng >= -180 &&
    lng <= 180 &&
    !isNaN(lat) &&
    !isNaN(lng)
  );
}

export function sanitizeString(input: string, maxLength: number = 1000): string {
  if (typeof input !== 'string') {
    return '';
  }
  
  return input
    .trim()
    .slice(0, maxLength)
    .replace(/[<>]/g, ''); // Basic XSS prevention
}

export function validateSimulationType(type: string): boolean {
  return ['pathfinding', 'sorting'].includes(type);
}

export function validateAlgorithmName(algorithm: string, type: string): boolean {
  const pathfindingAlgorithms = [
    'astar', 'dijkstra', 'bfs', 'dfs', 'greedy', 'bidirectional'
  ];
  
  const sortingAlgorithms = [
    'bubble', 'merge', 'quick', 'heap', 'insertion', 'selection', 'radix'
  ];
  
  if (type === 'pathfinding') {
    return pathfindingAlgorithms.includes(algorithm);
  } else if (type === 'sorting') {
    return sortingAlgorithms.includes(algorithm);
  }
  
  return false;
}
