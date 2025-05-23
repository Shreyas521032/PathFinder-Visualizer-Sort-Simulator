// api-server/src/utils/validation.ts

export function validateEnvironment(): void {
  const requiredEnvVars = [
    'MAPBOX_SECRET_KEY',
  ];

  const missingVars = requiredEnvVars.filter(varName => !process.env[varName]);

  if (missingVars.length > 0) {
    console.error('❌ Missing required environment variables:');
    missingVars.forEach(varName => {
      console.error(`   - ${varName}`);
    });
    console.error('\nPlease set these variables in your .env file or environment.');
    process.exit(1);
  }

  // Validate Mapbox token format
  const mapboxToken = process.env.MAPBOX_SECRET_KEY;
  if (mapboxToken && !mapboxToken.startsWith('sk.')) {
    console.warn('⚠️  Warning: MAPBOX_SECRET_KEY should start with "sk." for secret tokens');
  }

  console.log('✅ Environment variables validated successfully');
}

export function validateMapboxToken(token: string): boolean {
  return token.startsWith('sk.') && token.length > 50;
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
