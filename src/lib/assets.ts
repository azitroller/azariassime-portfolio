/**
 * Get the correct asset path for both development and production environments
 * Uses Vite's BASE_URL for proper GitHub Pages deployment
 */
export const getAssetPath = (path: string): string => {
  // Remove leading slash if present
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;
  
  // Use Vite's BASE_URL which handles the correct base path
  const fullPath = import.meta.env.BASE_URL + cleanPath;
  
  // Debug logging in production to verify paths
  if (import.meta.env.PROD) {
    console.debug('Asset path resolved:', { original: path, resolved: fullPath });
  }
  
  return fullPath;
};