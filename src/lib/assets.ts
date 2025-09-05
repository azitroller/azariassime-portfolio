/**
 * Get the correct asset path for both development and production environments
 * In production (GitHub Pages), assets need the base path prefix
 */
export const getAssetPath = (path: string): string => {
  // Remove leading slash if present
  const cleanPath = path.startsWith('/') ? path.slice(1) : path;
  
  // In production, use the base path
  if (import.meta.env.PROD) {
    return `/azariassime-portfolio/${cleanPath}`;
  }
  
  // In development, use the path as-is
  return `/${cleanPath}`;
};