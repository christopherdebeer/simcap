/**
 * SIMCAP API Package
 *
 * Shared types and client for SIMCAP APIs.
 *
 * Usage:
 *   import { apiClient, SessionInfo, UploadResult } from '@api/client';
 *
 *   // List sessions
 *   const { sessions } = await apiClient.listSessions();
 *
 *   // Get combined session + visualization data
 *   const entries = await apiClient.listSessionsWithVisualizations();
 */

// Export all types
export * from './types';

// Export client and utilities
export {
  ApiClient,
  ApiClientError,
  apiClient,
  getUploadSecret,
  setUploadSecret,
  hasUploadSecret,
  uploadToBlob,
  uploadWithRetry,
} from './client';

// Default export is the client instance
export { default } from './client';
