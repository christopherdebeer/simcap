/**
 * Sessions Listing API Handler
 *
 * Lists all session files stored in Vercel Blob.
 * Returns JSON manifest compatible with VIZ explorer and session playback.
 *
 * No authentication required - sessions list is public.
 */

import { list } from '@vercel/blob';

interface SessionInfo {
  filename: string;
  pathname: string;
  url: string;
  downloadUrl: string;
  size: number;
  uploadedAt: string;
  timestamp: string;
}

interface SessionsResponse {
  sessions: SessionInfo[];
  count: number;
  generatedAt: string;
}

interface ErrorResponse {
  error: string;
  message?: string;
}

export default async function handler(request: Request): Promise<Response> {
  // Only allow GET requests
  if (request.method !== 'GET') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  try {
    // List all sessions from blob storage
    const { blobs } = await list({
      prefix: 'sessions/',
      limit: 1000, // Max sessions to list
    });

    // Transform blob list into session manifest format
    const sessions: SessionInfo[] = blobs
      .filter(blob => blob.pathname.endsWith('.json'))
      .map(blob => {
        // Extract timestamp from filename (e.g., "sessions/2025-12-15T22_35_15.567Z.json")
        const filename = blob.pathname.replace('sessions/', '');
        const timestamp = filename.replace('.json', '').replace(/_/g, ':');

        return {
          filename,
          pathname: blob.pathname,
          url: blob.url,
          downloadUrl: blob.downloadUrl,
          size: blob.size,
          uploadedAt: blob.uploadedAt.toISOString(),
          timestamp,
        };
      })
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()); // Newest first

    const responseData: SessionsResponse = {
      sessions,
      count: sessions.length,
      generatedAt: new Date().toISOString(),
    };

    return new Response(JSON.stringify(responseData), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        'Cache-Control': 's-maxage=60, stale-while-revalidate=300',
      },
    });
  } catch (error) {
    console.error('Sessions list error:', error);
    const message = error instanceof Error ? error.message : 'Unknown error';
    
    const errorResponse: ErrorResponse = {
      error: 'Failed to list sessions',
      message,
    };

    return new Response(JSON.stringify(errorResponse), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
