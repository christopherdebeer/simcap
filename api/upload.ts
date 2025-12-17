/**
 * Vercel Blob Upload API Handler
 *
 * Handles client-side upload token generation for session data.
 * Uses the two-phase upload protocol from @vercel/blob/client.
 *
 * Requires SIMCAP_UPLOAD_SECRET environment variable for authorization.
 * Client must provide x-upload-secret header matching the env var.
 */

import { handleUpload } from '@vercel/blob/client';
import type { VercelRequest, VercelResponse } from '@vercel/node';

export const config = {
  api: {
    bodyParser: false, // Required for handleUpload
  },
};

export default async function handler(
  request: VercelRequest,
  response: VercelResponse
) {
  // Only allow POST requests
  if (request.method !== 'POST') {
    return response.status(405).json({ error: 'Method not allowed' });
  }

  const serverSecret = process.env.SIMCAP_UPLOAD_SECRET;

  if (!serverSecret) {
    console.error('SIMCAP_UPLOAD_SECRET environment variable not configured');
    return response.status(500).json({ error: 'Server configuration error' });
  }

  try {
    // Use the updated handleUpload API
    // Note: Type definitions may not match latest API - using type assertion
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const jsonResponse = await (handleUpload as any)({
      body: request,
      request,
      onBeforeGenerateToken: async (pathname: string, clientPayload?: string) => {
        // Parse and validate the client payload containing auth secret
        let payload: { secret?: string } = {};
        try {
          payload = clientPayload ? JSON.parse(clientPayload) : {};
        } catch {
          throw new Error('Invalid client payload');
        }

        // Validate upload secret from clientPayload
        if (!payload.secret || payload.secret !== serverSecret) {
          throw new Error('Unauthorized: Invalid upload secret');
        }

        // Validate pathname - only allow sessions directory
        if (!pathname.startsWith('sessions/')) {
          throw new Error('Invalid upload path: must be in sessions/ directory');
        }

        // Validate file extension
        if (!pathname.endsWith('.json')) {
          throw new Error('Invalid file type: only .json files allowed');
        }

        return {
          allowedContentTypes: ['application/json'],
          maximumSizeInBytes: 10 * 1024 * 1024, // 10 MB max
          addRandomSuffix: false, // Use exact filename (timestamp-based)
        };
      },
      onUploadCompleted: async ({ blob }: { blob: { pathname: string; size: number } }) => {
        console.log('Session uploaded:', blob.pathname, 'Size:', blob.size);
      },
    });

    return response.status(200).json(jsonResponse);
  } catch (error) {
    console.error('Upload error:', error);
    // Return 401 for auth errors, 400 for others
    const message = error instanceof Error ? error.message : 'Upload failed';
    const status = message.includes('Unauthorized') ? 401 : 400;
    return response.status(status).json({
      error: message
    });
  }
}
