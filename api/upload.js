/**
 * Vercel Blob Upload API Handler
 *
 * Handles client-side upload token generation for session data.
 * Requires SIMCAP_UPLOAD_SECRET environment variable for authorization.
 *
 * Client must provide x-upload-secret header matching the env var.
 */

import { handleUpload } from '@vercel/blob/client';

export const config = {
  api: {
    bodyParser: false, // Required for handleUpload
  },
};

export default async function handler(request, response) {
  // Only allow POST requests
  if (request.method !== 'POST') {
    return response.status(405).json({ error: 'Method not allowed' });
  }

  // Validate upload secret
  const clientSecret = request.headers['x-upload-secret'];
  const serverSecret = process.env.SIMCAP_UPLOAD_SECRET;

  if (!serverSecret) {
    console.error('SIMCAP_UPLOAD_SECRET environment variable not configured');
    return response.status(500).json({ error: 'Server configuration error' });
  }

  if (!clientSecret || clientSecret !== serverSecret) {
    return response.status(401).json({ error: 'Unauthorized: Invalid upload secret' });
  }

  try {
    const jsonResponse = await handleUpload({
      request,
      onBeforeGenerateToken: async (pathname, clientPayload) => {
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
          validUntil: Date.now() + 60 * 1000, // Token valid for 60 seconds
          addRandomSuffix: false, // Use exact filename (timestamp-based)
        };
      },
      onUploadCompleted: async ({ blob, tokenPayload }) => {
        // Called after successful upload
        // Could update a database or trigger visualization generation here
        console.log('Session uploaded:', blob.pathname, 'Size:', blob.size);

        // Future: trigger webhook or update session index
      },
    });

    return response.status(200).json(jsonResponse);
  } catch (error) {
    console.error('Upload error:', error);
    return response.status(400).json({
      error: error.message || 'Upload failed'
    });
  }
}
