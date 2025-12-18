/**
 * Vercel Blob Upload API Handler
 *
 * Handles client-side upload token generation for session data.
 * Uses the two-phase upload protocol from @vercel/blob/client.
 *
 * Requires SIMCAP_UPLOAD_SECRET environment variable for authorization.
 * Client must provide secret in clientPayload.
 */

import { handleUpload, type HandleUploadBody } from '@vercel/blob/client';
import type { UploadClientPayload } from '@api/types';

/**
 * Web API handler for Vercel Edge/Serverless
 * Uses Request/Response instead of VercelRequest/VercelResponse
 */
export default async function handler(request: Request): Promise<Response> {
  // Only allow POST requests
  if (request.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  const serverSecret = process.env.SIMCAP_UPLOAD_SECRET;

  if (!serverSecret) {
    console.error('SIMCAP_UPLOAD_SECRET environment variable not configured');
    return new Response(JSON.stringify({ error: 'Server configuration error' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  try {
    // Parse the request body as HandleUploadBody
    const body = (await request.json()) as HandleUploadBody;

    // Call handleUpload with parsed body
    const jsonResponse = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (pathname: string, clientPayload: string | null, _multipart: boolean) => {
        // Parse and validate the client payload containing auth secret
        let payload: UploadClientPayload = { secret: '' };
        try {
          payload = clientPayload ? JSON.parse(clientPayload) : { secret: '' };
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
      onUploadCompleted: async ({ blob }) => {
        console.log('Session uploaded:', blob.pathname);
      },
    });

    return new Response(JSON.stringify(jsonResponse), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('Upload error:', error);
    // Return 401 for auth errors, 400 for others
    const message = error instanceof Error ? error.message : 'Upload failed';
    const status = message.includes('Unauthorized') ? 401 : 400;
    return new Response(JSON.stringify({ error: message }), {
      status,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}
