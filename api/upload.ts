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
import type { VercelRequest, VercelResponse } from '@vercel/node';

/** Client payload sent with upload token request */
interface UploadClientPayload {
  secret: string;
}

/**
 * Read request body as JSON
 * Handles both VercelRequest (Node.js) and Web API Request
 */
async function readBodyAsJson(request: VercelRequest | Request): Promise<HandleUploadBody> {
  // Check if it's a Web API Request (has .json method)
  if ('json' in request && typeof request.json === 'function') {
    return (await (request as Request).json()) as HandleUploadBody;
  }

  // It's a VercelRequest - read from body stream
  const vercelReq = request as VercelRequest;

  // If body is already parsed (e.g., by body-parser), return it
  if (vercelReq.body && typeof vercelReq.body === 'object') {
    return vercelReq.body as HandleUploadBody;
  }

  // Read body from stream
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    vercelReq.on('data', (chunk: Buffer) => chunks.push(chunk));
    vercelReq.on('end', () => {
      try {
        const body = Buffer.concat(chunks).toString();
        resolve(JSON.parse(body) as HandleUploadBody);
      } catch (e) {
        reject(new Error('Failed to parse request body'));
      }
    });
    vercelReq.on('error', reject);
  });
}

/**
 * Vercel Serverless Function handler
 */
export default async function handler(
  request: VercelRequest,
  response: VercelResponse
): Promise<void> {
  // Only allow POST requests
  if (request.method !== 'POST') {
    response.status(405).json({ error: 'Method not allowed' });
    return;
  }

  const serverSecret = process.env.SIMCAP_UPLOAD_SECRET;

  if (!serverSecret) {
    console.error('SIMCAP_UPLOAD_SECRET environment variable not configured');
    response.status(500).json({ error: 'Server configuration error' });
    return;
  }

  try {
    // Parse the request body
    const body = await readBodyAsJson(request);

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

    response.status(200).json(jsonResponse);
  } catch (error) {
    console.error('Upload error:', error);
    // Return 401 for auth errors, 400 for others
    const message = error instanceof Error ? error.message : 'Upload failed';
    const status = message.includes('Unauthorized') ? 401 : 400;
    response.status(status).json({ error: message });
  }
}
