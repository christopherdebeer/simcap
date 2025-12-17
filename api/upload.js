/**
 * Vercel Blob Upload API Handler
 *
 * Handles server-side upload of session data to Vercel Blob.
 * Requires SIMCAP_UPLOAD_SECRET environment variable for authorization.
 *
 * Client must provide:
 * - x-upload-secret header matching the env var
 * - x-vercel-blob-pathname header with the target path (e.g., 'sessions/filename.json')
 * - JSON content in the request body
 */

import { put } from '@vercel/blob';

export const config = {
  api: {
    bodyParser: false, // Handle raw body ourselves
  },
};

// Helper to read request body as buffer
async function readBody(request) {
  const chunks = [];
  for await (const chunk of request) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

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

  // Get pathname from header
  const pathname = request.headers['x-vercel-blob-pathname'];
  if (!pathname) {
    return response.status(400).json({ error: 'Missing x-vercel-blob-pathname header' });
  }

  // Validate pathname - only allow sessions directory
  if (!pathname.startsWith('sessions/')) {
    return response.status(400).json({ error: 'Invalid upload path: must be in sessions/ directory' });
  }

  // Validate file extension
  if (!pathname.endsWith('.json')) {
    return response.status(400).json({ error: 'Invalid file type: only .json files allowed' });
  }

  try {
    // Read the request body
    const body = await readBody(request);

    // Validate size (10 MB max)
    if (body.length > 10 * 1024 * 1024) {
      return response.status(400).json({ error: 'File too large: maximum 10 MB' });
    }

    // Upload to Vercel Blob
    const blob = await put(pathname, body, {
      access: 'public',
      contentType: 'application/json',
      addRandomSuffix: false, // Use exact filename (timestamp-based)
    });

    console.log('Session uploaded:', blob.pathname, 'Size:', body.length);

    return response.status(200).json({
      url: blob.url,
      pathname: blob.pathname,
      size: body.length,
    });
  } catch (error) {
    console.error('Upload error:', error);
    return response.status(400).json({
      error: error.message || 'Upload failed'
    });
  }
}
