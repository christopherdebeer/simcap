/**
 * GitHub Upload API Proxy
 *
 * Proxies file uploads to GitHub using the Contents API.
 * Keeps the GitHub PAT server-side for security.
 *
 * Environment Variables:
 *   GITHUB_TOKEN - GitHub PAT with repo write access
 *   SIMCAP_UPLOAD_SECRET - Client authentication secret
 *
 * Endpoints:
 *   POST /api/github-upload - Upload a file to GitHub
 *
 * Request Body:
 *   {
 *     secret: string,       // Client auth secret
 *     branch: string,       // Target branch (e.g., 'data')
 *     path: string,         // File path in repo
 *     content: string,      // File content
 *     message: string,      // Commit message
 *     validate?: boolean    // If true, only validate secret (don't upload)
 *   }
 */

// Use Edge Runtime for Web API Response support
export const config = {
  runtime: 'edge',
};

const GITHUB_API_URL = 'https://api.github.com';
const DEFAULT_OWNER = 'christopherdebeer';
const DEFAULT_REPO = 'simcap';

// Allowed branches for upload
const ALLOWED_BRANCHES = ['data', 'images'];

// Allowed path prefixes per branch
const ALLOWED_PATHS: Record<string, string[]> = {
  data: ['GAMBIT/'],
  images: [''], // Allow any path in images branch
};

// CORS headers for all responses
const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
};

interface UploadRequest {
  secret: string;
  branch: string;
  path: string;
  content: string;
  message: string;
  validate?: boolean;
  compressed?: boolean; // If true, content is gzip+base64 encoded
}

interface GitHubContentResponse {
  content?: {
    name: string;
    path: string;
    sha: string;
    html_url: string;
    download_url: string;
  };
  commit: {
    sha: string;
    html_url: string;
  };
}

/**
 * Base64 encode a string (handles Unicode)
 */
function base64Encode(str: string): string {
  const bytes = new TextEncoder().encode(str);
  const binString = Array.from(bytes, (byte) => String.fromCodePoint(byte)).join('');
  return btoa(binString);
}

/**
 * Decompress gzip+base64 encoded content
 * Uses native DecompressionStream API (available in Edge runtime)
 */
async function decompressFromBase64(compressedBase64: string): Promise<string> {
  // Decode base64 to bytes
  const binaryString = atob(compressedBase64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }

  // Decompress using DecompressionStream
  const ds = new DecompressionStream('gzip');
  const writer = ds.writable.getWriter();
  writer.write(bytes);
  writer.close();

  const decompressedChunks: Uint8Array[] = [];
  const reader = ds.readable.getReader();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    decompressedChunks.push(value);
  }

  // Concatenate chunks
  const totalLength = decompressedChunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const decompressed = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of decompressedChunks) {
    decompressed.set(chunk, offset);
    offset += chunk.length;
  }

  // Decode UTF-8
  return new TextDecoder().decode(decompressed);
}

/**
 * Get the SHA of an existing file
 */
async function getFileSha(
  owner: string,
  repo: string,
  path: string,
  branch: string,
  token: string
): Promise<string | null> {
  const url = `${GITHUB_API_URL}/repos/${owner}/${repo}/contents/${path}?ref=${branch}`;

  try {
    const response = await fetch(url, {
      method: 'GET',
      headers: {
        Authorization: `token ${token}`,
        Accept: 'application/vnd.github.v3+json',
      },
    });

    if (response.ok) {
      const data = await response.json();
      return data.sha;
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * Construct raw.githubusercontent.com URL
 */
function getRawUrl(owner: string, repo: string, branch: string, path: string): string {
  return `https://raw.githubusercontent.com/${owner}/${repo}/${branch}/${path}`;
}

/**
 * Validate upload request
 */
function validateRequest(
  body: UploadRequest,
  serverSecret: string
): { valid: boolean; error?: string } {
  // Validate secret
  if (!body.secret || body.secret !== serverSecret) {
    return { valid: false, error: 'Unauthorized: Invalid upload secret' };
  }

  // If validation-only request, we're done
  if (body.validate) {
    return { valid: true };
  }

  // Validate branch
  if (!body.branch || !ALLOWED_BRANCHES.includes(body.branch)) {
    return {
      valid: false,
      error: `Invalid branch: must be one of ${ALLOWED_BRANCHES.join(', ')}`,
    };
  }

  // Validate path
  if (!body.path) {
    return { valid: false, error: 'Missing path' };
  }

  const allowedPrefixes = ALLOWED_PATHS[body.branch];
  if (allowedPrefixes.length > 0) {
    const hasValidPrefix = allowedPrefixes.some((prefix) => body.path.startsWith(prefix));
    if (!hasValidPrefix) {
      return {
        valid: false,
        error: `Invalid path for branch ${body.branch}: must start with ${allowedPrefixes.join(' or ')}`,
      };
    }
  }

  // Validate content
  if (!body.content) {
    return { valid: false, error: 'Missing content' };
  }

  // Validate message
  if (!body.message) {
    return { valid: false, error: 'Missing commit message' };
  }

  return { valid: true };
}

export default async function handler(request: Request): Promise<Response> {
  // Handle CORS preflight
  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 204,
      headers: CORS_HEADERS,
    });
  }

  // Only allow POST
  if (request.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json', ...CORS_HEADERS },
    });
  }

  // Check environment variables
  const serverSecret = process.env.SIMCAP_UPLOAD_SECRET;
  const githubToken = process.env.GITHUB_TOKEN;

  if (!serverSecret) {
    console.error('SIMCAP_UPLOAD_SECRET environment variable not configured');
    return new Response(JSON.stringify({ error: 'Server configuration error' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json', ...CORS_HEADERS },
    });
  }

  if (!githubToken) {
    console.error('GITHUB_TOKEN environment variable not configured');
    return new Response(JSON.stringify({ error: 'Server configuration error' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json', ...CORS_HEADERS },
    });
  }

  try {
    const body: UploadRequest = await request.json();

    // Validate request
    const validation = validateRequest(body, serverSecret);
    if (!validation.valid) {
      const status = validation.error?.includes('Unauthorized') ? 401 : 400;
      return new Response(JSON.stringify({ error: validation.error }), {
        status,
        headers: { 'Content-Type': 'application/json', ...CORS_HEADERS },
      });
    }

    // Validation-only request
    if (body.validate) {
      return new Response(JSON.stringify({ valid: true }), {
        status: 200,
        headers: { 'Content-Type': 'application/json', ...CORS_HEADERS },
      });
    }

    const owner = DEFAULT_OWNER;
    const repo = DEFAULT_REPO;

    // Decompress content if it was compressed by the client
    let actualContent = body.content;
    if (body.compressed) {
      try {
        console.log(`Decompressing content for ${body.path}...`);
        actualContent = await decompressFromBase64(body.content);
        console.log(`Decompressed: ${body.content.length} -> ${actualContent.length} bytes`);
      } catch (decompressError) {
        console.error('Decompression failed:', decompressError);
        return new Response(
          JSON.stringify({ error: 'Failed to decompress content' }),
          {
            status: 400,
            headers: { 'Content-Type': 'application/json', ...CORS_HEADERS },
          }
        );
      }
    }

    // Check if file exists (for update)
    const existingSha = await getFileSha(owner, repo, body.path, body.branch, githubToken);

    // Encode content for GitHub API
    const encodedContent = base64Encode(actualContent);

    // Prepare request
    const requestBody: Record<string, string> = {
      message: body.message,
      content: encodedContent,
      branch: body.branch,
    };

    if (existingSha) {
      requestBody.sha = existingSha;
    }

    // Upload to GitHub
    const url = `${GITHUB_API_URL}/repos/${owner}/${repo}/contents/${body.path}`;
    const response = await fetch(url, {
      method: 'PUT',
      headers: {
        Authorization: `token ${githubToken}`,
        'Content-Type': 'application/json',
        Accept: 'application/vnd.github.v3+json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const error = await response.json();
      console.error('GitHub API error:', error);
      return new Response(
        JSON.stringify({
          error: error.message || `GitHub API error: ${response.status}`,
        }),
        {
          status: response.status >= 500 ? 502 : response.status,
          headers: { 'Content-Type': 'application/json', ...CORS_HEADERS },
        }
      );
    }

    const result: GitHubContentResponse = await response.json();
    const rawUrl = getRawUrl(owner, repo, body.branch, body.path);

    console.log(`File uploaded: ${body.path} to branch ${body.branch}`);

    return new Response(
      JSON.stringify({
        success: true,
        url: rawUrl,
        pathname: body.path,
        branch: body.branch,
        commitSha: result.commit?.sha,
        htmlUrl: result.content?.html_url,
      }),
      {
        status: 200,
        headers: { 'Content-Type': 'application/json', ...CORS_HEADERS },
      }
    );
  } catch (error) {
    console.error('Upload error:', error);
    const message = error instanceof Error ? error.message : 'Upload failed';
    return new Response(JSON.stringify({ error: message }), {
      status: 500,
      headers: { 'Content-Type': 'application/json', ...CORS_HEADERS },
    });
  }
}
