#!/usr/bin/env npx tsx
/**
 * Markdown to HTML Compiler for SIMCAP Documentation
 *
 * Finds all markdown files, converts to static HTML with consistent
 * header/footer and simcap styling, and generates an auto-indexed docs page.
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync, statSync, copyFileSync } from 'fs';
import { join, dirname, relative, basename, extname } from 'path';
import { marked, Renderer } from 'marked';
import { fileURLToPath } from 'url';
import { execSync } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const ROOT_DIR = join(__dirname, '..');
const OUTPUT_DIR = join(ROOT_DIR, 'dist', 'docs');

// Directories to exclude from markdown search
const EXCLUDE_DIRS = [
  'node_modules',
  'dist',
  '.git',
  '.worktrees',
  'data',
  'images',
  '.claude'
];

// Markdown file structure for organizing in index
interface DocFile {
  path: string;        // Original path relative to root
  htmlPath: string;    // Output HTML path relative to docs/
  title: string;       // Extracted title from file
  directory: string;   // Parent directory for grouping
  content: string;     // Raw markdown content (without frontmatter)
  createdAt: number;   // First git commit timestamp (ms since epoch)
  lastModified: number; // Last git commit timestamp (ms since epoch)
  wordCount: number;   // Approximate word count
  excerpt: string;     // First paragraph or 200 chars
  dateSource: 'frontmatter' | 'inline' | 'git' | 'none'; // Source of date info
}

// Frontmatter data structure
interface Frontmatter {
  created?: string;    // Created date (ISO or YYYY-MM-DD)
  updated?: string;    // Updated date (ISO or YYYY-MM-DD)
  date?: string;       // Generic date (used as created if no created field)
  title?: string;      // Optional title override
  [key: string]: unknown;
}

// Searchable document index entry for JSON export
interface DocIndexEntry {
  path: string;
  htmlPath: string;
  title: string;
  directory: string;
  createdAt: number;
  lastModified: number;
  wordCount: number;
  excerpt: string;
  dateSource: 'frontmatter' | 'inline' | 'git' | 'none';
}

// Backlink information
interface Backlink {
  htmlPath: string;    // Path to the linking document
  title: string;       // Title of the linking document
}

/**
 * Recursively find all markdown files in a directory
 */
function findMarkdownFiles(dir: string, baseDir: string = dir): string[] {
  const files: string[] = [];

  if (!existsSync(dir)) return files;

  const entries = readdirSync(dir);

  for (const entry of entries) {
    const fullPath = join(dir, entry);
    const relativePath = relative(baseDir, fullPath);

    // Skip excluded directories
    if (EXCLUDE_DIRS.some(excluded => relativePath.startsWith(excluded) || entry === excluded)) {
      continue;
    }

    const stat = statSync(fullPath);

    if (stat.isDirectory()) {
      files.push(...findMarkdownFiles(fullPath, baseDir));
    } else if (entry.endsWith('.md')) {
      files.push(fullPath);
    }
  }

  return files;
}

/**
 * Extract title from markdown content (first H1 or filename)
 */
function extractTitle(content: string, filepath: string): string {
  // Try to find first H1
  const h1Match = content.match(/^#\s+(.+)$/m);
  if (h1Match) {
    return h1Match[1].trim();
  }

  // Fall back to filename
  const name = basename(filepath, '.md');
  return name.replace(/[-_]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/**
 * Extract word count from markdown content
 */
function extractWordCount(content: string): number {
  // Remove code blocks
  const withoutCodeBlocks = content.replace(/```[\s\S]*?```/g, '');
  // Remove inline code
  const withoutInlineCode = withoutCodeBlocks.replace(/`[^`]+`/g, '');
  // Split by whitespace and count
  const words = withoutInlineCode.trim().split(/\s+/);
  return words.filter(w => w.length > 0).length;
}

/**
 * Extract excerpt from markdown content (first paragraph or 200 chars)
 */
function extractExcerpt(content: string): string {
  // Remove title line if present
  const withoutTitle = content.replace(/^#\s+.+$/m, '').trim();

  // Find first paragraph (text before double newline or end)
  const paragraphMatch = withoutTitle.match(/^[\s\S]*?(?:\n\n|$)/);
  let excerpt = paragraphMatch ? paragraphMatch[0].trim() : withoutTitle;

  // Remove markdown formatting
  excerpt = excerpt
    .replace(/```[\s\S]*?```/g, '') // Remove code blocks
    .replace(/`[^`]+`/g, '')        // Remove inline code
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Convert links to text
    .replace(/[*_~#]/g, '')         // Remove formatting chars
    .trim();

  // Truncate to 200 chars with ellipsis
  if (excerpt.length > 200) {
    excerpt = excerpt.substring(0, 200).trim() + '...';
  }

  return excerpt || 'No description available';
}

/**
 * Parse YAML frontmatter from markdown content
 * Returns frontmatter data and content without frontmatter
 */
function parseFrontmatter(content: string): { frontmatter: Frontmatter; content: string } {
  // Check if content starts with frontmatter delimiter
  if (!content.startsWith('---')) {
    return { frontmatter: {}, content };
  }

  // Find the closing delimiter
  const endMatch = content.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/);
  if (!endMatch) {
    return { frontmatter: {}, content };
  }

  const yamlContent = endMatch[1];
  const restContent = content.slice(endMatch[0].length);

  // Simple YAML parser for key: value pairs
  const frontmatter: Frontmatter = {};
  const lines = yamlContent.split('\n');

  for (const line of lines) {
    const match = line.match(/^(\w+):\s*(.+)$/);
    if (match) {
      const key = match[1].toLowerCase();
      let value = match[2].trim();
      // Remove quotes if present
      if ((value.startsWith('"') && value.endsWith('"')) ||
          (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }
      frontmatter[key] = value;
    }
  }

  return { frontmatter, content: restContent };
}

/**
 * Extract inline date from markdown content
 * Looks for patterns like:
 * - **Date:** 2025-12-12
 * - *Date:* 2025-12-12
 * - Date: 2025-12-12
 * - Created: 2025-12-12
 * - Updated: 2025-12-12
 */
function extractInlineDates(content: string): { created?: number; updated?: number } {
  const result: { created?: number; updated?: number } = {};

  // Patterns for date extraction (with optional markdown formatting)
  const datePatterns = [
    // **Date:** YYYY-MM-DD or *Date:* YYYY-MM-DD or Date: YYYY-MM-DD
    /\*{0,2}(?:Date|Created|Written|Published)\*{0,2}:\s*(\d{4}-\d{2}-\d{2})/i,
    // **Updated:** YYYY-MM-DD
    /\*{0,2}(?:Updated|Modified|Revised|Last Updated|Last Modified)\*{0,2}:\s*(\d{4}-\d{2}-\d{2})/i,
  ];

  // Extract created date
  const createdMatch = content.match(datePatterns[0]);
  if (createdMatch) {
    const date = new Date(createdMatch[1]);
    if (!isNaN(date.getTime())) {
      result.created = date.getTime();
    }
  }

  // Extract updated date
  const updatedMatch = content.match(datePatterns[1]);
  if (updatedMatch) {
    const date = new Date(updatedMatch[1]);
    if (!isNaN(date.getTime())) {
      result.updated = date.getTime();
    }
  }

  return result;
}

/**
 * Parse a date string to timestamp
 * Supports ISO dates and YYYY-MM-DD format
 */
function parseDate(dateStr: string | undefined): number | undefined {
  if (!dateStr) return undefined;

  // Try parsing as-is (handles ISO and YYYY-MM-DD)
  const date = new Date(dateStr);
  if (!isNaN(date.getTime())) {
    return date.getTime();
  }

  return undefined;
}

/**
 * Get document dates with precedence: frontmatter > inline > git
 * Returns dates and the source they came from
 */
function getDocumentDates(
  rawContent: string,
  frontmatter: Frontmatter,
  filepath: string
): { createdAt: number; lastModified: number; dateSource: 'frontmatter' | 'inline' | 'git' | 'none' } {
  // 1. Try frontmatter dates first
  const fmCreated = parseDate(frontmatter.created as string) || parseDate(frontmatter.date as string);
  const fmUpdated = parseDate(frontmatter.updated as string);

  if (fmCreated || fmUpdated) {
    return {
      createdAt: fmCreated || fmUpdated || 0,
      lastModified: fmUpdated || fmCreated || 0,
      dateSource: 'frontmatter'
    };
  }

  // 2. Try inline dates from content
  const inlineDates = extractInlineDates(rawContent);
  if (inlineDates.created || inlineDates.updated) {
    return {
      createdAt: inlineDates.created || inlineDates.updated || 0,
      lastModified: inlineDates.updated || inlineDates.created || 0,
      dateSource: 'inline'
    };
  }

  // 3. Fall back to git history
  const gitDates = getGitDates(filepath);
  if (gitDates.createdAt || gitDates.lastModified) {
    return {
      ...gitDates,
      dateSource: 'git'
    };
  }

  // 4. No dates available
  return { createdAt: 0, lastModified: 0, dateSource: 'none' };
}

/**
 * Get git dates for a file (created and last modified)
 * Returns timestamps in ms since epoch for consistency with search index
 */
function getGitDates(filepath: string): { createdAt: number; lastModified: number } {
  try {
    // Get last modified date (most recent commit)
    const lastModifiedOutput = execSync(
      `git log -1 --format="%aI" -- "${filepath}"`,
      { cwd: ROOT_DIR, encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] }
    ).trim();

    // Get created date (first commit that added the file)
    const createdOutput = execSync(
      `git log --follow --format="%aI" --diff-filter=A -- "${filepath}"`,
      { cwd: ROOT_DIR, encoding: 'utf-8', stdio: ['pipe', 'pipe', 'pipe'] }
    ).trim();

    return {
      createdAt: createdOutput ? new Date(createdOutput).getTime() : 0,
      lastModified: lastModifiedOutput ? new Date(lastModifiedOutput).getTime() : 0
    };
  } catch {
    // File might not be tracked by git yet
    return { createdAt: 0, lastModified: 0 };
  }
}

/**
 * Format timestamp for display
 */
function formatDate(timestamp: number): string {
  if (!timestamp) return 'Unknown';
  return new Date(timestamp).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  });
}

/**
 * Get friendly directory name for display
 */
function getFriendlyDirName(dir: string): string {
  const dirMap: Record<string, string> = {
    '': 'Project Root',
    'docs': 'Documentation',
    'docs/design': 'Design & Research',
    'docs/technical': 'Technical Notes',
    'docs/procedures': 'Procedures & Guides',
    'apps/gambit': 'GAMBIT App',
    'apps/gambit/modules': 'GAMBIT Modules',
    'apps/gambit/analysis': 'GAMBIT Analysis',
    'apps/loader': 'Firmware Loader',
    'ml': 'ML Pipeline',
    'src/device/GAMBIT': 'GAMBIT Firmware',
    'src/device/MOUSE': 'MOUSE Firmware',
    'src/device/KEYBOARD': 'KEYBOARD Firmware',
    'src/device/BAE': 'BAE Firmware',
    'src/web/JOYPAD': 'JOYPAD Research',
    'src/web/FFO$$': 'FFO$$ Research'
  };

  return dirMap[dir] || dir.split('/').pop()?.toUpperCase() || dir;
}

/**
 * Generate breadcrumb navigation
 */
function generateBreadcrumb(relativePath: string): string {
  const parts = relativePath.split('/');
  const breadcrumbs: string[] = ['<a href="/docs/">Docs</a>'];

  let currentPath = '';
  for (let i = 0; i < parts.length - 1; i++) {
    currentPath += (currentPath ? '/' : '') + parts[i];
    breadcrumbs.push(`<a href="/docs/${currentPath}/">${parts[i]}</a>`);
  }

  return breadcrumbs.join(' / ') + ' / <span>' + parts[parts.length - 1].replace('.html', '') + '</span>';
}

/**
 * Generate HTML page from markdown
 */
function generateHtmlPage(title: string, htmlContent: string, relativePath: string, doc: DocFile, backlinks: Backlink[] = []): string {
  const breadcrumb = generateBreadcrumb(relativePath);
  const backlinksHtml = generateBacklinksHtml(backlinks);

  const metaHtml = `
        <div class="doc-meta">
            ${doc.createdAt ? `<span class="meta-item"><span class="meta-label">Created:</span> ${formatDate(doc.createdAt)}</span>` : ''}
            ${doc.lastModified ? `<span class="meta-item"><span class="meta-label">Updated:</span> ${formatDate(doc.lastModified)}</span>` : ''}
        </div>`;

  return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title} · SIMCAP Docs</title>
    <link rel="icon" href="/favicon.ico">
    <link rel="stylesheet" href="/docs/docs.css">
</head>
<body>
    <header class="doc-header">
        <nav class="breadcrumb">${breadcrumb}</nav>
        <a href="/" class="home-link">SIMCAP</a>
    </header>

    <main class="doc-content">
        <article class="markdown-body">
${htmlContent}
        </article>
${metaHtml}
    </main>
${backlinksHtml}
    <footer class="doc-footer">
        <span>SIMCAP Documentation</span>
        <div class="footer-links">
            <a href="/">Home</a>
            <a href="/docs/">Docs Index</a>
            <a href="https://github.com/christopherdebeer/simcap">GitHub</a>
        </div>
    </footer>
</body>
</html>`;
}

/**
 * Generate the docs index page
 */
function generateIndexPage(docs: DocFile[]): string {
  // Calculate statistics
  const totalWords = docs.reduce((sum, doc) => sum + doc.wordCount, 0);
  const directories = new Set(docs.map(doc => doc.directory)).size;

  return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documentation · SIMCAP</title>
    <link rel="icon" href="/favicon.ico">
    <link rel="stylesheet" href="/docs/docs.css">
    <script type="module" src="/docs/docs-viewer.js"></script>
</head>
<body>
    <header class="doc-header index-header">
        <a href="/" class="home-link">SIMCAP</a>
        <h1 class="index-title">Documentation</h1>
        <p class="index-subtitle">Interactive documentation with fuzzy search and filtering</p>
    </header>

    <main class="index-content">
        <div id="docs-viewer-container"></div>
    </main>

    <footer class="doc-footer">
        <span>SIMCAP Documentation</span>
        <div class="footer-links">
            <a href="/">Home</a>
            <a href="https://github.com/christopherdebeer/simcap">GitHub</a>
        </div>
    </footer>
</body>
</html>`;
}

/**
 * Generate docs.css stylesheet
 */
function generateDocsStylesheet(): string {
  return `/**
 * SIMCAP Documentation Styles
 * Brutalist scientific aesthetic matching main site
 */

:root {
    --bg: #fafafa;
    --bg-surface: #fff;
    --fg: #0a0a0a;
    --fg-muted: #666;
    --border: #ddd;
    --accent: #0a0a0a;
    --code-bg: #f5f5f5;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg: #0a0a0a;
        --bg-surface: #111;
        --fg: #fafafa;
        --fg-muted: #888;
        --border: #333;
        --accent: #fafafa;
        --code-bg: #1a1a1a;
    }
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html {
    font-size: 16px;
}

body {
    font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace;
    background: var(--bg);
    color: var(--fg);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    line-height: 1.6;
}

a {
    color: inherit;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
    text-underline-offset: 2px;
}

/* Header */
.doc-header {
    border-bottom: 1px solid var(--border);
    padding: 1rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}

.doc-header.index-header {
    flex-direction: column;
    align-items: flex-start;
    padding: 2rem 1.5rem;
}

.breadcrumb {
    font-size: 0.75rem;
    color: var(--fg-muted);
}

.breadcrumb a:hover {
    color: var(--fg);
}

.breadcrumb span {
    color: var(--fg);
}

.home-link {
    font-weight: 700;
    font-size: 0.875rem;
    letter-spacing: -0.02em;
}

.index-title {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin-top: 0.5rem;
}

.index-subtitle {
    font-size: 0.875rem;
    color: var(--fg-muted);
    margin-top: 0.25rem;
}

.index-stats {
    display: flex;
    gap: 1.5rem;
    margin-top: 1rem;
    font-size: 0.75rem;
    color: var(--fg-muted);
}

/* Main content */
.doc-content {
    flex: 1;
    padding: 2rem 1.5rem;
    max-width: 900px;
}

.index-content {
    flex: 1;
    padding: 2rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 2rem;
}

/* Doc sections */
.doc-section {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.section-header {
    font-size: 0.625rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--fg-muted);
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

.doc-links {
    display: flex;
    flex-direction: column;
}

.doc-link {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.5rem 0;
    text-decoration: none;
}

.doc-link:hover {
    background: var(--fg);
    color: var(--bg);
    margin: 0 -0.5rem;
    padding: 0.5rem;
    text-decoration: none;
}

.doc-title {
    font-weight: 500;
}

.doc-path {
    font-size: 0.75rem;
    color: var(--fg-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 300px;
}

.doc-link:hover .doc-path {
    color: var(--bg);
    opacity: 0.7;
}

/* Markdown body */
.markdown-body {
    line-height: 1.7;
}

.markdown-body h1,
.markdown-body h2,
.markdown-body h3,
.markdown-body h4,
.markdown-body h5,
.markdown-body h6 {
    font-weight: 600;
    letter-spacing: -0.02em;
    margin-top: 2rem;
    margin-bottom: 0.75rem;
}

.markdown-body h1 {
    font-size: 1.75rem;
    margin-top: 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

.markdown-body h2 {
    font-size: 1.375rem;
}

.markdown-body h3 {
    font-size: 1.125rem;
}

.markdown-body h4 {
    font-size: 1rem;
}

.markdown-body p {
    margin-bottom: 1rem;
}

.markdown-body ul,
.markdown-body ol {
    margin-bottom: 1rem;
    padding-left: 1.5rem;
}

.markdown-body li {
    margin-bottom: 0.25rem;
}

.markdown-body code {
    font-family: inherit;
    background: var(--code-bg);
    padding: 0.125rem 0.375rem;
    font-size: 0.875em;
}

.markdown-body pre {
    background: var(--code-bg);
    padding: 1rem;
    overflow-x: auto;
    margin-bottom: 1rem;
    border: 1px solid var(--border);
}

.markdown-body pre code {
    background: none;
    padding: 0;
}

.markdown-body blockquote {
    border-left: 3px solid var(--border);
    padding-left: 1rem;
    color: var(--fg-muted);
    margin-bottom: 1rem;
}

.markdown-body table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 1rem;
    font-size: 0.875rem;
}

.markdown-body th,
.markdown-body td {
    border: 1px solid var(--border);
    padding: 0.5rem;
    text-align: left;
}

.markdown-body th {
    background: var(--code-bg);
    font-weight: 600;
}

.markdown-body img {
    max-width: 100%;
    height: auto;
}

.markdown-body hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

.markdown-body a {
    text-decoration: underline;
    text-underline-offset: 2px;
}

/* Document metadata */
.doc-meta {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 2rem;
    font-size: 0.75rem;
    color: var(--fg-muted);
}

.meta-item {
    display: flex;
    gap: 0.5rem;
}

.meta-label {
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 0.625rem;
}

/* Backlinks */
.backlinks {
    border-top: 1px solid var(--border);
    padding: 1.5rem;
    max-width: 900px;
}

.backlinks-header {
    font-size: 0.625rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--fg-muted);
    margin-bottom: 0.75rem;
}

.backlinks-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.backlinks-list a {
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    background: var(--code-bg);
    border: 1px solid var(--border);
    text-decoration: none;
    transition: all 0.15s ease;
}

.backlinks-list a:hover {
    background: var(--fg);
    color: var(--bg);
    border-color: var(--fg);
    text-decoration: none;
}

/* Footer */
.doc-footer {
    border-top: 1px solid var(--border);
    padding: 1.5rem;
    font-size: 0.75rem;
    color: var(--fg-muted);
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 0.5rem 2rem;
}

.footer-links {
    display: flex;
    gap: 1.5rem;
}

/* Responsive */
@media (min-width: 640px) {
    .doc-header.index-header {
        padding: 3rem 2.5rem;
    }

    .index-title {
        font-size: 2.5rem;
    }

    .doc-content,
    .index-content {
        padding: 3rem 2.5rem;
    }

    .doc-footer {
        padding: 2rem 2.5rem;
    }
}

@media (min-width: 1024px) {
    .doc-header.index-header {
        padding: 4rem;
    }

    .index-title {
        font-size: 3rem;
    }

    .doc-content {
        padding: 4rem;
        max-width: 1000px;
    }

    .index-content {
        padding: 4rem;
        max-width: 1200px;
    }

    .doc-footer {
        padding: 2rem 4rem;
    }
}

@media print {
    body {
        background: white;
        color: black;
    }

    .doc-header,
    .doc-footer {
        display: none;
    }
}

/* Documentation Viewer Styles */
.docs-viewer {
    width: 100%;
}

.docs-controls {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 2rem;
    padding: 1.5rem;
    background: var(--bg-surface);
    border: 1px solid var(--border);
}

.search-box {
    width: 100%;
}

.search-box input {
    width: 100%;
    padding: 0.75rem 1rem;
    font-family: inherit;
    font-size: 0.875rem;
    background: var(--bg);
    color: var(--fg);
    border: 1px solid var(--border);
    outline: none;
}

.search-box input:focus {
    border-color: var(--accent);
}

.filter-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
}

.filter-group {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.filter-group label {
    font-size: 0.75rem;
    color: var(--fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.filter-group select {
    padding: 0.5rem;
    font-family: inherit;
    font-size: 0.75rem;
    background: var(--bg);
    color: var(--fg);
    border: 1px solid var(--border);
    outline: none;
    cursor: pointer;
}

.filter-group select:focus {
    border-color: var(--accent);
}

.docs-stats {
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
    font-size: 0.75rem;
    color: var(--fg-muted);
    padding: 0 0.5rem;
}

.docs-results {
    display: flex;
    flex-direction: column;
    gap: 2rem;
}

.doc-link {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.75rem 0;
    text-decoration: none;
    border-bottom: 1px solid transparent;
}

.doc-link:hover {
    background: var(--fg);
    color: var(--bg);
    margin: 0 -0.75rem;
    padding: 0.75rem;
    text-decoration: none;
    border-bottom-color: var(--fg);
}

.doc-link-main {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
}

.doc-meta {
    display: flex;
    gap: 1rem;
    font-size: 0.625rem;
    color: var(--fg-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.doc-link:hover .doc-meta {
    color: var(--bg);
    opacity: 0.7;
}

.doc-excerpt {
    font-size: 0.75rem;
    color: var(--fg-muted);
    line-height: 1.5;
    margin: 0;
}

.doc-link:hover .doc-excerpt {
    color: var(--bg);
    opacity: 0.8;
}

.no-results {
    padding: 3rem 1rem;
    text-align: center;
    color: var(--fg-muted);
    font-size: 0.875rem;
}

.error-message {
    padding: 2rem 1rem;
    background: var(--code-bg);
    border: 1px solid var(--border);
    color: var(--fg-muted);
}

.error-message p {
    margin-bottom: 0.5rem;
}

@media (min-width: 640px) {
    .filter-controls {
        flex-wrap: nowrap;
    }
}
`;
}

/**
 * Extract all internal markdown links from content
 * Returns normalized paths relative to root
 */
function extractInternalLinks(content: string, sourceDir: string): string[] {
  const links: string[] = [];
  // Match markdown links: [text](url)
  const linkRegex = /\[([^\]]*)\]\(([^)]+)\)/g;
  let match;

  while ((match = linkRegex.exec(content)) !== null) {
    const href = match[2];

    // Skip external links, anchors, and non-md links
    if (!href ||
        href.startsWith('http://') ||
        href.startsWith('https://') ||
        href.startsWith('mailto:') ||
        href.startsWith('#') ||
        !href.includes('.md')) {
      continue;
    }

    // Remove anchor from href
    const hrefWithoutAnchor = href.split('#')[0];

    // Resolve relative path to get normalized path from root
    const resolvedPath = join(sourceDir, hrefWithoutAnchor);
    // Normalize to use forward slashes and remove leading ./
    const normalizedPath = resolvedPath.replace(/\\/g, '/').replace(/^\.\//, '');

    links.push(normalizedPath);
  }

  return links;
}

/**
 * Build backlinks map: target path -> array of sources that link to it
 */
function buildBacklinksMap(docs: DocFile[]): Map<string, Backlink[]> {
  const backlinks = new Map<string, Backlink[]>();

  for (const doc of docs) {
    const sourceDir = dirname(doc.path);
    const internalLinks = extractInternalLinks(doc.content, sourceDir);

    // Deduplicate links from same source
    const uniqueTargets = [...new Set(internalLinks)];

    for (const targetPath of uniqueTargets) {
      if (!backlinks.has(targetPath)) {
        backlinks.set(targetPath, []);
      }
      backlinks.get(targetPath)!.push({
        htmlPath: doc.htmlPath,
        title: doc.title
      });
    }
  }

  return backlinks;
}

/**
 * Generate backlinks HTML section
 */
function generateBacklinksHtml(backlinks: Backlink[]): string {
  if (backlinks.length === 0) {
    return '';
  }

  const linksHtml = backlinks
    .sort((a, b) => a.title.localeCompare(b.title))
    .map(bl => `<a href="/docs/${bl.htmlPath}">${bl.title}</a>`)
    .join('');

  return `
    <aside class="backlinks">
        <div class="backlinks-header">Linked from</div>
        <nav class="backlinks-list">${linksHtml}</nav>
    </aside>`;
}

/**
 * Transform markdown links (.md) to HTML links (.html)
 * Handles relative paths, anchors, and preserves external links
 */
function transformMdLink(href: string): string {
  // Skip external links, anchors-only, and non-md links
  if (!href ||
      href.startsWith('http://') ||
      href.startsWith('https://') ||
      href.startsWith('mailto:') ||
      href.startsWith('#')) {
    return href;
  }

  // Transform .md extension to .html (handles anchors like file.md#section)
  if (href.includes('.md')) {
    return href.replace(/\.md(#|$)/, '.html$1');
  }

  return href;
}

/**
 * Create custom renderer with link transformation
 */
function createCustomRenderer(): Renderer {
  const renderer = new Renderer();

  // Override link rendering to transform .md -> .html
  renderer.link = function({ href, title, text }) {
    const transformedHref = transformMdLink(href);
    const titleAttr = title ? ` title="${title}"` : '';
    return `<a href="${transformedHref}"${titleAttr}>${text}</a>`;
  };

  return renderer;
}

/**
 * Configure marked with custom renderer
 */
function configureMarked() {
  marked.setOptions({
    gfm: true,
    breaks: false,
    renderer: createCustomRenderer()
  });
}

/**
 * Main compilation function
 */
async function main() {
  console.log('📚 SIMCAP Markdown Compiler');
  console.log('===========================\n');

  configureMarked();

  // Find all markdown files
  console.log('🔍 Scanning for markdown files...');
  const markdownFiles = findMarkdownFiles(ROOT_DIR);
  console.log(`   Found ${markdownFiles.length} markdown files\n`);

  // Create output directory
  mkdirSync(OUTPUT_DIR, { recursive: true });

  // PASS 1: Collect all docs and their content
  console.log('📖 Pass 1: Collecting documents and extracting dates...');
  const docs: DocFile[] = [];

  // Track date sources for summary
  const dateSources = { frontmatter: 0, inline: 0, git: 0, none: 0 };

  for (const filepath of markdownFiles) {
    const relativePath = relative(ROOT_DIR, filepath);
    const rawContent = readFileSync(filepath, 'utf-8');

    // Parse frontmatter (removes it from content)
    const { frontmatter, content } = parseFrontmatter(rawContent);

    // Get title: frontmatter title > extracted from content
    const title = (frontmatter.title as string) || extractTitle(content, filepath);

    const htmlRelativePath = relativePath.replace(/\.md$/, '.html');

    // Get dates with precedence: frontmatter > inline > git
    const { createdAt, lastModified, dateSource } = getDocumentDates(rawContent, frontmatter, relativePath);
    dateSources[dateSource]++;

    docs.push({
      path: relativePath,
      htmlPath: htmlRelativePath,
      title,
      directory: dirname(relativePath),
      content,
      createdAt,
      lastModified,
      wordCount: extractWordCount(content),
      excerpt: extractExcerpt(content),
      dateSource
    });
  }

  // Log date source summary
  console.log(`   Date sources: ${dateSources.frontmatter} frontmatter, ${dateSources.inline} inline, ${dateSources.git} git, ${dateSources.none} none`);

  // Build backlinks map
  console.log('🔗 Building backlinks map...');
  const backlinksMap = buildBacklinksMap(docs);
  const totalBacklinks = Array.from(backlinksMap.values()).reduce((sum, arr) => sum + arr.length, 0);
  console.log(`   Found ${totalBacklinks} backlinks across ${backlinksMap.size} documents\n`);

  // PASS 2: Generate HTML with backlinks
  console.log('📝 Pass 2: Compiling markdown to HTML...');

  for (const doc of docs) {
    // Convert to HTML
    const htmlContent = marked(doc.content) as string;

    // Get backlinks for this document
    const backlinks = backlinksMap.get(doc.path) || [];

    // Determine output path
    const outputPath = join(OUTPUT_DIR, doc.htmlPath);
    const outputDir = dirname(outputPath);

    // Create output directory
    mkdirSync(outputDir, { recursive: true });

    // Generate full HTML page with backlinks
    const fullHtml = generateHtmlPage(doc.title, htmlContent, doc.htmlPath, doc, backlinks);

    // Write file
    writeFileSync(outputPath, fullHtml);

    const backlinkInfo = backlinks.length > 0 ? ` (${backlinks.length} backlinks)` : '';
    console.log(`   ✓ ${doc.path} → ${doc.htmlPath}${backlinkInfo}`);
  }

  console.log(`\n📋 Generating index page...`);

  // Generate and write index page
  const indexHtml = generateIndexPage(docs);
  writeFileSync(join(OUTPUT_DIR, 'index.html'), indexHtml);
  console.log(`   ✓ docs/index.html`);

  // Generate and write stylesheet
  console.log(`\n🎨 Generating stylesheet...`);
  const stylesheet = generateDocsStylesheet();
  writeFileSync(join(OUTPUT_DIR, 'docs.css'), stylesheet);
  console.log(`   ✓ docs/docs.css`);

  // Generate JSON index for search
  console.log(`\n🔍 Generating search index...`);
  const searchIndex: DocIndexEntry[] = docs.map(doc => ({
    path: doc.path,
    htmlPath: doc.htmlPath,
    title: doc.title,
    directory: doc.directory,
    createdAt: doc.createdAt,
    lastModified: doc.lastModified,
    wordCount: doc.wordCount,
    excerpt: doc.excerpt,
    dateSource: doc.dateSource
  }));
  writeFileSync(
    join(OUTPUT_DIR, 'docs-index.json'),
    JSON.stringify(searchIndex, null, 2)
  );
  console.log(`   ✓ docs/docs-index.json (${searchIndex.length} entries)`);

  console.log(`\n✅ Done! Compiled ${docs.length} documents to dist/docs/`);
}

main().catch(console.error);
