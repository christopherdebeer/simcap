#!/usr/bin/env npx tsx
/**
 * Markdown to HTML Compiler for SIMCAP Documentation
 *
 * Finds all markdown files, converts to static HTML with consistent
 * header/footer and simcap styling, and generates an auto-indexed docs page.
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync, statSync, copyFileSync } from 'fs';
import { join, dirname, relative, basename, extname } from 'path';
import { marked } from 'marked';
import { fileURLToPath } from 'url';

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
function generateHtmlPage(title: string, htmlContent: string, relativePath: string): string {
  const breadcrumb = generateBreadcrumb(relativePath);

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
    </main>

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
  // Group docs by directory
  const grouped = new Map<string, DocFile[]>();

  for (const doc of docs) {
    const dir = doc.directory;
    if (!grouped.has(dir)) {
      grouped.set(dir, []);
    }
    grouped.get(dir)!.push(doc);
  }

  // Sort directories - root first, then docs, then alphabetically
  const sortedDirs = Array.from(grouped.keys()).sort((a, b) => {
    if (a === '') return -1;
    if (b === '') return 1;
    if (a.startsWith('docs') && !b.startsWith('docs')) return -1;
    if (!a.startsWith('docs') && b.startsWith('docs')) return 1;
    return a.localeCompare(b);
  });

  // Generate sections HTML
  let sectionsHtml = '';

  for (const dir of sortedDirs) {
    const files = grouped.get(dir)!;
    const friendlyName = getFriendlyDirName(dir);

    // Sort files: README first, then alphabetically
    files.sort((a, b) => {
      if (a.title.toLowerCase().includes('readme')) return -1;
      if (b.title.toLowerCase().includes('readme')) return 1;
      return a.title.localeCompare(b.title);
    });

    sectionsHtml += `
        <section class="doc-section">
            <div class="section-header">${friendlyName}</div>
            <div class="doc-links">`;

    for (const file of files) {
      const desc = dir || 'root';
      sectionsHtml += `
                <a href="/docs/${file.htmlPath}" class="doc-link">
                    <span class="doc-title">${file.title}</span>
                    <span class="doc-path">${file.path}</span>
                </a>`;
    }

    sectionsHtml += `
            </div>
        </section>`;
  }

  return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documentation · SIMCAP</title>
    <link rel="icon" href="/favicon.ico">
    <link rel="stylesheet" href="/docs/docs.css">
</head>
<body>
    <header class="doc-header index-header">
        <a href="/" class="home-link">SIMCAP</a>
        <h1 class="index-title">Documentation</h1>
        <p class="index-subtitle">All markdown documentation compiled to static HTML</p>
        <div class="index-stats">
            <span class="stat"><strong>${docs.length}</strong> documents</span>
            <span class="stat"><strong>${sortedDirs.length}</strong> sections</span>
        </div>
    </header>

    <main class="index-content">
        ${sectionsHtml}
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
`;
}

/**
 * Configure marked for better code highlighting
 */
function configureMarked() {
  marked.setOptions({
    gfm: true,
    breaks: false
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

  // Process each markdown file
  const docs: DocFile[] = [];

  console.log('📝 Compiling markdown to HTML...');

  for (const filepath of markdownFiles) {
    const relativePath = relative(ROOT_DIR, filepath);
    const content = readFileSync(filepath, 'utf-8');
    const title = extractTitle(content, filepath);

    // Convert to HTML
    const htmlContent = marked(content) as string;

    // Determine output path (preserve directory structure)
    const htmlRelativePath = relativePath.replace(/\.md$/, '.html');
    const outputPath = join(OUTPUT_DIR, htmlRelativePath);
    const outputDir = dirname(outputPath);

    // Create output directory
    mkdirSync(outputDir, { recursive: true });

    // Generate full HTML page
    const fullHtml = generateHtmlPage(title, htmlContent, htmlRelativePath);

    // Write file
    writeFileSync(outputPath, fullHtml);

    // Track for index
    docs.push({
      path: relativePath,
      htmlPath: htmlRelativePath,
      title,
      directory: dirname(relativePath)
    });

    console.log(`   ✓ ${relativePath} → ${htmlRelativePath}`);
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

  console.log(`\n✅ Done! Compiled ${docs.length} documents to dist/docs/`);
}

main().catch(console.error);
