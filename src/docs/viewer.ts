/**
 * Documentation Viewer Module
 * Handles UI rendering and interactions for documentation search/filter
 */

import {
  DocsSearch,
  SearchOptions,
  SearchResult,
  formatDate,
  formatWordCount,
  SortField,
  SortOrder
} from './search';

/**
 * Interactive documentation viewer with search and filtering
 */
export class DocsViewer {
  private search: DocsSearch;
  private container: HTMLElement;
  private searchInput: HTMLInputElement | null = null;
  private directorySelect: HTMLSelectElement | null = null;
  private sortFieldSelect: HTMLSelectElement | null = null;
  private sortOrderSelect: HTMLSelectElement | null = null;
  private resultsContainer: HTMLElement | null = null;
  private statsContainer: HTMLElement | null = null;

  constructor(containerId: string) {
    const container = document.getElementById(containerId);
    if (!container) {
      throw new Error(`Container element #${containerId} not found`);
    }
    this.container = container;
    this.search = new DocsSearch();
  }

  /**
   * Initialize the viewer
   */
  async init(): Promise<void> {
    try {
      await this.search.load();
      this.render();
      this.attachEventListeners();
      this.performSearch();
    } catch (error) {
      console.error('Failed to initialize docs viewer:', error);
      this.container.innerHTML = `
        <div class="error-message">
          <p>Failed to load documentation index.</p>
          <p>${error instanceof Error ? error.message : 'Unknown error'}</p>
        </div>
      `;
    }
  }

  /**
   * Render the viewer UI
   */
  private render(): void {
    this.container.innerHTML = `
      <div class="docs-viewer">
        <div class="docs-controls">
          <div class="search-box">
            <input
              type="text"
              id="docs-search-input"
              placeholder="Search documentation..."
              autocomplete="off"
            />
          </div>

          <div class="filter-controls">
            <div class="filter-group">
              <label for="docs-directory-select">Directory:</label>
              <select id="docs-directory-select">
                <option value="all">All Directories</option>
                ${this.renderDirectoryOptions()}
              </select>
            </div>

            <div class="filter-group">
              <label for="docs-sort-field">Sort by:</label>
              <select id="docs-sort-field">
                <option value="title">Title</option>
                <option value="lastModified">Last Modified</option>
                <option value="wordCount">Word Count</option>
              </select>
            </div>

            <div class="filter-group">
              <label for="docs-sort-order">Order:</label>
              <select id="docs-sort-order">
                <option value="asc">Ascending</option>
                <option value="desc">Descending</option>
              </select>
            </div>
          </div>
        </div>

        <div class="docs-stats" id="docs-stats"></div>
        <div class="docs-results" id="docs-results"></div>
      </div>
    `;

    this.searchInput = document.getElementById('docs-search-input') as HTMLInputElement;
    this.directorySelect = document.getElementById('docs-directory-select') as HTMLSelectElement;
    this.sortFieldSelect = document.getElementById('docs-sort-field') as HTMLSelectElement;
    this.sortOrderSelect = document.getElementById('docs-sort-order') as HTMLSelectElement;
    this.resultsContainer = document.getElementById('docs-results');
    this.statsContainer = document.getElementById('docs-stats');
  }

  /**
   * Render directory filter options
   */
  private renderDirectoryOptions(): string {
    const directories = this.search.getDirectories();
    return directories
      .map(dir => `<option value="${dir}">${this.getFriendlyDirName(dir)}</option>`)
      .join('');
  }

  /**
   * Get friendly directory name
   */
  private getFriendlyDirName(dir: string): string {
    const dirMap: Record<string, string> = {
      '': 'Project Root',
      'docs': 'Documentation',
      'docs/design': 'Design & Research',
      'docs/technical': 'Technical Notes',
      'docs/procedures': 'Procedures & Guides',
      'apps/gambit': 'GAMBIT App',
      'ml': 'ML Pipeline'
    };
    return dirMap[dir] || dir;
  }

  /**
   * Attach event listeners
   */
  private attachEventListeners(): void {
    // Search input with debounce
    let searchTimeout: number;
    this.searchInput?.addEventListener('input', () => {
      clearTimeout(searchTimeout);
      searchTimeout = window.setTimeout(() => this.performSearch(), 300);
    });

    // Filter controls
    this.directorySelect?.addEventListener('change', () => this.performSearch());
    this.sortFieldSelect?.addEventListener('change', () => this.performSearch());
    this.sortOrderSelect?.addEventListener('change', () => this.performSearch());
  }

  /**
   * Perform search with current options
   */
  private performSearch(): void {
    const options: SearchOptions = {
      query: this.searchInput?.value || '',
      directory: this.directorySelect?.value || 'all',
      sortField: (this.sortFieldSelect?.value as SortField) || 'title',
      sortOrder: (this.sortOrderSelect?.value as SortOrder) || 'asc'
    };

    const results = this.search.search(options);
    this.renderResults(results);
    this.renderStats(results.length);
  }

  /**
   * Render search results
   */
  private renderResults(results: SearchResult[]): void {
    if (!this.resultsContainer) return;

    if (results.length === 0) {
      this.resultsContainer.innerHTML = `
        <div class="no-results">
          <p>No documents found matching your criteria.</p>
        </div>
      `;
      return;
    }

    const grouped = this.groupByDirectory(results);
    const sortedDirs = this.sortDirectories(Array.from(grouped.keys()));

    let html = '';
    for (const dir of sortedDirs) {
      const docs = grouped.get(dir)!;
      const friendlyName = this.getFriendlyDirName(dir);

      html += `
        <section class="doc-section">
          <div class="section-header">${friendlyName}</div>
          <div class="doc-links">
            ${docs.map(result => this.renderDocLink(result)).join('')}
          </div>
        </section>
      `;
    }

    this.resultsContainer.innerHTML = html;
  }

  /**
   * Render a single document link
   */
  private renderDocLink(result: SearchResult): string {
    const { item } = result;
    const lastModified = formatDate(item.lastModified);
    const wordCount = formatWordCount(item.wordCount);

    return `
      <a href="/docs/${item.htmlPath}" class="doc-link">
        <div class="doc-link-main">
          <span class="doc-title">${this.escapeHtml(item.title)}</span>
          <span class="doc-path">${this.escapeHtml(item.path)}</span>
        </div>
        <div class="doc-meta">
          <span class="doc-meta-item">${lastModified}</span>
          <span class="doc-meta-item">${wordCount} words</span>
        </div>
        ${item.excerpt ? `<p class="doc-excerpt">${this.escapeHtml(item.excerpt)}</p>` : ''}
      </a>
    `;
  }

  /**
   * Group results by directory
   */
  private groupByDirectory(results: SearchResult[]): Map<string, SearchResult[]> {
    const grouped = new Map<string, SearchResult[]>();

    for (const result of results) {
      const dir = result.item.directory;
      if (!grouped.has(dir)) {
        grouped.set(dir, []);
      }
      grouped.get(dir)!.push(result);
    }

    return grouped;
  }

  /**
   * Sort directories (root first, then docs, then alphabetically)
   */
  private sortDirectories(dirs: string[]): string[] {
    return dirs.sort((a, b) => {
      if (a === '') return -1;
      if (b === '') return 1;
      if (a.startsWith('docs') && !b.startsWith('docs')) return -1;
      if (!a.startsWith('docs') && b.startsWith('docs')) return 1;
      return a.localeCompare(b);
    });
  }

  /**
   * Render statistics
   */
  private renderStats(resultCount: number): void {
    if (!this.statsContainer) return;

    const stats = this.search.getStats();
    const query = this.searchInput?.value || '';

    let statsHtml = '';
    if (query) {
      statsHtml = `
        <span class="stat">Found <strong>${resultCount}</strong> of <strong>${stats.totalDocs}</strong> documents</span>
      `;
    } else {
      statsHtml = `
        <span class="stat"><strong>${stats.totalDocs}</strong> documents</span>
        <span class="stat"><strong>${stats.directories}</strong> sections</span>
        <span class="stat"><strong>${formatWordCount(stats.totalWords)}</strong> total words</span>
      `;
    }

    this.statsContainer.innerHTML = statsHtml;
  }

  /**
   * Escape HTML to prevent XSS
   */
  private escapeHtml(text: string): string {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

/**
 * Initialize the docs viewer when DOM is ready
 */
export function initDocsViewer(containerId: string = 'docs-viewer-container'): void {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      const viewer = new DocsViewer(containerId);
      viewer.init();
    });
  } else {
    const viewer = new DocsViewer(containerId);
    viewer.init();
  }
}
