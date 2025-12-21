/**
 * Documentation Search Module
 * Provides fuzzy search and filtering capabilities for compiled documentation
 */

import Fuse, { FuseResult } from 'fuse.js';

// Document index entry structure (matches compile-docs.ts)
export interface DocIndexEntry {
  path: string;
  htmlPath: string;
  title: string;
  directory: string;
  lastModified: number;
  wordCount: number;
  excerpt: string;
}

export type SortField = 'title' | 'lastModified' | 'wordCount';
export type SortOrder = 'asc' | 'desc';

export interface SearchOptions {
  query?: string;
  directory?: string;
  sortField?: SortField;
  sortOrder?: SortOrder;
}

export interface SearchResult {
  item: DocIndexEntry;
  score?: number;
  matches?: readonly any[];
}

/**
 * Documentation search and filter engine
 */
export class DocsSearch {
  private docs: DocIndexEntry[] = [];
  private fuse: Fuse<DocIndexEntry> | null = null;
  private loaded = false;

  constructor() {}

  /**
   * Load documentation index from JSON
   */
  async load(indexUrl: string = '/docs/docs-index.json'): Promise<void> {
    const response = await fetch(indexUrl);
    if (!response.ok) {
      throw new Error(`Failed to load docs index: ${response.statusText}`);
    }

    this.docs = await response.json();
    this.initializeFuse();
    this.loaded = true;
  }

  /**
   * Initialize Fuse.js search engine
   */
  private initializeFuse(): void {
    this.fuse = new Fuse(this.docs, {
      keys: [
        { name: 'title', weight: 0.5 },
        { name: 'excerpt', weight: 0.3 },
        { name: 'path', weight: 0.2 }
      ],
      threshold: 0.4,
      includeScore: true,
      includeMatches: true,
      minMatchCharLength: 2,
      ignoreLocation: true
    });
  }

  /**
   * Check if search engine is loaded
   */
  isLoaded(): boolean {
    return this.loaded;
  }

  /**
   * Get all documents (unfiltered)
   */
  getAllDocs(): DocIndexEntry[] {
    return [...this.docs];
  }

  /**
   * Get unique directories
   */
  getDirectories(): string[] {
    const dirs = new Set(this.docs.map(doc => doc.directory));
    return Array.from(dirs).sort();
  }

  /**
   * Search and filter documents
   */
  search(options: SearchOptions = {}): SearchResult[] {
    if (!this.loaded || !this.fuse) {
      return [];
    }

    let results: SearchResult[];

    // Perform fuzzy search if query provided
    if (options.query && options.query.trim().length > 0) {
      const fuseResults = this.fuse.search(options.query);
      results = fuseResults.map(r => ({
        item: r.item,
        score: r.score,
        matches: r.matches
      }));
    } else {
      // No query - return all docs
      results = this.docs.map(item => ({ item }));
    }

    // Filter by directory if specified
    if (options.directory && options.directory !== 'all') {
      results = results.filter(r => r.item.directory === options.directory);
    }

    // Sort results
    const sortField = options.sortField || 'title';
    const sortOrder = options.sortOrder || 'asc';

    results.sort((a, b) => {
      let comparison = 0;

      switch (sortField) {
        case 'title':
          comparison = a.item.title.localeCompare(b.item.title);
          break;
        case 'lastModified':
          comparison = a.item.lastModified - b.item.lastModified;
          break;
        case 'wordCount':
          comparison = a.item.wordCount - b.item.wordCount;
          break;
      }

      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return results;
  }

  /**
   * Get statistics about the documentation
   */
  getStats() {
    return {
      totalDocs: this.docs.length,
      totalWords: this.docs.reduce((sum, doc) => sum + doc.wordCount, 0),
      directories: this.getDirectories().length,
      lastUpdated: Math.max(...this.docs.map(doc => doc.lastModified))
    };
  }
}

/**
 * Format timestamp to readable date
 */
export function formatDate(timestamp: number): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays === 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays} days ago`;
  if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
  if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;

  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  });
}

/**
 * Format word count with commas
 */
export function formatWordCount(count: number): string {
  return count.toLocaleString();
}
