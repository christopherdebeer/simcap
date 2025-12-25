/**
 * FFO$$ Gesture Recognizer
 *
 * Main recognizer class that implements the $Q-3D algorithm for
 * gesture recognition from IMU sensor data.
 *
 * Usage:
 * ```typescript
 * const recognizer = new FFORecognizer({ numPoints: 32 });
 *
 * // Add templates
 * recognizer.addTemplateFromSamples('wave', waveSamples);
 * recognizer.addTemplateFromSamples('circle', circleSamples);
 *
 * // Recognize gestures
 * const result = recognizer.recognize(inputSamples);
 * if (!result.rejected) {
 *   console.log(`Detected: ${result.template.name} (score: ${result.score})`);
 * }
 * ```
 *
 * @module ffo/recognizer
 */

import type {
  GestureTemplate,
  GestureVocabulary,
  RecognitionResult,
  RecognitionCandidate,
  RecognizerConfig,
  TelemetrySample3D,
  TemplatePoint3D,
  DEFAULT_CONFIG,
} from './types';

import { resampleImmutable, extractTrajectory, removeGravityApprox } from './resample';
import { quickNormalize, centroid } from './normalize';
import {
  cloudDistance,
  lookupDistance,
  buildLookupTable,
  distanceToScore,
} from './distance';

/**
 * Generate a unique ID for a template.
 */
function generateId(): string {
  return `tmpl_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

/**
 * FFO$$ Gesture Recognizer
 *
 * Template-based gesture recognition using the $Q-3D algorithm.
 * Designed for real-time recognition of IMU sensor data.
 */
export class FFORecognizer {
  private templates: GestureTemplate[] = [];
  private config: RecognizerConfig;

  /**
   * Create a new FFO$$ recognizer.
   *
   * @param config - Partial configuration (merged with defaults)
   */
  constructor(config: Partial<RecognizerConfig> = {}) {
    this.config = {
      numPoints: 32,
      useOrientation: true,
      rejectThreshold: null,
      useLookupTable: true,
      minSamples: 10,
      removeGravity: true,
      ...config,
    };
  }

  /**
   * Get current configuration.
   */
  getConfig(): Readonly<RecognizerConfig> {
    return { ...this.config };
  }

  /**
   * Update configuration.
   *
   * Note: Changing numPoints after adding templates will cause
   * mismatches. Clear templates first if changing this value.
   */
  setConfig(config: Partial<RecognizerConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get the number of templates in the recognizer.
   */
  get templateCount(): number {
    return this.templates.length;
  }

  /**
   * Get all template names.
   */
  get templateNames(): string[] {
    return this.templates.map((t) => t.name);
  }

  /**
   * Check if a template with the given name exists.
   */
  hasTemplate(name: string): boolean {
    return this.templates.some((t) => t.name === name);
  }

  /**
   * Get a template by name.
   */
  getTemplate(name: string): GestureTemplate | undefined {
    return this.templates.find((t) => t.name === name);
  }

  /**
   * Process telemetry samples into normalized template points.
   *
   * @param samples - Raw telemetry samples
   * @returns Normalized, resampled points ready for matching
   */
  private processInput(samples: TelemetrySample3D[]): TemplatePoint3D[] {
    // Extract trajectory from accelerometer data
    let trajectory = extractTrajectory(samples);

    // Optionally remove gravity
    if (this.config.removeGravity) {
      trajectory = removeGravityApprox(trajectory);
    }

    // Resample to fixed point count
    const resampled = resampleImmutable(trajectory, this.config.numPoints);

    // Normalize (translate + scale)
    const normalized = quickNormalize(resampled);

    return normalized;
  }

  /**
   * Add a template from pre-normalized points.
   *
   * @param name - Gesture name
   * @param points - Normalized 3D points (must have length = numPoints)
   * @param source - Optional source identifier
   * @returns The created template
   */
  addTemplate(name: string, points: TemplatePoint3D[], source: string = 'manual'): GestureTemplate {
    if (points.length !== this.config.numPoints) {
      throw new Error(
        `Template must have ${this.config.numPoints} points, got ${points.length}`
      );
    }

    const template: GestureTemplate = {
      id: generateId(),
      name,
      points,
      meta: {
        n: this.config.numPoints,
        source,
        created: new Date().toISOString(),
        lookupTable: this.config.useLookupTable ? buildLookupTable(points) : undefined,
      },
    };

    this.templates.push(template);
    return template;
  }

  /**
   * Add a template from raw telemetry samples.
   *
   * Automatically processes the samples (extract, resample, normalize).
   *
   * @param name - Gesture name
   * @param samples - Raw telemetry samples
   * @param source - Optional source identifier
   * @returns The created template
   */
  addTemplateFromSamples(
    name: string,
    samples: TelemetrySample3D[],
    source: string = 'recorded'
  ): GestureTemplate {
    if (samples.length < this.config.minSamples) {
      throw new Error(
        `Need at least ${this.config.minSamples} samples, got ${samples.length}`
      );
    }

    const points = this.processInput(samples);
    const duration = samples[samples.length - 1].t - samples[0].t;

    const template = this.addTemplate(name, points, source);
    template.meta.duration = duration;

    return template;
  }

  /**
   * Remove a template by name.
   *
   * @param name - Template name to remove
   * @returns True if removed, false if not found
   */
  removeTemplate(name: string): boolean {
    const index = this.templates.findIndex((t) => t.name === name);
    if (index === -1) return false;

    this.templates.splice(index, 1);
    return true;
  }

  /**
   * Remove a template by ID.
   *
   * @param id - Template ID to remove
   * @returns True if removed, false if not found
   */
  removeTemplateById(id: string): boolean {
    const index = this.templates.findIndex((t) => t.id === id);
    if (index === -1) return false;

    this.templates.splice(index, 1);
    return true;
  }

  /**
   * Clear all templates.
   */
  clearTemplates(): void {
    this.templates = [];
  }

  /**
   * Recognize a gesture from telemetry samples.
   *
   * @param samples - Raw telemetry samples
   * @returns Recognition result with best match and score
   */
  recognize(samples: TelemetrySample3D[]): RecognitionResult {
    if (samples.length < this.config.minSamples) {
      return {
        template: null,
        distance: Infinity,
        score: 0,
        rejected: true,
        candidates: [],
      };
    }

    if (this.templates.length === 0) {
      return {
        template: null,
        distance: Infinity,
        score: 0,
        rejected: true,
        candidates: [],
      };
    }

    // Process input
    const inputPoints = this.processInput(samples);
    const inputTable = this.config.useLookupTable ? buildLookupTable(inputPoints) : undefined;

    // Match against all templates
    const candidates: RecognitionCandidate[] = [];

    for (const template of this.templates) {
      let distance: number;

      if (this.config.useLookupTable && template.meta.lookupTable && inputTable) {
        // Fast $Q-style distance
        distance = lookupDistance(
          inputPoints,
          inputTable,
          template.points,
          template.meta.lookupTable
        );
      } else {
        // Standard cloud distance
        distance = cloudDistance(inputPoints, template.points);
      }

      candidates.push({
        template,
        distance,
        score: distanceToScore(distance),
      });
    }

    // Sort by distance (ascending)
    candidates.sort((a, b) => a.distance - b.distance);

    const best = candidates[0];
    const rejected =
      this.config.rejectThreshold !== null && best.distance > this.config.rejectThreshold;

    return {
      template: rejected ? null : best.template,
      distance: best.distance,
      score: best.score,
      rejected,
      candidates,
    };
  }

  /**
   * Recognize from pre-normalized points.
   *
   * Useful when you want to handle preprocessing yourself.
   *
   * @param points - Normalized points (must have length = numPoints)
   * @returns Recognition result
   */
  recognizePoints(points: TemplatePoint3D[]): RecognitionResult {
    if (points.length !== this.config.numPoints) {
      throw new Error(
        `Input must have ${this.config.numPoints} points, got ${points.length}`
      );
    }

    if (this.templates.length === 0) {
      return {
        template: null,
        distance: Infinity,
        score: 0,
        rejected: true,
        candidates: [],
      };
    }

    const inputTable = this.config.useLookupTable ? buildLookupTable(points) : undefined;
    const candidates: RecognitionCandidate[] = [];

    for (const template of this.templates) {
      let distance: number;

      if (this.config.useLookupTable && template.meta.lookupTable && inputTable) {
        distance = lookupDistance(points, inputTable, template.points, template.meta.lookupTable);
      } else {
        distance = cloudDistance(points, template.points);
      }

      candidates.push({
        template,
        distance,
        score: distanceToScore(distance),
      });
    }

    candidates.sort((a, b) => a.distance - b.distance);

    const best = candidates[0];
    const rejected =
      this.config.rejectThreshold !== null && best.distance > this.config.rejectThreshold;

    return {
      template: rejected ? null : best.template,
      distance: best.distance,
      score: best.score,
      rejected,
      candidates,
    };
  }

  /**
   * Export all templates as a vocabulary.
   *
   * @param name - Optional vocabulary name
   * @returns GestureVocabulary object
   */
  export(name?: string): GestureVocabulary {
    return {
      version: '1.0.0',
      templates: this.templates.map((t) => ({ ...t, points: [...t.points] })),
      rejectThreshold: this.config.rejectThreshold ?? undefined,
      meta: {
        name,
        created: new Date().toISOString(),
      },
    };
  }

  /**
   * Import templates from a vocabulary.
   *
   * @param vocabulary - Vocabulary to import
   * @param replace - If true, replace existing templates; if false, merge
   */
  import(vocabulary: GestureVocabulary, replace: boolean = false): void {
    if (replace) {
      this.templates = [];
    }

    for (const template of vocabulary.templates) {
      // Validate template
      if (template.points.length !== this.config.numPoints) {
        console.warn(
          `Skipping template '${template.name}': expected ${this.config.numPoints} points, got ${template.points.length}`
        );
        continue;
      }

      // Regenerate lookup table if needed
      const importedTemplate: GestureTemplate = {
        ...template,
        points: [...template.points],
        meta: {
          ...template.meta,
          lookupTable:
            this.config.useLookupTable && !template.meta.lookupTable
              ? buildLookupTable(template.points)
              : template.meta.lookupTable,
        },
      };

      this.templates.push(importedTemplate);
    }

    // Update reject threshold if specified in vocabulary
    if (vocabulary.rejectThreshold !== undefined) {
      this.config.rejectThreshold = vocabulary.rejectThreshold;
    }
  }

  /**
   * Export vocabulary as JSON string.
   */
  toJSON(name?: string): string {
    return JSON.stringify(this.export(name), null, 2);
  }

  /**
   * Import vocabulary from JSON string.
   */
  fromJSON(json: string, replace: boolean = false): void {
    const vocabulary = JSON.parse(json) as GestureVocabulary;
    this.import(vocabulary, replace);
  }

  /**
   * Get statistics about the current template set.
   */
  getStats(): {
    templateCount: number;
    names: string[];
    numPoints: number;
    useLookupTable: boolean;
    rejectThreshold: number | null;
  } {
    return {
      templateCount: this.templates.length,
      names: this.templateNames,
      numPoints: this.config.numPoints,
      useLookupTable: this.config.useLookupTable,
      rejectThreshold: this.config.rejectThreshold,
    };
  }
}

/**
 * Create a new FFO$$ recognizer with default settings.
 *
 * Convenience factory function.
 */
export function createRecognizer(config?: Partial<RecognizerConfig>): FFORecognizer {
  return new FFORecognizer(config);
}
