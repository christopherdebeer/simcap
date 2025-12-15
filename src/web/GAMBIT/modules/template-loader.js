/**
 * Wizard Template Loader Module
 * Loads and validates wizard templates from JSON files
 */

// Template cache
const templateCache = new Map();

// Available templates (will be populated by scanning directory or manifest)
const availableTemplates = [
    { id: 'tier1_minimal', path: 'wizard-templates/tier1_minimal.json', name: 'Tier 1: Minimal (3 poses)' }
    // More templates will be added as they're created
];

/**
 * Load a template from a JSON file
 * @param {string} templateId - Template identifier
 * @returns {Promise<Object>} Parsed and validated template object
 */
export async function loadTemplate(templateId) {
    // Check cache first
    if (templateCache.has(templateId)) {
        return templateCache.get(templateId);
    }

    // Find template info
    const templateInfo = availableTemplates.find(t => t.id === templateId);
    if (!templateInfo) {
        throw new Error(`Template not found: ${templateId}`);
    }

    try {
        // Fetch template JSON
        const response = await fetch(templateInfo.path);
        if (!response.ok) {
            throw new Error(`Failed to load template: ${response.statusText}`);
        }

        const template = await response.json();

        // Validate template structure
        validateTemplate(template);

        // Cache the template
        templateCache.set(templateId, template);

        return template;
    } catch (error) {
        throw new Error(`Error loading template ${templateId}: ${error.message}`);
    }
}

/**
 * Get list of available templates
 * @returns {Array<Object>} Array of template info objects
 */
export function getAvailableTemplates() {
    return [...availableTemplates];
}

/**
 * Validate template structure
 * @param {Object} template - Template object to validate
 * @throws {Error} If template is invalid
 */
function validateTemplate(template) {
    // Required top-level fields
    const requiredFields = ['id', 'name', 'tier', 'steps'];
    for (const field of requiredFields) {
        if (!(field in template)) {
            throw new Error(`Missing required field: ${field}`);
        }
    }

    // Validate tier
    if (typeof template.tier !== 'number' || template.tier < 1 || template.tier > 3) {
        throw new Error(`Invalid tier: must be 1, 2, or 3`);
    }

    // Validate steps array
    if (!Array.isArray(template.steps) || template.steps.length === 0) {
        throw new Error(`Steps must be a non-empty array`);
    }

    // Validate each step
    template.steps.forEach((step, index) => {
        validateStep(step, index);
    });

    return true;
}

/**
 * Validate a single step
 * @param {Object} step - Step object to validate
 * @param {number} index - Step index (for error messages)
 * @throws {Error} If step is invalid
 */
function validateStep(step, index) {
    // Required step fields
    const requiredFields = ['id', 'title', 'labels', 'timing'];
    for (const field of requiredFields) {
        if (!(field in step)) {
            throw new Error(`Step ${index}: Missing required field: ${field}`);
        }
    }

    // Validate timing
    if (!step.timing.record_duration || step.timing.record_duration < 1) {
        throw new Error(`Step ${index}: record_duration must be >= 1`);
    }

    // Validate labels
    if (!step.labels || typeof step.labels !== 'object') {
        throw new Error(`Step ${index}: labels must be an object`);
    }

    // Validate labels.motion (required)
    if (!step.labels.motion) {
        throw new Error(`Step ${index}: labels.motion is required`);
    }

    // Validate labels.fingers structure if present
    if (step.labels.fingers) {
        const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];
        for (const finger of fingers) {
            if (!(finger in step.labels.fingers)) {
                throw new Error(`Step ${index}: Missing finger state for ${finger}`);
            }
        }
    }

    return true;
}

/**
 * Parse step labels and return a normalized labels object
 * @param {Object} stepLabels - Labels object from template step
 * @returns {Object} Normalized labels object for application state
 */
export function parseStepLabels(stepLabels) {
    return {
        pose: stepLabels.pose || null,
        fingers: stepLabels.fingers ? { ...stepLabels.fingers } : {
            thumb: null,
            index: null,
            middle: null,
            ring: null,
            pinky: null
        },
        motion: stepLabels.motion || 'static',
        custom: stepLabels.custom ? [...stepLabels.custom] : []
    };
}

/**
 * Get template metadata (without loading full template)
 * @param {string} templateId - Template identifier
 * @returns {Object|null} Template metadata or null if not found
 */
export function getTemplateMetadata(templateId) {
    return availableTemplates.find(t => t.id === templateId) || null;
}

/**
 * Clear template cache (useful for testing/development)
 */
export function clearCache() {
    templateCache.clear();
}

// Export for testing
export const _test = {
    validateTemplate,
    validateStep,
    templateCache
};
