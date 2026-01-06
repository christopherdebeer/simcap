# SimCap Documentation Index

Comprehensive documentation for the SimCap magnetic finger tracking system.

## 📚 Documentation Structure

```
docs/
├── ml/              # Machine Learning & AI
├── technical/       # Technical implementation
├── research/        # Research & experiments
├── design/          # System design
├── procedures/      # Operational procedures
└── gambit/          # GAMBIT hardware docs
```

## 🚀 Quick Start

### For ML/AI Developers
- **[ML Documentation](./ml/)** - Start here for ML-related work
- **[Physics-to-ML Insights](./ml/physics/physics-to-ml-insights.md)** - How physics improves ML models
- **[Action Plan](./ml/physics/action-plan.md)** - Next steps for model improvement

### For Hardware Engineers
- **[GAMBIT Documentation](./gambit/)** - Hardware system analysis
- **[Capacitive Wiring](./technical/gambit-capacitive-wiring.md)** - Wiring diagrams
- **[Firmware Improvements](./technical/gambit-firmware-improvements.md)** - Firmware enhancements

### For System Developers
- **[TypeScript Migration](./technical/typescript-migration.md)** - Codebase migration guide
- **[Sensor Units Policy](./sensor-units-policy.md)** - Unit conventions
- **[Calibration Guide](./calibration-filtering-guide.md)** - Sensor calibration

## 📁 Documentation by Category

### Machine Learning
| Document | Description | Updated |
|----------|-------------|---------|
| [ML Overview](./ml/README.md) | ML documentation index | 2026-01-06 |
| [Physics Action Plan](./ml/physics/action-plan.md) | Next steps for physics-based models | 2026-01-06 |
| [Optimization Report](./ml/physics/optimization-report.md) | Complete physics optimization analysis | 2026-01-06 |
| [Physics-to-ML Insights](./ml/physics/physics-to-ml-insights.md) | How physics improves ML | 2026-01-06 |
| [Clustering Analysis](./ml/clustering-analysis.md) | Clustering approach | 2025-12-19 |
| [Physics Simulation](./ml/physics-simulation-findings.md) | Magnetic simulation findings | 2025-12-29 |

### Research
| Document | Description | Updated |
|----------|-------------|---------|
| [Literature Techniques Summary](./research/literature-techniques-summary.md) | **Summary of all 3 experiments (all negative)** | 2026-01-06 |
| [arXiv Literature Review](./research/arxiv-literature-review.md) | Survey of small model training techniques | 2026-01-06 |
| [V5 Context-Aware Experiment](./research/v5-context-experiment-results.md) | Context-aware gated fusion (negative result) | 2026-01-06 |
| [DropDim Experiment](./research/dropdim-experiment-results.md) | DropDim regularization (negative result) | 2026-01-06 |
| [V4 Architecture Exploration](./research/v4-architecture-exploration.md) | New architectures for improved generalization | 2026-01-06 |
| [V2 vs V3 Benchmark](./research/v2-v3-benchmark-comparison.md) | Model comparison with held-out validation | 2026-01-06 |
| [Physics Model Analysis](./research/physics-model-analysis.md) | Physics-based model evaluation | 2025-12-30 |
| [Ablation Study](./research/ablation-study-results.md) | Model component analysis | 2025-12-29 |
| [Cross-Orientation Study](./research/cross-orientation-ablation-results.md) | Orientation robustness | 2025-12-29 |
| [Template Matching](./research/template-matching-magnetometer-analysis.md) | Template-based approach | 2025-12-19 |

### Technical Implementation
| Document | Description | Updated |
|----------|-------------|---------|
| [TypeScript Migration](./technical/typescript-migration.md) | Codebase migration | 2025-12-11 |
| [Unit Conversion Bug](./technical/critical-unit-conversion-bug.md) | Critical bug analysis | 2025-12-19 |
| [GAMBIT Wiring](./technical/gambit-capacitive-wiring.md) | Hardware wiring | 2025-12-17 |
| [Firmware Improvements](./technical/gambit-firmware-improvements.md) | Firmware enhancements | 2025-12-17 |
| [Mag Calibration](./technical/magnetometer-calibration-complete-analysis.md) | Calibration deep dive | 2025-12-16 |
| [Gyro Bias Fix](./technical/gyro-bias-calibration-fix-2025-12-19.md) | Gyroscope calibration | 2025-12-19 |
| [Earth Field Analysis](./technical/earth-field-subtraction-investigation.md) | Earth field handling | 2025-12-16 |

### System Design
| Document | Description | Updated |
|----------|-------------|---------|
| [Design Overview](./design/README.md) | Design documentation index | - |
| [Magnetic Tracking Pipeline](./design/magnetic-tracking-pipeline-analysis.md) | Complete pipeline | 2025-12-19 |
| [ML Finger Tracking](./design/ml-finger-tracking-design.md) | ML design approach | 2025-12-19 |
| [Finger State Taxonomy](./design/finger-state-taxonomy.md) | State classification | 2025-12-18 |
| [GAMBIT Workflow](./design/gambit-workflow-review.md) | Workflow analysis | 2025-12-17 |

### GAMBIT Hardware
| Document | Description | Updated |
|----------|-------------|---------|
| [Orientation System](./gambit/orientation-magnetometer-system.md) | Sensor system analysis | 2025-12-19 |
| [Diagnostic Report](./gambit/orientation-diagnostic-report.md) | System diagnostics | 2025-12-19 |
| [Firmware Interface](./gambit-firmware-interface.md) | API documentation | 2025-12-17 |
| [Future Enhancements](./gambit-future-enhancements-imp.md) | Roadmap | 2025-12-17 |

### Calibration & Procedures
| Document | Description | Updated |
|----------|-------------|---------|
| [Calibration Guide](./calibration-filtering-guide.md) | Calibration procedures | 2025-12-16 |
| [Magnet Attachment](./procedures/magnet-attachment-guide.md) | Hardware setup | 2025-12-17 |
| [Orientation Validation](./procedures/orientation-validation-protocol.md) | Validation protocol | 2025-12-19 |

## 🔍 Finding Documentation

### By Topic
- **Calibration:** `calibration-*.md`, `technical/mag-calibration-*.md`
- **Physics:** `ml/physics/*.md`, `ml/physics-simulation-*.md`
- **Firmware:** `technical/gambit-firmware-*.md`, `gambit-firmware-*.md`
- **Machine Learning:** `ml/*.md`, `research/*.md`
- **Hardware:** `gambit/*.md`, `technical/gambit-*.md`

### By Date
- **Latest (Jan 2026):** ML physics optimization docs
- **Dec 2025:** Most technical and research docs
- **Ongoing:** See individual file headers for update dates

## 📝 Documentation Standards

All documentation follows these conventions:

### Front Matter
```yaml
---
title: Document Title
created: YYYY-MM-DD
updated: YYYY-MM-DD
original_location: path/to/original/file.md (if moved)
---
```

### Naming Conventions
- Use lowercase with hyphens: `my-document.md`
- Be descriptive: `physics-optimization-analysis.md` not `analysis.md`
- Include dates for time-specific docs: `gyro-fix-2025-12-19.md`

### Organization
- **Technical implementation** → `technical/`
- **Research & experiments** → `research/`
- **ML/AI content** → `ml/`
- **System design** → `design/`
- **Hardware specifics** → `gambit/`
- **Procedures** → `procedures/`

## 🔧 Recent Reorganization

On 2026-01-06, ALL_CAPS.md files were reorganized:

**Moved to `docs/ml/physics/`:**
- `ACTION_PLAN.md` → `action-plan.md`
- `FINAL_PHYSICS_OPTIMIZATION_REPORT.md` → `optimization-report.md`
- `PHYSICS_OPTIMIZATION_ANALYSIS.md` → `optimization-analysis.md`
- `PHYSICS_TO_ML_INSIGHTS.md` → `physics-to-ml-insights.md`

**Moved to `docs/ml/`:**
- `CLUSTERING.md` → `clustering-analysis.md`
- `PHYSICS_SIMULATION_FINDINGS.md` → `physics-simulation-findings.md`
- `RESIDUAL_ANALYSIS_SUMMARY.md` → `residual-analysis-summary.md`

**Moved to `docs/technical/`:**
- `TYPESCRIPT_MIGRATION.md` → `typescript-migration.md`
- `CRITICAL-unit-conversion-bug.md` → `critical-unit-conversion-bug.md`
- `GAMBIT-*.md` → `gambit-*.md` (lowercase)

**Moved to `docs/gambit/`:**
- Orientation analysis docs from `apps/gambit/analysis/`

Git history preserved using `git mv`.

## 📞 Contributing

When adding new documentation:

1. Place in appropriate category folder
2. Use lowercase-with-hyphens naming
3. Add front matter with title and dates
4. Update this index
5. Link from related docs

---

**Index Last Updated:** 2026-01-06
**Total Documents:** 60+
**Categories:** 6
