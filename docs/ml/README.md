# Machine Learning Documentation

This directory contains ML/AI related documentation for the SimCap project.

## Physics-Based Modeling

Located in [`physics/`](./physics/):

- **[Action Plan](./physics/action-plan.md)** - Next steps for improving models with physics
- **[Optimization Report](./physics/optimization-report.md)** - Complete physics optimization results
- **[Optimization Analysis](./physics/optimization-analysis.md)** - Detailed analysis of physics model
- **[Physics-to-ML Insights](./physics/physics-to-ml-insights.md)** - How to use physics for ML improvement

## Analysis Reports

- **[Clustering Analysis](./clustering-analysis.md)** - Clustering approach for finger tracking
- **[Physics Simulation Findings](./physics-simulation-findings.md)** - Findings from magnetic simulations
- **[Residual Analysis Summary](./residual-analysis-summary.md)** - Analysis of model residuals

## Quick Links

### Getting Started with Physics-Augmented ML

1. Read: [Physics-to-ML Insights](./physics/physics-to-ml-insights.md)
2. Execute: [Action Plan](./physics/action-plan.md)
3. Reference: [Optimization Report](./physics/optimization-report.md)

### Key Findings

**Physics Model Performance:**
- Classification accuracy: 14.3% (not suitable for direct use)
- Regression accuracy: 95% improvement (excellent for field prediction)
- Enables synthetic data generation for all 32 finger state combinations

**ML Model with Physics Augmentation:**
- Baseline (real data only): 100% accuracy on 10/32 combos
- Augmented (real + synthetic): 98.8% accuracy on 32/32 combos
- **3.2× more data efficient**
- **100% state space coverage**

## Related Documentation

- [Research](../research/) - Ablation studies and model analysis
- [Technical](../technical/) - Implementation details
- [Design](../design/) - System design documents

---

**Last Updated:** 2026-01-06
**Maintainer:** ML Team
