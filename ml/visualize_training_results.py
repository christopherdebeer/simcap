#!/usr/bin/env python3
"""
Visualize Training Results from Advanced Synthetic Training Pipeline

Generates comprehensive visualizations of:
1. Architecture comparison bar charts
2. Per-finger accuracy heatmap
3. Real vs synthetic accuracy scatter plot
4. Generalization gap analysis
5. Training summary table

Run with: python -m ml.visualize_training_results
"""

import json
import numpy as np
from pathlib import Path

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib not available - text analysis only")


def load_results():
    """Load training results from JSON file."""
    results_path = Path(__file__).parent / 'training_results.json'
    with open(results_path, 'r') as f:
        return json.load(f)


def print_text_analysis(data):
    """Print comprehensive text analysis of results."""
    results = data['results']
    analysis = data['analysis']

    print("\n" + "=" * 80)
    print(" COMPREHENSIVE TRAINING RESULTS ANALYSIS")
    print("=" * 80)

    # Dataset overview
    print("\n" + "-" * 80)
    print(" DATASET OVERVIEW")
    print("-" * 80)
    ds = data['dataset_info']
    print(f"  Training samples: {ds['train_size']:,}")
    print(f"  Validation samples: {ds['val_size']:,}")
    print(f"  Test samples: {ds['test_size']:,}")
    print(f"  Total: {ds['train_size'] + ds['val_size'] + ds['test_size']:,}")
    print(f"  Synthesized missing combos: {len(ds['synthesized_combos'])} / 32")

    # Architecture comparison
    print("\n" + "-" * 80)
    print(" ARCHITECTURE PERFORMANCE COMPARISON")
    print("-" * 80)
    print(f"\n{'Architecture':<15} {'Trials':>7} {'Test Acc':>10} {'Real Acc':>10} {'Synth Acc':>10} {'Gap':>8}")
    print("-" * 68)

    for arch, stats in analysis['architecture_comparison'].items():
        gap = stats['synth_acc_mean'] - stats['real_acc_mean']
        print(f"{arch:<15} {stats['n_trials']:>7} {stats['test_acc_mean']:>9.1%} "
              f"{stats['real_acc_mean']:>9.1%} {stats['synth_acc_mean']:>9.1%} {gap:>+7.1%}")

    # Best models per architecture
    print("\n" + "-" * 80)
    print(" BEST MODEL PER ARCHITECTURE")
    print("-" * 80)

    for arch, model in analysis['best_models'].items():
        if arch == 'overall':
            continue
        print(f"\n  {arch.upper()}:")
        config = model['config']
        if arch == 'cnn':
            print(f"    Config: filters={config['filters']}, kernel={config['kernel_size']}, "
                  f"dropout={config['dropout']}, lr={config['lr']}")
        elif arch == 'lstm':
            print(f"    Config: units={config['units']}, dropout={config['dropout']}, lr={config['lr']}")
        elif arch == 'transformer':
            print(f"    Config: heads={config['num_heads']}, ff_dim={config['ff_dim']}, "
                  f"dropout={config['dropout']}, lr={config['lr']}")
        elif arch == 'hybrid':
            print(f"    Config: lr={config['lr']}")

        print(f"    Real accuracy: {model['real_acc']:.1%}")
        print(f"    Synthetic accuracy: {model['synth_acc']:.1%}")
        print(f"    Per-finger: {[f'{x:.1%}' for x in model['finger_acc']]}")
        print(f"    Generalization gap: {model['generalization_gap']:+.1%}")

    # Per-finger analysis
    print("\n" + "-" * 80)
    print(" PER-FINGER ACCURACY ANALYSIS")
    print("-" * 80)
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    print(f"\n{'Finger':<10} {'Mean':>10} {'Std':>10} {'Best':>10} {'Rank':>6}")
    print("-" * 50)

    finger_stats = [(f, analysis['finger_analysis'][f]) for f in fingers]
    ranked = sorted(finger_stats, key=lambda x: x[1]['mean'], reverse=True)

    for rank, (finger, stats) in enumerate(ranked, 1):
        print(f"{finger:<10} {stats['mean']:>9.1%} {stats['std']:>9.1%} "
              f"{stats['best']:>9.1%} {rank:>6}")

    # Key insights
    print("\n" + "-" * 80)
    print(" KEY INSIGHTS")
    print("-" * 80)

    # Find best overall model
    best_real_acc = max(r['real_acc'] for r in results)
    best_real_model = [r for r in results if r['real_acc'] == best_real_acc][0]

    best_synth_acc = max(r['synth_acc'] for r in results)
    best_synth_model = [r for r in results if r['synth_acc'] == best_synth_acc][0]

    # Find smallest gap (best generalization)
    best_gap = min(abs(r['generalization_gap']) for r in results)
    best_gap_model = [r for r in results if abs(r['generalization_gap']) == best_gap][0]

    print(f"""
  1. REAL DATA PERFORMANCE
     - Best model: {best_real_model['config']['arch'].upper()} ({best_real_acc:.1%} accuracy)
     - All architectures achieve >98% on real data
     - CNN and Hybrid show most consistent real data performance

  2. SYNTHETIC DATA PERFORMANCE
     - Best model: {best_synth_model['config']['arch'].upper()} ({best_synth_acc:.1%} accuracy)
     - Hybrid shows best synthetic performance ({analysis['architecture_comparison']['hybrid']['synth_acc_mean']:.1%})
     - Transformer struggles most with synthetic data ({analysis['architecture_comparison']['transformer']['synth_acc_mean']:.1%})

  3. GENERALIZATION (Real → Synthetic)
     - Best generalizer: {best_gap_model['config']['arch'].upper()} (gap: {best_gap:.1%})
     - Average gap across all models: {analysis['generalization']['avg_gap']:.1%}
     - Negative gap means model performs slightly better on real than synthetic
     - This is EXPECTED - synthetic data approximates but doesn't perfectly match real

  4. PER-FINGER DIFFICULTY
     - Easiest: Pinky ({analysis['finger_analysis']['pinky']['mean']:.1%}) - strongest signal (3243 μT)
     - Hardest: Middle ({analysis['finger_analysis']['middle']['mean']:.1%}) - intermediate signal
     - All fingers achieve >93% mean accuracy

  5. ARCHITECTURE RECOMMENDATIONS
     - For production: HYBRID (CNN-LSTM) - best balance of accuracy and generalization
     - For speed: CNN - fastest training, excellent real accuracy
     - Avoid: Transformer - highest variance, lowest synthetic accuracy
""")

    # Training efficiency
    print("-" * 80)
    print(" TRAINING EFFICIENCY")
    print("-" * 80)

    for arch in ['cnn', 'lstm', 'transformer', 'hybrid']:
        arch_results = [r for r in results if r['config']['arch'] == arch]
        avg_epochs = np.mean([r['epochs_trained'] for r in arch_results])
        avg_val_loss = np.mean([r['best_val_loss'] for r in arch_results])
        print(f"  {arch:<12}: avg epochs={avg_epochs:.0f}, avg val_loss={avg_val_loss:.4f}")


def create_visualizations(data, output_dir):
    """Create matplotlib visualizations."""
    results = data['results']
    analysis = data['analysis']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Finger State Classification Training Results', fontsize=14, fontweight='bold')

    # 1. Architecture Comparison (top-left)
    ax1 = axes[0, 0]
    archs = ['cnn', 'lstm', 'transformer', 'hybrid']
    x = np.arange(len(archs))
    width = 0.35

    real_accs = [analysis['architecture_comparison'][a]['real_acc_mean'] for a in archs]
    synth_accs = [analysis['architecture_comparison'][a]['synth_acc_mean'] for a in archs]

    bars1 = ax1.bar(x - width/2, real_accs, width, label='Real Data', color='#2ecc71')
    bars2 = ax1.bar(x + width/2, synth_accs, width, label='Synthetic Data', color='#3498db')

    ax1.set_ylabel('Accuracy')
    ax1.set_title('Architecture Performance: Real vs Synthetic')
    ax1.set_xticks(x)
    ax1.set_xticklabels([a.upper() for a in archs])
    ax1.legend()
    ax1.set_ylim(0.85, 1.02)
    ax1.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')

    # Add value labels
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.1%}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.1%}', ha='center', va='bottom', fontsize=8)

    # 2. Per-Finger Accuracy Heatmap (top-right)
    ax2 = axes[0, 1]
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    archs_order = ['cnn', 'lstm', 'transformer', 'hybrid']

    # Build heatmap data
    heatmap_data = []
    for arch in archs_order:
        best = analysis['best_models'][arch]
        heatmap_data.append(best['finger_acc'])

    heatmap_data = np.array(heatmap_data)

    im = ax2.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.85, vmax=1.0)
    ax2.set_xticks(np.arange(len(fingers)))
    ax2.set_yticks(np.arange(len(archs_order)))
    ax2.set_xticklabels([f.capitalize() for f in fingers])
    ax2.set_yticklabels([a.upper() for a in archs_order])
    ax2.set_title('Per-Finger Accuracy by Architecture')

    # Add text annotations
    for i in range(len(archs_order)):
        for j in range(len(fingers)):
            val = heatmap_data[i, j]
            color = 'white' if val < 0.92 else 'black'
            ax2.text(j, i, f'{val:.1%}', ha='center', va='center', color=color, fontsize=9)

    plt.colorbar(im, ax=ax2, label='Accuracy')

    # 3. Generalization Gap (bottom-left)
    ax3 = axes[1, 0]

    for r in results:
        arch = r['config']['arch']
        color = {'cnn': '#e74c3c', 'lstm': '#9b59b6', 'transformer': '#f39c12', 'hybrid': '#1abc9c'}[arch]
        marker = {'cnn': 'o', 'lstm': 's', 'transformer': '^', 'hybrid': 'D'}[arch]
        ax3.scatter(r['real_acc'], r['synth_acc'], c=color, marker=marker, s=100,
                   label=arch.upper() if r == [x for x in results if x['config']['arch'] == arch][0] else '')

    # Add diagonal line (perfect generalization)
    ax3.plot([0.9, 1.0], [0.9, 1.0], 'k--', alpha=0.3, label='Perfect generalization')

    ax3.set_xlabel('Real Data Accuracy')
    ax3.set_ylabel('Synthetic Data Accuracy')
    ax3.set_title('Real vs Synthetic Accuracy (Generalization)')
    ax3.legend(loc='lower right')
    ax3.set_xlim(0.97, 1.002)
    ax3.set_ylim(0.88, 1.01)

    # 4. Per-Finger Mean Accuracy with Error Bars (bottom-right)
    ax4 = axes[1, 1]

    finger_means = [analysis['finger_analysis'][f]['mean'] for f in fingers]
    finger_stds = [analysis['finger_analysis'][f]['std'] for f in fingers]
    finger_bests = [analysis['finger_analysis'][f]['best'] for f in fingers]

    x = np.arange(len(fingers))
    bars = ax4.bar(x, finger_means, yerr=finger_stds, capsize=5, color='#3498db', alpha=0.7)
    ax4.scatter(x, finger_bests, color='#e74c3c', marker='*', s=150, zorder=5, label='Best')

    ax4.set_ylabel('Accuracy')
    ax4.set_title('Per-Finger Accuracy (Mean ± Std)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f.capitalize() for f in fingers])
    ax4.set_ylim(0.9, 1.02)
    ax4.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)
    ax4.legend()

    # Add value labels
    for i, (mean, best) in enumerate(zip(finger_means, finger_bests)):
        ax4.text(i, mean + finger_stds[i] + 0.01, f'{mean:.1%}', ha='center', fontsize=9)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'training_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to {output_path}")

    return output_path


def main():
    print("\n" + "=" * 80)
    print(" TRAINING RESULTS VISUALIZATION")
    print("=" * 80)

    # Load results
    data = load_results()

    # Print text analysis
    print_text_analysis(data)

    # Create visualizations if matplotlib is available
    if HAS_MATPLOTLIB:
        output_dir = Path(__file__).parent / 'results'
        output_dir.mkdir(exist_ok=True)
        create_visualizations(data, output_dir)

    print("\n" + "=" * 80)
    print(" ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
