#!/usr/bin/env python3
"""
SIMCAP Training Script

Train gesture classification models on labeled GAMBIT data.

Usage:
    python -m ml.train --data-dir data/GAMBIT --epochs 50
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

from .schema import Gesture
from .data_loader import GambitDataset
from .model import (
    create_cnn_model_keras, train_model_keras,
    evaluate_model, save_model_for_inference,
    HAS_TF, HAS_TORCH
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train SIMCAP gesture classification model'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/GAMBIT',
        help='Path to data directory'
    )
    parser.add_argument(
        '--output-dir', type=str, default='ml/models',
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--window-size', type=int, default=50,
        help='Window size in samples (50 = 1 second at 50Hz)'
    )
    parser.add_argument(
        '--stride', type=int, default=25,
        help='Window stride (25 = 50% overlap)'
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Maximum training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.2,
        help='Validation split ratio'
    )
    parser.add_argument(
        '--framework', type=str, default='keras',
        choices=['keras', 'pytorch'],
        help='ML framework to use'
    )
    parser.add_argument(
        '--summary-only', action='store_true',
        help='Only print dataset summary, do not train'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SIMCAP Gesture Classification Training")
    print("=" * 60)

    # Check framework availability
    if args.framework == 'keras' and not HAS_TF:
        print("ERROR: TensorFlow/Keras not installed. Run: pip install tensorflow")
        sys.exit(1)
    if args.framework == 'pytorch' and not HAS_TORCH:
        print("ERROR: PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    # Load dataset
    print(f"\nLoading data from: {args.data_dir}")
    dataset = GambitDataset(
        args.data_dir,
        window_size=args.window_size,
        stride=args.stride
    )

    # Print summary
    summary = dataset.summary()
    print(f"\nDataset Summary:")
    print(f"  Total sessions: {summary['total_sessions']}")
    print(f"  Labeled sessions: {summary['labeled_sessions']}")
    print(f"  Unlabeled sessions: {summary['unlabeled_sessions']}")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"\nGesture sample counts:")
    for gesture, count in summary['gesture_counts'].items():
        if count > 0:
            print(f"  {gesture}: {count}")

    if args.summary_only:
        return

    # Load train/val split
    print(f"\nLoading labeled data...")
    X_train, y_train, X_val, y_val = dataset.get_train_val_split(args.val_ratio)

    if len(X_train) == 0:
        print("\nERROR: No labeled data found!")
        print("Please label some sessions first using the labeling tool.")
        print("See: python -m ml.label --help")
        sys.exit(1)

    print(f"  Training windows: {len(X_train)}")
    print(f"  Validation windows: {len(X_val)}")
    print(f"  Window shape: {X_train.shape[1:]}")

    # Check class distribution
    train_classes, train_counts = np.unique(y_train, return_counts=True)
    print(f"\nTraining class distribution:")
    for cls, count in zip(train_classes, train_counts):
        print(f"  {Gesture(cls).name}: {count} ({100*count/len(y_train):.1f}%)")

    # Create model
    print(f"\nCreating {args.framework} model...")
    num_classes = len(Gesture)

    if args.framework == 'keras':
        model = create_cnn_model_keras(
            window_size=args.window_size,
            num_classes=num_classes
        )
        model.summary()

        # Train
        print(f"\nTraining for up to {args.epochs} epochs...")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = str(output_dir / 'best_model.keras')
        history = train_model_keras(
            model, X_train, y_train, X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            checkpoint_path=checkpoint_path
        )

        # Evaluate
        print("\nEvaluating on validation set...")
        metrics = evaluate_model(model, X_val, y_val)
        print(f"  Validation Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"\n  Per-class accuracy:")
        for cls, acc in metrics['per_class_accuracy'].items():
            if acc is not None:
                print(f"    {cls}: {acc*100:.1f}%")

        # Save model
        print("\nSaving models...")
        saved = save_model_for_inference(model, args.output_dir, 'gesture_model')
        for fmt, path in saved.items():
            print(f"  {fmt}: {path}")

        # Save training results
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': vars(args),
            'summary': summary,
            'history': {k: [float(v) for v in vals] for k, vals in history.items()},
            'metrics': metrics,
            'saved_models': saved
        }
        results_path = output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Results: {results_path}")

    else:
        # PyTorch training
        from .model import GestureCNN
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        model = GestureCNN(window_size=args.window_size, num_classes=num_classes)
        print(model)

        # Prepare data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        # Training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience_counter = 0

        print(f"\nTraining for up to {args.epochs} epochs...")
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()

            val_acc = correct / total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}, "
                  f"val_acc={val_acc*100:.2f}%")

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), output_dir / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print("Early stopping triggered")
                    break

        # Load best model and evaluate
        model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
        metrics = evaluate_model(model, X_val, y_val)
        print(f"\nFinal Validation Accuracy: {metrics['accuracy']*100:.2f}%")

        # Save model
        saved = save_model_for_inference(model, args.output_dir, 'gesture_model')
        for fmt, path in saved.items():
            print(f"  Saved {fmt}: {path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
