#!/usr/bin/env python3
"""
Simple CLI script to train ML model

Usage:
    python train_model.py                    # Quick train with defaults
    python train_model.py --epochs 100       # Custom epochs
    python train_model.py --help             # Show all options
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.ml_engine.training_orchestrator import TrainingOrchestrator
# UPDATED: Используем оптимизированные v2 версии
from backend.ml_engine.models.hybrid_cnn_lstm_v2 import ModelConfigV2 as ModelConfig
from backend.ml_engine.training.model_trainer_v2 import TrainerConfigV2 as TrainerConfig
from backend.ml_engine.training.data_loader import DataConfig


async def train(
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 20,
    export_onnx: bool = True,
    auto_promote: bool = True
):
    """Train model with specified parameters"""

    print("\n" + "=" * 60)
    print("ML MODEL TRAINING")
    print("=" * 60)
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Early Stopping Patience: {early_stopping_patience}")
    print(f"Export ONNX: {export_onnx}")
    print(f"Auto Promote: {auto_promote}")
    print("=" * 60 + "\n")

    # Create configs
    model_config = ModelConfig()
    trainer_config = TrainerConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience
    )
    data_config = DataConfig(
        batch_size=batch_size
    )

    # Create orchestrator
    orchestrator = TrainingOrchestrator(
        model_config=model_config,
        trainer_config=trainer_config,
        data_config=data_config
    )

    # Train
    result = await orchestrator.train_model(
        export_onnx=export_onnx,
        auto_promote=auto_promote
    )

    # Print results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    if result["success"]:
        print(f"✅ Success!")
        print(f"Version: {result['version']}")
        print(f"Model Path: {result['model_path']}")

        if result.get('test_metrics'):
            metrics = result['test_metrics']
            print(f"\nTest Metrics:")
            print(f"  Accuracy:  {metrics.get('accuracy', 0):.4f}")
            print(f"  Precision: {metrics.get('precision', 0):.4f}")
            print(f"  Recall:    {metrics.get('recall', 0):.4f}")
            print(f"  F1 Score:  {metrics.get('f1', 0):.4f}")

        if result.get('onnx_path'):
            print(f"\nONNX Model: {result['onnx_path']}")

        if result.get('promoted_to_production'):
            print(f"\n🚀 Model promoted to PRODUCTION!")
        else:
            print(f"\n⚠️  Model not promoted (accuracy threshold not met)")

    else:
        print(f"❌ Training failed: {result.get('error', 'Unknown error')}")

    print("=" * 60 + "\n")

    return result


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train ML model for trading bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_model.py                                    # Quick train with defaults
  python train_model.py --epochs 100                       # Train for 100 epochs
  python train_model.py --epochs 50 --lr 0.0001            # Custom epochs and learning rate
  python train_model.py --patience 30                      # Increase early stopping patience
  python train_model.py --epochs 100 --patience 0          # Disable early stopping (patience=0)
  python train_model.py --no-onnx --no-promote             # Skip ONNX export and promotion
        """
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience in epochs (default: 20)"
    )

    parser.add_argument(
        "--no-onnx",
        action="store_true",
        help="Skip ONNX export"
    )

    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="Skip automatic promotion to production"
    )

    args = parser.parse_args()

    # Run training
    result = asyncio.run(train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        early_stopping_patience=args.patience,
        export_onnx=not args.no_onnx,
        auto_promote=not args.no_promote
    ))

    # Exit code based on success
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
