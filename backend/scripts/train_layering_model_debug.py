#!/usr/bin/env python3
"""
Training Pipeline for Adaptive Layering Model (Debug Version).

This version uses print() instead of logger for better Windows debugging.

Usage (from project root):
  python train_layering_model_debug.py

Requirements:
  - Python 3.8+
  - pandas, pyarrow, scikit-learn installed
  - Collected training data in data/ml_training/layering/
  - At least 100 labeled samples
"""

import sys
import warnings
from pathlib import Path

# Suppress known warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print("=" * 80)
print("🎓 ADAPTIVE LAYERING MODEL TRAINING PIPELINE (DEBUG)")
print("=" * 80)
print()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print(f"✓ Project root: {project_root}")
print(f"✓ Python path updated")
print()

try:
    print("Loading modules...")
    from backend.ml_engine.detection.layering_data_collector import LayeringDataCollector
    print("  ✓ LayeringDataCollector imported")

    from backend.ml_engine.detection.adaptive_layering_model import AdaptiveLayeringModel
    print("  ✓ AdaptiveLayeringModel imported")
    print()

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nУстановите зависимости:")
    print("  pip install pandas pyarrow scikit-learn")
    sys.exit(1)


def main():
    """Main training pipeline with debug output."""

    print("=" * 80)
    print("STEP 1: Initialize Data Collector")
    print("=" * 80)

    try:
        # 1. Load data collector
        data_collector = LayeringDataCollector(
            data_dir="data/ml_training/layering",
            enabled=True
        )
        print(f"✓ Data collector initialized")
        print()
    except Exception as e:
        print(f"❌ Error initializing data collector: {e}")
        import traceback
        traceback.print_exc()
        return

    print("=" * 80)
    print("STEP 2: Load Training Data")
    print("=" * 80)

    try:
        # 2. Get labeled data
        labeled_df = data_collector.get_labeled_data()
        print(f"✓ Loaded {len(labeled_df)} labeled samples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    if len(labeled_df) < 10:
        print()
        print(f"❌ Insufficient labeled data: {len(labeled_df)} samples")
        print(f"   Minimum required: 10 samples")
        print(f"   Recommended: 100+ samples")
        print()
        print("Для обучения модели необходимо:")
        print("1. Собрать данные в режиме ONLY_TRAINING или full mode")
        print("2. Пометить данные используя data_collector.update_label()")
        print("3. Минимум 10 образцов с метками (рекомендуется 100+)")
        return

    # Check class balance
    true_count = int(labeled_df['is_true_layering'].sum())
    false_count = len(labeled_df) - true_count

    print(f"   True positives:  {true_count} ({true_count/len(labeled_df)*100:.1f}%)")
    print(f"   False positives: {false_count} ({false_count/len(labeled_df)*100:.1f}%)")
    print(f"   Balance: {true_count / len(labeled_df):.1%} positive")
    print()

    if true_count < 5 or false_count < 5:
        print("⚠️  Unbalanced dataset - нужно минимум 5 примеров каждого класса")
        print()

    print("=" * 80)
    print("STEP 3: Initialize ML Model")
    print("=" * 80)

    try:
        # 3. Initialize model
        model = AdaptiveLayeringModel(enabled=True)
        print(f"✓ Model initialized")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return

    if not model.enabled:
        print("❌ sklearn not available - cannot train model")
        print("   Install: pip install scikit-learn")
        return

    print()

    print("=" * 80)
    print("STEP 4: Train Model (this may take 2-10 minutes)")
    print("=" * 80)

    try:
        # 4. Train model
        print(f"Training with {len(labeled_df)} samples...")
        print("Please wait...")
        print()

        metrics = model.train(
            training_data=labeled_df,
            target_column='is_true_layering',
            test_size=0.2
        )

        print()
        print("✓ Training completed successfully!")
        print()
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Display results
    print("=" * 80)
    print("STEP 5: Training Results")
    print("=" * 80)
    print(f"Accuracy:  {metrics.accuracy:.3f}")
    print(f"Precision: {metrics.precision:.3f}")
    print(f"Recall:    {metrics.recall:.3f}")
    print(f"F1 Score:  {metrics.f1_score:.3f}")
    print(f"ROC AUC:   {metrics.roc_auc:.3f}")
    print()
    print("Confusion Matrix:")
    print(f"  TN: {metrics.true_negatives:4d}  FP: {metrics.false_positives:4d}")
    print(f"  FN: {metrics.false_negatives:4d}  TP: {metrics.true_positives:4d}")
    print()

    # 6. Save model
    print("=" * 80)
    print("STEP 6: Save Model")
    print("=" * 80)

    try:
        output_path = "data/models/layering_adaptive_v1.pkl"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        model.save(output_path)
        print(f"✓ Model saved: {output_path}")
        print()
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 7. Feature importance
    print("=" * 80)
    print("STEP 7: Feature Importance (Top 10)")
    print("=" * 80)

    for i, (feature, importance) in enumerate(
        list(model.feature_importance.items())[:10], 1
    ):
        print(f"  {i:2d}. {feature:30s}: {importance:.4f}")

    print()
    print("=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Restart your bot to load the trained model")
    print("2. The model will automatically improve layering detection")
    print("3. Monitor detection accuracy in production")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
