#!/usr/bin/env python3
"""
Training Pipeline for Adaptive Layering Model.

Usage (from project root):
  python backend/scripts/train_layering_model.py

Требования:
  - Collected training data в data/ml_training/layering/
  - Labeled samples (is_true_layering не None)
  - sklearn установлен

Path: backend/scripts/train_layering_model.py
"""

import sys
import warnings
from pathlib import Path

# Suppress known warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# Add project root to path
# Script is in: backend/scripts/train_layering_model.py
# We need to add the directory that contains 'backend' folder
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.ml_engine.detection.layering_data_collector import LayeringDataCollector
from backend.ml_engine.detection.adaptive_layering_model import AdaptiveLayeringModel
from backend.core.logger import get_logger

logger = get_logger(__name__)


def main():
  """Main training pipeline."""
  logger.info("=" * 80)
  logger.info("🎓 ADAPTIVE LAYERING MODEL TRAINING PIPELINE")
  logger.info("=" * 80)

  # 1. Load data collector
  data_collector = LayeringDataCollector(
    data_dir="data/ml_training/layering",
    enabled=True
  )

  # 2. Get labeled data
  logger.info("📚 Loading training data...")
  labeled_df = data_collector.get_labeled_data()

  if len(labeled_df) < 10:
    logger.error(
      f"❌ Insufficient labeled data: {len(labeled_df)} samples "
      f"(minimum 10 required)"
    )
    logger.info("\nДля обучения модели необходимо:")
    logger.info("1. Собрать данные в режиме ONLY_TRAINING или full mode")
    logger.info("2. Пометить данные используя data_collector.update_label()")
    logger.info("3. Минимум 10 образцов с метками (рекомендуется 100+)")
    return

  logger.info(f"✅ Loaded {len(labeled_df)} labeled samples")

  # Check class balance
  true_count = labeled_df['is_true_layering'].sum()
  false_count = len(labeled_df) - true_count

  logger.info(f"   True positives: {true_count}")
  logger.info(f"   False positives: {false_count}")
  logger.info(f"   Balance: {true_count / len(labeled_df):.1%} positive")

  if true_count < 5 or false_count < 5:
    logger.warning(
      "⚠️  Unbalanced dataset - нужно минимум 5 примеров каждого класса"
    )

  # 3. Initialize model
  model = AdaptiveLayeringModel(enabled=True)

  if not model.enabled:
    logger.error("❌ sklearn not available - cannot train model")
    return

  # 4. Train model
  logger.info("🎓 Training model...")
  metrics = model.train(
    training_data=labeled_df,
    target_column='is_true_layering',
    test_size=0.2
  )

  # 5. Display results
  logger.info("=" * 80)
  logger.info("📊 TRAINING RESULTS")
  logger.info("=" * 80)
  logger.info(f"Accuracy:  {metrics.accuracy:.3f}")
  logger.info(f"Precision: {metrics.precision:.3f}")
  logger.info(f"Recall:    {metrics.recall:.3f}")
  logger.info(f"F1 Score:  {metrics.f1_score:.3f}")
  logger.info(f"ROC AUC:   {metrics.roc_auc:.3f}")
  logger.info("")
  logger.info("Confusion Matrix:")
  logger.info(f"  TN: {metrics.true_negatives:4d}  FP: {metrics.false_positives:4d}")
  logger.info(f"  FN: {metrics.false_negatives:4d}  TP: {metrics.true_positives:4d}")

  # 6. Save model
  output_path = "data/models/layering_adaptive_v1.pkl"
  Path(output_path).parent.mkdir(parents=True, exist_ok=True)

  model.save(output_path)
  logger.info("")
  logger.info(f"✅ Model saved: {output_path}")

  # 7. Feature importance
  logger.info("")
  logger.info("🔝 Top 10 Features:")
  for i, (feature, importance) in enumerate(
      list(model.feature_importance.items())[:10], 1
  ):
    logger.info(f"  {i:2d}. {feature:30s}: {importance:.4f}")

  logger.info("=" * 80)
  logger.info("✅ TRAINING COMPLETE")
  logger.info("=" * 80)


if __name__ == "__main__":
  main()
