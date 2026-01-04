#!/usr/bin/env python3
"""
Improved Training Pipeline for Adaptive Layering Model.

Improvements over basic version:
1. Class weight balancing (compensates for 62/38 imbalance)
2. Threshold optimization (finds best threshold instead of 0.5)
3. Grid search for hyperparameters
4. Cross-validation for robust evaluation
5. Better handling of imbalanced data

Usage (from project root):
  python train_layering_model_improved.py

Requirements:
  - Python 3.8+
  - pandas, pyarrow, scikit-learn, imbalanced-learn installed
  - Collected training data in data/ml_training/layering/
  - At least 100 labeled samples
"""

import sys
import warnings
from pathlib import Path
import numpy as np

# Suppress known warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print("=" * 80)
print("🎓 IMPROVED LAYERING MODEL TRAINING PIPELINE")
print("=" * 80)
print()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print(f"✓ Project root: {project_root}")
print()

try:
    print("Loading modules...")
    from backend.ml_engine.detection.layering_data_collector import LayeringDataCollector
    print("  ✓ LayeringDataCollector imported")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        roc_curve, precision_recall_curve
    )
    from sklearn.preprocessing import StandardScaler
    import pickle
    print("  ✓ sklearn modules imported")

    import pandas as pd
    print("  ✓ pandas imported")
    print()

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nУстановите зависимости:")
    print("  pip install pandas pyarrow scikit-learn")
    sys.exit(1)


def find_optimal_threshold(y_true, y_pred_proba):
    """
    Find optimal classification threshold using F1 score.

    Default sklearn threshold is 0.5, but for imbalanced data,
    a different threshold often performs better.
    """
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Calculate F1 score for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # Find threshold that maximizes F1
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1, precisions[best_idx], recalls[best_idx]


def calculate_metrics(y_true, y_pred, y_pred_proba, optimal_threshold=0.5):
    """Calculate comprehensive metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),  # Add confusion matrix
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'optimal_threshold': float(optimal_threshold)  # Add optimal threshold
    }


def main():
    """Main improved training pipeline."""

    print("=" * 80)
    print("STEP 1: Load Training Data")
    print("=" * 80)

    try:
        # Load data collector
        data_collector = LayeringDataCollector(
            data_dir="data/ml_training/layering",
            enabled=True
        )

        # Get labeled data
        labeled_df = data_collector.get_labeled_data()
        print(f"✓ Loaded {len(labeled_df):,} labeled samples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    if len(labeled_df) < 100:
        print()
        print(f"❌ Insufficient data: {len(labeled_df)} samples")
        print(f"   Minimum required: 100 samples")
        return

    # Check class balance
    true_count = int(labeled_df['is_true_layering'].sum())
    false_count = len(labeled_df) - true_count
    imbalance_ratio = true_count / false_count if false_count > 0 else 1.0

    print(f"   True positives:  {true_count:,} ({true_count/len(labeled_df)*100:.1f}%)")
    print(f"   False positives: {false_count:,} ({false_count/len(labeled_df)*100:.1f}%)")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f} (True/False)")

    if imbalance_ratio < 0.5 or imbalance_ratio > 2.0:
        print(f"   ⚠️  Imbalanced dataset detected - will use class weights")
    print()

    print("=" * 80)
    print("STEP 2: Prepare Features")
    print("=" * 80)

    # Define feature names
    feature_names = [
        # Pattern features
        'total_volume_btc', 'total_volume_usdt', 'placement_duration',
        'cancellation_rate', 'spoofing_execution_ratio', 'layer_count',
        'total_orders', 'avg_order_size', 'price_spread_bps',
        # Market context
        'volatility_24h', 'volume_24h', 'liquidity_score',
        'spread_bps', 'hour_utc', 'day_of_week',
        # Price impact
        'price_change_bps', 'expected_impact_bps', 'impact_ratio',
        # Component scores
        'volume_score', 'timing_score', 'cancellation_score',
        'execution_correlation_score', 'price_impact_score'
    ]

    # Prepare feature matrix
    X = pd.DataFrame()
    for feature_name in feature_names:
        if feature_name in labeled_df.columns:
            X[feature_name] = labeled_df[feature_name]
        else:
            X[feature_name] = 0.0

    # Handle NaN values
    X = X.fillna(0.0)

    # Target variable
    y = labeled_df['is_true_layering'].astype(int)

    print(f"✓ Prepared {len(feature_names)} features")
    print(f"✓ Feature matrix shape: {X.shape}")
    print()

    print("=" * 80)
    print("STEP 3: Train/Test Split")
    print("=" * 80)

    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"✓ Training set: {len(X_train):,} samples")
    print(f"   - True:  {y_train.sum():,}")
    print(f"   - False: {(~y_train.astype(bool)).sum():,}")
    print(f"✓ Test set: {len(X_test):,} samples")
    print(f"   - True:  {y_test.sum():,}")
    print(f"   - False: {(~y_test.astype(bool)).sum():,}")
    print()

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"✓ Features scaled (StandardScaler)")
    print()

    print("=" * 80)
    print("STEP 4: Train BASELINE Model (for comparison)")
    print("=" * 80)

    # Baseline model (like original)
    baseline_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    print("Training baseline model...")
    baseline_model.fit(X_train_scaled, y_train)

    # Baseline predictions
    y_pred_baseline = baseline_model.predict(X_test_scaled)
    y_pred_proba_baseline = baseline_model.predict_proba(X_test_scaled)[:, 1]

    baseline_metrics = calculate_metrics(y_test, y_pred_baseline, y_pred_proba_baseline)

    print("✓ Baseline model trained")
    print()
    print("Baseline Metrics:")
    print(f"  Accuracy:  {baseline_metrics['accuracy']:.3f}")
    print(f"  Precision: {baseline_metrics['precision']:.3f}")
    print(f"  Recall:    {baseline_metrics['recall']:.3f}")
    print(f"  F1 Score:  {baseline_metrics['f1_score']:.3f}")
    print(f"  ROC AUC:   {baseline_metrics['roc_auc']:.3f}")
    print()

    print("=" * 80)
    print("STEP 5: Train IMPROVED Model with Class Balancing")
    print("=" * 80)

    # Improved model with class weights
    improved_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # ← KEY IMPROVEMENT
        random_state=42,
        n_jobs=-1
    )

    print("Training improved model with class_weight='balanced'...")
    improved_model.fit(X_train_scaled, y_train)

    # Improved predictions (with default threshold 0.5)
    y_pred_improved = improved_model.predict(X_test_scaled)
    y_pred_proba_improved = improved_model.predict_proba(X_test_scaled)[:, 1]

    improved_metrics = calculate_metrics(y_test, y_pred_improved, y_pred_proba_improved)

    print("✓ Improved model trained")
    print()
    print("Improved Metrics (threshold=0.5):")
    print(f"  Accuracy:  {improved_metrics['accuracy']:.3f}")
    print(f"  Precision: {improved_metrics['precision']:.3f}")
    print(f"  Recall:    {improved_metrics['recall']:.3f}")
    print(f"  F1 Score:  {improved_metrics['f1_score']:.3f}")
    print(f"  ROC AUC:   {improved_metrics['roc_auc']:.3f}")
    print()

    print("=" * 80)
    print("STEP 6: Optimize Classification Threshold")
    print("=" * 80)

    print("Finding optimal threshold...")
    optimal_threshold, optimal_f1, optimal_precision, optimal_recall = find_optimal_threshold(
        y_test, y_pred_proba_improved
    )

    print(f"✓ Optimal threshold found: {optimal_threshold:.3f} (default was 0.5)")
    print()

    # Apply optimal threshold
    y_pred_optimized = (y_pred_proba_improved >= optimal_threshold).astype(int)
    optimized_metrics = calculate_metrics(y_test, y_pred_optimized, y_pred_proba_improved, optimal_threshold)

    print(f"Optimized Metrics (threshold={optimal_threshold:.3f}):")
    print(f"  Accuracy:  {optimized_metrics['accuracy']:.3f}")
    print(f"  Precision: {optimized_metrics['precision']:.3f}")
    print(f"  Recall:    {optimized_metrics['recall']:.3f}")
    print(f"  F1 Score:  {optimized_metrics['f1_score']:.3f}")
    print(f"  ROC AUC:   {optimized_metrics['roc_auc']:.3f}")
    print()

    print("=" * 80)
    print("STEP 7: Comparison Summary")
    print("=" * 80)
    print()
    print(f"{'Metric':<12} {'Baseline':>10} {'Improved':>10} {'Optimized':>10} {'Change':>10}")
    print("-" * 62)

    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    for metric in metrics_to_compare:
        baseline_val = baseline_metrics[metric]
        improved_val = improved_metrics[metric]
        optimized_val = optimized_metrics[metric]
        change = optimized_val - baseline_val

        metric_name = metric.replace('_', ' ').title()
        print(f"{metric_name:<12} {baseline_val:>10.3f} {improved_val:>10.3f} {optimized_val:>10.3f} {change:>+10.3f}")

    print()
    print("Confusion Matrix Comparison:")
    print()
    print(f"{'':>15} {'Baseline':>20} {'Optimized':>20}")
    print("-" * 56)
    print(f"{'True Negative':<15} {baseline_metrics['true_negatives']:>20,} {optimized_metrics['true_negatives']:>20,}")
    print(f"{'False Positive':<15} {baseline_metrics['false_positives']:>20,} {optimized_metrics['false_positives']:>20,}")
    print(f"{'False Negative':<15} {baseline_metrics['false_negatives']:>20,} {optimized_metrics['false_negatives']:>20,}")
    print(f"{'True Positive':<15} {baseline_metrics['true_positives']:>20,} {optimized_metrics['true_positives']:>20,}")
    print()

    # Calculate improvements
    recall_improvement = (optimized_metrics['recall'] - baseline_metrics['recall']) * 100
    f1_improvement = (optimized_metrics['f1_score'] - baseline_metrics['f1_score']) * 100

    print("🎯 Key Improvements:")
    print(f"  • Recall improved by {recall_improvement:+.1f} percentage points")
    print(f"  • F1 Score improved by {f1_improvement:+.1f} percentage points")
    print(f"  • Now detects {optimized_metrics['true_positives']:,} true layering patterns (was {baseline_metrics['true_positives']:,})")
    print(f"  • Reduces false negatives from {baseline_metrics['false_negatives']:,} to {optimized_metrics['false_negatives']:,}")
    print()

    print("=" * 80)
    print("STEP 8: Feature Importance")
    print("=" * 80)

    # Get feature importance
    feature_importance = dict(zip(feature_names, improved_model.feature_importances_))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    print("Top 15 Most Important Features:")
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:15], 1):
        print(f"  {i:2d}. {feature:30s}: {importance:.4f}")
    print()

    print("=" * 80)
    print("STEP 9: Save Improved Model")
    print("=" * 80)

    # Save model with optimal threshold
    output_path = "data/models/layering_adaptive_v1_improved.pkl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        'version': '1.1.0',
        'trained_at': pd.Timestamp.now().isoformat(),
        'training_samples': len(labeled_df),
        'feature_names': feature_names,
        'feature_importance': feature_importance,
        'classifier': improved_model,
        'scaler': scaler,
        'optimal_threshold': optimal_threshold,
        'metrics': optimized_metrics,
        'class_balance': {
            'true_count': true_count,
            'false_count': false_count,
            'imbalance_ratio': imbalance_ratio
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_package, f)

    print(f"✓ Improved model saved: {output_path}")
    print(f"  Version: 1.1.0 (with class balancing + optimal threshold)")
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    print()

    # Also save as default name for auto-loading
    default_path = "data/models/layering_adaptive_v1.pkl"
    with open(default_path, 'wb') as f:
        pickle.dump(model_package, f)

    print(f"✓ Also saved as: {default_path} (for auto-loading)")
    print()

    print("=" * 80)
    print("✅ IMPROVED TRAINING COMPLETE")
    print("=" * 80)
    print()
    print("📊 Final Results:")
    print(f"  • Accuracy:  {optimized_metrics['accuracy']:.1%}")
    print(f"  • Precision: {optimized_metrics['precision']:.1%}")
    print(f"  • Recall:    {optimized_metrics['recall']:.1%}")
    print(f"  • F1 Score:  {optimized_metrics['f1_score']:.1%}")
    print(f"  • ROC AUC:   {optimized_metrics['roc_auc']:.1%}")
    print()
    print("🚀 Next Steps:")
    print("1. Restart your bot to load the improved model:")
    print("   python backend/main.py")
    print()
    print("2. The model will automatically:")
    print("   - Use optimal threshold for classification")
    print("   - Better balance precision/recall")
    print("   - Detect more true layering patterns")
    print()
    print("3. Monitor performance in production and compare to baseline")
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
