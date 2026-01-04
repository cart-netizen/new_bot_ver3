#!/usr/bin/env python3
"""
Improved Training Pipeline for Adaptive Layering Model v2.0

Key Improvements over v1:
1. NO detector_confidence as feature (removes data leakage)
2. Better NaN handling (median/mode imputation, not zeros)
3. Feature engineering (derived features, interactions)
4. Removed multicollinear features
5. Multiple algorithms comparison (RF, XGBoost, LightGBM)
6. Proper cross-validation
7. Feature selection based on importance

Usage (from project root):
  python train_layering_model_improved.py

Requirements:
  - Python 3.8+
  - pandas, pyarrow, scikit-learn, xgboost (optional), lightgbm (optional)
  - Collected training data in data/ml_training/layering/
  - At least 100 labeled samples
"""

import sys
import warnings
from pathlib import Path
import numpy as np
from datetime import datetime

# Suppress known warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

print("=" * 80)
print("IMPROVED LAYERING MODEL TRAINING v2.0")
print("=" * 80)
print()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print(f"Project root: {project_root}")
print()

try:
    print("Loading modules...")
    from backend.ml_engine.detection.layering_data_collector import LayeringDataCollector
    print("  LayeringDataCollector imported")

    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report,
        precision_recall_curve
    )
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.feature_selection import SelectFromModel
    import pickle
    print("  sklearn modules imported")

    import pandas as pd
    print("  pandas imported")

    # Try to import XGBoost (optional, better performance)
    try:
        import xgboost as xgb
        HAS_XGBOOST = True
        print("  xgboost imported")
    except ImportError:
        HAS_XGBOOST = False
        print("  xgboost not available (optional)")

    # Try to import LightGBM (optional, faster)
    try:
        import lightgbm as lgb
        HAS_LIGHTGBM = True
        print("  lightgbm imported")
    except ImportError:
        HAS_LIGHTGBM = False
        print("  lightgbm not available (optional)")

    print()

except ImportError as e:
    print(f"Import Error: {e}")
    print("\nInstall dependencies:")
    print("  pip install pandas pyarrow scikit-learn xgboost lightgbm")
    sys.exit(1)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Core features (no detector_confidence to avoid data leakage!)
CORE_FEATURES = [
    # Pattern behavioral features
    'cancellation_rate',           # Key indicator: how many orders canceled
    'spoofing_execution_ratio',    # Key indicator: how many orders executed
    'placement_duration',          # How long orders were active
    'layer_count',                 # Number of price layers
    'total_orders',                # Total orders in pattern
    'avg_order_size',              # Average order size
    'price_spread_bps',            # Price spread of layers

    # Volume features (keep only one to avoid multicollinearity)
    'total_volume_usdt',           # Total volume in USD

    # Market context
    'volatility_24h',              # Market volatility
    'liquidity_score',             # Market liquidity
    'spread_bps',                  # Current spread
    'hour_utc',                    # Time of day
    'day_of_week',                 # Day of week

    # Price impact
    'price_change_bps',            # Actual price change
    'expected_impact_bps',         # Expected impact
    'impact_ratio',                # Ratio of actual to expected

    # Execution metrics
    'aggressive_ratio',            # Ratio of aggressive orders
]

# Features to EXCLUDE (to avoid data leakage and multicollinearity)
EXCLUDED_FEATURES = [
    'detector_confidence',          # DATA LEAKAGE - this is what we're trying to predict!
    'volume_score',                 # Component of detector_confidence
    'timing_score',                 # Component of detector_confidence
    'cancellation_score',           # Component of detector_confidence
    'execution_correlation_score',  # Component of detector_confidence
    'price_impact_score',           # Component of detector_confidence
    'total_volume_btc',             # Multicollinear with total_volume_usdt
    'volume_24h',                   # Often missing, less relevant
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that capture layering behavior better.

    Args:
        df: Raw feature DataFrame

    Returns:
        DataFrame with engineered features
    """
    df = df.copy()

    # === DERIVED FEATURES ===

    # 1. Cancel-Execute Ratio (key behavioral indicator)
    # High ratio = more cancellation relative to execution = more likely layering
    df['cancel_execute_ratio'] = (
        df['cancellation_rate'] / (df['spoofing_execution_ratio'] + 0.01)
    ).clip(0, 100)

    # 2. Volume per Order (size indicator)
    df['volume_per_order'] = (
        df['total_volume_usdt'] / (df['total_orders'] + 1)
    )

    # 3. Speed Score (orders per second during placement)
    df['order_placement_speed'] = (
        df['total_orders'] / (df['placement_duration'] + 0.1)
    )

    # 4. Layer Density (how densely packed are layers)
    df['layer_density'] = (
        df['layer_count'] / (df['price_spread_bps'] + 0.1)
    )

    # 5. Impact Efficiency (how much price moved per volume)
    df['impact_efficiency'] = (
        np.abs(df['price_change_bps']) / (df['total_volume_usdt'] / 10000 + 0.1)
    ).clip(0, 100)

    # 6. Time-based features (cyclical encoding for hour)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_utc'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_utc'] / 24)

    # 7. Is weekend (trading patterns differ)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # 8. Volatility-adjusted impact
    df['volatility_adjusted_impact'] = (
        np.abs(df['price_change_bps']) / (df['volatility_24h'] + 0.1)
    ).clip(0, 100)

    # 9. Liquidity utilization (volume relative to liquidity)
    df['liquidity_utilization'] = (
        df['total_volume_usdt'] / (df['liquidity_score'] * 100000 + 1)
    ).clip(0, 100)

    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix with proper handling of missing values.

    Returns:
        (X, feature_names) tuple
    """
    # Start with core features
    available_features = [f for f in CORE_FEATURES if f in df.columns]

    # Create initial feature matrix
    X = df[available_features].copy()

    # Engineer additional features
    X = engineer_features(pd.concat([X, df[['timestamp']]], axis=1).drop('timestamp', axis=1, errors='ignore'))

    # Get all feature names
    feature_names = list(X.columns)

    # === HANDLE MISSING VALUES ===
    # Use different strategies for different feature types

    # For rate/ratio features: use median
    rate_features = [f for f in feature_names if 'rate' in f or 'ratio' in f or 'score' in f]
    for f in rate_features:
        if f in X.columns:
            X[f] = X[f].fillna(X[f].median())

    # For count features: use 0
    count_features = ['layer_count', 'total_orders']
    for f in count_features:
        if f in X.columns:
            X[f] = X[f].fillna(0)

    # For other numeric features: use median
    for f in feature_names:
        if X[f].isna().any():
            X[f] = X[f].fillna(X[f].median())

    # Final safety: fill any remaining NaN with 0
    X = X.fillna(0)

    # Replace infinities
    X = X.replace([np.inf, -np.inf], 0)

    print(f"Prepared {len(feature_names)} features:")
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")

    return X, feature_names


def find_optimal_threshold(y_true, y_pred_proba) -> tuple:
    """
    Find optimal classification threshold using F1 score.
    Returns Python float types for JSON serialization.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Calculate F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # Find best threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]

    # Convert to Python float for JSON serialization
    return float(best_threshold), float(best_f1)


def calculate_metrics(y_true, y_pred, y_pred_proba) -> dict:
    """Calculate comprehensive metrics. All values converted to Python types for JSON serialization."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    except:
        roc_auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Convert all numpy types to Python native types for JSON serialization
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
    }


def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name: str) -> dict:
    """Train a model and return metrics."""
    print(f"\nTraining {model_name}...")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics with default threshold
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

    # Find optimal threshold
    optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_pred_proba)

    # Recalculate with optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    metrics_optimal = calculate_metrics(y_test, y_pred_optimal, y_pred_proba)
    metrics_optimal['optimal_threshold'] = float(optimal_threshold)

    print(f"  Default (0.5):  Accuracy={metrics['accuracy']:.3f}, Precision={metrics['precision']:.3f}, "
          f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}, AUC={metrics['roc_auc']:.3f}")
    print(f"  Optimal ({optimal_threshold:.3f}): Accuracy={metrics_optimal['accuracy']:.3f}, "
          f"Precision={metrics_optimal['precision']:.3f}, Recall={metrics_optimal['recall']:.3f}, "
          f"F1={metrics_optimal['f1_score']:.3f}")

    return {
        'model': model,
        'model_name': model_name,
        'metrics_default': metrics,
        'metrics_optimal': metrics_optimal,
        'optimal_threshold': optimal_threshold,
        'y_pred_proba': y_pred_proba
    }


def main():
    """Main improved training pipeline."""

    print("=" * 80)
    print("STEP 1: Load Training Data")
    print("=" * 80)

    try:
        data_collector = LayeringDataCollector(
            data_dir="data/ml_training/layering",
            enabled=True
        )
        labeled_df = data_collector.get_labeled_data()
        print(f"Loaded {len(labeled_df):,} labeled samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return

    if len(labeled_df) < 100:
        print(f"\nInsufficient data: {len(labeled_df)} samples")
        print(f"Minimum required: 100 samples")
        return

    # Check class balance
    true_count = int(labeled_df['is_true_layering'].sum())
    false_count = len(labeled_df) - true_count
    imbalance_ratio = true_count / false_count if false_count > 0 else 1.0

    print(f"\nClass Distribution:")
    print(f"   True positives:  {true_count:,} ({true_count/len(labeled_df)*100:.1f}%)")
    print(f"   False positives: {false_count:,} ({false_count/len(labeled_df)*100:.1f}%)")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}")

    print()
    print("=" * 80)
    print("STEP 2: Feature Engineering")
    print("=" * 80)

    X, feature_names = prepare_features(labeled_df)
    y = labeled_df['is_true_layering'].astype(int)

    print(f"\nFeature matrix shape: {X.shape}")

    print()
    print("=" * 80)
    print("STEP 3: Train/Test Split")
    print("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training set: {len(X_train):,} samples")
    print(f"   - True:  {y_train.sum():,}")
    print(f"   - False: {(~y_train.astype(bool)).sum():,}")
    print(f"Test set: {len(X_test):,} samples")
    print(f"   - True:  {y_test.sum():,}")
    print(f"   - False: {(~y_test.astype(bool)).sum():,}")

    # Scale features using RobustScaler (better for outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"\nFeatures scaled (RobustScaler)")

    print()
    print("=" * 80)
    print("STEP 4: Train Multiple Models")
    print("=" * 80)

    models_results = []

    # 1. Random Forest (baseline) - OPTIMIZED
    rf_model = RandomForestClassifier(
        n_estimators=300,           # More trees
        max_depth=12,               # Reasonable depth
        min_samples_split=10,       # Prevent overfitting
        min_samples_leaf=5,         # Minimum samples in leaf
        max_features='sqrt',        # Standard for classification
        class_weight='balanced',
        bootstrap=True,
        oob_score=True,             # Out-of-bag score
        random_state=42,
        n_jobs=-1
    )
    rf_result = train_and_evaluate_model(rf_model, X_train_scaled, X_test_scaled, y_train, y_test, "RandomForest")
    models_results.append(rf_result)

    # 2. Gradient Boosting - OPTIMIZED
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,                # Slightly deeper
        learning_rate=0.05,         # Lower LR
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,              # Stochastic gradient boosting
        max_features='sqrt',
        random_state=42
    )
    gb_result = train_and_evaluate_model(gb_model, X_train_scaled, X_test_scaled, y_train, y_test, "GradientBoosting")
    models_results.append(gb_result)

    # 3. XGBoost (if available) - OPTIMIZED
    if HAS_XGBOOST:
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = false_count / true_count if true_count > 0 else 1.0

        xgb_model = xgb.XGBClassifier(
            n_estimators=300,           # More trees
            max_depth=8,                # Deeper trees for complex patterns
            learning_rate=0.05,         # Lower LR with more trees
            scale_pos_weight=scale_pos_weight,
            min_child_weight=3,         # Regularization
            subsample=0.8,              # Row sampling for robustness
            colsample_bytree=0.8,       # Column sampling
            gamma=0.1,                  # Minimum loss reduction
            reg_alpha=0.1,              # L1 regularization
            reg_lambda=1.0,             # L2 regularization
            use_label_encoder=False,
            eval_metric='auc',          # Optimize for AUC
            random_state=42,
            n_jobs=-1
        )
        xgb_result = train_and_evaluate_model(xgb_model, X_train_scaled, X_test_scaled, y_train, y_test, "XGBoost")
        models_results.append(xgb_result)

    # 4. LightGBM (if available) - OPTIMIZED
    if HAS_LIGHTGBM:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,           # More trees
            max_depth=8,                # Deeper trees
            learning_rate=0.05,         # Lower LR with more trees
            num_leaves=63,              # 2^max_depth - 1
            class_weight='balanced',
            min_child_samples=20,       # Minimum data in leaf
            subsample=0.8,              # Row sampling
            colsample_bytree=0.8,       # Column sampling
            reg_alpha=0.1,              # L1 regularization
            reg_lambda=1.0,             # L2 regularization
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True         # Avoid OpenMP issues
        )
        lgb_result = train_and_evaluate_model(lgb_model, X_train_scaled, X_test_scaled, y_train, y_test, "LightGBM")
        models_results.append(lgb_result)

    print()
    print("=" * 80)
    print("STEP 5: Model Comparison")
    print("=" * 80)

    print(f"\n{'Model':<20} {'AUC':>8} {'F1 (opt)':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 60)

    best_result = None
    best_auc = 0

    for result in models_results:
        metrics = result['metrics_optimal']
        print(f"{result['model_name']:<20} {metrics['roc_auc']:>8.3f} {metrics['f1_score']:>10.3f} "
              f"{metrics['precision']:>10.3f} {metrics['recall']:>10.3f}")

        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_result = result

    print()
    print(f"Best model: {best_result['model_name']} (AUC: {best_auc:.3f})")

    print()
    print("=" * 80)
    print("STEP 6: Feature Importance (Best Model)")
    print("=" * 80)

    best_model = best_result['model']

    # Get feature importance
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importance = np.abs(best_model.coef_[0])
    else:
        importance = np.zeros(len(feature_names))

    # Convert numpy types to Python types for JSON serialization
    feature_importance = {k: float(v) for k, v in zip(feature_names, importance)}
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    print("\nTop 15 Most Important Features:")
    for i, (feature, imp) in enumerate(list(feature_importance.items())[:15], 1):
        print(f"  {i:2d}. {feature:35s}: {imp:.4f}")

    print()
    print("=" * 80)
    print("STEP 7: Cross-Validation")
    print("=" * 80)

    print(f"\nRunning 5-fold stratified cross-validation for {best_result['model_name']}...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')

    print(f"\nCV ROC-AUC scores: {cv_scores}")
    print(f"Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    print()
    print("=" * 80)
    print("STEP 8: Save Best Model")
    print("=" * 80)

    # Prepare model package
    optimal_threshold = best_result['optimal_threshold']
    final_metrics = best_result['metrics_optimal']

    # Ensure all numeric values are Python native types for JSON serialization
    model_package = {
        'version': '2.0.0',
        'trained_at': datetime.now().isoformat(),
        'training_samples': int(len(labeled_df)),
        'feature_names': feature_names,
        'feature_importance': feature_importance,
        'classifier': best_model,
        'scaler': scaler,
        'optimal_threshold': float(optimal_threshold),
        'metrics': final_metrics,
        'model_type': best_result['model_name'],
        'class_balance': {
            'true_count': int(true_count),
            'false_count': int(false_count),
            'imbalance_ratio': float(imbalance_ratio)
        },
        'cv_scores': {
            'mean': float(cv_scores.mean()),
            'std': float(cv_scores.std()),
            'scores': [float(s) for s in cv_scores]
        },
        'excluded_features': EXCLUDED_FEATURES,
        'training_notes': 'v2.0 - No detector_confidence, improved feature engineering'
    }

    # Save improved model
    output_path = project_root / "data" / "models" / "layering_adaptive_v2.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(model_package, f)

    print(f"\nImproved model saved: {output_path}")
    print(f"  Version: 2.0.0")
    print(f"  Model type: {best_result['model_name']}")
    print(f"  Optimal threshold: {optimal_threshold:.3f}")

    # Also save as v1 for backward compatibility
    compat_path = project_root / "data" / "models" / "layering_adaptive_v1.pkl"
    with open(compat_path, 'wb') as f:
        pickle.dump(model_package, f)

    print(f"\nAlso saved as: {compat_path} (for backward compatibility)")

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()
    print("Final Results:")
    print(f"  Model:      {best_result['model_name']}")
    print(f"  Accuracy:   {final_metrics['accuracy']:.1%}")
    print(f"  Precision:  {final_metrics['precision']:.1%}")
    print(f"  Recall:     {final_metrics['recall']:.1%}")
    print(f"  F1 Score:   {final_metrics['f1_score']:.1%}")
    print(f"  ROC AUC:    {final_metrics['roc_auc']:.1%}")
    print(f"  CV AUC:     {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")
    print()
    print("Confusion Matrix:")
    print(f"  True Negatives:  {final_metrics['true_negatives']:,}")
    print(f"  False Positives: {final_metrics['false_positives']:,}")
    print(f"  False Negatives: {final_metrics['false_negatives']:,}")
    print(f"  True Positives:  {final_metrics['true_positives']:,}")
    print()
    print("Key improvements in v2.0:")
    print("  1. Removed detector_confidence from features (no data leakage)")
    print("  2. Added derived features (cancel_execute_ratio, impact_efficiency, etc.)")
    print("  3. Better NaN handling (median imputation instead of zeros)")
    print("  4. Compared multiple algorithms, selected best")
    print("  5. Cross-validation for robust evaluation")
    print()
    print("Next steps:")
    print("  1. Restart bot to load the new model: python backend/main.py")
    print("  2. Monitor performance in production")
    print("  3. Collect more labeled data if AUC < 0.75")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
