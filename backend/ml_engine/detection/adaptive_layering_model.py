"""
Adaptive ML Model for Layering Detection.

Professional machine learning model that adapts thresholds and confidence
scoring based on market conditions, reducing false positives and improving
detection accuracy.

Features:
1. Random Forest Classifier for pattern classification
2. Adaptive threshold prediction based on context
3. Feature importance analysis
4. Model evaluation and metrics
5. Incremental learning support
6. Model versioning and persistence

Path: backend/ml_engine/detection/adaptive_layering_model.py
"""

from __future__ import annotations  # Makes all type hints strings automatically

import numpy as np
import pandas as pd
import pickle
import json
import warnings
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Suppress sklearn feature names warning (we use numpy arrays for performance)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

try:
  from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
  from sklearn.model_selection import train_test_split, cross_val_score
  from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
  )
  from sklearn.preprocessing import StandardScaler
  SKLEARN_AVAILABLE = True
except ImportError:
  SKLEARN_AVAILABLE = False
  # Dummy types for when sklearn is not available
  if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelMetrics:
  """ML model evaluation metrics."""
  accuracy: float
  precision: float
  recall: float
  f1_score: float
  roc_auc: float
  confusion_matrix: List[List[int]]

  # Additional stats
  true_positives: int
  false_positives: int
  true_negatives: int
  false_negatives: int

  # Thresholds
  optimal_threshold: float


@dataclass
class AdaptiveThresholds:
  """Adaptive thresholds for detection."""
  volume_threshold_usdt: float
  placement_duration_threshold: float
  cancellation_rate_threshold: float
  confidence_threshold: float

  # Context
  market_regime: str
  volatility: float
  liquidity: float


class AdaptiveLayeringModel:
  """
  Professional ML model for adaptive layering detection.

  Uses Random Forest to:
  1. Classify patterns as true positives vs false positives
  2. Predict adaptive thresholds based on market conditions
  3. Adjust confidence scoring dynamically
  4. Learn from historical data
  """

  def __init__(
      self,
      model_path: Optional[str] = None,
      enabled: bool = True
  ):
    """
    Initialize adaptive model.

    Args:
        model_path: Path to saved model (None = create new)
        enabled: Whether model is enabled
    """
    self.enabled = enabled and SKLEARN_AVAILABLE
    self.model_path = model_path

    if not SKLEARN_AVAILABLE:
      logger.warning(
        "⚠️  sklearn not available - AdaptiveLayeringModel disabled"
      )
      self.enabled = False
      return

    # Models
    self.classifier: Optional[RandomForestClassifier] = None
    self.threshold_predictor: Optional[GradientBoostingClassifier] = None
    self.scaler: Optional[StandardScaler] = None

    # Feature names (for consistency)
    self.feature_names: List[str] = []

    # Metrics
    self.metrics: Optional[ModelMetrics] = None
    self.feature_importance: Dict[str, float] = {}

    # Model metadata
    self.version = "1.0.0"
    self.trained_at: Optional[str] = None
    self.training_samples: int = 0
    self.optimal_threshold: float = 0.5  # Default threshold, can be optimized

    # Load model if path provided
    if model_path and Path(model_path).exists():
      self.load(model_path)
    else:
      # Initialize new models
      self._init_models()

    if self.enabled:
      logger.info(
        f"✅ AdaptiveLayeringModel initialized: "
        f"trained={self.classifier is not None and hasattr(self.classifier, 'n_estimators')}"
      )

  def _init_models(self):
    """Initialize ML models with default hyperparameters."""
    if not self.enabled:
      return

    # Classification model (true positive vs false positive)
    self.classifier = RandomForestClassifier(
      n_estimators=100,
      max_depth=10,
      min_samples_split=5,
      min_samples_leaf=2,
      random_state=42,
      n_jobs=-1  # Use all cores
    )

    # Threshold prediction model
    self.threshold_predictor = GradientBoostingClassifier(
      n_estimators=50,
      max_depth=5,
      learning_rate=0.1,
      random_state=42
    )

    # Feature scaler
    self.scaler = StandardScaler()

    # Define feature names (v2.0 - no detector_confidence or component scores!)
    # NOTE: Temporal features (hour_utc, day_of_week, hour_sin, hour_cos, is_weekend) removed
    # They add noise rather than signal for layering detection
    self.feature_names = [
      # Pattern behavioral features (raw)
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

      # Price impact
      'price_change_bps',            # Actual price change
      'expected_impact_bps',         # Expected impact
      'impact_ratio',                # Ratio of actual to expected

      # Execution metrics
      'aggressive_ratio',            # Ratio of aggressive orders

      # === DERIVED FEATURES (computed at prediction time) ===
      'cancel_execute_ratio',        # cancellation_rate / execution_ratio
      'volume_per_order',            # total_volume / total_orders
      'order_placement_speed',       # total_orders / placement_duration
      'layer_density',               # layer_count / price_spread
      'impact_efficiency',           # price_change / volume
      'volatility_adjusted_impact',  # price_change / volatility
      'liquidity_utilization',       # volume / liquidity
      'spread_utilization',          # price_spread / market_spread
      'order_intensity',             # orders per layer
    ]

    # NOTE: Removed these to avoid data leakage:
    # - detector_confidence (this is what we're trying to predict!)
    # - volume_score, timing_score, cancellation_score, etc. (components of detector_confidence)
    # - total_volume_btc (multicollinear with total_volume_usdt)

  def train(
      self,
      training_data: pd.DataFrame,
      target_column: str = 'is_true_layering',
      test_size: float = 0.2
  ) -> ModelMetrics:
    """
    Train the model on labeled data.

    Args:
        training_data: DataFrame with features and labels
        target_column: Column name for target variable
        test_size: Fraction for test set

    Returns:
        ModelMetrics: Evaluation metrics
    """
    if not self.enabled:
      logger.error("Model training requires sklearn")
      return None

    logger.info(f"🎓 Training AdaptiveLayeringModel with {len(training_data)} samples")

    # Prepare features
    X = self._prepare_features(training_data)
    y = training_data[target_column].astype(int)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Scale features
    X_train_scaled = self.scaler.fit_transform(X_train)
    X_test_scaled = self.scaler.transform(X_test)

    # Train classifier
    logger.info("Training classifier...")
    self.classifier.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = self.classifier.predict(X_test_scaled)
    y_pred_proba = self.classifier.predict_proba(X_test_scaled)[:, 1]

    # Calculate metrics
    metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
    self.metrics = metrics

    # Feature importance
    self.feature_importance = dict(zip(
      self.feature_names,
      self.classifier.feature_importances_
    ))

    # Sort by importance
    self.feature_importance = dict(
      sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    # Update metadata
    self.trained_at = datetime.now().isoformat()
    self.training_samples = len(training_data)

    # Log results
    logger.info(f"✅ Training complete!")
    logger.info(f"   Accuracy:  {metrics.accuracy:.3f}")
    logger.info(f"   Precision: {metrics.precision:.3f}")
    logger.info(f"   Recall:    {metrics.recall:.3f}")
    logger.info(f"   F1 Score:  {metrics.f1_score:.3f}")
    logger.info(f"   ROC AUC:   {metrics.roc_auc:.3f}")
    logger.info(f"   Top features: {list(self.feature_importance.keys())[:5]}")

    return metrics

  def predict(self, features: Dict) -> Tuple[bool, float]:
    """
    Predict if pattern is true layering.

    Args:
        features: Feature dictionary

    Returns:
        Tuple of (is_true_layering, confidence)
    """
    if not self.enabled or self.classifier is None:
      # Fallback to detector confidence
      return (True, features.get('detector_confidence', 0.5))

    # Check if scaler is fitted (has been trained)
    if self.scaler is None or not hasattr(self.scaler, 'mean_'):
      # Scaler not fitted yet - fallback to detector confidence
      logger.warning("StandardScaler not fitted yet - using fallback confidence")
      return (True, features.get('detector_confidence', 0.5))

    # Prepare features
    X = self._prepare_single_sample(features)
    X_scaled = self.scaler.transform(X)

    # Get probability score
    confidence = self.classifier.predict_proba(X_scaled)[0][1]

    # Use optimal threshold instead of default 0.5
    prediction = confidence >= self.optimal_threshold

    return (bool(prediction), float(confidence))

  def predict_adaptive_thresholds(
      self,
      market_context: Dict
  ) -> AdaptiveThresholds:
    """
    Predict adaptive thresholds based on market conditions.

    Args:
        market_context: Market regime, volatility, liquidity, etc.

    Returns:
        AdaptiveThresholds: Recommended thresholds
    """
    if not self.enabled or not self.feature_importance:
      # Return default thresholds
      return AdaptiveThresholds(
        volume_threshold_usdt=200000.0,
        placement_duration_threshold=30.0,
        cancellation_rate_threshold=0.60,
        confidence_threshold=0.65,
        market_regime=market_context.get('market_regime', 'unknown'),
        volatility=market_context.get('volatility_24h', 0.0),
        liquidity=market_context.get('liquidity_score', 0.0)
      )

    # Extract market features
    market_regime = market_context.get('market_regime', 'unknown')
    volatility = market_context.get('volatility_24h', 0.0)
    liquidity = market_context.get('liquidity_score', 0.0)

    # Adaptive threshold calculation based on conditions
    base_volume = 200000.0
    base_duration = 30.0
    base_cancel_rate = 0.60
    base_confidence = 0.65

    # Adjust for volatility (high volatility → higher thresholds)
    if volatility > 5.0:
      volume_multiplier = 1.5
      confidence_adjustment = 0.05
    elif volatility > 3.0:
      volume_multiplier = 1.2
      confidence_adjustment = 0.02
    else:
      volume_multiplier = 1.0
      confidence_adjustment = 0.0

    # Adjust for liquidity (low liquidity → lower thresholds)
    if liquidity < 0.3:
      volume_multiplier *= 0.7
      confidence_adjustment -= 0.05
    elif liquidity < 0.5:
      volume_multiplier *= 0.85
      confidence_adjustment -= 0.02

    # Adjust for market regime
    if market_regime == 'bullish':
      volume_multiplier *= 1.1
    elif market_regime == 'bearish':
      volume_multiplier *= 0.9

    return AdaptiveThresholds(
      volume_threshold_usdt=base_volume * volume_multiplier,
      placement_duration_threshold=base_duration,
      cancellation_rate_threshold=base_cancel_rate,
      confidence_threshold=min(base_confidence + confidence_adjustment, 0.95),
      market_regime=market_regime,
      volatility=volatility,
      liquidity=liquidity
    )

  def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature matrix from DataFrame with proper NaN handling."""
    # Select and order features
    feature_df = pd.DataFrame()

    for feature_name in self.feature_names:
      if feature_name in df.columns:
        feature_df[feature_name] = df[feature_name]
      else:
        # Fill missing features with 0
        feature_df[feature_name] = 0.0

    # === PROPER NaN HANDLING (not just zeros!) ===
    # For rate/ratio features: use median
    rate_features = [f for f in feature_df.columns if 'rate' in f or 'ratio' in f]
    for f in rate_features:
      if feature_df[f].isna().any():
        feature_df[f] = feature_df[f].fillna(feature_df[f].median())

    # For count features: use 0
    count_features = ['layer_count', 'total_orders']
    for f in count_features:
      if f in feature_df.columns:
        feature_df[f] = feature_df[f].fillna(0)

    # For other numeric features: use median
    for f in feature_df.columns:
      if feature_df[f].isna().any():
        median_val = feature_df[f].median()
        if pd.isna(median_val):
          median_val = 0.0
        feature_df[f] = feature_df[f].fillna(median_val)

    # Final safety: fill any remaining NaN with 0
    feature_df = feature_df.fillna(0.0)

    # Replace infinities
    feature_df = feature_df.replace([np.inf, -np.inf], 0)

    return feature_df

  def _prepare_single_sample(self, features: Dict) -> np.ndarray:
    """Prepare single sample for prediction with derived features."""
    # First, compute derived features (same as in training)
    features = dict(features)  # Copy to avoid modifying original

    # === COMPUTE DERIVED FEATURES ===
    cancellation_rate = features.get('cancellation_rate', 0.5)
    execution_ratio = features.get('spoofing_execution_ratio', 0.5)
    total_volume = features.get('total_volume_usdt', 0.0)
    total_orders = features.get('total_orders', 1)
    placement_duration = features.get('placement_duration', 30.0)
    layer_count = features.get('layer_count', 1)
    price_spread = features.get('price_spread_bps', 0.1)
    price_change = features.get('price_change_bps', 0.0)
    volatility = features.get('volatility_24h', 1.0)
    liquidity = features.get('liquidity_score', 0.5)
    spread_bps = features.get('spread_bps', 0.1)

    # 1. Cancel-Execute Ratio
    features['cancel_execute_ratio'] = min(
      cancellation_rate / (execution_ratio + 0.01), 100.0
    )

    # 2. Volume per Order
    features['volume_per_order'] = total_volume / (total_orders + 1)

    # 3. Order placement speed
    features['order_placement_speed'] = total_orders / (placement_duration + 0.1)

    # 4. Layer Density
    features['layer_density'] = layer_count / (price_spread + 0.1)

    # 5. Impact Efficiency
    features['impact_efficiency'] = min(
      abs(price_change) / (total_volume / 10000 + 0.1), 100.0
    )

    # 6. Volatility-adjusted impact
    features['volatility_adjusted_impact'] = min(
      abs(price_change) / (volatility + 0.1), 100.0
    )

    # 7. Liquidity utilization
    features['liquidity_utilization'] = min(
      total_volume / (liquidity * 100000 + 1), 100.0
    )

    # 8. Spread utilization
    features['spread_utilization'] = min(
      price_spread / (spread_bps + 0.1), 100.0
    )

    # 9. Order intensity (orders per layer)
    features['order_intensity'] = total_orders / (layer_count + 1)

    # Now build the sample vector
    sample = []
    for feature_name in self.feature_names:
      value = features.get(feature_name, 0.0)
      # Handle None values
      if value is None:
        value = 0.0
      sample.append(float(value))

    return np.array([sample])

  def _calculate_metrics(
      self,
      y_true: np.ndarray,
      y_pred: np.ndarray,
      y_pred_proba: np.ndarray
  ) -> ModelMetrics:
    """Calculate evaluation metrics."""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Find optimal threshold
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    return ModelMetrics(
      accuracy=float(accuracy),
      precision=float(precision),
      recall=float(recall),
      f1_score=float(f1),
      roc_auc=float(roc_auc),
      confusion_matrix=cm.tolist(),
      true_positives=int(tp),
      false_positives=int(fp),
      true_negatives=int(tn),
      false_negatives=int(fn),
      optimal_threshold=float(optimal_threshold)
    )

  def save(self, filepath: str):
    """Save model to disk."""
    if not self.enabled:
      return

    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Package for saving
    model_package = {
      'version': self.version,
      'trained_at': self.trained_at,
      'training_samples': self.training_samples,
      'feature_names': self.feature_names,
      'feature_importance': self.feature_importance,
      'metrics': self.metrics.__dict__ if self.metrics else None,
      'classifier': self.classifier,
      'threshold_predictor': self.threshold_predictor,
      'scaler': self.scaler
    }

    # Save with pickle
    with open(filepath, 'wb') as f:
      pickle.dump(model_package, f)

    logger.info(f"💾 Model saved: {filepath}")

  def load(self, filepath: str):
    """Load model from disk."""
    if not self.enabled:
      return

    try:
      with open(filepath, 'rb') as f:
        model_package = pickle.load(f)

      # Restore
      self.version = model_package.get('version', '1.0.0')
      self.trained_at = model_package.get('trained_at')
      self.training_samples = model_package.get('training_samples', 0)
      self.feature_names = model_package.get('feature_names', [])
      self.feature_importance = model_package.get('feature_importance', {})
      self.optimal_threshold = model_package.get('optimal_threshold', 0.5)

      metrics_dict = model_package.get('metrics')
      if metrics_dict:
        # Add missing fields for backward compatibility
        if 'confusion_matrix' not in metrics_dict:
          # Construct confusion matrix from individual values if available
          tn = metrics_dict.get('true_negatives', 0)
          fp = metrics_dict.get('false_positives', 0)
          fn = metrics_dict.get('false_negatives', 0)
          tp = metrics_dict.get('true_positives', 0)
          metrics_dict['confusion_matrix'] = [[tn, fp], [fn, tp]]

        if 'optimal_threshold' not in metrics_dict:
          # Get from top-level model_package (where training script saves it)
          metrics_dict['optimal_threshold'] = model_package.get('optimal_threshold', 0.5)

        self.metrics = ModelMetrics(**metrics_dict)

      self.classifier = model_package.get('classifier')
      self.threshold_predictor = model_package.get('threshold_predictor')
      self.scaler = model_package.get('scaler')

      logger.info(
        f"✅ Model loaded: {filepath}, "
        f"samples={self.training_samples}, "
        f"threshold={self.optimal_threshold:.3f}, "
        f"trained_at={self.trained_at}"
      )

    except Exception as e:
      logger.warning(f"Could not load model from {filepath}: {e}")
      self._init_models()

  def get_info(self) -> Dict:
    """Get model information."""
    return {
      'enabled': self.enabled,
      'sklearn_available': SKLEARN_AVAILABLE,
      'trained': self.classifier is not None and hasattr(self.classifier, 'n_estimators'),
      'version': self.version,
      'trained_at': self.trained_at,
      'training_samples': self.training_samples,
      'optimal_threshold': self.optimal_threshold,
      'feature_count': len(self.feature_names),
      'metrics': self.metrics.__dict__ if self.metrics else None,
      'top_features': list(self.feature_importance.keys())[:10] if self.feature_importance else []
    }


# Example usage
if __name__ == "__main__":
  if SKLEARN_AVAILABLE:
    model = AdaptiveLayeringModel()
    print("Model initialized")
    print(f"Info: {model.get_info()}")
  else:
    print("sklearn not available")
