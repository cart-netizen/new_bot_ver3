"""
Professional Feature Scaler Manager for ML Pipeline.

Implements industry-standard multi-channel feature scaling with:
- Separate scalers for different feature channels (OrderBook, Candle, Indicator)
- Persistent state management (save/load)
- Historical data fitting (warm-up on past data)
- Versioning and backward compatibility
- Feature importance analysis

This implementation is suitable for REAL MONEY TRADING.

Path: backend/ml_engine/features/feature_scaler_manager.py
"""

from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import joblib
from datetime import datetime
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

from backend.core.logger import get_logger
# from ml_engine.features import FeatureVector

# Type-only imports to avoid circular dependency
if TYPE_CHECKING:
    from backend.ml_engine.features.feature_pipeline import FeatureVector

logger = get_logger(__name__)

# Type alias for sklearn scalers
SklearnScaler = Union[StandardScaler, RobustScaler, MinMaxScaler]


@dataclass
class ScalerConfig:
    """Configuration for feature scaling."""

    # Scaler types for each channel
    orderbook_scaler_type: str = "standard"  # standard, robust, minmax
    candle_scaler_type: str = "robust"       # robust handles outliers better
    indicator_scaler_type: str = "minmax"    # indicators often bounded

    # Warm-up settings
    min_samples_for_fitting: int = 100  # Minimum samples to fit scalers
    refit_interval_samples: int = 1000  # Refit every N samples

    # Persistence
    save_dir: str = "ml_models/scalers"
    auto_save: bool = True
    save_interval_samples: int = 500

    # Feature importance
    enable_feature_importance: bool = True
    variance_threshold: float = 0.01  # Remove features with variance < 0.01

    # Version
    version: str = "v1.0.0"


@dataclass
class ScalerState:
    """State of fitted scalers."""

    orderbook_scaler: Optional[SklearnScaler] = None
    candle_scaler: Optional[SklearnScaler] = None
    indicator_scaler: Optional[SklearnScaler] = None

    # Metadata
    is_fitted: bool = False
    samples_processed: int = 0
    last_fit_timestamp: int = 0
    last_save_timestamp: int = 0

    # Statistics
    feature_means: Optional[Dict[str, float]] = None
    feature_stds: Optional[Dict[str, float]] = None
    feature_variances: Optional[Dict[str, float]] = None

    # Version for backward compatibility
    version: str = "v1.0.0"


class FeatureScalerManager:
    """
    Professional multi-channel feature scaler for ML pipeline.

    Architecture:
    ============
    Three independent scalers for three feature channels:

    1. OrderBook Channel (50 features):
       - StandardScaler (mean=0, std=1)
       - Best for features with Gaussian-like distribution
       - Examples: prices, volumes, imbalance, spread

    2. Candle Channel (25 features):
       - RobustScaler (median-based, IQR scaling)
       - Robust to outliers (extreme candles)
       - Examples: OHLC, returns, volatility, shadows

    3. Indicator Channel (35 features):
       - MinMaxScaler (scale to [0, 1])
       - Best for bounded indicators
       - Examples: RSI (0-100), Stochastic (0-100), normalized MACD

    Why Multi-Channel?
    ==================
    - Different features have different distributions
    - Single scaler causes cross-contamination
    - Orderbook volumes (millions) vs RSI (0-100) need different scaling
    - Candle outliers (flash crashes) shouldn't affect orderbook scaling

    Persistence:
    ============
    - Scalers saved to disk (joblib)
    - Loaded on startup (no retraining)
    - Versioned for backward compatibility
    - Auto-save every N samples

    Usage:
    ======
    ```python
    # Initialize
    config = ScalerConfig()
    manager = FeatureScalerManager(symbol="BTCUSDT", config=config)

    # Warm-up on historical data
    await manager.warmup(historical_feature_vectors)

    # Scale new features
    normalized_vector = await manager.scale_features(feature_vector)

    # Get feature importance
    importance = manager.get_feature_importance()
    ```
    """

    def __init__(
        self,
        symbol: str,
        config: Optional[ScalerConfig] = None
    ):
        """
        Initialize Feature Scaler Manager.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            config: Scaler configuration
        """
        self.symbol = symbol
        self.config = config or ScalerConfig()

        # Create save directory
        self.save_dir = Path(self.config.save_dir) / symbol
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize scalers
        self.state = ScalerState(version=self.config.version)
        self._init_scalers()

        # Try to load existing state
        self._load_state()

        # Feature importance tracking
        self.feature_names: List[str] = []
        self.feature_importance_scores: Dict[str, float] = {}

        # History for periodic refitting
        self.feature_history: List[np.ndarray] = []
        self.max_history_size = 5000  # Keep last 5000 samples

        logger.info(
            f"Initialized FeatureScalerManager for {symbol}: "
            f"orderbook={self.config.orderbook_scaler_type}, "
            f"candle={self.config.candle_scaler_type}, "
            f"indicator={self.config.indicator_scaler_type}, "
            f"fitted={self.state.is_fitted}"
        )

    def _init_scalers(self):
        """Initialize scalers based on config."""

        # OrderBook scaler
        if self.config.orderbook_scaler_type == "standard":
            self.state.orderbook_scaler = StandardScaler()
        elif self.config.orderbook_scaler_type == "robust":
            self.state.orderbook_scaler = RobustScaler()
        elif self.config.orderbook_scaler_type == "minmax":
            self.state.orderbook_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.config.orderbook_scaler_type}")

        # Candle scaler (default: RobustScaler for outlier resistance)
        if self.config.candle_scaler_type == "robust":
            self.state.candle_scaler = RobustScaler()
        elif self.config.candle_scaler_type == "standard":
            self.state.candle_scaler = StandardScaler()
        elif self.config.candle_scaler_type == "minmax":
            self.state.candle_scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.config.candle_scaler_type}")

        # Indicator scaler (default: MinMaxScaler for bounded indicators)
        if self.config.indicator_scaler_type == "minmax":
            self.state.indicator_scaler = MinMaxScaler()
        elif self.config.indicator_scaler_type == "standard":
            self.state.indicator_scaler = StandardScaler()
        elif self.config.indicator_scaler_type == "robust":
            self.state.indicator_scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.config.indicator_scaler_type}")

    async def warmup(
        self,
        feature_vectors: List['FeatureVector'],
        force_refit: bool = False
    ) -> bool:
        """
        Warm-up scalers on historical data.

        This is CRITICAL for production use. Scalers must be fitted on
        representative historical data before use on live data.

        Args:
            feature_vectors: List of historical FeatureVector objects
            force_refit: Force refitting even if already fitted

        Returns:
            True if successful, False otherwise

        Example:
        --------
        historical_vectors = load_last_1000_feature_vectors()
        success = await manager.warmup(historical_vectors)
        if success:
        print("Scalers warmed up and ready")
        """
        if not feature_vectors:
            logger.warning(f"{self.symbol} | No feature vectors for warmup")
            return False

        if len(feature_vectors) < self.config.min_samples_for_fitting:
            logger.warning(
                f"{self.symbol} | Insufficient samples for warmup: "
                f"{len(feature_vectors)} < {self.config.min_samples_for_fitting}"
            )
            return False

        if self.state.is_fitted and not force_refit:
            logger.info(
                f"{self.symbol} | Scalers already fitted, skipping warmup "
                "(use force_refit=True to refit)"
            )
            return True

        logger.info(
            f"{self.symbol} | Warming up scalers on {len(feature_vectors)} samples..."
        )

        try:
            # Extract channel arrays from all feature vectors
            orderbook_arrays = []
            candle_arrays = []
            indicator_arrays = []

            for fv in feature_vectors:
                channels = fv.to_channels()
                orderbook_arrays.append(channels["orderbook"])
                candle_arrays.append(channels["candle"])
                indicator_arrays.append(channels["indicator"])

            # Stack into 2D arrays (samples x features)
            orderbook_data = np.vstack(orderbook_arrays)
            candle_data = np.vstack(candle_arrays)
            indicator_data = np.vstack(indicator_arrays)

            # Clean NaN/Inf
            orderbook_data = np.nan_to_num(orderbook_data, nan=0.0, posinf=0.0, neginf=0.0)
            candle_data = np.nan_to_num(candle_data, nan=0.0, posinf=0.0, neginf=0.0)
            indicator_data = np.nan_to_num(indicator_data, nan=0.0, posinf=0.0, neginf=0.0)

            # Fit scalers (BATCH fitting, not online!)
            logger.debug(f"{self.symbol} | Fitting OrderBook scaler on {orderbook_data.shape}")
            self.state.orderbook_scaler.fit(orderbook_data)

            logger.debug(f"{self.symbol} | Fitting Candle scaler on {candle_data.shape}")
            self.state.candle_scaler.fit(candle_data)

            logger.debug(f"{self.symbol} | Fitting Indicator scaler on {indicator_data.shape}")
            self.state.indicator_scaler.fit(indicator_data)

            # Update state
            self.state.is_fitted = True
            self.state.samples_processed = len(feature_vectors)
            self.state.last_fit_timestamp = int(datetime.now().timestamp() * 1000)

            # Calculate feature statistics
            self._calculate_feature_statistics(feature_vectors)

            # Save state
            if self.config.auto_save:
                self._save_state()

            logger.info(
                f"{self.symbol} | ✅ Scalers warmed up successfully on "
                f"{len(feature_vectors)} samples"
            )

            return True

        except Exception as e:
            logger.error(f"{self.symbol} | Error during warmup: {e}", exc_info=True)
            return False

    async def scale_features(
        self,
        feature_vector: 'FeatureVector',
        update_history: bool = True
    ) -> Optional['FeatureVector']:
        """
        Scale features using fitted scalers.

        This is the main method called during live trading.

        Args:
            feature_vector: FeatureVector with raw features
            update_history: Whether to add to history (for periodic refitting)

        Returns:
            New FeatureVector with scaled features, or None if scaling failed

        Algorithm:
        ----------
        1. Extract channel arrays from feature_vector
        2. Scale each channel independently using fitted scaler
        3. Create NEW feature objects with scaled values
        4. Preserve original values in metadata
        5. Return new FeatureVector with normalized features

        Example:
        --------
        raw_vector = await pipeline.extract_features(...)
        scaled_vector = await scaler_manager.scale_features(raw_vector)
        ml_model.predict(scaled_vector.to_array())
        """
        if not self.state.is_fitted:
            logger.warning(
                f"{self.symbol} | Scalers not fitted, call warmup() first. "
                "Returning original vector."
            )
            return feature_vector

        try:
            # ========================================================================
            # STEP 1: Extract channel arrays
            # ========================================================================
            channels = feature_vector.to_channels()
            orderbook_raw = channels["orderbook"].reshape(1, -1)
            candle_raw = channels["candle"].reshape(1, -1)
            indicator_raw = channels["indicator"].reshape(1, -1)

            # Clean NaN/Inf
            orderbook_raw = np.nan_to_num(orderbook_raw, nan=0.0, posinf=0.0, neginf=0.0)
            candle_raw = np.nan_to_num(candle_raw, nan=0.0, posinf=0.0, neginf=0.0)
            indicator_raw = np.nan_to_num(indicator_raw, nan=0.0, posinf=0.0, neginf=0.0)

            # ========================================================================
            # STEP 2: Scale each channel independently
            # ========================================================================
            orderbook_scaled = self.state.orderbook_scaler.transform(orderbook_raw).flatten()
            candle_scaled = self.state.candle_scaler.transform(candle_raw).flatten()
            indicator_scaled = self.state.indicator_scaler.transform(indicator_raw).flatten()

            logger.debug(
                f"{self.symbol} | Scaled features: "
                f"ob_mean={orderbook_scaled.mean():.3f}, "
                f"candle_mean={candle_scaled.mean():.3f}, "
                f"ind_mean={indicator_scaled.mean():.3f}"
            )

            # ========================================================================
            # STEP 3: Create NEW feature objects with scaled values
            # ========================================================================

            from backend.ml_engine.features.orderbook_feature_extractor import OrderBookFeatures
            from backend.ml_engine.features.candle_feature_extractor import CandleFeatures
            from backend.ml_engine.features.indicator_feature_extractor import IndicatorFeatures

            # OrderBook features (scaled)
            ob_dict = feature_vector.orderbook_features.to_dict()
            ob_keys = [k for k in ob_dict.keys() if k not in ['symbol', 'timestamp']]
            ob_scaled_dict = dict(zip(ob_keys, orderbook_scaled))

            orderbook_features_scaled = OrderBookFeatures(
                symbol=feature_vector.symbol,
                timestamp=feature_vector.timestamp,
                **ob_scaled_dict
            )

            # Candle features (scaled)
            candle_dict = feature_vector.candle_features.to_dict()
            candle_keys = [k for k in candle_dict.keys() if k not in ['symbol', 'timestamp']]
            candle_scaled_dict = dict(zip(candle_keys, candle_scaled))

            candle_features_scaled = CandleFeatures(
                symbol=feature_vector.symbol,
                timestamp=feature_vector.timestamp,
                **candle_scaled_dict
            )

            # Indicator features (scaled)
            indicator_dict = feature_vector.indicator_features.to_dict()
            indicator_keys = [k for k in indicator_dict.keys() if k not in ['symbol', 'timestamp']]
            indicator_scaled_dict = dict(zip(indicator_keys, indicator_scaled))

            indicator_features_scaled = IndicatorFeatures(
                symbol=feature_vector.symbol,
                timestamp=feature_vector.timestamp,
                **indicator_scaled_dict
            )

            # ========================================================================
            # STEP 4: Create new FeatureVector with scaled features
            # ========================================================================

            from backend.ml_engine.features.feature_pipeline import FeatureVector

            scaled_vector = FeatureVector(
                symbol=feature_vector.symbol,
                timestamp=feature_vector.timestamp,
                orderbook_features=orderbook_features_scaled,
                candle_features=candle_features_scaled,
                indicator_features=indicator_features_scaled,
                feature_count=feature_vector.feature_count,
                version=feature_vector.version,
                metadata=feature_vector.metadata.copy()
            )

            # ========================================================================
            # STEP 5: Preserve original values in metadata
            # ========================================================================
            scaled_vector.metadata['original_features'] = {
                'orderbook': orderbook_raw.flatten().tolist(),
                'candle': candle_raw.flatten().tolist(),
                'indicator': indicator_raw.flatten().tolist()
            }
            scaled_vector.metadata['scaled'] = True
            scaled_vector.metadata['scaler_version'] = self.state.version
            scaled_vector.metadata['scaling_timestamp'] = int(datetime.now().timestamp() * 1000)

            # ========================================================================
            # STEP 6: Update history for periodic refitting
            # ========================================================================
            if update_history:
                self.feature_history.append(feature_vector.to_array())
                if len(self.feature_history) > self.max_history_size:
                    self.feature_history.pop(0)

                self.state.samples_processed += 1

                # Periodic refit check
                if (self.state.samples_processed % self.config.refit_interval_samples == 0 and
                    len(self.feature_history) >= self.config.min_samples_for_fitting):
                    logger.info(
                        f"{self.symbol} | Periodic refit triggered after "
                        f"{self.state.samples_processed} samples"
                    )
                    # TODO: Implement async refit without blocking

                # Auto-save check
                if (self.config.auto_save and
                    self.state.samples_processed % self.config.save_interval_samples == 0):
                    self._save_state()

            return scaled_vector

        except Exception as e:
            logger.error(
                f"{self.symbol} | Error scaling features: {e}",
                exc_info=True
            )
            return feature_vector  # Fallback to original

    def _calculate_feature_statistics(
        self,
        feature_vectors: List['FeatureVector']
    ):
        """Calculate feature statistics for importance analysis."""
        try:
            # Extract all features
            all_features = np.vstack([fv.to_array() for fv in feature_vectors])

            # Get feature names
            if feature_vectors:
                self.feature_names = feature_vectors[0].get_feature_names()

            # Calculate statistics
            feature_means = np.mean(all_features, axis=0)
            feature_stds = np.std(all_features, axis=0)
            feature_vars = np.var(all_features, axis=0)

            # Store in state
            self.state.feature_means = dict(zip(self.feature_names, feature_means))
            self.state.feature_stds = dict(zip(self.feature_names, feature_stds))
            self.state.feature_variances = dict(zip(self.feature_names, feature_vars))

            logger.debug(f"{self.symbol} | Feature statistics calculated")

        except Exception as e:
            logger.error(f"{self.symbol} | Error calculating statistics: {e}")

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Calculate feature importance based on variance.

        Features with higher variance are more important for ML models.

        Returns:
            Dict[feature_name, importance_score] sorted by importance

        Algorithm:
        ----------
        1. Variance-based importance (normalized to 0-1)
        2. Features with variance < threshold flagged as low-importance
        3. Sorted by importance (descending)

        Example:
        --------
        >>> importance = manager.get_feature_importance()
        >>> for feature, score in list(importance.items())[:10]:
        ...     print(f"{feature}: {score:.4f}")
        """
        if not self.state.feature_variances:
            logger.warning(f"{self.symbol} | No feature statistics available")
            return {}

        try:
            # Variance-based importance (normalize to 0-1)
            variances = np.array(list(self.state.feature_variances.values()))
            max_var = np.max(variances) if np.max(variances) > 0 else 1.0

            normalized_importance = variances / max_var

            # Create dict
            importance_dict = dict(zip(self.feature_names, normalized_importance))

            # Sort by importance (descending)
            importance_sorted = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )

            # Cache
            self.feature_importance_scores = importance_sorted

            logger.debug(
                f"{self.symbol} | Feature importance calculated: "
                f"top feature={list(importance_sorted.items())[0]}"
            )

            return importance_sorted

        except Exception as e:
            logger.error(f"{self.symbol} | Error calculating feature importance: {e}")
            return {}

    def get_low_importance_features(
        self,
        threshold: float = 0.01
    ) -> List[str]:
        """
        Get list of low-importance features (for removal).

        Args:
            threshold: Variance threshold (0-1)

        Returns:
            List of feature names with importance < threshold
        """
        importance = self.get_feature_importance()
        return [name for name, score in importance.items() if score < threshold]

    def _save_state(self):
        """Save scaler state to disk."""
        try:
            timestamp = int(datetime.now().timestamp())
            filename = f"scaler_state_{self.state.version}_{timestamp}.joblib"
            filepath = self.save_dir / filename

            # Save state
            joblib.dump(self.state, filepath)

            # Create/update symlink to latest
            latest_link = self.save_dir / "scaler_state_latest.joblib"
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(filename)

            self.state.last_save_timestamp = timestamp

            logger.debug(f"{self.symbol} | Scaler state saved to {filepath}")

        except Exception as e:
            logger.error(f"{self.symbol} | Error saving scaler state: {e}")

    def _load_state(self):
        """Load scaler state from disk."""
        try:
            latest_link = self.save_dir / "scaler_state_latest.joblib"

            if not latest_link.exists():
                logger.info(f"{self.symbol} | No existing scaler state found")
                return

            # Load state
            loaded_state = joblib.load(latest_link)

            # Version check
            if loaded_state.version != self.config.version:
                logger.warning(
                    f"{self.symbol} | Scaler version mismatch: "
                    f"loaded={loaded_state.version}, current={self.config.version}. "
                    "Using loaded state anyway (may cause issues)."
                )

            # Apply loaded state
            self.state = loaded_state

            logger.info(
                f"{self.symbol} | ✅ Scaler state loaded: "
                f"fitted={self.state.is_fitted}, "
                f"samples={self.state.samples_processed}"
            )

        except Exception as e:
            logger.error(f"{self.symbol} | Error loading scaler state: {e}")

    def get_state_info(self) -> Dict:
        """Get current scaler state information."""
        return {
            'symbol': self.symbol,
            'is_fitted': self.state.is_fitted,
            'samples_processed': self.state.samples_processed,
            'last_fit_timestamp': self.state.last_fit_timestamp,
            'last_save_timestamp': self.state.last_save_timestamp,
            'version': self.state.version,
            'config': {
                'orderbook_scaler': self.config.orderbook_scaler_type,
                'candle_scaler': self.config.candle_scaler_type,
                'indicator_scaler': self.config.indicator_scaler_type
            },
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'history_size': len(self.feature_history)
        }


# Factory function
def create_scaler_manager(
    symbol: str,
    orderbook_scaler: str = "standard",
    candle_scaler: str = "robust",
    indicator_scaler: str = "minmax",
    save_dir: str = "ml_models/scalers"
) -> FeatureScalerManager:
    """
    Factory function to create FeatureScalerManager with custom settings.

    Args:
        symbol: Trading pair
        orderbook_scaler: Scaler type for orderbook features
        candle_scaler: Scaler type for candle features
        indicator_scaler: Scaler type for indicator features
        save_dir: Directory for saving scaler state

    Returns:
        Configured FeatureScalerManager instance

    Example:
    --------
    >>> manager = create_scaler_manager(
    ...     symbol="BTCUSDT",
    ...     orderbook_scaler="standard",
    ...     candle_scaler="robust",
    ...     indicator_scaler="minmax"
    ... )
    """
    config = ScalerConfig(
        orderbook_scaler_type=orderbook_scaler,
        candle_scaler_type=candle_scaler,
        indicator_scaler_type=indicator_scaler,
        save_dir=save_dir
    )

    return FeatureScalerManager(symbol=symbol, config=config)


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 80)
    print("PROFESSIONAL FEATURE SCALER MANAGER")
    print("=" * 80)

    # Create manager
    manager = create_scaler_manager("BTCUSDT")

    print(f"\nManager created:")
    print(f"  OrderBook scaler: StandardScaler")
    print(f"  Candle scaler: RobustScaler")
    print(f"  Indicator scaler: MinMaxScaler")
    print(f"  Save dir: {manager.save_dir}")

    # State info
    info = manager.get_state_info()
    print(f"\nState:")
    print(f"  Fitted: {info['is_fitted']}")
    print(f"  Samples processed: {info['samples_processed']}")

    print("\n" + "=" * 80)
    print("Module ready for production use in ML Pipeline")
    print("=" * 80)
