"""
Feature Store Module

Централизованное хранилище фич для:
- Consistency между training и serving
- Feature versioning
- Feature reuse
- Online/Offline feature serving
"""

from backend.ml_engine.feature_store.feature_store import (
    FeatureStore,
    get_feature_store,
    FeatureMetadata
)

__all__ = [
    "FeatureStore",
    "get_feature_store",
    "FeatureMetadata"
]
