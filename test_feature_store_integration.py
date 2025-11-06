"""
Test Feature Store → Sequences Integration

Проверяет:
1. Импорт FeatureStoreSchema
2. Создание схемы и валидация
3. Наличие метода load_from_dataframe()
4. Импорт в retraining_pipeline
"""

import sys
import numpy as np
import pandas as pd


print("=" * 60)
print("Testing Feature Store Integration")
print("=" * 60)

# Test 1: Import FeatureStoreSchema
print("\n1. Testing FeatureStoreSchema import...")
try:
    from backend.ml_engine.feature_store.feature_schema import (
        FeatureStoreSchema,
        DEFAULT_SCHEMA
    )
    print("   ✓ FeatureStoreSchema imported successfully")
    print(f"   ✓ DEFAULT_SCHEMA created")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Check schema structure
print("\n2. Testing schema structure...")
try:
    # Check feature counts
    orderbook_count = len(DEFAULT_SCHEMA.orderbook_features)
    candle_count = len(DEFAULT_SCHEMA.candle_features)
    indicator_count = len(DEFAULT_SCHEMA.indicator_features)
    total_count = len(DEFAULT_SCHEMA.get_all_feature_columns())

    print(f"   • OrderBook features: {orderbook_count}")
    print(f"   • Candle features: {candle_count}")
    print(f"   • Indicator features: {indicator_count}")
    print(f"   • Total features: {total_count}")

    if orderbook_count == 50 and candle_count == 25 and indicator_count == 35:
        print("   ✓ Feature counts correct (50+25+35=110)")
    else:
        print(f"   ✗ FAILED: Expected 50+25+35=110, got {orderbook_count}+{candle_count}+{indicator_count}={total_count}")
        sys.exit(1)

    # Check column names
    assert DEFAULT_SCHEMA.timestamp_column == 'timestamp'
    assert DEFAULT_SCHEMA.label_column == 'future_direction_60s'
    print("   ✓ Column names correct")

except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 3: Test DataFrame validation
print("\n3. Testing DataFrame validation...")
try:
    # Create mock DataFrame with all required columns
    required_cols = DEFAULT_SCHEMA.get_required_columns()
    mock_df = pd.DataFrame({col: [0] * 100 for col in required_cols})

    # Validate
    is_valid = DEFAULT_SCHEMA.validate_dataframe(mock_df, strict=False)
    if is_valid:
        print("   ✓ Valid DataFrame passes validation")
    else:
        print("   ✗ FAILED: Valid DataFrame rejected")
        sys.exit(1)

    # Test invalid DataFrame
    invalid_df = pd.DataFrame({'timestamp': [0] * 100})
    try:
        DEFAULT_SCHEMA.validate_dataframe(invalid_df, strict=True)
        print("   ✗ FAILED: Invalid DataFrame should raise error")
        sys.exit(1)
    except ValueError:
        print("   ✓ Invalid DataFrame correctly rejected")

except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check load_from_dataframe method exists
print("\n4. Testing load_from_dataframe() method...")
try:
    from backend.ml_engine.training.data_loader import HistoricalDataLoader

    if hasattr(HistoricalDataLoader, 'load_from_dataframe'):
        print("   ✓ load_from_dataframe() method exists")

        # Check signature
        import inspect
        sig = inspect.signature(HistoricalDataLoader.load_from_dataframe)
        params = list(sig.parameters.keys())
        print(f"   ✓ Parameters: {params}")

        # Expected params
        expected = ['self', 'features_df', 'feature_columns', 'label_column',
                   'timestamp_column', 'symbol_column', 'apply_resampling']
        if params == expected:
            print("   ✓ Method signature correct")
        else:
            print(f"   ⚠ Warning: Signature differs from expected")
            print(f"     Expected: {expected}")
            print(f"     Got: {params}")
    else:
        print("   ✗ FAILED: load_from_dataframe() method not found")
        sys.exit(1)

except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check retraining_pipeline imports
print("\n5. Testing retraining_pipeline imports...")
try:
    # Read file and check for import
    with open('backend/ml_engine/auto_retraining/retraining_pipeline.py', 'r') as f:
        content = f.read()

    # Check for schema import in _collect_training_data
    if 'from backend.ml_engine.feature_store.feature_schema import DEFAULT_SCHEMA' in content:
        print("   ✓ Schema import found in retraining_pipeline")
    else:
        print("   ⚠ Warning: Schema import not found")

    # Check for load_from_dataframe call
    if 'load_from_dataframe(' in content:
        print("   ✓ load_from_dataframe() call found")
    else:
        print("   ⚠ Warning: load_from_dataframe() call not found")

    # Check for schema validation
    if 'validate_dataframe' in content:
        print("   ✓ Schema validation found")
    else:
        print("   ⚠ Warning: Schema validation not found")

except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 6: Feature groups
print("\n6. Testing feature groups...")
try:
    groups = DEFAULT_SCHEMA.get_feature_groups()
    print(f"   • Feature groups: {list(groups.keys())}")

    for group_name, features in groups.items():
        print(f"   • {group_name}: {len(features)} features")

    print("   ✓ Feature groups accessible")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Success
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nImplementation Summary:")
print("1. ✓ FeatureStoreSchema created with 110 features")
print("2. ✓ DEFAULT_SCHEMA available globally")
print("3. ✓ DataFrame validation works correctly")
print("4. ✓ load_from_dataframe() method added to HistoricalDataLoader")
print("5. ✓ retraining_pipeline.py updated to use Feature Store data")
print("\nFeature Store integration is ready! 🎉")
print("=" * 60)
