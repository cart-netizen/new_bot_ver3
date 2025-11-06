"""
Test script to verify fixes for:
1. get_drift_detector() function
2. HistoricalDataLoader.load_and_split() method
"""

import sys

print("=" * 60)
print("Testing imports and function availability")
print("=" * 60)

# Test 1: Import get_drift_detector
print("\n1. Testing get_drift_detector import...")
try:
    from backend.ml_engine.monitoring.drift_detector import get_drift_detector
    print("   ✓ get_drift_detector imported successfully")

    # Try to instantiate
    detector = get_drift_detector()
    print(f"   ✓ DriftDetector instance created: {type(detector).__name__}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Check HistoricalDataLoader.load_and_split method
print("\n2. Testing HistoricalDataLoader.load_and_split...")
try:
    from backend.ml_engine.training.data_loader import HistoricalDataLoader, DataConfig
    print("   ✓ HistoricalDataLoader imported successfully")

    # Check method exists
    if hasattr(HistoricalDataLoader, 'load_and_split'):
        print("   ✓ load_and_split method exists")

        # Check signature
        import inspect
        sig = inspect.signature(HistoricalDataLoader.load_and_split)
        print(f"   ✓ Method signature: {sig}")
    else:
        print("   ✗ FAILED: load_and_split method not found")
        sys.exit(1)

except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

# Test 3: Check retraining_pipeline imports
print("\n3. Testing retraining_pipeline imports...")
try:
    from backend.ml_engine.auto_retraining.retraining_pipeline import RetrainingPipeline
    print("   ✓ RetrainingPipeline imported successfully")
    print("   ✓ All imports in retraining_pipeline.py work correctly")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success
print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\nFixed issues:")
print("1. ✓ get_drift_detector() function added to drift_detector.py")
print("2. ✓ load_and_split() method added to HistoricalDataLoader")
print("3. ✓ All imports in retraining_pipeline.py work correctly")
print("=" * 60)
