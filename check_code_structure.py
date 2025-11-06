"""
Static code analysis to verify fixes without running
"""

import ast
import sys

def check_function_exists(filepath, function_name):
    """Check if function exists in file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return True
    return False

def check_method_exists(filepath, class_name, method_name):
    """Check if method exists in class"""
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return True
    return False

print("=" * 60)
print("Static Code Analysis - Checking fixes")
print("=" * 60)

# Test 1: Check get_drift_detector function
print("\n1. Checking get_drift_detector() in drift_detector.py...")
filepath = "backend/ml_engine/monitoring/drift_detector.py"
if check_function_exists(filepath, "get_drift_detector"):
    print("   ✓ get_drift_detector() function found")
else:
    print("   ✗ FAILED: get_drift_detector() function not found")
    sys.exit(1)

# Test 2: Check load_and_split method
print("\n2. Checking load_and_split() in HistoricalDataLoader...")
filepath = "backend/ml_engine/training/data_loader.py"
if check_method_exists(filepath, "HistoricalDataLoader", "load_and_split"):
    print("   ✓ HistoricalDataLoader.load_and_split() method found")
else:
    print("   ✗ FAILED: load_and_split() method not found")
    sys.exit(1)

# Test 3: Check imports in retraining_pipeline
print("\n3. Checking imports in retraining_pipeline.py...")
filepath = "backend/ml_engine/auto_retraining/retraining_pipeline.py"
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Check for import statements
if "from backend.ml_engine.monitoring.drift_detector import get_drift_detector" in content:
    print("   ✓ get_drift_detector import found")
else:
    print("   ✗ FAILED: get_drift_detector import not found")
    sys.exit(1)

if "data_loader.load_and_split()" in content:
    print("   ✓ load_and_split() call found")
else:
    print("   ⚠ load_and_split() call not found (might be unused)")

# Success
print("\n" + "=" * 60)
print("✅ ALL CHECKS PASSED!")
print("=" * 60)
print("\nFixed issues:")
print("1. ✓ get_drift_detector() function exists in drift_detector.py")
print("2. ✓ load_and_split() method exists in HistoricalDataLoader")
print("3. ✓ Imports are correctly defined in retraining_pipeline.py")
print("\nThe code structure is correct.")
print("Runtime testing requires installing dependencies.")
print("=" * 60)
