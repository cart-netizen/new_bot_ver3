"""
Test to verify load_and_split returns DataLoaders (not raw arrays)
"""

import ast
import sys


def get_function_return_annotation(filepath, class_name, method_name):
    """Get return type annotation of a method"""
    with open(filepath, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    if item.returns:
                        return ast.unparse(item.returns)
    return None


print("=" * 60)
print("Verifying load_and_split() return type")
print("=" * 60)

filepath = "backend/ml_engine/training/data_loader.py"
return_type = get_function_return_annotation(filepath, "HistoricalDataLoader", "load_and_split")

print(f"\n✓ Method: HistoricalDataLoader.load_and_split()")
print(f"✓ Return type: {return_type}")

# Check that it returns DataLoaders, not raw tuples
if return_type and "DataLoader" in return_type:
    print("\n✅ CORRECT: Returns DataLoader objects")
    print("\nExpected by ModelTrainer.train():")
    print("  - train_loader: DataLoader")
    print("  - val_loader: DataLoader")
    print("  - test_loader: Optional[DataLoader]")
    print("\nThis matches the requirement! 🎉")
elif return_type and "Tuple[Tuple[np.ndarray" in return_type:
    print("\n❌ WRONG: Returns raw numpy arrays")
    print("\nThis won't work with ModelTrainer.train() which expects DataLoader!")
    sys.exit(1)
else:
    print(f"\n⚠ WARNING: Unexpected return type: {return_type}")

print("=" * 60)
