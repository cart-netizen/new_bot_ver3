#!/usr/bin/env python3
"""
Wrapper script for training Layering ML model.

This is a convenience wrapper that can be run from project root.
It calls the actual training script in backend/scripts/

Usage:
  python train_layering_model.py

Requirements:
  - Python 3.8+
  - pandas, pyarrow, scikit-learn installed
  - Collected training data in data/ml_training/layering/
  - At least 100 labeled samples
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the actual training script
if __name__ == "__main__":
    try:
        from backend.scripts.train_layering_model import main
        main()
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\nУстановите зависимости:")
        print("  pip install pandas pyarrow scikit-learn")
        print("\nИли активируйте виртуальное окружение:")
        print("  # Windows:")
        print("  .venv\\Scripts\\activate")
        print("  # Linux/Mac:")
        print("  source .venv/bin/activate")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
