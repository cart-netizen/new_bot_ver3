#!/usr/bin/env python3
"""
Automatic Labeling for Collected Layering ML Data

This script labels unlabeled layering ML data by analyzing:
1. Pattern confidence scores from detector
2. Market impact metrics (price_change, impact_ratio)
3. Execution correlation (aggressive_ratio, spoofing_execution_ratio)

Labeling criteria:
- TRUE POSITIVE: High confidence + significant market impact + low execution
- FALSE POSITIVE: Low confidence + minimal impact + high execution rate

Run: python label_layering_data.py
"""

import sys
from pathlib import Path
import warnings

# Check dependencies before importing
try:
    import pandas as pd
except ImportError:
    print("=" * 80)
    print("❌ ERROR: Required dependencies not installed")
    print("=" * 80)
    print()
    print("The following packages are required:")
    print("  - pandas")
    print("  - pyarrow")
    print()
    print("Install with:")
    print("  pip install pandas pyarrow")
    print()
    print("=" * 80)
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.config import settings
from backend.core.logger import get_logger

logger = get_logger(__name__)


def label_data_automatic():
    """
    Automatically label collected data based on pattern characteristics.

    Uses detector confidence, market impact, and execution metrics to determine
    if a pattern is a true layering attempt or false positive.
    """

    # Load data
    # Use relative path from project root
    data_dir = Path(__file__).parent / "data" / "ml_training" / "layering"

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    parquet_files = sorted(data_dir.glob("layering_data_*.parquet"))

    if not parquet_files:
        print(f"❌ No parquet files found in {data_dir}")
        return

    print(f"Found {len(parquet_files)} data files")
    print(f"Loading data...")

    # Load all data
    dfs = []
    for filepath in parquet_files:
        try:
            df = pd.read_parquet(filepath)
            if not df.empty and not df.isna().all().all():
                dfs.append(df)
        except Exception as e:
            print(f"Error loading {filepath.name}: {e}")

    if not dfs:
        print("❌ No valid data loaded")
        return

    # Combine all data
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
        combined_df = pd.concat(dfs, ignore_index=True)

    print(f"✓ Loaded {len(combined_df)} total samples")

    # Count unlabeled
    unlabeled = combined_df['is_true_layering'].isna()
    unlabeled_count = unlabeled.sum()

    print(f"\n📊 Status:")
    print(f"   Total samples: {len(combined_df)}")
    print(f"   Unlabeled: {unlabeled_count}")
    print(f"   Already labeled: {len(combined_df) - unlabeled_count}")

    if unlabeled_count == 0:
        print("\n✓ All data is already labeled!")
        return

    print(f"\n🏷️  Starting automatic labeling...")

    # Get unlabeled data
    to_label = combined_df[unlabeled].copy()

    # === LABELING LOGIC ===
    # Based on detector confidence, market impact, and execution metrics

    # Criteria for TRUE POSITIVE (real layering):
    # 1. High detector confidence (>= 0.7)
    # 2. Low execution rate (< 0.3) - fake orders were not executed
    # 3. Significant market impact OR high cancellation rate

    true_positive_mask = (
        (to_label['detector_confidence'] >= 0.7) &  # High confidence
        (
            (to_label['spoofing_execution_ratio'].fillna(1.0) < 0.3) |  # Low execution
            (to_label['cancellation_rate'] > 0.7)  # High cancellations
        )
    )

    # Criteria for FALSE POSITIVE (not real layering):
    # 1. Low detector confidence (< 0.5)
    # OR
    # 2. High execution rate (> 0.6) - orders were actually filled
    # OR
    # 3. Low cancellation rate + low detector confidence

    false_positive_mask = (
        (to_label['detector_confidence'] < 0.5) |
        (to_label['spoofing_execution_ratio'].fillna(0.0) > 0.6) |
        (
            (to_label['cancellation_rate'] < 0.5) &
            (to_label['detector_confidence'] < 0.6)
        )
    )

    # Apply labels
    to_label.loc[true_positive_mask, 'is_true_layering'] = True
    to_label.loc[false_positive_mask & ~true_positive_mask, 'is_true_layering'] = False
    to_label.loc[true_positive_mask | false_positive_mask, 'label_source'] = 'automatic_heuristic'
    to_label.loc[true_positive_mask | false_positive_mask, 'label_confidence'] = 0.8

    # Count labeled
    newly_labeled_true = true_positive_mask.sum()
    newly_labeled_false = (false_positive_mask & ~true_positive_mask).sum()
    still_unlabeled = unlabeled_count - newly_labeled_true - newly_labeled_false

    print(f"\n📈 Labeling Results:")
    print(f"   ✅ Labeled as TRUE (real layering): {newly_labeled_true}")
    print(f"   ❌ Labeled as FALSE (false positive): {newly_labeled_false}")
    print(f"   ⚪ Remain unlabeled (uncertain): {still_unlabeled}")
    print(f"   📊 Labeling rate: {((newly_labeled_true + newly_labeled_false) / unlabeled_count * 100):.1f}%")

    if newly_labeled_true + newly_labeled_false == 0:
        print("\n⚠️  No samples could be labeled with current criteria")
        print("   Consider adjusting thresholds or manual labeling")
        return

    # Update combined dataframe
    combined_df.loc[unlabeled, 'is_true_layering'] = to_label['is_true_layering']
    combined_df.loc[unlabeled, 'label_source'] = to_label['label_source']
    combined_df.loc[unlabeled, 'label_confidence'] = to_label['label_confidence']

    # Save back to files
    print(f"\n💾 Saving labeled data...")

    # Group by original file (approximate based on timestamp)
    combined_df['file_group'] = pd.to_datetime(combined_df['timestamp'], unit='ms').dt.strftime('%Y%m%d_%H')

    saved_count = 0
    for file_group, group_df in combined_df.groupby('file_group'):
        # Find matching file or create new one
        matching_files = [f for f in parquet_files if file_group[:8] in f.name]

        if matching_files:
            output_file = matching_files[0]
        else:
            output_file = data_dir / f"layering_data_{file_group}00_labeled.parquet"

        # Save
        try:
            group_df.drop('file_group', axis=1).to_parquet(output_file, index=False)
            saved_count += 1
        except Exception as e:
            print(f"Error saving {output_file.name}: {e}")

    print(f"✓ Saved to {saved_count} files")

    # Final statistics
    final_labeled = combined_df['is_true_layering'].notna().sum()
    final_true = (combined_df['is_true_layering'] == True).sum()
    final_false = (combined_df['is_true_layering'] == False).sum()

    print(f"\n{'=' * 80}")
    print(f"✅ LABELING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Total samples: {len(combined_df)}")
    print(f"Labeled: {final_labeled} ({final_labeled / len(combined_df) * 100:.1f}%)")
    print(f"  - True positives: {final_true}")
    print(f"  - False positives: {final_false}")
    print(f"  - Class balance: {final_true / final_labeled * 100:.1f}% / {final_false / final_labeled * 100:.1f}%")
    print(f"Unlabeled: {len(combined_df) - final_labeled}")
    print(f"{'=' * 80}")

    # Update statistics.json for frontend
    try:
        import json
        from datetime import datetime

        stats_file = data_dir / "statistics.json"
        stats = {
            "total_collected": len(combined_df),
            "total_labeled": int(final_labeled),
            "total_saved": len(combined_df),
            "last_updated": datetime.now().isoformat(),
            "exists": True
        }

        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n✓ Updated statistics.json ({final_labeled} labeled samples)")
    except Exception as e:
        print(f"\n⚠️  Warning: Could not update statistics.json: {e}")

    print(f"\n📚 Ready for training!")
    print(f"   Run: python train_layering_model_improved.py")


if __name__ == "__main__":
    print("=" * 80)
    print("🏷️  LAYERING ML DATA AUTO-LABELING")
    print("=" * 80)
    print()

    try:
        label_data_automatic()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
