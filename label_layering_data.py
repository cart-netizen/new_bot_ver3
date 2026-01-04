#!/usr/bin/env python3
"""
Improved Automatic Labeling for Collected Layering ML Data

FIXED: Removed circular dependency on detector_confidence!

This script labels unlabeled layering ML data by analyzing:
1. OUTCOME-BASED metrics (price movement after pattern)
2. BEHAVIORAL metrics (cancellation rate, execution ratio)
3. MARKET IMPACT metrics (price_change_bps, impact_ratio)

Labeling Strategy:
- TRUE POSITIVE: Orders were canceled (not filled) + price moved as expected
- FALSE POSITIVE: Orders were filled OR price didn't move OR low cancellation

Key principle: Label based on WHAT HAPPENED, not detector confidence!

Run: python label_layering_data.py
"""

import sys
from pathlib import Path
import warnings
import json
from datetime import datetime

# Check dependencies before importing
try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("=" * 80)
    print("ERROR: Required dependencies not installed")
    print("=" * 80)
    print()
    print("The following packages are required:")
    print("  - pandas")
    print("  - pyarrow")
    print("  - numpy")
    print()
    print("Install with:")
    print("  pip install pandas pyarrow numpy")
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


# ============================================================
# LABELING CONFIGURATION
# ============================================================

class LabelingConfig:
    """
    Configuration for labeling criteria.

    Key insight: Real layering has these characteristics:
    1. Orders are placed but NOT executed (high cancellation)
    2. Price moves in the direction the manipulator wants
    3. Orders are canceled quickly after achieving the goal

    v2.1: Relaxed criteria to get more balanced TRUE/FALSE distribution
    """

    # === BEHAVIORAL CRITERIA (RELAXED) ===
    # Cancellation rate: real layering has HIGH cancellation (orders are fake)
    MIN_CANCELLATION_FOR_TRUE = 0.50  # Relaxed from 0.65 to 0.50
    MAX_CANCELLATION_FOR_FALSE = 0.35  # Relaxed from 0.40 to 0.35

    # Execution ratio: real layering has LOW execution (orders shouldn't be filled)
    MAX_EXECUTION_FOR_TRUE = 0.50  # Relaxed from 0.35 to 0.50
    MIN_EXECUTION_FOR_FALSE = 0.60  # Relaxed from 0.50 to 0.60

    # === PRICE IMPACT CRITERIA (RELAXED) ===
    # Real layering causes price movement
    MIN_PRICE_CHANGE_BPS_FOR_TRUE = 1.5  # Relaxed from 3.0 to 1.5 bps

    # Impact ratio: expected vs actual impact
    MIN_IMPACT_RATIO_FOR_TRUE = 0.2  # Relaxed from 0.3 to 0.2

    # === TIMING CRITERIA (RELAXED) ===
    # Real layering: orders placed and canceled quickly
    MAX_DURATION_FOR_TRUE = 90.0  # Relaxed from 60 to 90 seconds
    MIN_DURATION_FOR_FALSE = 180.0  # Relaxed from 120 to 180 seconds

    # === VOLUME CRITERIA (RELAXED) ===
    # Minimum volume to be considered significant
    MIN_VOLUME_USDT = 2000  # Relaxed from $5k to $2k

    # === CONFIDENCE THRESHOLDS ===
    # How confident we are in the label
    HIGH_CONFIDENCE = 0.9
    MEDIUM_CONFIDENCE = 0.7
    LOW_CONFIDENCE = 0.5


def label_data_improved():
    """
    Improved automatic labeling based on OUTCOMES, not detector confidence.

    This removes the circular dependency where we were labeling based on
    the same detector output we're trying to improve.
    """

    # Load data
    data_dir = Path(__file__).parent / "data" / "ml_training" / "layering"

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return

    parquet_files = sorted(data_dir.glob("layering_data_*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
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
        print("No valid data loaded")
        return

    # Combine all data
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
        combined_df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(combined_df):,} total samples")

    # Show available columns for debugging
    print(f"\nAvailable columns: {len(combined_df.columns)}")

    # Count current state
    already_labeled = combined_df['is_true_layering'].notna().sum()
    unlabeled_count = len(combined_df) - already_labeled

    print(f"\nCurrent Status:")
    print(f"   Total samples: {len(combined_df):,}")
    print(f"   Already labeled: {already_labeled:,}")
    print(f"   Unlabeled: {unlabeled_count:,}")

    # Re-label ALL data with improved logic (optional: only unlabeled)
    # For now, let's re-label everything to fix the circular dependency
    print(f"\n{'='*80}")
    print("IMPROVED LABELING (outcome-based, no detector_confidence dependency)")
    print(f"{'='*80}")

    config = LabelingConfig()

    # Create working copy
    df = combined_df.copy()

    # Fill NaN with safe defaults for comparison
    df['cancellation_rate'] = df['cancellation_rate'].fillna(0.5)
    df['spoofing_execution_ratio'] = df['spoofing_execution_ratio'].fillna(0.5)
    df['price_change_bps'] = df['price_change_bps'].fillna(0.0)
    df['impact_ratio'] = df['impact_ratio'].fillna(0.0)
    df['placement_duration'] = df['placement_duration'].fillna(30.0)
    df['total_volume_usdt'] = df['total_volume_usdt'].fillna(0.0)

    # Also check for outcome fields
    if 'price_moved_as_expected' in df.columns:
        df['price_moved_as_expected'] = df['price_moved_as_expected'].fillna(False)
    else:
        df['price_moved_as_expected'] = False

    if 'manipulation_successful' in df.columns:
        df['manipulation_successful'] = df['manipulation_successful'].fillna(False)
    else:
        df['manipulation_successful'] = False

    # ================================================================
    # STEP 1: Strong TRUE POSITIVE indicators
    # ================================================================
    print("\nStep 1: Identifying TRUE POSITIVES...")

    # High confidence TRUE: Clear behavioral + outcome signals
    strong_true_mask = (
        # HIGH cancellation (orders were fake)
        (df['cancellation_rate'] >= config.MIN_CANCELLATION_FOR_TRUE) &
        # LOW execution (orders weren't meant to be filled)
        (df['spoofing_execution_ratio'] <= config.MAX_EXECUTION_FOR_TRUE) &
        # Significant volume
        (df['total_volume_usdt'] >= config.MIN_VOLUME_USDT) &
        # Price actually moved
        (np.abs(df['price_change_bps']) >= config.MIN_PRICE_CHANGE_BPS_FOR_TRUE)
    )

    # Medium confidence TRUE: Good behavioral signals (relaxed thresholds)
    medium_true_mask = (
        # Good cancellation rate (slightly below strong threshold)
        (df['cancellation_rate'] >= config.MIN_CANCELLATION_FOR_TRUE - 0.10) &
        # Reasonable execution ratio (slightly above strong threshold)
        (df['spoofing_execution_ratio'] <= config.MAX_EXECUTION_FOR_TRUE + 0.10) &
        # Fast pattern (typical for manipulation)
        (df['placement_duration'] <= config.MAX_DURATION_FOR_TRUE) &
        # Not already matched
        ~strong_true_mask
    )

    # Outcome-based TRUE: If we have explicit outcome data
    outcome_true_mask = (
        (df['price_moved_as_expected'] == True) |
        (df['manipulation_successful'] == True)
    ) & ~strong_true_mask & ~medium_true_mask

    # ================================================================
    # STEP 2: Strong FALSE POSITIVE indicators
    # ================================================================
    print("Step 2: Identifying FALSE POSITIVES...")

    # High confidence FALSE: Clear signs of legitimate trading
    strong_false_mask = (
        # LOW cancellation (orders were actually filled)
        (df['cancellation_rate'] <= config.MAX_CANCELLATION_FOR_FALSE) &
        # HIGH execution (real orders being executed)
        (df['spoofing_execution_ratio'] >= config.MIN_EXECUTION_FOR_FALSE)
    )

    # Medium confidence FALSE: Indicators of non-manipulation
    medium_false_mask = (
        # Moderate-low cancellation (slightly above strong threshold)
        (df['cancellation_rate'] <= config.MAX_CANCELLATION_FOR_FALSE + 0.10) &
        # Slow pattern (less typical for manipulation)
        (df['placement_duration'] >= config.MIN_DURATION_FOR_FALSE) &
        # Not already matched
        ~strong_false_mask
    )

    # Volume-based FALSE: Very small volume unlikely to be manipulation
    volume_false_mask = (
        (df['total_volume_usdt'] < config.MIN_VOLUME_USDT * 0.5) &
        ~strong_false_mask & ~medium_false_mask
    )

    # ================================================================
    # STEP 3: Apply labels
    # ================================================================
    print("Step 3: Applying labels...")

    # Initialize new label columns
    df['is_true_layering_new'] = None
    df['label_source_new'] = 'unknown'
    df['label_confidence_new'] = 0.0

    # Apply TRUE labels (order matters - higher confidence first)
    df.loc[strong_true_mask, 'is_true_layering_new'] = True
    df.loc[strong_true_mask, 'label_source_new'] = 'automatic_behavioral_strong'
    df.loc[strong_true_mask, 'label_confidence_new'] = config.HIGH_CONFIDENCE

    df.loc[medium_true_mask, 'is_true_layering_new'] = True
    df.loc[medium_true_mask, 'label_source_new'] = 'automatic_behavioral_medium'
    df.loc[medium_true_mask, 'label_confidence_new'] = config.MEDIUM_CONFIDENCE

    df.loc[outcome_true_mask, 'is_true_layering_new'] = True
    df.loc[outcome_true_mask, 'label_source_new'] = 'automatic_outcome'
    df.loc[outcome_true_mask, 'label_confidence_new'] = config.HIGH_CONFIDENCE

    # Apply FALSE labels
    df.loc[strong_false_mask, 'is_true_layering_new'] = False
    df.loc[strong_false_mask, 'label_source_new'] = 'automatic_behavioral_strong'
    df.loc[strong_false_mask, 'label_confidence_new'] = config.HIGH_CONFIDENCE

    df.loc[medium_false_mask, 'is_true_layering_new'] = False
    df.loc[medium_false_mask, 'label_source_new'] = 'automatic_behavioral_medium'
    df.loc[medium_false_mask, 'label_confidence_new'] = config.MEDIUM_CONFIDENCE

    df.loc[volume_false_mask, 'is_true_layering_new'] = False
    df.loc[volume_false_mask, 'label_source_new'] = 'automatic_volume_filter'
    df.loc[volume_false_mask, 'label_confidence_new'] = config.LOW_CONFIDENCE

    # ================================================================
    # STEP 4: Statistics
    # ================================================================
    total_samples = len(df)
    labeled_true = (df['is_true_layering_new'] == True).sum()
    labeled_false = (df['is_true_layering_new'] == False).sum()
    still_unlabeled = df['is_true_layering_new'].isna().sum()

    print(f"\n{'='*80}")
    print("LABELING RESULTS")
    print(f"{'='*80}")
    print(f"\nTotal samples: {total_samples:,}")
    print(f"\nLabeled:")
    print(f"   TRUE (real layering):    {labeled_true:>10,} ({labeled_true/total_samples*100:>5.1f}%)")
    print(f"   FALSE (false positive):  {labeled_false:>10,} ({labeled_false/total_samples*100:>5.1f}%)")
    print(f"   Unlabeled (uncertain):   {still_unlabeled:>10,} ({still_unlabeled/total_samples*100:>5.1f}%)")

    if labeled_true + labeled_false > 0:
        class_balance = labeled_true / (labeled_true + labeled_false)
        print(f"\nClass balance: {class_balance*100:.1f}% TRUE / {(1-class_balance)*100:.1f}% FALSE")

    # Breakdown by confidence
    print(f"\nBreakdown by label source:")
    for source in df['label_source_new'].unique():
        if source != 'unknown':
            count = (df['label_source_new'] == source).sum()
            print(f"   {source}: {count:,}")

    # ================================================================
    # STEP 5: Update original columns
    # ================================================================
    df['is_true_layering'] = df['is_true_layering_new']
    df['label_source'] = df['label_source_new']
    df['label_confidence'] = df['label_confidence_new']

    # Drop temp columns
    df = df.drop(columns=['is_true_layering_new', 'label_source_new', 'label_confidence_new'])

    # ================================================================
    # STEP 6: Save to files
    # ================================================================
    print(f"\nSaving labeled data...")

    # Group by original file (approximate based on timestamp)
    df['file_group'] = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime('%Y%m%d_%H')

    saved_count = 0
    for file_group, group_df in df.groupby('file_group'):
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

    print(f"Saved to {saved_count} files")

    # Update statistics.json
    try:
        stats_file = data_dir / "statistics.json"
        final_labeled = labeled_true + labeled_false
        stats = {
            "total_collected": total_samples,
            "total_labeled": int(final_labeled),
            "total_saved": total_samples,
            "last_updated": datetime.now().isoformat(),
            "labeling_method": "outcome_based_v2",
            "class_balance": {
                "true_count": int(labeled_true),
                "false_count": int(labeled_false),
                "balance_ratio": float(labeled_true / final_labeled) if final_labeled > 0 else 0.0
            },
            "exists": True
        }

        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\nUpdated statistics.json")
    except Exception as e:
        print(f"\nWarning: Could not update statistics.json: {e}")

    print(f"\n{'='*80}")
    print("LABELING COMPLETE")
    print(f"{'='*80}")
    print(f"\nNext step: Run training with improved features:")
    print(f"   python train_layering_model_improved.py")


if __name__ == "__main__":
    print("=" * 80)
    print("IMPROVED LAYERING DATA LABELING")
    print("(Outcome-based, no detector_confidence dependency)")
    print("=" * 80)
    print()

    try:
        label_data_improved()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
