#!/usr/bin/env python3
"""
Labeling Layering Data with Real Price Outcomes

This script creates TRUE outcome-based labels by matching:
1. Layering pattern timestamps
2. Feature Store data with actual future_movement_Xs

This removes ALL circular dependencies - labels are based on
ACTUAL PRICE MOVEMENT, not behavioral features.

Run: python label_layering_with_price_outcome.py
"""

import sys
from pathlib import Path
import warnings
import json
from datetime import datetime

# Check dependencies
try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("=" * 80)
    print("ERROR: Required dependencies not installed")
    print("Install with: pip install pandas pyarrow numpy")
    print("=" * 80)
    sys.exit(1)

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# ============================================================
# CONFIGURATION
# ============================================================

class LabelingConfig:
    """
    Outcome-based labeling configuration.

    Key insight: Use ACTUAL price movement as ground truth,
    completely independent of pattern features.
    """

    # Time window for matching (milliseconds)
    MATCH_WINDOW_MS = 30_000  # ±30 seconds

    # Price movement thresholds (basis points)
    # TRUE: Price moved significantly (manipulation worked)
    MIN_MOVEMENT_FOR_TRUE = 5.0  # > 5 bps = significant movement

    # FALSE: Price barely moved (no manipulation effect)
    MAX_MOVEMENT_FOR_FALSE = 1.5  # < 1.5 bps = no effect

    # UNCERTAIN: Between thresholds (ambiguous)
    # 1.5 - 5.0 bps = uncertain

    # Which future movement to use (10s, 30s, 60s)
    PRIMARY_HORIZON = "30s"
    SECONDARY_HORIZON = "60s"  # Fallback if primary is None

    # Minimum volume for valid pattern
    MIN_VOLUME_USDT = 1000


def load_layering_data() -> pd.DataFrame:
    """Load all layering pattern data."""
    data_dir = project_root / "data" / "ml_training" / "layering"

    if not data_dir.exists():
        print(f"❌ Layering data directory not found: {data_dir}")
        return pd.DataFrame()

    parquet_files = sorted(data_dir.glob("layering_data_*.parquet"))

    if not parquet_files:
        print(f"❌ No layering parquet files found")
        return pd.DataFrame()

    print(f"📂 Found {len(parquet_files)} layering data files")

    dfs = []
    for filepath in parquet_files:
        try:
            df = pd.read_parquet(filepath)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"  ⚠️ Error loading {filepath.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    print(f"✅ Loaded {len(combined):,} layering samples")

    return combined


def load_feature_store_data() -> pd.DataFrame:
    """Load Feature Store data with future_movement fields."""
    fs_dir = project_root / "data" / "feature_store" / "offline" / "training_features"

    if not fs_dir.exists():
        print(f"❌ Feature Store directory not found: {fs_dir}")
        return pd.DataFrame()

    # Find all parquet files in date partitions
    parquet_files = list(fs_dir.glob("**/features_*.parquet"))

    if not parquet_files:
        print(f"❌ No Feature Store parquet files found")
        return pd.DataFrame()

    print(f"📂 Found {len(parquet_files)} Feature Store files")

    # Load only needed columns to save memory
    needed_columns = [
        'symbol', 'timestamp', 'current_mid_price',
        'future_movement_10s', 'future_movement_30s', 'future_movement_60s',
        'future_direction_10s', 'future_direction_30s', 'future_direction_60s'
    ]

    dfs = []
    for filepath in parquet_files:
        try:
            df = pd.read_parquet(filepath)
            # Select only columns that exist
            cols_to_use = [c for c in needed_columns if c in df.columns]
            if cols_to_use:
                dfs.append(df[cols_to_use])
        except Exception as e:
            print(f"  ⚠️ Error loading {filepath.name}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    # Check for future_movement columns
    movement_cols = [c for c in combined.columns if 'future_movement' in c]
    print(f"✅ Loaded {len(combined):,} Feature Store samples")
    print(f"   Movement columns: {movement_cols}")

    # Check how many have valid future_movement
    if 'future_movement_30s' in combined.columns:
        valid_count = combined['future_movement_30s'].notna().sum()
        print(f"   Valid future_movement_30s: {valid_count:,} ({valid_count/len(combined)*100:.1f}%)")

    return combined


def match_and_label(
    layering_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    config: LabelingConfig
) -> pd.DataFrame:
    """
    Match layering patterns with feature store data and create labels.

    Uses timestamp matching with tolerance window.
    """
    print(f"\n{'='*80}")
    print("MATCHING LAYERING PATTERNS WITH PRICE OUTCOMES")
    print(f"{'='*80}")

    if layering_df.empty or feature_df.empty:
        print("❌ Empty data, cannot match")
        return layering_df

    # Ensure timestamps are numeric (milliseconds)
    layering_df['timestamp'] = pd.to_numeric(layering_df['timestamp'], errors='coerce')
    feature_df['timestamp'] = pd.to_numeric(feature_df['timestamp'], errors='coerce')

    # Sort by timestamp for efficient matching
    feature_df = feature_df.sort_values('timestamp').reset_index(drop=True)

    # Get movement column
    movement_col = f"future_movement_{config.PRIMARY_HORIZON}"
    fallback_col = f"future_movement_{config.SECONDARY_HORIZON}"

    if movement_col not in feature_df.columns:
        print(f"❌ Column {movement_col} not found in Feature Store")
        print(f"   Available columns: {list(feature_df.columns)}")
        return layering_df

    # Statistics
    matched_count = 0
    true_count = 0
    false_count = 0
    uncertain_count = 0
    no_match_count = 0

    # Create new label columns
    layering_df['outcome_label'] = None
    layering_df['outcome_movement_bps'] = None
    layering_df['outcome_source'] = 'no_match'
    layering_df['outcome_confidence'] = 0.0

    print(f"\nMatching with ±{config.MATCH_WINDOW_MS/1000:.0f}s window...")
    print(f"Movement thresholds: TRUE > {config.MIN_MOVEMENT_FOR_TRUE} bps, FALSE < {config.MAX_MOVEMENT_FOR_FALSE} bps")

    # Group feature data by symbol for faster matching
    feature_by_symbol = {
        symbol: group.reset_index(drop=True)
        for symbol, group in feature_df.groupby('symbol')
    }

    # Process each layering pattern
    total = len(layering_df)
    for idx, row in layering_df.iterrows():
        if idx % 10000 == 0:
            print(f"  Processing {idx:,}/{total:,}...")

        symbol = row.get('symbol', '')
        ts = row['timestamp']

        if pd.isna(ts) or symbol not in feature_by_symbol:
            no_match_count += 1
            continue

        symbol_features = feature_by_symbol[symbol]

        # Find matching timestamps within window
        mask = (
            (symbol_features['timestamp'] >= ts - config.MATCH_WINDOW_MS) &
            (symbol_features['timestamp'] <= ts + config.MATCH_WINDOW_MS)
        )

        matches = symbol_features[mask]

        if matches.empty:
            no_match_count += 1
            continue

        # Get closest match
        closest_idx = (matches['timestamp'] - ts).abs().idxmin()
        closest = matches.loc[closest_idx]

        # Get movement value
        movement = closest.get(movement_col)
        if pd.isna(movement):
            movement = closest.get(fallback_col)

        if pd.isna(movement):
            no_match_count += 1
            continue

        matched_count += 1
        abs_movement = abs(movement)

        # Store movement
        layering_df.at[idx, 'outcome_movement_bps'] = movement

        # Apply labeling thresholds
        if abs_movement >= config.MIN_MOVEMENT_FOR_TRUE:
            layering_df.at[idx, 'outcome_label'] = True
            layering_df.at[idx, 'outcome_source'] = 'price_movement_strong'
            layering_df.at[idx, 'outcome_confidence'] = min(abs_movement / 10.0, 1.0)
            true_count += 1
        elif abs_movement <= config.MAX_MOVEMENT_FOR_FALSE:
            layering_df.at[idx, 'outcome_label'] = False
            layering_df.at[idx, 'outcome_source'] = 'price_movement_weak'
            layering_df.at[idx, 'outcome_confidence'] = 1.0 - (abs_movement / config.MAX_MOVEMENT_FOR_FALSE)
            false_count += 1
        else:
            layering_df.at[idx, 'outcome_label'] = None
            layering_df.at[idx, 'outcome_source'] = 'uncertain'
            uncertain_count += 1

    # Print statistics
    print(f"\n{'='*80}")
    print("MATCHING RESULTS")
    print(f"{'='*80}")
    print(f"\nTotal layering patterns: {total:,}")
    print(f"Matched with price data: {matched_count:,} ({matched_count/total*100:.1f}%)")
    print(f"No match found:          {no_match_count:,} ({no_match_count/total*100:.1f}%)")
    print(f"\nLabeled:")
    print(f"  TRUE (price moved):     {true_count:>10,} ({true_count/total*100:.1f}%)")
    print(f"  FALSE (no movement):    {false_count:>10,} ({false_count/total*100:.1f}%)")
    print(f"  UNCERTAIN (ambiguous):  {uncertain_count:>10,} ({uncertain_count/total*100:.1f}%)")

    if true_count + false_count > 0:
        balance = true_count / (true_count + false_count)
        print(f"\nClass balance: {balance*100:.1f}% TRUE / {(1-balance)*100:.1f}% FALSE")

    return layering_df


def update_original_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Update original is_true_layering based on outcome labels.

    Priority:
    1. If outcome_label is not None, use it
    2. Otherwise keep existing label
    """
    # Create backup of old labels
    df['old_is_true_layering'] = df['is_true_layering']
    df['old_label_source'] = df['label_source']

    # Update with outcome labels where available
    outcome_mask = df['outcome_label'].notna()
    df.loc[outcome_mask, 'is_true_layering'] = df.loc[outcome_mask, 'outcome_label']
    df.loc[outcome_mask, 'label_source'] = df.loc[outcome_mask, 'outcome_source']
    df.loc[outcome_mask, 'label_confidence'] = df.loc[outcome_mask, 'outcome_confidence']

    # Statistics on changes
    changed_count = (df['is_true_layering'] != df['old_is_true_layering']).sum()
    print(f"\n📊 Labels updated: {changed_count:,} patterns")

    return df


def save_labeled_data(df: pd.DataFrame):
    """Save labeled data back to files."""
    data_dir = project_root / "data" / "ml_training" / "layering"

    print(f"\n💾 Saving labeled data...")

    # Group by original file (approximate based on timestamp)
    df['file_group'] = pd.to_datetime(df['timestamp'], unit='ms').dt.strftime('%Y%m%d_%H')

    saved_count = 0
    for file_group, group_df in df.groupby('file_group'):
        output_file = data_dir / f"layering_data_{file_group}00_outcome.parquet"

        try:
            # Drop temporary columns
            save_df = group_df.drop(['file_group', 'old_is_true_layering', 'old_label_source'], axis=1, errors='ignore')
            save_df.to_parquet(output_file, index=False)
            saved_count += 1
        except Exception as e:
            print(f"  ⚠️ Error saving {output_file.name}: {e}")

    print(f"✅ Saved to {saved_count} files")

    # Update statistics.json
    stats_file = data_dir / "statistics.json"
    labeled_true = (df['is_true_layering'] == True).sum()
    labeled_false = (df['is_true_layering'] == False).sum()

    stats = {
        "total_collected": len(df),
        "total_labeled": int(labeled_true + labeled_false),
        "total_saved": len(df),
        "last_updated": datetime.now().isoformat(),
        "labeling_method": "price_outcome_based",
        "class_balance": {
            "true_count": int(labeled_true),
            "false_count": int(labeled_false),
            "balance_ratio": float(labeled_true / (labeled_true + labeled_false)) if (labeled_true + labeled_false) > 0 else 0.0
        },
        "outcome_matching": {
            "matched": int((df['outcome_label'].notna()).sum()),
            "unmatched": int((df['outcome_label'].isna()).sum())
        },
        "exists": True
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✅ Updated statistics.json")


def main():
    """Main function."""
    print("=" * 80)
    print("OUTCOME-BASED LAYERING LABELING")
    print("Using actual price movements from Feature Store")
    print("=" * 80)
    print()

    config = LabelingConfig()

    # Step 1: Load layering data
    print("\n📥 STEP 1: Loading Layering Data")
    print("-" * 40)
    layering_df = load_layering_data()

    if layering_df.empty:
        print("❌ No layering data found")
        return

    # Step 2: Load Feature Store data
    print("\n📥 STEP 2: Loading Feature Store Data")
    print("-" * 40)
    feature_df = load_feature_store_data()

    if feature_df.empty:
        print("❌ No Feature Store data found")
        print("\n⚠️ Feature Store data is required for outcome-based labeling.")
        print("Make sure the bot has been running with ML data collection enabled.")
        return

    # Step 3: Match and label
    print("\n🔄 STEP 3: Matching & Labeling")
    print("-" * 40)
    labeled_df = match_and_label(layering_df, feature_df, config)

    # Step 4: Update original labels
    print("\n📝 STEP 4: Updating Labels")
    print("-" * 40)
    final_df = update_original_labels(labeled_df)

    # Step 5: Save
    print("\n💾 STEP 5: Saving Results")
    print("-" * 40)
    save_labeled_data(final_df)

    print("\n" + "=" * 80)
    print("LABELING COMPLETE")
    print("=" * 80)
    print("\nNext step: Run training:")
    print("  python train_layering_model_improved.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
