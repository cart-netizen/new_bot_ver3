#!/usr/bin/env python3
"""
Feature Quality Analyzer - Analyzes feature predictiveness for ML training.

This script helps diagnose mode collapse issues by analyzing:
1. Feature-label correlations
2. Class separability
3. Feature distributions
4. Information gain

Run this before hyperopt to ensure features are predictive.

File: backend/ml_engine/feature_quality_analyzer.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

from backend.core.logger import get_logger

logger = get_logger(__name__)


class FeatureQualityAnalyzer:
    """
    Analyzes feature quality for ML training.

    Key diagnostics:
    1. Feature-label correlation: Do features predict the label?
    2. Class separability: Can we distinguish classes by feature values?
    3. Feature distributions: Are features normalized/scaled properly?
    4. Information gain: How much info does each feature provide?
    """

    def __init__(self, output_dir: str = "data/feature_analysis"):
        """
        Initialize analyzer.

        Args:
            output_dir: Directory for output reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.report_path = self.output_dir / "feature_quality_report.md"
        self.json_path = self.output_dir / "feature_quality_data.json"

    def analyze(
        self,
        features_df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str = "future_direction_60s",
        top_n: int = 20
    ) -> Dict[str, Any]:
        """
        Run full feature quality analysis.

        Args:
            features_df: DataFrame with features and labels
            feature_columns: List of feature column names
            label_column: Name of label column
            top_n: Number of top features to report

        Returns:
            Dict with analysis results
        """
        logger.info(f"Starting feature quality analysis on {len(features_df)} samples")
        logger.info(f"Analyzing {len(feature_columns)} features")

        results = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": len(features_df),
            "n_features": len(feature_columns),
            "label_column": label_column
        }

        # 1. Class distribution analysis
        logger.info("Analyzing class distribution...")
        class_dist = self._analyze_class_distribution(features_df, label_column)
        results["class_distribution"] = class_dist

        # 2. Feature statistics
        logger.info("Computing feature statistics...")
        feature_stats = self._compute_feature_statistics(features_df, feature_columns)
        results["feature_statistics"] = feature_stats

        # 3. Feature-label correlations
        logger.info("Computing feature-label correlations...")
        correlations = self._compute_correlations(features_df, feature_columns, label_column)
        results["correlations"] = correlations

        # 4. Class separability
        logger.info("Analyzing class separability...")
        separability = self._analyze_class_separability(features_df, feature_columns, label_column)
        results["separability"] = separability

        # 5. Information metrics
        logger.info("Computing information metrics...")
        info_metrics = self._compute_information_metrics(features_df, feature_columns, label_column)
        results["information_metrics"] = info_metrics

        # 6. Overall quality score
        quality_score = self._compute_quality_score(results)
        results["quality_score"] = quality_score

        # Generate reports
        self._generate_markdown_report(results, top_n)
        self._save_json(results)

        logger.info(f"Feature quality analysis complete. Score: {quality_score['overall']:.2f}/100")
        logger.info(f"Reports saved to: {self.output_dir}")

        return results

    def _analyze_class_distribution(
        self,
        df: pd.DataFrame,
        label_column: str
    ) -> Dict[str, Any]:
        """Analyze class distribution."""
        if label_column not in df.columns:
            return {"error": f"Label column '{label_column}' not found"}

        labels = df[label_column].dropna()
        counts = Counter(labels)
        total = len(labels)

        class_names = {0: "SELL", 1: "HOLD", 2: "BUY"}

        distribution = {}
        for cls in [0, 1, 2]:
            count = counts.get(cls, 0)
            distribution[class_names.get(cls, str(cls))] = {
                "count": count,
                "percentage": count / total * 100 if total > 0 else 0
            }

        # Calculate imbalance metrics
        if len(counts) > 0:
            max_class = max(counts.values())
            min_class = min(counts.values()) if min(counts.values()) > 0 else 1
            imbalance_ratio = max_class / min_class
        else:
            imbalance_ratio = 0

        return {
            "distribution": distribution,
            "total_samples": total,
            "imbalance_ratio": imbalance_ratio,
            "is_balanced": imbalance_ratio < 2.0,
            "recommendation": self._get_balance_recommendation(imbalance_ratio)
        }

    def _get_balance_recommendation(self, ratio: float) -> str:
        """Get recommendation based on imbalance ratio."""
        if ratio < 1.5:
            return "Classes are well balanced. No special handling needed."
        elif ratio < 3.0:
            return "Moderate imbalance. Consider class weights or oversampling."
        elif ratio < 5.0:
            return "Significant imbalance. Use oversampling + focal loss."
        else:
            return "SEVERE imbalance! May cause mode collapse. Consider SMOTE or aggressive oversampling."

    def _compute_feature_statistics(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> Dict[str, Any]:
        """Compute basic statistics for each feature."""
        stats = {}

        for col in feature_columns:
            if col not in df.columns:
                continue

            values = df[col].dropna()

            if len(values) == 0:
                stats[col] = {"error": "No valid values"}
                continue

            stats[col] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "median": float(values.median()),
                "null_pct": float(df[col].isnull().sum() / len(df) * 100),
                "unique_count": int(values.nunique()),
                "is_constant": values.nunique() <= 1,
                "is_normalized": -10 <= values.mean() <= 10 and values.std() <= 100
            }

        # Summary
        n_constant = sum(1 for s in stats.values() if isinstance(s, dict) and s.get("is_constant", False))
        n_high_null = sum(1 for s in stats.values() if isinstance(s, dict) and s.get("null_pct", 0) > 10)

        return {
            "features": stats,
            "summary": {
                "constant_features": n_constant,
                "high_null_features": n_high_null,
                "total_analyzed": len(stats)
            }
        }

    def _compute_correlations(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str
    ) -> Dict[str, Any]:
        """Compute feature-label correlations."""
        correlations = {}

        if label_column not in df.columns:
            return {"error": f"Label column '{label_column}' not found"}

        labels = df[label_column]

        for col in feature_columns:
            if col not in df.columns:
                continue

            try:
                # Pearson correlation
                corr = df[col].corr(labels)
                correlations[col] = {
                    "pearson": float(corr) if not np.isnan(corr) else 0.0,
                    "abs_corr": abs(float(corr)) if not np.isnan(corr) else 0.0
                }
            except Exception:
                correlations[col] = {"pearson": 0.0, "abs_corr": 0.0}

        # Sort by absolute correlation
        sorted_corr = sorted(
            correlations.items(),
            key=lambda x: x[1]["abs_corr"],
            reverse=True
        )

        top_features = sorted_corr[:20]
        bottom_features = sorted_corr[-10:]

        return {
            "correlations": correlations,
            "top_correlated": [{"feature": f, **c} for f, c in top_features],
            "least_correlated": [{"feature": f, **c} for f, c in bottom_features],
            "mean_abs_correlation": np.mean([c["abs_corr"] for c in correlations.values()])
        }

    def _analyze_class_separability(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str
    ) -> Dict[str, Any]:
        """Analyze how well features separate classes."""
        separability = {}

        if label_column not in df.columns:
            return {"error": f"Label column '{label_column}' not found"}

        # Group by class
        grouped = df.groupby(label_column)

        for col in feature_columns[:50]:  # Analyze first 50 for speed
            if col not in df.columns:
                continue

            try:
                # Get mean for each class
                class_means = grouped[col].mean()
                class_stds = grouped[col].std()

                # Calculate Fisher's discriminant ratio
                # Higher = better separation
                if len(class_means) >= 2:
                    overall_mean = df[col].mean()
                    between_var = sum(
                        grouped.size()[c] * (class_means[c] - overall_mean) ** 2
                        for c in class_means.index
                    ) / len(df)
                    within_var = sum(
                        grouped.size()[c] * class_stds[c] ** 2
                        for c in class_stds.index if not np.isnan(class_stds[c])
                    ) / len(df)

                    fisher_ratio = between_var / (within_var + 1e-10)
                else:
                    fisher_ratio = 0

                separability[col] = {
                    "fisher_ratio": float(fisher_ratio),
                    "class_means": {str(k): float(v) for k, v in class_means.items()},
                    "has_separation": fisher_ratio > 0.01
                }
            except Exception as e:
                separability[col] = {"fisher_ratio": 0.0, "error": str(e)}

        # Sort by Fisher ratio
        sorted_sep = sorted(
            separability.items(),
            key=lambda x: x[1].get("fisher_ratio", 0),
            reverse=True
        )

        top_separating = sorted_sep[:20]

        # Count features with good separation
        n_good_sep = sum(1 for s in separability.values() if s.get("fisher_ratio", 0) > 0.01)

        return {
            "features": separability,
            "top_separating": [{"feature": f, **s} for f, s in top_separating],
            "n_features_with_separation": n_good_sep,
            "separation_rate": n_good_sep / len(separability) if separability else 0
        }

    def _compute_information_metrics(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        label_column: str
    ) -> Dict[str, Any]:
        """Compute information-theoretic metrics."""
        # Mutual information would be ideal but slow
        # Use correlation-based proxy

        if label_column not in df.columns:
            return {"error": f"Label column '{label_column}' not found"}

        # Calculate feature redundancy (inter-feature correlation)
        sample_features = feature_columns[:30]  # Sample for speed
        valid_features = [f for f in sample_features if f in df.columns]

        if len(valid_features) < 2:
            return {"redundancy": 0, "error": "Not enough features"}

        try:
            corr_matrix = df[valid_features].corr().abs()
            # Average off-diagonal correlation
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            mean_redundancy = corr_matrix.where(mask).mean().mean()
        except Exception:
            mean_redundancy = 0

        return {
            "feature_redundancy": float(mean_redundancy) if not np.isnan(mean_redundancy) else 0,
            "redundancy_assessment": (
                "HIGH redundancy - consider PCA or feature selection"
                if mean_redundancy > 0.5 else
                "MODERATE redundancy" if mean_redundancy > 0.3 else
                "LOW redundancy - features are diverse"
            )
        }

    def _compute_quality_score(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall quality score (0-100)."""
        scores = {}

        # 1. Class balance score (0-25)
        imbalance = results.get("class_distribution", {}).get("imbalance_ratio", 10)
        if imbalance < 1.5:
            scores["balance"] = 25
        elif imbalance < 3:
            scores["balance"] = 20
        elif imbalance < 5:
            scores["balance"] = 10
        else:
            scores["balance"] = 5

        # 2. Correlation score (0-25)
        mean_corr = results.get("correlations", {}).get("mean_abs_correlation", 0)
        scores["correlation"] = min(25, mean_corr * 100)

        # 3. Separability score (0-25)
        sep_rate = results.get("separability", {}).get("separation_rate", 0)
        scores["separability"] = sep_rate * 25

        # 4. Feature health score (0-25)
        feature_stats = results.get("feature_statistics", {}).get("summary", {})
        n_constant = feature_stats.get("constant_features", 0)
        n_high_null = feature_stats.get("high_null_features", 0)
        total = feature_stats.get("total_analyzed", 1)

        health_pct = 1 - (n_constant + n_high_null) / total
        scores["health"] = health_pct * 25

        overall = sum(scores.values())

        return {
            "overall": overall,
            "breakdown": scores,
            "assessment": (
                "EXCELLENT - Features are highly predictive"
                if overall >= 80 else
                "GOOD - Features have reasonable predictive power"
                if overall >= 60 else
                "MODERATE - May struggle with prediction"
                if overall >= 40 else
                "POOR - High risk of mode collapse! Check features and labels."
            )
        }

    def _generate_markdown_report(self, results: Dict[str, Any], top_n: int):
        """Generate markdown report."""
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write("# Feature Quality Analysis Report\n\n")
            f.write(f"**Generated:** {results['timestamp']}\n")
            f.write(f"**Samples:** {results['n_samples']:,}\n")
            f.write(f"**Features:** {results['n_features']}\n\n")

            # Quality Score
            score = results.get("quality_score", {})
            f.write("## Overall Quality Score\n\n")
            f.write(f"**Score: {score.get('overall', 0):.1f}/100**\n\n")
            f.write(f"**Assessment:** {score.get('assessment', 'Unknown')}\n\n")

            f.write("### Score Breakdown\n\n")
            for component, value in score.get("breakdown", {}).items():
                bar = "█" * int(value) + "░" * (25 - int(value))
                f.write(f"- {component.capitalize()}: {bar} {value:.1f}/25\n")
            f.write("\n")

            # Class Distribution
            f.write("## Class Distribution\n\n")
            dist = results.get("class_distribution", {}).get("distribution", {})
            for cls, info in dist.items():
                f.write(f"- **{cls}:** {info['count']:,} ({info['percentage']:.1f}%)\n")
            f.write(f"\n**Imbalance Ratio:** {results.get('class_distribution', {}).get('imbalance_ratio', 0):.2f}\n")
            f.write(f"\n**Recommendation:** {results.get('class_distribution', {}).get('recommendation', '')}\n\n")

            # Top Correlated Features
            f.write("## Top Correlated Features\n\n")
            f.write("| Rank | Feature | Correlation |\n")
            f.write("|------|---------|-------------|\n")
            for i, item in enumerate(results.get("correlations", {}).get("top_correlated", [])[:top_n], 1):
                f.write(f"| {i} | {item['feature']} | {item['pearson']:.4f} |\n")
            f.write("\n")

            # Top Separating Features
            f.write("## Top Separating Features (Fisher Ratio)\n\n")
            f.write("| Rank | Feature | Fisher Ratio |\n")
            f.write("|------|---------|-------------|\n")
            for i, item in enumerate(results.get("separability", {}).get("top_separating", [])[:top_n], 1):
                f.write(f"| {i} | {item['feature']} | {item['fisher_ratio']:.4f} |\n")
            f.write("\n")

            # Feature Redundancy
            f.write("## Feature Redundancy\n\n")
            info = results.get("information_metrics", {})
            f.write(f"**Mean Inter-feature Correlation:** {info.get('feature_redundancy', 0):.4f}\n")
            f.write(f"\n**Assessment:** {info.get('redundancy_assessment', 'Unknown')}\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            overall = score.get("overall", 0)
            if overall < 40:
                f.write("1. **CRITICAL:** Feature quality is poor. Mode collapse is likely.\n")
                f.write("2. Check if labels are correctly generated (Triple Barrier parameters)\n")
                f.write("3. Consider increasing prediction horizon or using different labeling\n")
                f.write("4. Verify feature calculation is correct (no look-ahead bias)\n")
            elif overall < 60:
                f.write("1. Consider feature selection to remove low-quality features\n")
                f.write("2. Try different feature engineering approaches\n")
                f.write("3. Ensure proper normalization/scaling\n")
            else:
                f.write("1. Features appear reasonable for training\n")
                f.write("2. Focus on model architecture and hyperparameters\n")

            f.write("\n---\n*Generated by FeatureQualityAnalyzer*\n")

    def _save_json(self, results: Dict[str, Any]):
        """Save results as JSON."""
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=2, default=str)


async def analyze_training_data():
    """Run analysis on training data from feature store."""
    print("=" * 60)
    print("FEATURE QUALITY ANALYZER")
    print("=" * 60)

    try:
        from backend.ml_engine.feature_store.feature_store import get_feature_store
        from backend.ml_engine.feature_store.feature_schema import DEFAULT_SCHEMA

        print("\n[1/4] Loading data from Feature Store...")

        feature_store = get_feature_store()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

        features_df = feature_store.read_offline_features(
            feature_group="training_features",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )

        if features_df is None or features_df.empty:
            print("\n❌ ERROR: No data found in Feature Store!")
            print("  Make sure you have data in data/feature_store/training_features/")
            print("  Or run preprocessing first to generate training features.")
            return None

        print(f"  ✓ Loaded {len(features_df):,} samples")
        print(f"  ✓ Columns: {len(features_df.columns)}")

        # Check if label column exists
        label_col = DEFAULT_SCHEMA.label_column
        if label_col not in features_df.columns:
            print(f"\n❌ ERROR: Label column '{label_col}' not found!")
            print(f"  Available columns: {list(features_df.columns)[:20]}...")
            print("  You may need to run preprocessing to add labels.")
            return None

        print(f"  ✓ Label column: {label_col}")

        # Run analysis
        print("\n[2/4] Running feature analysis...")
        analyzer = FeatureQualityAnalyzer()

        feature_columns = DEFAULT_SCHEMA.get_all_feature_columns()
        # Filter to only existing columns
        existing_features = [c for c in feature_columns if c in features_df.columns]
        print(f"  Features to analyze: {len(existing_features)}/{len(feature_columns)}")

        if len(existing_features) == 0:
            print("\n❌ ERROR: No feature columns found in data!")
            print(f"  Expected: {feature_columns[:10]}...")
            print(f"  Found: {list(features_df.columns)[:10]}...")
            return None

        results = analyzer.analyze(
            features_df=features_df,
            feature_columns=existing_features,
            label_column=label_col
        )

        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)

        score = results.get("quality_score", {})
        print(f"\n📊 OVERALL QUALITY SCORE: {score.get('overall', 0):.1f}/100")
        print(f"   {score.get('assessment', 'Unknown')}")

        print("\n📈 Score Breakdown:")
        for component, value in score.get("breakdown", {}).items():
            bar = "█" * int(value) + "░" * (25 - int(value))
            print(f"   {component.capitalize():15} {bar} {value:.1f}/25")

        # Class distribution
        dist = results.get("class_distribution", {})
        print(f"\n📦 Class Distribution:")
        for cls, info in dist.get("distribution", {}).items():
            print(f"   {cls}: {info['count']:,} ({info['percentage']:.1f}%)")
        print(f"   Imbalance ratio: {dist.get('imbalance_ratio', 0):.2f}")

        # Top features
        print(f"\n🔝 Top 5 Correlated Features:")
        for i, item in enumerate(results.get("correlations", {}).get("top_correlated", [])[:5], 1):
            print(f"   {i}. {item['feature']}: {item['pearson']:.4f}")

        print(f"\n📁 Reports saved to: {analyzer.output_dir}")
        print(f"   - {analyzer.report_path.name}")
        print(f"   - {analyzer.json_path.name}")

        print("\n" + "=" * 60)

        return results

    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import asyncio
    print("Starting Feature Quality Analyzer...")
    result = asyncio.run(analyze_training_data())
    if result is None:
        print("\nAnalysis failed or no data available.")
        exit(1)
    else:
        print("\nAnalysis completed successfully!")
        exit(0)
