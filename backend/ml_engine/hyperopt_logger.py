#!/usr/bin/env python3
"""
Structured Hyperopt Logger - Comprehensive logging for hyperparameter optimization.

This module provides structured logging to files for later analysis by AI or humans.
All important events are logged in both human-readable and JSON formats.

File: backend/ml_engine/hyperopt_logger.py
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import numpy as np

from backend.core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrialLogEntry:
    """Structured log entry for a single trial."""
    trial_number: int
    group: str
    timestamp: str

    # Parameters
    params: Dict[str, Any]
    fixed_params: Dict[str, Any]

    # Training results
    epochs_completed: int
    best_val_loss: float
    best_val_f1: float
    final_val_loss: float
    final_val_f1: float

    # Test results
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float

    # Class distribution analysis
    class_distribution: Dict[str, int]
    prediction_distribution: Dict[str, int]
    mode_collapse_detected: bool
    mode_collapse_severity: str  # "none", "mild", "severe", "total"
    majority_class_percentage: float

    # Confusion matrix (as list for JSON serialization)
    confusion_matrix: List[List[int]]

    # Per-class metrics
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]

    # Confidence analysis
    mean_confidence_correct: float
    mean_confidence_incorrect: float

    # Training dynamics
    loss_trend: str  # "decreasing", "stable", "increasing", "oscillating"
    learning_curve_quality: str  # "good", "overfitting", "underfitting", "collapsed"

    # Time
    duration_seconds: float

    # Status
    status: str  # "completed", "pruned", "failed", "mode_collapse"
    error_message: Optional[str] = None


class HyperoptLogger:
    """
    Structured logger for hyperparameter optimization.

    Saves all trial information to:
    - hyperopt_log.jsonl - JSON Lines format for programmatic analysis
    - hyperopt_report.md - Human-readable markdown report
    - trial_summaries/ - Individual trial summaries
    """

    CLASS_NAMES = {0: "SELL", 1: "HOLD", 2: "BUY"}

    def __init__(self, output_dir: str = "data/hyperopt"):
        """
        Initialize the logger.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "trial_summaries").mkdir(exist_ok=True)

        # File paths
        self.jsonl_path = self.output_dir / "hyperopt_log.jsonl"
        self.report_path = self.output_dir / "hyperopt_report.md"
        self.analysis_path = self.output_dir / "analysis_data.json"

        # Session info
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")

        # Statistics
        self.total_trials = 0
        self.completed_trials = 0
        self.mode_collapse_trials = 0
        self.best_trial: Optional[TrialLogEntry] = None

        # Initialize report
        self._init_report()

        logger.info(f"HyperoptLogger initialized: {self.output_dir}")

    def _init_report(self):
        """Initialize the markdown report file."""
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Hyperopt Optimization Report\n\n")
            f.write(f"**Session ID:** {self.session_id}\n")
            f.write(f"**Started:** {self.session_start.isoformat()}\n\n")
            f.write(f"---\n\n")

    def log_group_start(self, group: str, fixed_params: Dict[str, Any], search_space: Dict[str, Any]):
        """Log the start of a parameter group optimization."""
        timestamp = datetime.now().isoformat()

        # Write to report
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(f"## Group: {group}\n\n")
            f.write(f"**Started:** {timestamp}\n\n")
            f.write(f"### Fixed Parameters (from previous groups)\n\n")
            f.write("```json\n")
            f.write(json.dumps(fixed_params, indent=2, default=str))
            f.write("\n```\n\n")
            f.write(f"### Search Space\n\n")
            for param_name, space in search_space.items():
                f.write(f"- **{param_name}**: {space}\n")
            f.write("\n---\n\n")

        logger.info(f"HYPEROPT_LOG: Started group '{group}' with {len(search_space)} parameters to optimize")

    def log_trial(
        self,
        trial_number: int,
        group: str,
        params: Dict[str, Any],
        fixed_params: Dict[str, Any],
        training_history: List[Dict[str, float]],
        test_results: Dict[str, Any],
        all_predictions: List[int],
        all_labels: List[int],
        all_confidences: List[float],
        duration_seconds: float,
        status: str = "completed",
        error_message: Optional[str] = None
    ) -> TrialLogEntry:
        """
        Log a complete trial with all metrics.

        Args:
            trial_number: Trial number
            group: Parameter group being optimized
            params: Parameters being tested
            fixed_params: Fixed parameters from previous groups
            training_history: List of epoch metrics dicts
            test_results: Test evaluation results
            all_predictions: All test predictions
            all_labels: All test labels
            all_confidences: All prediction confidences
            duration_seconds: Trial duration
            status: Trial status
            error_message: Error message if failed

        Returns:
            TrialLogEntry for further processing
        """
        from collections import Counter
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

        timestamp = datetime.now().isoformat()

        # Analyze predictions
        pred_counter = Counter(all_predictions)
        label_counter = Counter(all_labels)

        # Convert to class name keys
        pred_dist = {self.CLASS_NAMES.get(k, str(k)): v for k, v in pred_counter.items()}
        label_dist = {self.CLASS_NAMES.get(k, str(k)): v for k, v in label_counter.items()}

        # Mode collapse detection
        total_preds = len(all_predictions)
        if total_preds > 0:
            most_common_count = pred_counter.most_common(1)[0][1]
            majority_pct = most_common_count / total_preds
        else:
            majority_pct = 0.0

        # Determine mode collapse severity
        if majority_pct >= 0.99:
            mode_collapse_severity = "total"
            mode_collapse_detected = True
        elif majority_pct >= 0.90:
            mode_collapse_severity = "severe"
            mode_collapse_detected = True
        elif majority_pct >= 0.75:
            mode_collapse_severity = "mild"
            mode_collapse_detected = True
        else:
            mode_collapse_severity = "none"
            mode_collapse_detected = False

        # Confusion matrix
        if len(all_predictions) > 0 and len(all_labels) > 0:
            cm = confusion_matrix(all_labels, all_predictions, labels=[0, 1, 2])
            cm_list = cm.tolist()
        else:
            cm_list = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        # Per-class metrics
        if len(all_predictions) > 0:
            prec, rec, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, labels=[0, 1, 2], zero_division=0
            )
            per_class_precision = {self.CLASS_NAMES[i]: float(prec[i]) for i in range(3)}
            per_class_recall = {self.CLASS_NAMES[i]: float(rec[i]) for i in range(3)}
            per_class_f1 = {self.CLASS_NAMES[i]: float(f1[i]) for i in range(3)}
        else:
            per_class_precision = {name: 0.0 for name in self.CLASS_NAMES.values()}
            per_class_recall = {name: 0.0 for name in self.CLASS_NAMES.values()}
            per_class_f1 = {name: 0.0 for name in self.CLASS_NAMES.values()}

        # Confidence analysis
        confidences = np.array(all_confidences)
        correct_mask = np.array(all_predictions) == np.array(all_labels)

        mean_conf_correct = float(confidences[correct_mask].mean()) if correct_mask.any() else 0.0
        mean_conf_incorrect = float(confidences[~correct_mask].mean()) if (~correct_mask).any() else 0.0

        # Training dynamics analysis
        if len(training_history) >= 2:
            losses = [h.get('val_loss', h.get('train_loss', 0)) for h in training_history]
            loss_trend = self._analyze_trend(losses)

            # Learning curve quality
            if mode_collapse_detected:
                learning_curve_quality = "collapsed"
            elif len(losses) >= 3:
                first_third = np.mean(losses[:len(losses)//3])
                last_third = np.mean(losses[-len(losses)//3:])
                if last_third < first_third * 0.9:
                    learning_curve_quality = "good"
                elif last_third > first_third * 1.1:
                    learning_curve_quality = "overfitting"
                else:
                    learning_curve_quality = "underfitting"
            else:
                learning_curve_quality = "underfitting"
        else:
            loss_trend = "unknown"
            learning_curve_quality = "unknown"

        # Extract training metrics
        if training_history:
            best_epoch = max(training_history, key=lambda x: x.get('val_f1', 0))
            final_epoch = training_history[-1]

            best_val_loss = min(h.get('val_loss', float('inf')) for h in training_history)
            best_val_f1 = max(h.get('val_f1', 0) for h in training_history)
        else:
            best_val_loss = float('inf')
            best_val_f1 = 0.0
            final_epoch = {}

        # Create log entry
        entry = TrialLogEntry(
            trial_number=trial_number,
            group=group,
            timestamp=timestamp,
            params=params,
            fixed_params=fixed_params,
            epochs_completed=len(training_history),
            best_val_loss=best_val_loss,
            best_val_f1=best_val_f1,
            final_val_loss=final_epoch.get('val_loss', float('inf')),
            final_val_f1=final_epoch.get('val_f1', 0.0),
            test_accuracy=test_results.get('accuracy', 0.0),
            test_precision=test_results.get('precision', 0.0),
            test_recall=test_results.get('recall', 0.0),
            test_f1=test_results.get('f1', 0.0),
            class_distribution=label_dist,
            prediction_distribution=pred_dist,
            mode_collapse_detected=mode_collapse_detected,
            mode_collapse_severity=mode_collapse_severity,
            majority_class_percentage=majority_pct,
            confusion_matrix=cm_list,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            mean_confidence_correct=mean_conf_correct,
            mean_confidence_incorrect=mean_conf_incorrect,
            loss_trend=loss_trend,
            learning_curve_quality=learning_curve_quality,
            duration_seconds=duration_seconds,
            status=status if not mode_collapse_detected else "mode_collapse",
            error_message=error_message
        )

        # Update statistics
        self.total_trials += 1
        if status == "completed":
            self.completed_trials += 1
        if mode_collapse_detected:
            self.mode_collapse_trials += 1

        # Update best trial
        if not mode_collapse_detected and entry.test_f1 > 0:
            if self.best_trial is None or entry.test_f1 > self.best_trial.test_f1:
                self.best_trial = entry

        # Write to files
        self._write_jsonl(entry)
        self._write_trial_summary(entry)
        self._append_to_report(entry)

        logger.info(
            f"HYPEROPT_LOG: Trial {trial_number} [{group}] - "
            f"test_f1={entry.test_f1:.4f}, mode_collapse={entry.mode_collapse_severity}, "
            f"status={entry.status}"
        )

        return entry

    def _analyze_trend(self, values: List[float]) -> str:
        """Analyze the trend of a time series."""
        if len(values) < 2:
            return "unknown"

        # Simple trend analysis
        diffs = [values[i+1] - values[i] for i in range(len(values)-1)]

        increasing = sum(1 for d in diffs if d > 0.01)
        decreasing = sum(1 for d in diffs if d < -0.01)

        if decreasing > increasing * 1.5:
            return "decreasing"
        elif increasing > decreasing * 1.5:
            return "increasing"
        elif max(values) - min(values) > np.mean(values) * 0.1:
            return "oscillating"
        else:
            return "stable"

    def _write_jsonl(self, entry: TrialLogEntry):
        """Write entry to JSON Lines file."""
        with open(self.jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(entry), default=str) + '\n')

    def _write_trial_summary(self, entry: TrialLogEntry):
        """Write individual trial summary."""
        summary_path = self.output_dir / "trial_summaries" / f"trial_{entry.trial_number:04d}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(entry), f, indent=2, default=str)

    def _append_to_report(self, entry: TrialLogEntry):
        """Append trial summary to markdown report."""
        with open(self.report_path, 'a', encoding='utf-8') as f:
            # Status emoji
            if entry.mode_collapse_severity == "total":
                status_emoji = "🔴"
            elif entry.mode_collapse_severity == "severe":
                status_emoji = "🟠"
            elif entry.mode_collapse_severity == "mild":
                status_emoji = "🟡"
            elif entry.status == "completed":
                status_emoji = "🟢"
            else:
                status_emoji = "⚪"

            f.write(f"### Trial {entry.trial_number} {status_emoji}\n\n")
            f.write(f"**Group:** {entry.group} | **Duration:** {entry.duration_seconds:.1f}s\n\n")

            # Parameters
            f.write(f"**Parameters:**\n```json\n{json.dumps(entry.params, indent=2, default=str)}\n```\n\n")

            # Metrics table
            f.write("| Metric | Validation | Test |\n")
            f.write("|--------|------------|------|\n")
            f.write(f"| Loss | {entry.best_val_loss:.4f} | - |\n")
            f.write(f"| F1 | {entry.best_val_f1:.4f} | {entry.test_f1:.4f} |\n")
            f.write(f"| Accuracy | - | {entry.test_accuracy:.4f} |\n")
            f.write(f"| Precision | - | {entry.test_precision:.4f} |\n")
            f.write(f"| Recall | - | {entry.test_recall:.4f} |\n\n")

            # Mode collapse warning
            if entry.mode_collapse_detected:
                f.write(f"⚠️ **MODE COLLAPSE DETECTED** ({entry.mode_collapse_severity})\n")
                f.write(f"- Majority class: {entry.majority_class_percentage:.1%}\n")
                f.write(f"- Prediction distribution: {entry.prediction_distribution}\n\n")

            # Per-class F1
            f.write("**Per-class F1:**\n")
            for cls, f1 in entry.per_class_f1.items():
                bar = "█" * int(f1 * 20) + "░" * (20 - int(f1 * 20))
                f.write(f"- {cls}: {bar} {f1:.4f}\n")
            f.write("\n")

            # Confusion matrix
            f.write("**Confusion Matrix:**\n```\n")
            f.write("           SELL  HOLD   BUY\n")
            for i, row in enumerate(entry.confusion_matrix):
                f.write(f"{['SELL', 'HOLD', 'BUY'][i]:>6}  {row[0]:>5}  {row[1]:>5}  {row[2]:>5}\n")
            f.write("```\n\n")

            f.write("---\n\n")

    def log_group_end(self, group: str, best_params: Dict[str, Any], best_value: float, total_trials: int):
        """Log the end of a parameter group optimization."""
        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write(f"### Group {group} Summary\n\n")
            f.write(f"- **Total trials:** {total_trials}\n")
            f.write(f"- **Best value:** {best_value:.4f}\n")
            f.write(f"- **Best parameters:**\n```json\n{json.dumps(best_params, indent=2, default=str)}\n```\n\n")
            f.write("---\n\n")

    def log_final_summary(self, results: Dict[str, Any]):
        """Log final optimization summary."""
        session_end = datetime.now()
        duration = (session_end - self.session_start).total_seconds()

        with open(self.report_path, 'a', encoding='utf-8') as f:
            f.write("# Final Summary\n\n")
            f.write(f"**Completed:** {session_end.isoformat()}\n")
            f.write(f"**Total duration:** {duration/3600:.2f} hours\n\n")

            f.write("## Statistics\n\n")
            f.write(f"- Total trials: {self.total_trials}\n")
            f.write(f"- Completed trials: {self.completed_trials}\n")
            f.write(f"- Mode collapse trials: {self.mode_collapse_trials} ({self.mode_collapse_trials/max(1,self.total_trials)*100:.1f}%)\n\n")

            f.write("## Best Parameters\n\n")
            f.write("```json\n")
            f.write(json.dumps(results.get('best_params', {}), indent=2, default=str))
            f.write("\n```\n\n")

            f.write(f"**Best metric value:** {results.get('best_value', 0):.4f}\n\n")

            if self.best_trial:
                f.write("## Best Trial Details\n\n")
                f.write(f"- Trial number: {self.best_trial.trial_number}\n")
                f.write(f"- Group: {self.best_trial.group}\n")
                f.write(f"- Test F1: {self.best_trial.test_f1:.4f}\n")
                f.write(f"- Test Accuracy: {self.best_trial.test_accuracy:.4f}\n")

        # Save analysis data for AI consumption
        analysis_data = {
            "session_id": self.session_id,
            "session_start": self.session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "duration_hours": duration / 3600,
            "total_trials": self.total_trials,
            "completed_trials": self.completed_trials,
            "mode_collapse_trials": self.mode_collapse_trials,
            "mode_collapse_rate": self.mode_collapse_trials / max(1, self.total_trials),
            "best_params": results.get('best_params', {}),
            "best_value": results.get('best_value', 0),
            "best_trial": asdict(self.best_trial) if self.best_trial else None,
            "group_results": results.get('group_results', {})
        }

        with open(self.analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, default=str)

        logger.info(f"HYPEROPT_LOG: Final summary saved. Mode collapse rate: {analysis_data['mode_collapse_rate']:.1%}")


# Global instance
_hyperopt_logger: Optional[HyperoptLogger] = None


def get_hyperopt_logger(output_dir: str = "data/hyperopt") -> HyperoptLogger:
    """Get or create the global hyperopt logger instance."""
    global _hyperopt_logger
    if _hyperopt_logger is None:
        _hyperopt_logger = HyperoptLogger(output_dir)
    return _hyperopt_logger


def reset_hyperopt_logger():
    """Reset the global hyperopt logger (for new sessions)."""
    global _hyperopt_logger
    _hyperopt_logger = None
