"""
Historical Pattern Database for Layering Detection.

Professional database system for storing, matching, and learning from
detected layering patterns over time.

Features:
1. SQLite storage for persistence
2. Pattern matching using feature similarity
3. Blacklist management for known manipulators
4. Success rate tracking
5. Pattern evolution analysis
6. Automatic confidence boosting for known patterns

Path: backend/ml_engine/detection/pattern_database.py
"""

import sqlite3
import hashlib
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from pathlib import Path

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PatternFingerprint:
  """Behavioral fingerprint of a layering pattern."""
  # Core features
  avg_layer_count: float
  avg_cancellation_rate: float
  avg_volume_btc: float
  avg_placement_duration: float

  # Advanced features
  typical_spread_pct: float
  typical_order_count: int
  spoofing_execution_ratio: Optional[float]

  # Temporal patterns
  time_of_day_pattern: List[int]  # Active hours UTC (0-23)
  avg_lifetime_seconds: float

  # Computed hash for fast matching
  fingerprint_hash: str


@dataclass
class HistoricalPattern:
  """Historical pattern record."""
  pattern_id: str
  first_seen: int  # timestamp ms
  last_seen: int
  occurrence_count: int

  # Fingerprint
  fingerprint: PatternFingerprint

  # Metadata
  symbols: List[str]
  success_rate: float  # How often manipulation succeeded
  avg_price_impact_bps: float
  avg_confidence: float

  # Risk assessment
  risk_level: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
  blacklist: bool

  # Notes
  notes: str


class HistoricalPatternDatabase:
  """
  Professional database for historical layering patterns.

  Features:
  - Pattern storage and retrieval
  - Similarity matching
  - Blacklist management
  - Statistics and analytics
  - Automatic learning from new patterns
  """

  def __init__(self, db_path: str = "data/layering_patterns.db"):
    """
    Initialize pattern database.

    Args:
        db_path: Path to SQLite database file
    """
    self.db_path = db_path

    # Ensure data directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Initialize database
    self._init_database()

    # In-memory cache for fast lookup
    self._pattern_cache: Dict[str, HistoricalPattern] = {}
    self._load_cache()

    logger.info(f"✅ HistoricalPatternDatabase initialized: {db_path}")
    logger.info(f"   Loaded {len(self._pattern_cache)} patterns from database")

  def _init_database(self):
    """Initialize SQLite database schema."""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    # Patterns table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patterns (
            pattern_id TEXT PRIMARY KEY,
            first_seen INTEGER NOT NULL,
            last_seen INTEGER NOT NULL,
            occurrence_count INTEGER DEFAULT 1,

            -- Fingerprint features
            avg_layer_count REAL,
            avg_cancellation_rate REAL,
            avg_volume_btc REAL,
            avg_placement_duration REAL,
            typical_spread_pct REAL,
            typical_order_count INTEGER,
            spoofing_execution_ratio REAL,
            time_of_day_pattern TEXT,
            avg_lifetime_seconds REAL,
            fingerprint_hash TEXT,

            -- Metadata
            symbols TEXT,
            success_rate REAL DEFAULT 0.0,
            avg_price_impact_bps REAL DEFAULT 0.0,
            avg_confidence REAL,

            -- Risk
            risk_level TEXT,
            blacklist INTEGER DEFAULT 0,
            notes TEXT
        )
    """)

    # Indices for fast lookup
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_fingerprint_hash
        ON patterns(fingerprint_hash)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_blacklist
        ON patterns(blacklist)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_last_seen
        ON patterns(last_seen)
    """)

    # Statistics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pattern_statistics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            total_patterns INTEGER,
            blacklisted_patterns INTEGER,
            total_occurrences INTEGER,
            avg_success_rate REAL
        )
    """)

    conn.commit()
    conn.close()

  def _load_cache(self):
    """Load all patterns into memory cache."""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM patterns")
    rows = cursor.fetchall()

    for row in rows:
      pattern = self._row_to_pattern(row)
      self._pattern_cache[pattern.pattern_id] = pattern

    conn.close()

  def save_pattern(
      self,
      pattern_features: Dict,
      symbol: str,
      confidence: float,
      price_impact_bps: Optional[float] = None
  ) -> str:
    """
    Save detected pattern to database.

    Args:
        pattern_features: Extracted features from LayeringPattern
        symbol: Trading symbol
        confidence: Detection confidence
        price_impact_bps: Price impact if available

    Returns:
        pattern_id: ID of saved/updated pattern
    """
    # Create fingerprint
    fingerprint = self._create_fingerprint(pattern_features)

    # Check if similar pattern exists
    existing_id = self._find_exact_match(fingerprint.fingerprint_hash)

    timestamp = int(datetime.now().timestamp() * 1000)

    if existing_id:
      # Update existing pattern
      self._update_pattern(
        existing_id,
        timestamp,
        symbol,
        confidence,
        price_impact_bps
      )
      pattern_id = existing_id
    else:
      # Insert new pattern
      pattern_id = self._insert_pattern(
        fingerprint,
        timestamp,
        symbol,
        confidence,
        price_impact_bps
      )

    # Update cache
    self._reload_pattern(pattern_id)

    return pattern_id

  def _create_fingerprint(self, features: Dict) -> PatternFingerprint:
    """Create behavioral fingerprint from features."""
    # Extract features
    avg_layer_count = features.get('avg_layer_count', 0.0)
    avg_cancellation_rate = features.get('cancellation_rate', 0.0)
    avg_volume_btc = features.get('total_volume', 0.0)
    avg_placement_duration = features.get('placement_duration', 0.0)
    typical_spread_pct = features.get('avg_spread_pct', 0.0)
    typical_order_count = features.get('total_orders', 0)
    spoofing_execution_ratio = features.get('spoofing_execution_ratio')
    avg_lifetime_seconds = features.get('avg_lifetime_seconds', 0.0)

    # Time of day pattern (hour UTC)
    current_hour = datetime.utcnow().hour
    time_of_day_pattern = [current_hour]

    # Create hash for fast matching
    hash_input = (
      f"{avg_layer_count:.1f}_{avg_cancellation_rate:.2f}_"
      f"{avg_volume_btc:.2f}_{avg_placement_duration:.1f}_"
      f"{typical_spread_pct:.3f}_{typical_order_count}"
    )
    fingerprint_hash = hashlib.md5(hash_input.encode()).hexdigest()[:16]

    return PatternFingerprint(
      avg_layer_count=avg_layer_count,
      avg_cancellation_rate=avg_cancellation_rate,
      avg_volume_btc=avg_volume_btc,
      avg_placement_duration=avg_placement_duration,
      typical_spread_pct=typical_spread_pct,
      typical_order_count=typical_order_count,
      spoofing_execution_ratio=spoofing_execution_ratio,
      time_of_day_pattern=time_of_day_pattern,
      avg_lifetime_seconds=avg_lifetime_seconds,
      fingerprint_hash=fingerprint_hash
    )

  def _find_exact_match(self, fingerprint_hash: str) -> Optional[str]:
    """Find pattern with exact hash match."""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute(
      "SELECT pattern_id FROM patterns WHERE fingerprint_hash = ?",
      (fingerprint_hash,)
    )

    result = cursor.fetchone()
    conn.close()

    return result[0] if result else None

  def _insert_pattern(
      self,
      fingerprint: PatternFingerprint,
      timestamp: int,
      symbol: str,
      confidence: float,
      price_impact_bps: Optional[float]
  ) -> str:
    """Insert new pattern into database."""
    pattern_id = f"pattern_{fingerprint.fingerprint_hash}"

    # Determine risk level
    risk_level = self._calculate_risk_level(confidence, fingerprint)

    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO patterns VALUES (
            ?, ?, ?, 1,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, 0.0, ?, ?,
            ?, 0, ''
        )
    """, (
      pattern_id,
      timestamp,
      timestamp,
      fingerprint.avg_layer_count,
      fingerprint.avg_cancellation_rate,
      fingerprint.avg_volume_btc,
      fingerprint.avg_placement_duration,
      fingerprint.typical_spread_pct,
      fingerprint.typical_order_count,
      fingerprint.spoofing_execution_ratio,
      json.dumps(fingerprint.time_of_day_pattern),
      fingerprint.avg_lifetime_seconds,
      fingerprint.fingerprint_hash,
      json.dumps([symbol]),
      price_impact_bps or 0.0,
      confidence,
      risk_level
    ))

    conn.commit()
    conn.close()

    logger.info(f"📝 New pattern saved: {pattern_id}, risk={risk_level}")

    return pattern_id

  def _update_pattern(
      self,
      pattern_id: str,
      timestamp: int,
      symbol: str,
      confidence: float,
      price_impact_bps: Optional[float]
  ):
    """Update existing pattern."""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    # Get current data
    cursor.execute(
      "SELECT symbols, occurrence_count, avg_confidence, avg_price_impact_bps FROM patterns WHERE pattern_id = ?",
      (pattern_id,)
    )
    row = cursor.fetchone()

    if not row:
      conn.close()
      return

    current_symbols = json.loads(row[0])
    occurrence_count = row[1]
    avg_confidence = row[2]
    avg_impact = row[3]

    # Update symbols list
    if symbol not in current_symbols:
      current_symbols.append(symbol)

    # Update averages
    new_occurrence = occurrence_count + 1
    new_avg_confidence = (
      (avg_confidence * occurrence_count + confidence) / new_occurrence
    )

    if price_impact_bps:
      new_avg_impact = (
        (avg_impact * occurrence_count + price_impact_bps) / new_occurrence
      )
    else:
      new_avg_impact = avg_impact

    # Update database
    cursor.execute("""
        UPDATE patterns
        SET last_seen = ?,
            occurrence_count = ?,
            symbols = ?,
            avg_confidence = ?,
            avg_price_impact_bps = ?
        WHERE pattern_id = ?
    """, (
      timestamp,
      new_occurrence,
      json.dumps(current_symbols),
      new_avg_confidence,
      new_avg_impact,
      pattern_id
    ))

    conn.commit()
    conn.close()

    logger.info(
      f"🔄 Pattern updated: {pattern_id}, "
      f"occurrences={new_occurrence}, "
      f"avg_confidence={new_avg_confidence:.2f}"
    )

  def find_similar_pattern(
      self,
      pattern_features: Dict,
      similarity_threshold: float = 0.80
  ) -> Optional[Tuple[str, float]]:
    """
    Find most similar pattern in database.

    Args:
        pattern_features: Features to match
        similarity_threshold: Minimum similarity (0.0-1.0)

    Returns:
        Tuple of (pattern_id, similarity) or None
    """
    if not self._pattern_cache:
      return None

    # Create fingerprint for comparison
    query_fingerprint = self._create_fingerprint(pattern_features)

    best_match_id = None
    best_similarity = 0.0

    # Compare with all cached patterns
    for pattern_id, pattern in self._pattern_cache.items():
      similarity = self._calculate_similarity(
        query_fingerprint,
        pattern.fingerprint
      )

      if similarity > best_similarity and similarity >= similarity_threshold:
        best_similarity = similarity
        best_match_id = pattern_id

    if best_match_id:
      return (best_match_id, best_similarity)

    return None

  def _calculate_similarity(
      self,
      fp1: PatternFingerprint,
      fp2: PatternFingerprint
  ) -> float:
    """
    Calculate similarity between two fingerprints.

    Uses weighted feature comparison.

    Returns:
        Similarity score (0.0-1.0)
    """
    # Feature differences (normalized)
    diff_layers = abs(fp1.avg_layer_count - fp2.avg_layer_count) / 10.0
    diff_cancel = abs(fp1.avg_cancellation_rate - fp2.avg_cancellation_rate)
    diff_volume = (
      abs(fp1.avg_volume_btc - fp2.avg_volume_btc) /
      max(fp1.avg_volume_btc, fp2.avg_volume_btc, 0.01)
    )
    diff_duration = abs(fp1.avg_placement_duration - fp2.avg_placement_duration) / 30.0
    diff_spread = abs(fp1.typical_spread_pct - fp2.typical_spread_pct) / 0.01

    # Weighted similarity (inverted differences)
    similarity = (
      (1.0 - min(diff_layers, 1.0)) * 0.20 +
      (1.0 - min(diff_cancel, 1.0)) * 0.30 +
      (1.0 - min(diff_volume, 1.0)) * 0.25 +
      (1.0 - min(diff_duration, 1.0)) * 0.15 +
      (1.0 - min(diff_spread, 1.0)) * 0.10
    )

    return similarity

  def _calculate_risk_level(
      self,
      confidence: float,
      fingerprint: PatternFingerprint
  ) -> str:
    """Calculate risk level based on pattern characteristics."""
    risk_score = 0

    # High confidence
    if confidence >= 0.90:
      risk_score += 3
    elif confidence >= 0.75:
      risk_score += 2
    elif confidence >= 0.65:
      risk_score += 1

    # High cancellation rate
    if fingerprint.avg_cancellation_rate >= 0.80:
      risk_score += 2
    elif fingerprint.avg_cancellation_rate >= 0.60:
      risk_score += 1

    # Large volume
    if fingerprint.avg_volume_btc >= 10.0:
      risk_score += 2
    elif fingerprint.avg_volume_btc >= 5.0:
      risk_score += 1

    # Fast placement
    if fingerprint.avg_placement_duration < 10.0:
      risk_score += 1

    # Classify risk
    if risk_score >= 7:
      return "CRITICAL"
    elif risk_score >= 5:
      return "HIGH"
    elif risk_score >= 3:
      return "MEDIUM"
    else:
      return "LOW"

  def get_pattern(self, pattern_id: str) -> Optional[HistoricalPattern]:
    """Get pattern by ID from cache."""
    return self._pattern_cache.get(pattern_id)

  def is_blacklisted(self, pattern_id: str) -> bool:
    """Check if pattern is blacklisted."""
    pattern = self.get_pattern(pattern_id)
    return pattern.blacklist if pattern else False

  def update_blacklist(
      self,
      pattern_id: str,
      blacklist: bool,
      reason: str = ""
  ):
    """Update blacklist status for pattern."""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE patterns
        SET blacklist = ?, notes = ?
        WHERE pattern_id = ?
    """, (1 if blacklist else 0, reason, pattern_id))

    conn.commit()
    conn.close()

    # Update cache
    self._reload_pattern(pattern_id)

    logger.info(f"🚫 Pattern blacklist updated: {pattern_id}, blacklist={blacklist}")

  def get_blacklisted_patterns(self) -> List[HistoricalPattern]:
    """Get all blacklisted patterns."""
    return [
      pattern for pattern in self._pattern_cache.values()
      if pattern.blacklist
    ]

  def _reload_pattern(self, pattern_id: str):
    """Reload pattern from database into cache."""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM patterns WHERE pattern_id = ?", (pattern_id,))
    row = cursor.fetchone()

    if row:
      pattern = self._row_to_pattern(row)
      self._pattern_cache[pattern_id] = pattern

    conn.close()

  def _row_to_pattern(self, row: Tuple) -> HistoricalPattern:
    """Convert database row to HistoricalPattern."""
    fingerprint = PatternFingerprint(
      avg_layer_count=row[4],
      avg_cancellation_rate=row[5],
      avg_volume_btc=row[6],
      avg_placement_duration=row[7],
      typical_spread_pct=row[8],
      typical_order_count=row[9],
      spoofing_execution_ratio=row[10],
      time_of_day_pattern=json.loads(row[11]) if row[11] else [],
      avg_lifetime_seconds=row[12],
      fingerprint_hash=row[13]
    )

    return HistoricalPattern(
      pattern_id=row[0],
      first_seen=row[1],
      last_seen=row[2],
      occurrence_count=row[3],
      fingerprint=fingerprint,
      symbols=json.loads(row[14]) if row[14] else [],
      success_rate=row[15],
      avg_price_impact_bps=row[16],
      avg_confidence=row[17],
      risk_level=row[18],
      blacklist=bool(row[19]),
      notes=row[20] if row[20] else ""
    )

  def get_statistics(self) -> Dict:
    """Get database statistics."""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    # Total patterns
    cursor.execute("SELECT COUNT(*) FROM patterns")
    total_patterns = cursor.fetchone()[0]

    # Blacklisted
    cursor.execute("SELECT COUNT(*) FROM patterns WHERE blacklist = 1")
    blacklisted = cursor.fetchone()[0]

    # Total occurrences
    cursor.execute("SELECT SUM(occurrence_count) FROM patterns")
    total_occurrences = cursor.fetchone()[0] or 0

    # Average confidence
    cursor.execute("SELECT AVG(avg_confidence) FROM patterns")
    avg_confidence = cursor.fetchone()[0] or 0.0

    # Risk distribution
    cursor.execute("SELECT risk_level, COUNT(*) FROM patterns GROUP BY risk_level")
    risk_dist = dict(cursor.fetchall())

    conn.close()

    return {
      'total_patterns': total_patterns,
      'blacklisted_patterns': blacklisted,
      'total_occurrences': total_occurrences,
      'avg_confidence': avg_confidence,
      'risk_distribution': risk_dist,
      'cache_size': len(self._pattern_cache)
    }

  def cleanup_old_patterns(self, days: int = 90):
    """Remove patterns older than specified days."""
    cutoff_time = int((datetime.now().timestamp() - (days * 86400)) * 1000)

    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()

    # Delete old patterns (except blacklisted)
    cursor.execute("""
        DELETE FROM patterns
        WHERE last_seen < ? AND blacklist = 0
    """, (cutoff_time,))

    deleted = cursor.rowcount
    conn.commit()
    conn.close()

    # Reload cache
    self._load_cache()

    logger.info(f"🧹 Cleaned up {deleted} old patterns (older than {days} days)")


# Example usage
if __name__ == "__main__":
  db = HistoricalPatternDatabase("test_patterns.db")

  # Test save
  pattern_features = {
    'avg_layer_count': 3.5,
    'cancellation_rate': 0.75,
    'total_volume': 5.2,
    'placement_duration': 12.5,
    'avg_spread_pct': 0.005,
    'total_orders': 15
  }

  pattern_id = db.save_pattern(pattern_features, "BTCUSDT", 0.85, 8.5)
  print(f"Saved pattern: {pattern_id}")

  # Test search
  result = db.find_similar_pattern(pattern_features, 0.80)
  if result:
    print(f"Found similar: {result[0]}, similarity={result[1]:.2f}")

  # Statistics
  stats = db.get_statistics()
  print(f"Statistics: {stats}")
