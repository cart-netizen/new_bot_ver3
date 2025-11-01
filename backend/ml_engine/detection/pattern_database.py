"""
Historical Pattern Database for Layering Detection.

Professional PostgreSQL-based database system for storing, matching, and learning from
detected layering patterns over time.

Features:
1. PostgreSQL + SQLAlchemy async storage for persistence
2. Pattern matching using feature similarity
3. Blacklist management for known manipulators
4. Success rate tracking
5. Pattern evolution analysis
6. Automatic confidence boosting for known patterns

Path: backend/ml_engine/detection/pattern_database.py
"""

import hashlib
import json
import asyncio
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

from sqlalchemy import select, update, delete, func, and_
from sqlalchemy.exc import IntegrityError

from core.logger import get_logger
from database.connection import db_manager
from database.models import LayeringPattern

logger = get_logger(__name__)


def _run_async(coro):
  """Helper to run async functions from sync context."""
  try:
    loop = asyncio.get_event_loop()
    if loop.is_running():
      # If loop is running, create a new task
      import nest_asyncio
      nest_asyncio.apply()
      return loop.run_until_complete(coro)
    else:
      return loop.run_until_complete(coro)
  except RuntimeError:
    # No event loop, create new one
    return asyncio.run(coro)


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
  first_seen: datetime
  last_seen: datetime
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
  Professional PostgreSQL database for historical layering patterns.

  Features:
  - Pattern storage and retrieval via SQLAlchemy async
  - Similarity matching
  - Blacklist management
  - Statistics and analytics
  - Automatic learning from new patterns
  """

  def __init__(self):
    """Initialize pattern database."""
    # In-memory cache for fast lookup
    self._pattern_cache: Dict[str, HistoricalPattern] = {}
    self._cache_loaded = False

    logger.info(f"✅ HistoricalPatternDatabase initialized (PostgreSQL)")

  async def initialize(self):
    """Load patterns into cache asynchronously."""
    if self._cache_loaded:
      return

    await self._load_cache()
    self._cache_loaded = True
    logger.info(f"   Loaded {len(self._pattern_cache)} patterns from PostgreSQL")

  async def _load_cache(self):
    """Load all patterns into memory cache."""
    async with db_manager.session() as session:
      result = await session.execute(select(LayeringPattern))
      patterns = result.scalars().all()

      for pattern_model in patterns:
        pattern = self._model_to_pattern(pattern_model)
        self._pattern_cache[pattern.pattern_id] = pattern

  async def save_pattern(
      self,
      pattern_features: Dict,
      symbol: str,
      confidence: float,
      success_rate: float = 0.0,
      price_impact_bps: float = 0.0,
      notes: str = ""
  ) -> str:
    """
    Save new pattern or update existing.

    Args:
        pattern_features: Dict with pattern metrics
        symbol: Trading symbol
        confidence: Detector confidence
        success_rate: Manipulation success rate
        price_impact_bps: Price impact in bps
        notes: Additional notes

    Returns:
        pattern_id: ID of saved/updated pattern
    """
    # Create fingerprint
    fingerprint = self._create_fingerprint(pattern_features)

    # Generate pattern ID from fingerprint
    pattern_id = self._generate_pattern_id(fingerprint)

    async with db_manager.session() as session:
      # Check if pattern exists
      result = await session.execute(
        select(LayeringPattern).where(LayeringPattern.pattern_id == pattern_id)
      )
      existing = result.scalar_one_or_none()

      now = datetime.utcnow()

      if existing:
        # Update existing pattern
        existing.last_seen = now
        existing.occurrence_count += 1

        # Update moving averages
        n = existing.occurrence_count
        existing.avg_confidence = (existing.avg_confidence * (n - 1) + confidence) / n
        existing.success_rate = (existing.success_rate * (n - 1) + success_rate) / n
        existing.avg_price_impact_bps = (
          existing.avg_price_impact_bps * (n - 1) + price_impact_bps
        ) / n

        # Add symbol if new
        symbols_list = existing.symbols
        if symbol not in symbols_list:
          symbols_list.append(symbol)
          existing.symbols = symbols_list

        # Update risk level
        existing.risk_level = self._calculate_risk_level(
          existing.success_rate,
          existing.avg_price_impact_bps,
          existing.occurrence_count
        )

        logger.info(
          f"📝 Updated pattern {pattern_id[:8]}... "
          f"(occurrences: {existing.occurrence_count})"
        )

      else:
        # Create new pattern
        new_pattern = LayeringPattern(
          pattern_id=pattern_id,
          first_seen=now,
          last_seen=now,
          occurrence_count=1,
          # Fingerprint
          avg_layer_count=fingerprint.avg_layer_count,
          avg_cancellation_rate=fingerprint.avg_cancellation_rate,
          avg_volume_btc=fingerprint.avg_volume_btc,
          avg_placement_duration=fingerprint.avg_placement_duration,
          typical_spread_pct=fingerprint.typical_spread_pct,
          typical_order_count=fingerprint.typical_order_count,
          spoofing_execution_ratio=fingerprint.spoofing_execution_ratio,
          time_of_day_pattern=fingerprint.time_of_day_pattern,
          avg_lifetime_seconds=fingerprint.avg_lifetime_seconds,
          fingerprint_hash=fingerprint.fingerprint_hash,
          # Metadata
          symbols=[symbol],
          success_rate=success_rate,
          avg_price_impact_bps=price_impact_bps,
          avg_confidence=confidence,
          risk_level=self._calculate_risk_level(success_rate, price_impact_bps, 1),
          blacklist=False,
          notes=notes
        )

        session.add(new_pattern)
        logger.info(f"✨ Created new pattern {pattern_id[:8]}...")

      await session.commit()

      # Update cache
      await self._load_cache()

      return pattern_id

  async def find_similar_pattern(
      self,
      pattern_features: Dict,
      similarity_threshold: float = 0.75
  ) -> Optional[Tuple[HistoricalPattern, float]]:
    """
    Find most similar historical pattern.

    Args:
        pattern_features: Current pattern features
        similarity_threshold: Minimum similarity (0-1)

    Returns:
        (pattern, similarity_score) or None
    """
    if not self._cache_loaded:
      await self.initialize()

    if not self._pattern_cache:
      return None

    query_fingerprint = self._create_fingerprint(pattern_features)

    best_match = None
    best_similarity = 0.0

    for pattern in self._pattern_cache.values():
      similarity = self._calculate_similarity(
        query_fingerprint,
        pattern.fingerprint
      )

      if similarity > best_similarity:
        best_similarity = similarity
        best_match = pattern

    if best_similarity >= similarity_threshold:
      return (best_match, best_similarity)

    return None

  async def get_all_patterns(
      self,
      limit: int = 100,
      sort_by: str = "occurrence_count",
      blacklist_only: bool = False
  ) -> List[Dict]:
    """
    Get all patterns with filters.

    Args:
        limit: Max number of patterns
        sort_by: Sort field (occurrence_count, last_seen, success_rate)
        blacklist_only: Show only blacklisted patterns

    Returns:
        List of pattern dicts
    """
    async with db_manager.session() as session:
      query = select(LayeringPattern)

      if blacklist_only:
        query = query.where(LayeringPattern.blacklist == True)

      # Sorting
      if sort_by == "occurrence_count":
        query = query.order_by(LayeringPattern.occurrence_count.desc())
      elif sort_by == "last_seen":
        query = query.order_by(LayeringPattern.last_seen.desc())
      elif sort_by == "success_rate":
        query = query.order_by(LayeringPattern.success_rate.desc())

      query = query.limit(limit)

      result = await session.execute(query)
      patterns = result.scalars().all()

      return [self._model_to_dict(p) for p in patterns]

  async def get_pattern_by_id(self, pattern_id: str) -> Optional[Dict]:
    """Get pattern by ID."""
    async with db_manager.session() as session:
      result = await session.execute(
        select(LayeringPattern).where(LayeringPattern.pattern_id == pattern_id)
      )
      pattern = result.scalar_one_or_none()

      if pattern:
        return self._model_to_dict(pattern)
      return None

  async def update_blacklist(self, pattern_id: str, blacklist: bool):
    """Update pattern blacklist status."""
    async with db_manager.session() as session:
      await session.execute(
        update(LayeringPattern)
        .where(LayeringPattern.pattern_id == pattern_id)
        .values(blacklist=blacklist)
      )
      await session.commit()

    # Update cache
    await self._load_cache()

    logger.info(
      f"🚫 Pattern {pattern_id[:8]}... "
      f"{'added to' if blacklist else 'removed from'} blacklist"
    )

  async def get_statistics(self) -> Dict:
    """Get database statistics."""
    async with db_manager.session() as session:
      # Total patterns
      total_result = await session.execute(select(func.count(LayeringPattern.id)))
      total_patterns = total_result.scalar()

      # Blacklisted
      blacklist_result = await session.execute(
        select(func.count(LayeringPattern.id))
        .where(LayeringPattern.blacklist == True)
      )
      blacklisted = blacklist_result.scalar()

      # Average success rate
      avg_success_result = await session.execute(
        select(func.avg(LayeringPattern.success_rate))
      )
      avg_success = avg_success_result.scalar() or 0.0

      # Unique symbols
      symbols_result = await session.execute(select(LayeringPattern.symbols))
      all_symbols = set()
      for (symbols_json,) in symbols_result:
        all_symbols.update(symbols_json)

      # Oldest pattern
      oldest_result = await session.execute(
        select(LayeringPattern.first_seen).order_by(LayeringPattern.first_seen.asc()).limit(1)
      )
      oldest = oldest_result.scalar()
      oldest_age_hours = 0
      if oldest:
        oldest_age_hours = (datetime.utcnow() - oldest).total_seconds() / 3600

      return {
        "total_patterns": total_patterns,
        "blacklisted_patterns": blacklisted,
        "unique_symbols": len(all_symbols),
        "avg_success_rate": float(avg_success),
        "oldest_pattern_age_hours": oldest_age_hours
      }

  def _create_fingerprint(self, features: Dict) -> PatternFingerprint:
    """Create pattern fingerprint from features."""
    # Extract features with defaults
    avg_layer_count = features.get("avg_layer_count", 0.0)
    avg_cancellation_rate = features.get("avg_cancellation_rate", 0.0)
    avg_volume_btc = features.get("avg_volume_btc", 0.0)
    avg_placement_duration = features.get("avg_placement_duration", 0.0)
    typical_spread_pct = features.get("typical_spread_pct", 0.0)
    typical_order_count = features.get("typical_order_count", 0)
    spoofing_execution_ratio = features.get("spoofing_execution_ratio")
    time_of_day_pattern = features.get("time_of_day_pattern", [])
    avg_lifetime_seconds = features.get("avg_lifetime_seconds", 0.0)

    # Compute hash
    hash_string = (
      f"{avg_layer_count:.2f}_{avg_cancellation_rate:.3f}_"
      f"{avg_volume_btc:.4f}_{avg_placement_duration:.1f}_"
      f"{typical_spread_pct:.3f}_{typical_order_count}"
    )
    fingerprint_hash = hashlib.sha256(hash_string.encode()).hexdigest()

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

  def _generate_pattern_id(self, fingerprint: PatternFingerprint) -> str:
    """Generate unique pattern ID."""
    return fingerprint.fingerprint_hash[:32]  # First 32 chars

  def _calculate_similarity(
      self,
      fp1: PatternFingerprint,
      fp2: PatternFingerprint
  ) -> float:
    """
    Calculate cosine similarity between fingerprints.

    Returns:
        Similarity score (0-1)
    """
    # Weighted similarity
    diff_layers = abs(fp1.avg_layer_count - fp2.avg_layer_count) / max(fp1.avg_layer_count, fp2.avg_layer_count, 1)
    diff_cancel = abs(fp1.avg_cancellation_rate - fp2.avg_cancellation_rate)
    diff_volume = abs(fp1.avg_volume_btc - fp2.avg_volume_btc) / max(fp1.avg_volume_btc, fp2.avg_volume_btc, 0.01)
    diff_duration = abs(fp1.avg_placement_duration - fp2.avg_placement_duration) / max(
      fp1.avg_placement_duration, fp2.avg_placement_duration, 1
    )
    diff_spread = abs(fp1.typical_spread_pct - fp2.typical_spread_pct) / max(
      fp1.typical_spread_pct, fp2.typical_spread_pct, 0.01
    )

    # Weighted similarity score
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
      success_rate: float,
      price_impact_bps: float,
      occurrence_count: int
  ) -> str:
    """Calculate risk level based on pattern characteristics."""
    score = 0

    # Success rate
    if success_rate >= 0.7:
      score += 3
    elif success_rate >= 0.5:
      score += 2
    elif success_rate >= 0.3:
      score += 1

    # Price impact
    if price_impact_bps >= 10.0:
      score += 3
    elif price_impact_bps >= 5.0:
      score += 2
    elif price_impact_bps >= 2.0:
      score += 1

    # Frequency
    if occurrence_count >= 50:
      score += 2
    elif occurrence_count >= 10:
      score += 1

    # Risk level
    if score >= 7:
      return "CRITICAL"
    elif score >= 5:
      return "HIGH"
    elif score >= 3:
      return "MEDIUM"
    else:
      return "LOW"

  def _model_to_pattern(self, model: LayeringPattern) -> HistoricalPattern:
    """Convert SQLAlchemy model to HistoricalPattern."""
    fingerprint = PatternFingerprint(
      avg_layer_count=model.avg_layer_count,
      avg_cancellation_rate=model.avg_cancellation_rate,
      avg_volume_btc=model.avg_volume_btc,
      avg_placement_duration=model.avg_placement_duration,
      typical_spread_pct=model.typical_spread_pct,
      typical_order_count=model.typical_order_count,
      spoofing_execution_ratio=model.spoofing_execution_ratio,
      time_of_day_pattern=model.time_of_day_pattern or [],
      avg_lifetime_seconds=model.avg_lifetime_seconds,
      fingerprint_hash=model.fingerprint_hash
    )

    return HistoricalPattern(
      pattern_id=model.pattern_id,
      first_seen=model.first_seen,
      last_seen=model.last_seen,
      occurrence_count=model.occurrence_count,
      fingerprint=fingerprint,
      symbols=model.symbols,
      success_rate=model.success_rate,
      avg_price_impact_bps=model.avg_price_impact_bps,
      avg_confidence=model.avg_confidence,
      risk_level=model.risk_level,
      blacklist=model.blacklist,
      notes=model.notes or ""
    )

  def _model_to_dict(self, model: LayeringPattern) -> Dict:
    """Convert SQLAlchemy model to dict."""
    return {
      "pattern_id": model.pattern_id,
      "first_seen": int(model.first_seen.timestamp() * 1000),
      "last_seen": int(model.last_seen.timestamp() * 1000),
      "occurrence_count": model.occurrence_count,
      "avg_layer_count": model.avg_layer_count,
      "avg_cancellation_rate": model.avg_cancellation_rate,
      "avg_volume_btc": model.avg_volume_btc,
      "symbols": model.symbols,
      "success_rate": model.success_rate,
      "risk_level": model.risk_level,
      "blacklist": model.blacklist
    }

  # ===== SYNC WRAPPERS for compatibility with sync code =====

  def save_pattern_sync(self, *args, **kwargs) -> str:
    """Sync wrapper for save_pattern."""
    return _run_async(self.save_pattern(*args, **kwargs))

  def find_similar_pattern_sync(self, *args, **kwargs) -> Optional[Tuple[HistoricalPattern, float]]:
    """Sync wrapper for find_similar_pattern."""
    return _run_async(self.find_similar_pattern(*args, **kwargs))

  def get_pattern_by_id_sync(self, *args, **kwargs) -> Optional[Dict]:
    """Sync wrapper for get_pattern_by_id."""
    return _run_async(self.get_pattern_by_id(*args, **kwargs))

  def get_statistics_sync(self) -> Dict:
    """Sync wrapper for get_statistics."""
    return _run_async(self.get_statistics())
