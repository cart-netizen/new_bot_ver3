"""
Comprehensive tests for Professional Volume Distribution Module.

Tests cover:
- Basic volume distribution accuracy
- Edge cases (doji candles, extreme ranges)
- Weight ratio validation
- Gaussian weighting
- Multi-candle distribution to bins
- Performance benchmarks

Path: backend/tests/test_volume_distributor.py
"""

import pytest
import numpy as np
from datetime import datetime
from typing import List

# Import from project
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from strategies.volume_distributor import (
    VolumeDistributor,
    VolumeDistributionConfig,
    create_distributor
)
from strategy.candle_manager import Candle


class TestVolumeDistributionConfig:
    """Test configuration validation."""

    def test_valid_config(self):
        """Test creating valid configuration."""
        config = VolumeDistributionConfig(
            price_points=30,
            body_weight_ratio=0.70,
            upper_wick_ratio=0.15,
            lower_wick_ratio=0.15,
            gaussian_sigma_factor=1.5
        )
        assert config.price_points == 30
        assert config.body_weight_ratio == 0.70

    def test_invalid_weight_sum(self):
        """Test that invalid weight ratios raise ValueError."""
        with pytest.raises(ValueError, match="Weight ratios must sum to 1.0"):
            VolumeDistributionConfig(
                price_points=30,
                body_weight_ratio=0.80,  # Sum > 1.0
                upper_wick_ratio=0.15,
                lower_wick_ratio=0.15
            )

    def test_invalid_price_points(self):
        """Test that too few price points raises ValueError."""
        with pytest.raises(ValueError, match="price_points must be >= 10"):
            VolumeDistributionConfig(
                price_points=5,  # Too low
                body_weight_ratio=0.70,
                upper_wick_ratio=0.15,
                lower_wick_ratio=0.15
            )

    def test_invalid_gaussian_sigma(self):
        """Test that invalid gaussian sigma raises ValueError."""
        with pytest.raises(ValueError, match="gaussian_sigma_factor must be > 0"):
            VolumeDistributionConfig(
                price_points=30,
                body_weight_ratio=0.70,
                upper_wick_ratio=0.15,
                lower_wick_ratio=0.15,
                gaussian_sigma_factor=-1.0  # Invalid
            )


class TestVolumeDistributor:
    """Test volume distributor functionality."""

    @pytest.fixture
    def distributor(self):
        """Create standard distributor for tests."""
        return create_distributor(price_points=30)

    @pytest.fixture
    def bullish_candle(self) -> Candle:
        """Create bullish test candle."""
        return Candle(
            timestamp=int(datetime.now().timestamp() * 1000),
            open=50000.0,
            high=50500.0,
            low=49500.0,
            close=50200.0,
            volume=1000.0
        )

    @pytest.fixture
    def bearish_candle(self) -> Candle:
        """Create bearish test candle."""
        return Candle(
            timestamp=int(datetime.now().timestamp() * 1000),
            open=50200.0,
            high=50500.0,
            low=49500.0,
            close=50000.0,
            volume=1000.0
        )

    @pytest.fixture
    def doji_candle(self) -> Candle:
        """Create doji test candle (open == close)."""
        return Candle(
            timestamp=int(datetime.now().timestamp() * 1000),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50000.0,  # Same as open
            volume=500.0
        )

    def test_distribute_bullish_candle(self, distributor, bullish_candle):
        """Test volume distribution for bullish candle."""
        result = distributor.distribute_candle_volume(bullish_candle)

        # Check basic properties
        assert len(result) > 0, "Distribution should return price points"

        # Verify total volume conservation
        total_volume = sum(vol for _, vol in result)
        assert abs(total_volume - bullish_candle.volume) < 1e-6, \
            f"Volume not conserved: expected {bullish_candle.volume}, got {total_volume}"

        # Verify all prices within candle range
        for price, volume in result:
            assert bullish_candle.low <= price <= bullish_candle.high, \
                f"Price {price} outside candle range [{bullish_candle.low}, {bullish_candle.high}]"
            assert volume >= 0, f"Negative volume: {volume}"

    def test_distribute_bearish_candle(self, distributor, bearish_candle):
        """Test volume distribution for bearish candle."""
        result = distributor.distribute_candle_volume(bearish_candle)

        total_volume = sum(vol for _, vol in result)
        assert abs(total_volume - bearish_candle.volume) < 1e-6

        # Bearish candles should still distribute correctly
        assert len(result) > 0

    def test_distribute_doji_candle(self, distributor, doji_candle):
        """Test volume distribution for doji candle (minimal body)."""
        result = distributor.distribute_candle_volume(doji_candle)

        # Doji might return single point or normal distribution
        assert len(result) > 0

        total_volume = sum(vol for _, vol in result)
        assert abs(total_volume - doji_candle.volume) < 1e-6

    def test_body_concentration(self, distributor, bullish_candle):
        """Test that more volume is allocated to candle body."""
        result = distributor.distribute_candle_volume(bullish_candle)

        # Body boundaries
        body_low = min(bullish_candle.open, bullish_candle.close)
        body_high = max(bullish_candle.open, bullish_candle.close)

        # Calculate volume in body vs wicks
        body_volume = sum(vol for price, vol in result if body_low <= price <= body_high)
        total_volume = sum(vol for _, vol in result)

        body_ratio = body_volume / total_volume if total_volume > 0 else 0

        # Should be approximately 70% (with some tolerance for gaussian weighting)
        assert 0.60 <= body_ratio <= 0.80, \
            f"Body volume ratio {body_ratio:.2%} not in expected range [60%, 80%]"

    def test_close_price_concentration(self, distributor, bullish_candle):
        """Test that volume concentrates near close price (Gaussian weighting)."""
        result = distributor.distribute_candle_volume(bullish_candle)

        # Find volume at points nearest to close
        sorted_by_distance = sorted(
            result,
            key=lambda x: abs(x[0] - bullish_candle.close)
        )

        # Top 10% of points nearest to close should have significant volume
        top_10_pct_count = max(1, len(result) // 10)
        near_close_volume = sum(vol for _, vol in sorted_by_distance[:top_10_pct_count])
        total_volume = sum(vol for _, vol in result)

        concentration_ratio = near_close_volume / total_volume if total_volume > 0 else 0

        # Should have at least 15% of volume in top 10% of nearest points
        assert concentration_ratio >= 0.15, \
            f"Close price concentration {concentration_ratio:.2%} too low"

    def test_distribute_to_bins(self, distributor, bullish_candle):
        """Test distributing multiple candles to fixed bins."""
        candles = [bullish_candle] * 10  # 10 identical candles

        bins = 50
        min_price = 49000.0
        max_price = 51000.0

        result = distributor.distribute_candles_to_bins(
            candles=candles,
            price_bins=bins,
            min_price=min_price,
            max_price=max_price
        )

        # Check shape
        assert result.shape == (bins,), f"Expected shape ({bins},), got {result.shape}"

        # Check volume conservation
        expected_total = sum(c.volume for c in candles)
        actual_total = np.sum(result)

        # Should be close (some volume might be outside range)
        assert actual_total <= expected_total
        assert actual_total >= expected_total * 0.95  # At least 95% should be in range

    def test_statistics(self, distributor, bullish_candle):
        """Test statistics tracking."""
        initial_stats = distributor.get_statistics()
        assert initial_stats['candles_processed'] == 0

        # Process a candle
        distributor.distribute_candle_volume(bullish_candle)

        stats = distributor.get_statistics()
        assert stats['candles_processed'] == 1
        assert stats['total_volume_distributed'] == bullish_candle.volume

    def test_multiple_candles_distribution(self, distributor):
        """Test distributing multiple different candles."""
        candles = []
        for i in range(20):
            candle = Candle(
                timestamp=int(datetime.now().timestamp() * 1000) + i * 60000,
                open=50000.0 + i * 10,
                high=50500.0 + i * 10,
                low=49500.0 + i * 10,
                close=50200.0 + i * 10,
                volume=1000.0 + i * 50
            )
            candles.append(candle)

        result = distributor.distribute_candles_to_bins(
            candles=candles,
            price_bins=100,
            min_price=49000.0,
            max_price=51000.0
        )

        total_input = sum(c.volume for c in candles)
        total_output = np.sum(result)

        # Volume should be conserved (within 1%)
        assert abs(total_output - total_input) / total_input < 0.01


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_candles_list(self):
        """Test that empty candle list raises error."""
        distributor = create_distributor()

        with pytest.raises(ValueError, match="Cannot distribute empty candle list"):
            distributor.distribute_candles_to_bins(
                candles=[],
                price_bins=50,
                min_price=50000.0,
                max_price=51000.0
            )

    def test_invalid_price_range(self):
        """Test that invalid price range raises error."""
        distributor = create_distributor()
        candle = Candle(
            timestamp=int(datetime.now().timestamp() * 1000),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,
            volume=1000.0
        )

        with pytest.raises(ValueError, match="min_price must be < max_price"):
            distributor.distribute_candles_to_bins(
                candles=[candle],
                price_bins=50,
                min_price=51000.0,  # Invalid: min > max
                max_price=50000.0
            )

    def test_extreme_volatility_candle(self):
        """Test candle with extreme price range."""
        distributor = create_distributor()

        extreme_candle = Candle(
            timestamp=int(datetime.now().timestamp() * 1000),
            open=50000.0,
            high=60000.0,  # 20% range
            low=40000.0,
            close=50000.0,
            volume=10000.0
        )

        result = distributor.distribute_candle_volume(extreme_candle)

        # Should still conserve volume
        total = sum(vol for _, vol in result)
        assert abs(total - extreme_candle.volume) < 1e-6


class TestPerformance:
    """Performance benchmarks."""

    def test_large_candle_set_performance(self):
        """Test performance with large number of candles."""
        import time

        distributor = create_distributor()

        # Create 1000 candles
        candles = []
        for i in range(1000):
            candle = Candle(
                timestamp=int(datetime.now().timestamp() * 1000) + i * 60000,
                open=50000.0 + np.random.uniform(-100, 100),
                high=50000.0 + np.random.uniform(50, 200),
                low=50000.0 + np.random.uniform(-200, -50),
                close=50000.0 + np.random.uniform(-100, 100),
                volume=1000.0 + np.random.uniform(-200, 200)
            )
            candles.append(candle)

        start = time.time()
        result = distributor.distribute_candles_to_bins(
            candles=candles,
            price_bins=100,
            min_price=48000.0,
            max_price=52000.0
        )
        elapsed = time.time() - start

        print(f"\nProcessed 1000 candles in {elapsed:.3f} seconds")
        print(f"Average: {elapsed/1000*1000:.3f} ms per candle")

        # Should complete in reasonable time (< 5 seconds for 1000 candles)
        assert elapsed < 5.0, f"Too slow: {elapsed:.3f}s for 1000 candles"


class TestFactoryFunction:
    """Test factory function."""

    def test_create_distributor(self):
        """Test factory function creates valid distributor."""
        dist = create_distributor(
            price_points=50,
            body_weight_ratio=0.75,
            gaussian_sigma_factor=2.0
        )

        assert dist.config.price_points == 50
        assert dist.config.body_weight_ratio == 0.75
        assert dist.config.gaussian_sigma_factor == 2.0

        # Wick ratios should be calculated automatically
        expected_wick = (1.0 - 0.75) / 2.0
        assert abs(dist.config.upper_wick_ratio - expected_wick) < 1e-6
        assert abs(dist.config.lower_wick_ratio - expected_wick) < 1e-6


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
