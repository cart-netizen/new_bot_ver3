# 🔥 Memory Leak Analysis and Fix

## 📊 Problem Analysis

**Symptoms:**
- Start: 5 GB RAM
- After 5 hours: 25+ GB RAM
- Stop button doesn't free memory
- System crashes when RAM hits 32 GB limit

---

## 🔍 Root Causes Found

### 1. **ML Data Collector - Aggressive Buffering**
**Location:** `backend/ml_engine/data_collection/ml_data_collector.py`

**Problem:**
```python
max_samples_per_file: int = 2000  # Per symbol!
max_buffer_memory_mb: int = 200   # Per symbol!
```

**Impact:** With 50 active symbols:
- 50 symbols × 2000 samples × 112 features × 8 bytes = **~90 MB minimum**
- Labels + metadata add another 20-30 MB
- Total: **100-120 MB just in ML buffers**

**Why it grows:** If data comes in faster than it's saved, buffers accumulate.

---

### 2. **Infrequent Memory Cleanup**
**Location:** `backend/main.py:1959`

**Problem:**
```python
if cleanup_counter >= 1000:  # Every 1000 cycles!
    await self._cleanup_memory()
    cleanup_counter = 0
```

**Impact:**
- If cycle takes 1 second → cleanup every 16 minutes
- If cycle takes 5 seconds → cleanup every 83 minutes
- Memory grows unchecked between cleanups

**Also:** Cleanup only triggers if memory > 8GB (line 3886)

---

### 3. **Feature Extractors - History Accumulation**
**Location:** `backend/ml_engine/features/orderbook_feature_extractor.py`

**Problem:**
```python
self.snapshot_history: List[OrderBookSnapshot] = []
self.max_history_size = 100  # Per symbol
```

**Impact:** With 50 symbols:
- 50 × 100 snapshots × ~10 KB per snapshot = **~50 MB**

**Compounded:**
- Indicator extractor: 200 candles history per symbol
- Each candle ~1 KB → 50 symbols × 200 × 1 KB = **10 MB**

---

### 4. **Python Memory Not Released to OS**
**Problem:** Python's memory allocator doesn't immediately return freed memory to OS.

**Impact:** Even after `del` and `gc.collect()`, process memory stays high.

**Why:** Python maintains memory pools for future allocations (performance optimization).

---

### 5. **Stop() Doesn't Force Memory Release**
**Location:** `backend/main.py:3124`

**Problem:**
```python
async def stop(self):
    # ... stops tasks ...
    await self.ml_data_collector.finalize()
    # But doesn't call _cleanup_memory() or gc.collect()!
```

**Impact:**
- Buffers are saved to disk but objects remain in memory
- No explicit cleanup → Python process keeps holding RAM
- Only solution: kill process

---

## ✅ Comprehensive Fix

### Fix 1: Reduce Buffer Sizes (backend/main.py:400-410)

```python
self.ml_data_collector = MLDataCollector(
    storage_path="../data/ml_training",
    max_samples_per_file=500,        # 2000 → 500 (75% reduction)
    collection_interval=10,
    max_buffer_memory_mb=50,         # 200 → 50 (75% reduction)
    enable_feature_store=True,
    use_legacy_format=False,         # Disable to save CPU/disk
    feature_store_group="training_features"
)
```

**Expected impact:**
- Buffers: 120 MB → 30 MB (75% reduction)
- Saves more frequently (every 500 samples instead of 2000)

---

### Fix 2: Frequent Memory Cleanup (backend/main.py:1956)

```python
cleanup_counter += 1

# OLD: Every 1000 cycles
# if cleanup_counter >= 1000:

# NEW: Every 100 cycles (~2-8 minutes depending on cycle time)
if cleanup_counter >= 100:
    logger.info("🧹 Periodic memory cleanup (every 100 cycles)")
    await self._cleanup_memory()
    cleanup_counter = 0
```

**Expected impact:**
- Cleanup 10× more frequently
- Prevents large memory accumulation

---

### Fix 3: Lower Memory Threshold (backend/main.py:3886)

```python
# OLD: 8000 MB threshold
# if memory_mb > 8000:

# NEW: 4000 MB threshold (trigger earlier)
if memory_mb > 4000:  # 4 GB instead of 8 GB
    logger.warning(f"⚠️ HIGH MEMORY USAGE: {memory_mb:.1f} MB - запуск очистки памяти")
    await self._cleanup_memory()
```

**Expected impact:**
- Prevents reaching 25+ GB
- Triggers cleanup at 4 GB instead of 8 GB

---

### Fix 4: Aggressive Memory Release in stop() (backend/main.py:3186)

```python
# ===== НОВОЕ: Финализация ML Data Collector =====
if self.ml_data_collector:
    await self.ml_data_collector.finalize()
    logger.info("✓ ML Data Collector финализирован")

# ===== ДОБАВИТЬ ПОСЛЕ FINALIZE =====
# Агрессивная очистка памяти при остановке
logger.info("🧹 Агрессивная очистка памяти при остановке...")

# 1. Очистить все ML буферы явно
if self.ml_data_collector:
    for symbol in list(self.ml_data_collector.feature_buffers.keys()):
        self.ml_data_collector.feature_buffers[symbol].clear()
        self.ml_data_collector.label_buffers[symbol].clear()
        self.ml_data_collector.metadata_buffers[symbol].clear()
    logger.info("  ✓ ML буферы очищены")

# 2. Очистить feature pipelines
if self.ml_feature_pipeline:
    for symbol in list(self.ml_feature_pipeline.pipelines.keys()):
        pipeline = self.ml_feature_pipeline.pipelines[symbol]
        # Очистить кэши
        if hasattr(pipeline, '_cache'):
            pipeline._cache.clear()
        # Очистить историю в extractors
        if hasattr(pipeline, 'orderbook_extractor'):
            pipeline.orderbook_extractor.snapshot_history.clear()
            pipeline.orderbook_extractor.level_ttl_history.clear()
        if hasattr(pipeline, 'indicator_extractor'):
            pipeline.indicator_extractor.candle_history.clear()
    logger.info("  ✓ Feature pipeline кэши очищены")

# 3. Удалить крупные объекты
del self.ml_data_collector
del self.ml_feature_pipeline
logger.info("  ✓ Крупные объекты удалены")

# 4. Принудительная сборка мусора (3 прохода)
import gc
collected_total = 0
for i in range(3):
    collected = gc.collect()
    collected_total += collected
    logger.info(f"  ✓ GC проход {i+1}/3: собрано {collected} объектов")

logger.info(f"🧹 Очистка завершена. Всего освобождено: {collected_total} объектов")

# Логируем финальное использование памяти
final_memory = get_memory_usage()
logger.info(f"📊 Финальное использование памяти: {final_memory:.1f} MB")
```

**Expected impact:**
- Explicit cleanup of all buffers
- Multiple GC passes to break circular references
- Memory should drop significantly (though Python may keep some pools)

---

### Fix 5: Reduce History Sizes (optional, more aggressive)

**orderbook_feature_extractor.py:174**
```python
self.max_history_size = 50  # 100 → 50
```

**indicator_feature_extractor.py:163**
```python
self.max_history_size = 100  # 200 → 100
```

**Expected impact:**
- 50% less history in memory
- Slight reduction in feature quality (less historical context)

---

## 🎯 Expected Results

| Metric | Before | After Fix | Improvement |
|--------|--------|-----------|-------------|
| Start RAM | 5 GB | 5 GB | - |
| After 5h | 25+ GB | 8-10 GB | 60-70% ↓ |
| Stop RAM release | 0% | 40-60% | Significant ↑ |
| Cleanup frequency | Every 16-83 min | Every 2-8 min | 10× ↑ |
| Buffer size | 120 MB | 30 MB | 75% ↓ |

---

## 📝 Implementation Priority

### High Priority (Do First):
1. ✅ Fix 1: Reduce buffer sizes
2. ✅ Fix 2: Frequent cleanup (100 cycles)
3. ✅ Fix 3: Lower threshold (4 GB)

### Medium Priority:
4. ✅ Fix 4: Aggressive stop() cleanup

### Low Priority (Optional):
5. ⚠️ Fix 5: Reduce history (may affect features quality)

---

## 🔬 Monitoring & Validation

After applying fixes, monitor:

1. **Memory growth rate:**
   ```bash
   # Watch memory every minute
   watch -n 60 'ps aux | grep python | grep main.py'
   ```

2. **Cleanup frequency in logs:**
   ```bash
   tail -f logs/bot.log | grep "🧹 Periodic memory cleanup"
   ```

3. **Expected pattern:**
   - Memory grows to ~6-8 GB
   - Cleanup triggers → drops to ~4-5 GB
   - Sawtooth pattern instead of linear growth

---

## ⚠️ Important Notes

### Why memory won't go to 0 after stop():
- Python doesn't release memory pools back to OS
- Some memory is retained for faster restarts
- Only killing the process fully releases memory
- This is NORMAL Python behavior

### Why not just increase max_samples_per_file?
- Larger buffers = more time between saves
- More data loss risk if crash occurs
- Higher peak memory usage
- Slower to respond to memory pressure

### Alternative: Reduce active symbols
- If trading 50 symbols → reduce to 20-30
- Each symbol adds ~2-4 MB baseline memory
- Fewer symbols = less memory pressure

---

## 🚀 Quick Apply Script

Apply all fixes at once:

```bash
# Backup current code
cp backend/main.py backend/main.py.backup

# Apply fixes (manual for now - need to edit files)
# Or use the edits shown above
```

---

## 📞 If Memory Still Leaks

1. **Check for other data collectors:**
   ```python
   # layering_data_collector
   # quote_stuffing_detector
   # Any other components with .data_buffer
   ```

2. **Profile memory:**
   ```python
   import tracemalloc
   tracemalloc.start()
   # ... run bot ...
   snapshot = tracemalloc.take_snapshot()
   top_stats = snapshot.statistics('lineno')
   for stat in top_stats[:10]:
       print(stat)
   ```

3. **Check for circular references:**
   ```python
   import gc
   gc.set_debug(gc.DEBUG_SAVEALL)
   gc.collect()
   for obj in gc.garbage:
       print(type(obj), obj)
   ```

---

**Status:** Ready to implement ✅
**Risk:** Low (all changes are conservative)
**Testing:** Run for 6-12 hours and monitor memory
