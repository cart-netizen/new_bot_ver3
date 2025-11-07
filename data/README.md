# ML Training Data

This directory contains data for ML model training.

## Directory Structure

```
data/
├── ml_training/          # Legacy .npy training data
│   ├── BTCUSDT/
│   │   ├── features/     # Feature .npy files
│   │   └── labels/       # Label .json files
│   └── ...
└── feature_store/        # Feature Store data (parquet files)
    └── training_features/
```

## Data Collection

Before you can train models, you need to collect training data:

### Option 1: Use Feature Store (Recommended)

1. Start the bot to begin collecting live data:
   ```bash
   python -m backend.main
   ```

2. Data will be automatically stored in the Feature Store

3. After collecting data for some time (recommended: at least 1 day), you can start training

### Option 2: Use Legacy Data Collector

1. Run the historical data collector:
   ```bash
   python -m backend.ml_engine.data_collection.data_collector
   ```

2. This will collect historical data and save to `ml_training/` directory

## Training

Once you have data, you can train models:

### Via API (Frontend)

1. Open the ML Management page in the frontend
2. Click "Start Training"
3. Configure training parameters
4. Monitor training progress

### Via CLI

```bash
python -m backend.ml_engine.training_orchestrator --epochs 50 --batch-size 64
```

## Data Requirements

- **Minimum samples**: 600+ (sequence_length × 10)
- **Recommended**: 10,000+ samples for good model performance
- **Feature dimension**: 110 features per sample
- **Labels**: 3 classes (DOWN=0, NEUTRAL=1, UP=2)

## Troubleshooting

### "Failed to load training data"

This error means no training data is available. Solutions:
- Check if data directories exist: `ls data/ml_training/`
- Check if Feature Store has data: `ls data/feature_store/`
- Run data collection first (see above)
- Wait for bot to collect more data

### "Insufficient data: X samples, need at least Y"

You don't have enough data yet. Let the bot run longer to collect more samples.
