"""
Layering ML Management API - REST API для управления Layering ML моделью

Endpoints:
- GET /api/ml-management/layering/status - Model status
- GET /api/ml-management/layering/data-status - Data collection status
- POST /api/ml-management/layering/check-data - Run data status check
- POST /api/ml-management/layering/train - Train improved model
- GET /api/ml-management/layering/metrics - Model metrics
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import subprocess
import json
import asyncio
import sys

from backend.core.logger import get_logger
from backend.config import get_project_data_path

logger = get_logger(__name__)

# Project root directory (where main.py and scripts are located)
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# Пути к данным (абсолютные, от project_root)
_LAYERING_DATA_DIR = Path(get_project_data_path("ml_training/layering"))

# Create router
router = APIRouter(prefix="/api/ml-management/layering", tags=["Layering ML"])


# ============================================================
# Helper Functions
# ============================================================

def run_script_sync(script_path: str, timeout: int = 300) -> Dict[str, Any]:
    """
    Запустить Python скрипт синхронно и вернуть результат.

    Uses sys.executable to run script with the same Python interpreter
    that's running the backend (includes virtual environment if active).

    Args:
        script_path: Путь к скрипту (относительный от project root)
        timeout: Таймаут в секундах

    Returns:
        Dict с output, error, return_code
    """
    try:
        # Resolve script path relative to project root
        full_script_path = _PROJECT_ROOT / script_path

        # Use sys.executable to ensure we use the same Python (with venv)
        result = subprocess.run(
            [sys.executable, str(full_script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_PROJECT_ROOT
        )

        return {
            "success": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr,
            "return_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "output": "",
            "error": f"Script timeout after {timeout} seconds",
            "return_code": -1
        }
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": str(e),
            "return_code": -1
        }


async def run_script_async(script_path: str, timeout: int = 300) -> Dict[str, Any]:
    """Async wrapper для run_script_sync"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_script_sync, script_path, timeout)


def load_layering_model_info() -> Optional[Dict[str, Any]]:
    """
    Загрузить информацию о Layering модели из файла.
    Легковесная версия - читает только метаданные без полной загрузки модели.

    Returns:
        Dict с информацией о модели или None
    """
    try:
        import pickle

        model_path = "data/models/layering_adaptive_v1.pkl"
        model_file = Path(model_path)

        if not model_file.exists():
            return None

        # Read model metadata without fully loading the model
        # This avoids triggering ERROR logs on every page load
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)

        # Extract metadata
        info = {
            'version': model_package.get('version', '1.0.0'),
            'trained_at': model_package.get('trained_at'),
            'training_samples': model_package.get('training_samples', 0),
            'optimal_threshold': model_package.get('optimal_threshold', 0.5),
            'feature_names': model_package.get('feature_names', []),
            'feature_importance': model_package.get('feature_importance', {}),
            'metrics': model_package.get('metrics', {}),
            'file_path': str(model_file.absolute()),
            'file_size_kb': model_file.stat().st_size / 1024,
            'file_modified': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
        }

        return info

    except Exception as e:
        # Use debug instead of error - model info not loading is not critical
        logger.debug(f"Could not load layering model info: {e}")
        return None


def load_data_statistics() -> Dict[str, Any]:
    """
    Загрузить статистику данных для обучения.

    Returns:
        Dict со статистикой данных
    """
    try:
        stats_file = _LAYERING_DATA_DIR / "statistics.json"

        if not stats_file.exists():
            return {
                "exists": False,
                "total_collected": 0,
                "total_labeled": 0,
                "total_saved": 0
            }

        with open(stats_file, 'r') as f:
            stats = json.load(f)

        stats['exists'] = True
        return stats

    except Exception as e:
        logger.error(f"Failed to load data statistics: {e}")
        return {
            "exists": False,
            "error": str(e)
        }


# ============================================================
# Endpoints
# ============================================================

@router.get("/status")
async def get_layering_model_status() -> Dict[str, Any]:
    """
    Получить статус Layering ML модели.

    Returns:
        Статус модели (загружена ли, метрики, версия и т.д.)
    """
    try:
        model_info = load_layering_model_info()

        if not model_info:
            return {
                "loaded": False,
                "message": "Layering model not found or not trained yet",
                "model_path": "data/models/layering_adaptive_v1.pkl"
            }

        return {
            "loaded": True,
            "model_info": model_info
        }

    except Exception as e:
        logger.error(f"Failed to get layering model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-status")
async def get_data_status() -> Dict[str, Any]:
    """
    Получить статус данных для обучения.

    Returns:
        Статистика собранных данных
    """
    try:
        stats = load_data_statistics()

        # Count parquet files
        data_dir = _LAYERING_DATA_DIR
        if data_dir.exists():
            parquet_files = list(data_dir.glob("layering_data_*.parquet"))
            stats['files_count'] = len(parquet_files)
            stats['data_dir_exists'] = True
        else:
            stats['files_count'] = 0
            stats['data_dir_exists'] = False

        # Check readiness for training
        labeled_count = stats.get('total_labeled', 0)
        stats['ready_for_training'] = labeled_count >= 100
        stats['minimum_required'] = 100
        stats['recommended'] = 500

        return stats

    except Exception as e:
        logger.debug(f"Failed to get data status: {e}")
        # Return empty stats instead of error
        return {
            "total_collected": 0,
            "total_labeled": 0,
            "files_count": 0,
            "data_dir_exists": False,
            "ready_for_training": False,
            "minimum_required": 100,
            "recommended": 500
        }


@router.post("/check-data")
async def check_data_status(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Запустить проверку статуса данных через check_layering_data_status.py

    Returns:
        Результат проверки
    """
    try:
        # Run check script
        script_path = "check_layering_data_status.py"

        if not (_PROJECT_ROOT / script_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Check script not found: {script_path} (looked in {_PROJECT_ROOT})"
            )

        result = await run_script_async(script_path, timeout=60)

        return {
            "success": result['success'],
            "output": result['output'],
            "error": result['error'] if not result['success'] else None,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to check data status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class LayeringTrainingRequest(BaseModel):
    """Request для обучения Layering модели"""
    use_improved: bool = Field(
        default=True,
        description="Use improved training script (with class balancing)"
    )
    timeout: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="Training timeout in seconds"
    )


@router.post("/train")
async def train_layering_model(
    request: LayeringTrainingRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Запустить обучение Layering ML модели.

    Автоматически запускает labeling для unlabeled данных перед обучением.

    Args:
        request: Параметры обучения
        background_tasks: FastAPI background tasks

    Returns:
        Результат обучения
    """
    try:
        output_lines = []

        # ===== STEP 1: Auto-labeling unlabeled data =====
        logger.info("Step 1: Checking for unlabeled data...")
        output_lines.append("=" * 80)
        output_lines.append("STEP 1: AUTO-LABELING UNLABELED DATA")
        output_lines.append("=" * 80)

        # Check data status
        data_status = load_data_statistics()
        unlabeled_count = data_status.get('total_collected', 0) - data_status.get('total_labeled', 0)

        if unlabeled_count > 0:
            logger.info(f"Found {unlabeled_count} unlabeled samples, running auto-labeling...")
            output_lines.append(f"Found {unlabeled_count} unlabeled samples")
            output_lines.append("Running automatic labeling...")

            # Run labeling script
            label_script = "label_layering_data.py"
            if (_PROJECT_ROOT / label_script).exists():
                label_result = await run_script_async(label_script, timeout=120)

                if label_result['success']:
                    output_lines.append("✓ Auto-labeling completed successfully")
                    output_lines.append(label_result['output'])
                    logger.info("Auto-labeling completed successfully")
                else:
                    output_lines.append("⚠ Auto-labeling failed, proceeding with existing labeled data")
                    output_lines.append("")

                    # Check if it's a dependency error
                    error_msg = label_result['output'] if label_result['output'] else str(label_result.get('error', ''))
                    if 'pandas' in error_msg or 'pyarrow' in error_msg or 'ModuleNotFoundError' in error_msg:
                        output_lines.append("ERROR: Missing required packages (pandas, pyarrow)")
                        output_lines.append("")
                        output_lines.append("To enable automatic labeling, install dependencies:")
                        output_lines.append("  pip install pandas pyarrow")
                        output_lines.append("")
                        output_lines.append("Or run labeling manually before training:")
                        output_lines.append("  python label_layering_data.py")
                        output_lines.append("")
                        output_lines.append("Note: Training will use existing labeled data if available.")
                    else:
                        output_lines.append(error_msg or "Unknown error")

                    logger.warning(f"Auto-labeling failed: {label_result.get('error', 'Unknown')}")
            else:
                output_lines.append("⚠ Labeling script not found, using existing labeled data")
                logger.warning(f"Labeling script not found: {label_script}")
        else:
            output_lines.append("✓ All data is already labeled")
            logger.info("All data is already labeled")

        output_lines.append("")

        # ===== STEP 2: Training =====
        logger.info("Step 2: Starting model training...")
        output_lines.append("=" * 80)
        output_lines.append("STEP 2: MODEL TRAINING")
        output_lines.append("=" * 80)

        # Choose training script
        if request.use_improved:
            script_path = "train_layering_model_improved.py"
        else:
            script_path = "train_layering_model_debug.py"

        if not (_PROJECT_ROOT / script_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Training script not found: {script_path} (looked in {_PROJECT_ROOT})"
            )

        logger.info(f"Starting layering model training with script: {script_path}")

        # Run training (this will take 2-10 minutes)
        result = await run_script_async(script_path, timeout=request.timeout)

        # Combine outputs from labeling and training
        output_lines.append(result['output'])
        combined_output = "\n".join(output_lines)

        if result['success']:
            # Reload model info after training
            model_info = load_layering_model_info()

            return {
                "success": True,
                "message": "Layering model training completed successfully",
                "output": combined_output,
                "model_info": model_info,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": "Training failed",
                "error": result['error'],
                "output": combined_output,
                "timestamp": datetime.now().isoformat()
            }

    except Exception as e:
        logger.error(f"Failed to train layering model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_layering_metrics() -> Dict[str, Any]:
    """
    Получить метрики Layering ML модели.

    Returns:
        Метрики модели (accuracy, precision, recall, F1, etc.)
    """
    try:
        model_info = load_layering_model_info()

        if not model_info:
            return {
                "available": False,
                "message": "Model not loaded or not trained"
            }

        metrics = model_info.get('metrics', {})

        return {
            "available": True,
            "metrics": metrics,
            "optimal_threshold": model_info.get('optimal_threshold', 0.5),
            "version": model_info.get('version', 'unknown'),
            "training_samples": model_info.get('training_samples', 0),
            "trained_at": model_info.get('trained_at'),
            "top_features": model_info.get('top_features', [])
        }

    except Exception as e:
        logger.debug(f"Failed to get layering metrics: {e}")
        # Return unavailable instead of error
        return {
            "available": False,
            "message": "Failed to load metrics"
        }


@router.post("/analyze-data")
async def analyze_data(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """
    Запустить детальный анализ данных через analyze_layering_ml_data.py

    Returns:
        Результат анализа
    """
    try:
        script_path = "analyze_layering_ml_data.py"

        if not (_PROJECT_ROOT / script_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Analysis script not found: {script_path} (looked in {_PROJECT_ROOT})"
            )

        result = await run_script_async(script_path, timeout=120)

        return {
            "success": result['success'],
            "output": result['output'],
            "error": result['error'] if not result['success'] else None,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to analyze data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
