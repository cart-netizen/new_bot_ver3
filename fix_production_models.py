#!/usr/bin/env python3
"""
Скрипт для исправления ситуации с несколькими PRODUCTION моделями.
Демотирует все старые модели в STAGING, оставляя только самую новую в PRODUCTION.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.ml_engine.inference.model_registry import get_model_registry, ModelStage


async def fix_production_models():
    """Fix multiple PRODUCTION models issue."""
    print("=" * 80)
    print("Исправление множественных PRODUCTION моделей")
    print("=" * 80)
    print()

    registry = get_model_registry()

    # Get all models
    all_models = await registry.list_models()

    # Group by model name
    models_by_name = {}
    for model in all_models:
        if model.name not in models_by_name:
            models_by_name[model.name] = []
        models_by_name[model.name].append(model)

    # Fix each model
    for model_name, versions in models_by_name.items():
        production_models = [m for m in versions if m.metadata.stage == ModelStage.PRODUCTION]

        if len(production_models) <= 1:
            print(f"✓ {model_name}: OK ({len(production_models)} PRODUCTION model)")
            continue

        print(f"⚠ {model_name}: Найдено {len(production_models)} PRODUCTION моделей")

        # Sort by version (newest first)
        production_models.sort(key=lambda m: m.version, reverse=True)

        # Keep the newest one
        newest = production_models[0]
        print(f"  Оставляем в PRODUCTION: {newest.version}")

        # Demote the rest
        for old_model in production_models[1:]:
            print(f"  Демотируем в STAGING: {old_model.version}")
            success = await registry.set_model_stage(
                old_model.name,
                old_model.version,
                ModelStage.STAGING
            )
            if success:
                print(f"    ✓ Успешно")
            else:
                print(f"    ✗ Ошибка")

        print()

    print("=" * 80)
    print("✅ Готово! Обновите страницу Model Registry во фронтенде.")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(fix_production_models())
