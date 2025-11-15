#!/usr/bin/env python3
"""
Простой скрипт для исправления ситуации с несколькими PRODUCTION моделями.
Работает напрямую с файловой системой без импорта backend модулей.
"""

import json
from pathlib import Path
from datetime import datetime


def fix_production_models():
    """Fix multiple PRODUCTION models issue."""
    print("=" * 80)
    print("Исправление множественных PRODUCTION моделей")
    print("=" * 80)
    print()

    # Registry directory
    registry_dir = Path("models")
    if not registry_dir.exists():
        print(f"✗ Registry directory not found: {registry_dir}")
        return

    # For each model
    for model_dir in registry_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        production_versions = []

        # Find all PRODUCTION versions
        for version_dir in model_dir.iterdir():
            if not version_dir.is_dir() or version_dir.name.startswith('.'):
                continue

            metadata_file = version_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            # Read metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            if metadata.get('stage') == 'PRODUCTION':
                production_versions.append({
                    'version': version_dir.name,
                    'path': version_dir,
                    'metadata_file': metadata_file,
                    'metadata': metadata
                })

        if len(production_versions) <= 1:
            print(f"✓ {model_name}: OK ({len(production_versions)} PRODUCTION model)")
            continue

        print(f"⚠ {model_name}: Найдено {len(production_versions)} PRODUCTION моделей:")
        for pv in production_versions:
            print(f"  - {pv['version']}")
        print()

        # Sort by version (newest first)
        production_versions.sort(key=lambda x: x['version'], reverse=True)

        # Keep the newest one
        newest = production_versions[0]
        print(f"  ✓ Оставляем в PRODUCTION: {newest['version']}")

        # Demote the rest to STAGING
        for old_model in production_versions[1:]:
            print(f"  ➜ Демотируем в STAGING: {old_model['version']}")

            # Update metadata
            old_model['metadata']['stage'] = 'STAGING'
            old_model['metadata']['updated_at'] = datetime.now().isoformat()

            # Save metadata
            with open(old_model['metadata_file'], 'w') as f:
                json.dump(old_model['metadata'], f, indent=2)

            print(f"    ✓ Успешно обновлен metadata.json")

        print()

    print("=" * 80)
    print("✅ Готово! Обновите страницу Model Registry во фронтенде (F5).")
    print("=" * 80)


if __name__ == "__main__":
    fix_production_models()
