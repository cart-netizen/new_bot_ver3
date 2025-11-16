#!/usr/bin/env python3
"""
Скрипт для запуска ML Model Server на порту 8001.

Использование:
    python start_ml_server.py
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn

    # Импортируем app из model_server_v2
    from backend.ml_engine.inference.model_server_v2 import app

    print("=" * 80)
    print("🚀 Запуск ML Model Server")
    print("=" * 80)
    print()
    print("📍 URL: http://localhost:8001")
    print("📚 Документация: http://localhost:8001/docs")
    print("🔍 Health check: http://localhost:8001/health")
    print("🤖 Predict endpoint: POST http://localhost:8001/predict")
    print()
    print("Нажмите Ctrl+C для остановки")
    print("=" * 80)
    print()

    # Запускаем сервер
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )
