#!/usr/bin/env python3
"""
ML Model Server - Launcher Script

Запускает ML Model Server v2 на порту 8001
"""

import sys
import os
from pathlib import Path

# Добавить корневую директорию проекта в Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    import uvicorn

    # Запустить сервер
    uvicorn.run(
        "backend.ml_engine.inference.model_server_v2:app",
        host="0.0.0.0",
        port=8001,
        reload=True,  # Auto-reload при изменении кода
        log_level="info"
    )
