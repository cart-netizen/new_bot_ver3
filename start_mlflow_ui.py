#!/usr/bin/env python3
"""
Start MLflow UI Server

This script starts the MLflow tracking server UI with PostgreSQL backend.
Access the UI at: http://localhost:5000
"""

import subprocess
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.config import settings

def main():
    """Start MLflow UI server"""

    tracking_uri = settings.MLFLOW_TRACKING_URI
    artifact_location = settings.MLFLOW_ARTIFACT_LOCATION

    # Ensure artifact location exists
    Path(artifact_location).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🚀 Starting MLflow UI Server")
    print("=" * 70)
    print(f"📊 Tracking URI: {tracking_uri}")
    print(f"📁 Artifact Location: {artifact_location}")
    print(f"🌐 UI will be available at: http://localhost:5000")
    print("=" * 70)
    print("\nPress Ctrl+C to stop the server\n")

    # Start MLflow UI
    cmd = [
        "mlflow", "ui",
        "--backend-store-uri", tracking_uri,
        "--default-artifact-root", artifact_location,
        "--host", "0.0.0.0",
        "--port", "5000"
    ]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n✓ MLflow UI server stopped")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting MLflow UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
