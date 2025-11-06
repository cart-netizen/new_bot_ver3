"""
Test script to verify MLflow PostgreSQL integration

This script:
1. Checks PostgreSQL connection
2. Initializes MLflow with PostgreSQL backend
3. Creates a test experiment and run
4. Verifies data is stored in PostgreSQL
"""

import mlflow
from mlflow.tracking import MlflowClient
from backend.config import config
from backend.core.logger import get_logger

logger = get_logger(__name__)


def test_mlflow_postgres():
    """Test MLflow PostgreSQL integration"""

    print("=" * 60)
    print("Testing MLflow PostgreSQL Integration")
    print("=" * 60)

    # 1. Print configuration
    print(f"\n1. Configuration:")
    print(f"   MLFLOW_TRACKING_URI: {config.MLFLOW_TRACKING_URI}")
    print(f"   MLFLOW_ARTIFACT_LOCATION: {config.MLFLOW_ARTIFACT_LOCATION}")
    print(f"   MLFLOW_EXPERIMENT_NAME: {config.MLFLOW_EXPERIMENT_NAME}")

    # 2. Set tracking URI
    print(f"\n2. Setting MLflow tracking URI...")
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    print(f"   ✓ Tracking URI set to: {mlflow.get_tracking_uri()}")

    # 3. Create test experiment
    print(f"\n3. Creating test experiment...")
    experiment_name = "test_postgres_integration"

    try:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            artifact_location=config.MLFLOW_ARTIFACT_LOCATION
        )
        print(f"   ✓ Created new experiment: {experiment_name} (ID: {experiment_id})")
    except mlflow.exceptions.MlflowException as e:
        if "already exists" in str(e):
            experiment = mlflow.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
            print(f"   ✓ Using existing experiment: {experiment_name} (ID: {experiment_id})")
        else:
            raise

    mlflow.set_experiment(experiment_name)

    # 4. Create test run
    print(f"\n4. Creating test run...")
    with mlflow.start_run(run_name="test_run") as run:
        run_id = run.info.run_id
        print(f"   ✓ Started run: {run_id}")

        # Log parameters
        mlflow.log_param("test_param", "test_value")
        mlflow.log_param("learning_rate", 0.001)
        print(f"   ✓ Logged parameters")

        # Log metrics
        mlflow.log_metric("accuracy", 0.95)
        mlflow.log_metric("loss", 0.05)
        print(f"   ✓ Logged metrics")

        # Log tags
        mlflow.set_tag("test_tag", "postgres_integration")
        mlflow.set_tag("version", "1.0")
        print(f"   ✓ Logged tags")

    print(f"   ✓ Run completed: {run_id}")

    # 5. Verify data in PostgreSQL
    print(f"\n5. Verifying data in PostgreSQL...")
    client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)

    # Get run details
    run_data = client.get_run(run_id)
    print(f"   ✓ Retrieved run from PostgreSQL")
    print(f"   Run ID: {run_data.info.run_id}")
    print(f"   Status: {run_data.info.status}")
    print(f"   Parameters: {dict(run_data.data.params)}")
    print(f"   Metrics: {dict(run_data.data.metrics)}")
    print(f"   Tags: {dict(run_data.data.tags)}")

    # List all experiments
    experiments = client.search_experiments()
    print(f"\n6. All experiments in PostgreSQL:")
    for exp in experiments:
        print(f"   - {exp.name} (ID: {exp.experiment_id})")

    # 7. Success
    print("\n" + "=" * 60)
    print("✅ MLflow PostgreSQL Integration Test PASSED!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start MLflow UI:")
    print(f"   mlflow ui --backend-store-uri {config.MLFLOW_TRACKING_URI}")
    print("2. Open http://localhost:5000")
    print("3. You should see the 'test_postgres_integration' experiment")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_mlflow_postgres()
    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
        print("\n" + "=" * 60)
        print("❌ MLflow PostgreSQL Integration Test FAILED!")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Check that PostgreSQL is running:")
        print("   Windows: sc query postgresql-x64-16")
        print("   Linux/Mac: systemctl status postgresql")
        print("2. Check DATABASE_URL in .env:")
        print(f"   Current: {config.MLFLOW_TRACKING_URI}")
        print("3. Check psycopg2-binary is installed:")
        print("   pip install psycopg2-binary")
        print("4. Test PostgreSQL connection:")
        print("   psql -h localhost -U trading_bot -d trading_bot")
        print("=" * 60)
        raise
