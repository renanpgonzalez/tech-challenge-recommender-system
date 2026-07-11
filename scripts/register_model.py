"""Register and promote a trained recommender model in MLflow."""

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient

from recommender.models import load_neural_model
from recommender.tracking import (
    build_model_version_tags,
    configure_mlflow,
    find_model_version_by_run_id,
    promote_to_production,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Register and promote a PyTorch recommender model.",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument(
        "--registered-model-name",
        type=str,
        default="retailrocket-neural-recommender",
    )
    parser.add_argument("--tracking-uri", type=str, default="sqlite:///mlflow.db")
    parser.add_argument("--experiment-name", type=str, default="product-recommender")
    parser.add_argument("--run-name", type=str, default="register_neural_recommender")

    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, float]:
    """Load numeric metrics from JSON.

    Args:
        path: Metrics JSON path.

    Returns:
        Numeric metrics.
    """
    metrics = json.loads(path.read_text(encoding="utf-8"))

    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, int | float)
    }


def log_and_register_pytorch_model(
    model: Any,
    metrics: dict[str, float],
    registered_model_name: str,
    model_path: Path,
    metrics_path: Path,
    run_name: str,
) -> str:
    """Log PyTorch model to MLflow and register it.

    Args:
        model: PyTorch model.
        metrics: Evaluation metrics.
        registered_model_name: Registered model name.
        model_path: Local model path.
        metrics_path: Local metrics path.
        run_name: MLflow run name.

    Returns:
        The MLflow run ID.
    """
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "registered_model_name": registered_model_name,
                "source_model_path": str(model_path),
                "source_metrics_path": str(metrics_path),
            },
        )
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            registered_model_name=registered_model_name,
            code_paths=["src"],
            pip_requirements=["torch", "mlflow"],
        )
        return str(run.info.run_id)


def tag_registered_model_version(
    client: MlflowClient,
    model_name: str,
    version: str,
) -> None:
    """Apply validation tags to a registered model version.

    Args:
        client: MLflow client.
        model_name: Registered model name.
        version: Model version.
    """
    tags = build_model_version_tags(
        model_type="neural_reranker",
        validation_status="approved",
        decision_metric="hit_rate_at_k",
    )

    for key, value in tags.items():
        client.set_model_version_tag(
            name=model_name,
            version=version,
            key=key,
            value=value,
        )


def register_model(
    model_path: Path,
    metrics_path: Path,
    registered_model_name: str,
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
) -> None:
    """Register and promote a trained neural recommender model."""
    configure_mlflow(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )

    model, _ = load_neural_model(model_path)
    metrics = load_metrics(metrics_path)

    run_id = log_and_register_pytorch_model(
        model,
        metrics,
        registered_model_name,
        model_path,
        metrics_path,
        run_name,
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    model_version = find_model_version_by_run_id(
        client=client,
        model_name=registered_model_name,
        run_id=run_id,
    )
    version = model_version.version

    tag_registered_model_version(client, registered_model_name, version)

    promotion = promote_to_production(
        client=client,
        model_name=registered_model_name,
        version=version,
    )

    print(f"Registered model: {promotion.model_name}")
    print(f"Version: {promotion.version}")
    print(f"Stage: {promotion.stage}")
    print(f"Alias: {promotion.champion_alias}")


def main() -> None:
    """Run model registration command line interface."""
    args = parse_args()

    register_model(
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        registered_model_name=args.registered_model_name,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
