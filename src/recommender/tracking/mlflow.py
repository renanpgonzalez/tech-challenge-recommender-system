"""MLflow tracking helpers."""

from collections.abc import Iterator
from contextlib import contextmanager

import mlflow

from recommender.config import get_settings


def configure_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """Configure MLflow tracking.

    Args:
        tracking_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.
    """
    settings = get_settings()

    mlflow.set_tracking_uri(tracking_uri or settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name or settings.mlflow_experiment_name)


@contextmanager
def start_mlflow_run(
    run_name: str,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> Iterator[object]:
    """Start a configured MLflow run.

    Args:
        run_name: MLflow run name.
        tracking_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.

    Yields:
        Active MLflow run.
    """
    configure_mlflow(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
    )

    with mlflow.start_run(run_name=run_name) as active_run:
        yield active_run
