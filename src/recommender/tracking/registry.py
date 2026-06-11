"""MLflow Model Registry helpers."""

from dataclasses import dataclass

from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion


@dataclass(frozen=True)
class RegistryPromotionResult:
    """Model registry promotion result."""

    model_name: str
    version: str
    stage: str
    champion_alias: str


def find_model_version_by_run_id(
    client: MlflowClient,
    model_name: str,
    run_id: str,
) -> ModelVersion:
    """Find a registered model version by MLflow run ID.

    Args:
        client: MLflow client.
        model_name: Registered model name.
        run_id: Source MLflow run ID.

    Returns:
        Matching model version.
    """
    versions = client.search_model_versions(f"name = '{model_name}'")
    matching_versions = [version for version in versions if version.run_id == run_id]

    if not matching_versions:
        message = f"No model version found for run_id={run_id}"
        raise ValueError(message)

    return max(matching_versions, key=lambda version: int(version.version))


def promote_to_production(
    client: MlflowClient,
    model_name: str,
    version: str,
) -> RegistryPromotionResult:
    """Promote a model version through Staging and Production.

    Args:
        client: MLflow client.
        model_name: Registered model name.
        version: Model version.

    Returns:
        Promotion result.
    """
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=True,
    )
    client.set_registered_model_alias(model_name, "staging", version)

    production_version = client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    client.set_registered_model_alias(model_name, "champion", version)

    return RegistryPromotionResult(
        model_name=model_name,
        version=version,
        stage=production_version.current_stage,
        champion_alias="champion",
    )


def build_model_version_tags(
    model_type: str,
    validation_status: str,
    decision_metric: str,
) -> dict[str, str]:
    """Build model version tags.

    Args:
        model_type: Model type.
        validation_status: Validation status.
        decision_metric: Main decision metric.

    Returns:
        Model version tags.
    """
    return {
        "model_type": model_type,
        "validation_status": validation_status,
        "decision_metric": decision_metric,
    }
