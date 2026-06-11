"""Tests for MLflow Model Registry helpers."""

from recommender.tracking.registry import (
    RegistryPromotionResult,
    build_model_version_tags,
)


def test_build_model_version_tags() -> None:
    """Validate model version tag creation."""
    result = build_model_version_tags(
        model_type="neural_reranker",
        validation_status="approved",
        decision_metric="hit_rate_at_k",
    )

    assert result == {
        "model_type": "neural_reranker",
        "validation_status": "approved",
        "decision_metric": "hit_rate_at_k",
    }


def test_registry_promotion_result() -> None:
    """Validate registry promotion result data."""
    result = RegistryPromotionResult(
        model_name="test-model",
        version="1",
        stage="Production",
        champion_alias="champion",
    )

    assert result.model_name == "test-model"
    assert result.version == "1"
    assert result.stage == "Production"
    assert result.champion_alias == "champion"
