"""Tests for project settings."""

from pathlib import Path

import pytest

from recommender.config.settings import Settings, get_settings


def test_default_settings_values() -> None:
    """Validate default settings values."""
    settings = Settings()

    assert settings.app_env == "development"
    assert settings.random_seed == 42
    assert settings.data_raw_dir == Path("data/raw")
    assert settings.data_interim_dir == Path("data/interim")
    assert settings.data_processed_dir == Path("data/processed")
    assert settings.mlflow_experiment_name == "product-recommender"


def test_settings_can_be_overridden_by_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Validate that environment variables override default settings."""
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("RANDOM_SEED", "123")

    settings = Settings()

    assert settings.app_env == "test"
    assert settings.random_seed == 123


def test_get_settings_returns_cached_instance() -> None:
    """Validate that get_settings returns a cached settings instance."""
    get_settings.cache_clear()

    first_settings = get_settings()
    second_settings = get_settings()

    assert first_settings is second_settings

    get_settings.cache_clear()
