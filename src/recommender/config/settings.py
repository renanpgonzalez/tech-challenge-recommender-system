"""Project settings management."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        app_env: Current application environment.
        random_seed: Global random seed for reproducibility.
        data_raw_dir: Directory for raw datasets.
        data_interim_dir: Directory for intermediate datasets.
        data_processed_dir: Directory for processed datasets.
        models_dir: Directory for trained model artifacts.
        reports_dir: Directory for reports and evaluation outputs.
        mlflow_tracking_uri: MLflow tracking server URI.
        mlflow_experiment_name: MLflow experiment name.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="development", alias="APP_ENV")
    random_seed: int = Field(default=42, alias="RANDOM_SEED")

    data_raw_dir: Path = Field(default=Path("data/raw"), alias="DATA_RAW_DIR")
    data_interim_dir: Path = Field(
        default=Path("data/interim"),
        alias="DATA_INTERIM_DIR",
    )
    data_processed_dir: Path = Field(
        default=Path("data/processed"),
        alias="DATA_PROCESSED_DIR",
    )

    models_dir: Path = Field(default=Path("models"), alias="MODEL_DIR")
    reports_dir: Path = Field(default=Path("reports"), alias="REPORT_DIR")

    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        alias="MLFLOW_TRACKING_URI",
    )
    mlflow_experiment_name: str = Field(
        default="product-recommender",
        alias="MLFLOW_EXPERIMENT_NAME",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings.

    Returns:
        Project settings instance.
    """
    return Settings()
