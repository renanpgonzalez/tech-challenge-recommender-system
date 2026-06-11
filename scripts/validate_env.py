"""Validate the local project environment."""

from importlib.metadata import version

REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "scikit-learn",
    "torch",
    "mlflow",
    "dvc",
    "pydantic",
    "pydantic-settings",
    "python-dotenv",
    "PyYAML",
]


def validate_packages(packages: list[str]) -> None:
    """Validate whether required packages are installed.

    Args:
        packages: List of package names to validate.
    """
    for package in packages:
        package_version = version(package)
        print(f"{package}: {package_version}")


def main() -> None:
    """Run environment validation."""
    print("Validating project environment...")
    validate_packages(REQUIRED_PACKAGES)
    print("Environment validation completed successfully.")


if __name__ == "__main__":
    main()
