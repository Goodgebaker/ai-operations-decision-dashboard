"""Fail fast when the public Streamlit deployment bundle is incomplete."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = PROJECT_ROOT / "dashboard" / "app.py"
REQUIRED_FILES = (
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / ".streamlit" / "config.toml",
    PROJECT_ROOT / "data" / "synthetic_logs_v2.csv",
    PROJECT_ROOT / "outputs" / "model_operating_scores.csv",
    PROJECT_ROOT / "outputs" / "model_operating_snapshot.csv",
    PROJECT_ROOT / "outputs" / "model_capability_scores.csv",
    PROJECT_ROOT / "outputs" / "model_fusion_diagnosis.csv",
    PROJECT_ROOT / "outputs" / "model_capability_profiles.csv",
    PROJECT_ROOT / "outputs" / "model_health_risks.csv",
    PROJECT_ROOT / "outputs" / "model_diagnostic_evidence.csv",
    PROJECT_ROOT / "docs" / "ai_monitoring_metric_dictionary.xlsx",
)
STREAMLIT_DEPENDENCY_FILES = (
    "uv.lock",
    "Pipfile",
    "environment.yml",
    "environment.yaml",
    "requirements.txt",
    "pyproject.toml",
)
FORBIDDEN_PUBLIC_FILES = (
    PROJECT_ROOT / ".env",
    PROJECT_ROOT / ".streamlit" / "secrets.toml",
)


def main() -> None:
    missing = [path.relative_to(PROJECT_ROOT) for path in (ENTRYPOINT, *REQUIRED_FILES) if not path.is_file()]
    if missing:
        formatted = ", ".join(map(str, missing))
        raise SystemExit(f"Missing deployment files: {formatted}")

    dependency_files = [name for name in STREAMLIT_DEPENDENCY_FILES if (PROJECT_ROOT / name).is_file()]
    if dependency_files != ["requirements.txt"]:
        formatted = ", ".join(dependency_files) or "none"
        raise SystemExit(
            "Streamlit Community Cloud must have exactly one root dependency file; "
            f"found: {formatted}"
        )

    exposed = [path.relative_to(PROJECT_ROOT) for path in FORBIDDEN_PUBLIC_FILES if path.exists()]
    if exposed:
        formatted = ", ".join(map(str, exposed))
        raise SystemExit(f"Local secret files must not be part of the deployment bundle: {formatted}")

    print("Deployment bundle is complete: dashboard/app.py with all required demo assets.")


if __name__ == "__main__":
    main()

