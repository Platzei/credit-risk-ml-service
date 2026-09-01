import json
import os
from pathlib import Path

from catboost import CatBoostClassifier

from credit_risk.data import PROJECT_ROOT

ARTIFACT_DIR = Path(
    os.environ.get("CREDIT_RISK_ARTIFACT_DIR", PROJECT_ROOT / "artifacts")
)
MODEL_PATH = ARTIFACT_DIR / "catboost_model.cbm"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"


def save_model_artifacts(
    model: CatBoostClassifier,
    feature_order: list[str],
    decision_threshold: float,
) -> None:
    """Save the fitted model and inference metadata."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))

    metadata = {
        "model_name": "credit_risk_catboost",
        "model_type": "CatBoostClassifier",
        "feature_order": feature_order,
        "decision_threshold": decision_threshold,
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def load_model_artifacts(
    artifact_dir: str | Path = ARTIFACT_DIR,
) -> tuple[CatBoostClassifier, dict]:
    """Load the persisted model and metadata."""
    artifact_path = Path(artifact_dir)
    model_path = artifact_path / MODEL_PATH.name
    metadata_path = artifact_path / METADATA_PATH.name
    missing_paths = [path for path in (model_path, metadata_path) if not path.is_file()]

    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"Required model artifacts are missing: {missing}. "
            "Run 'python -m credit_risk.train' first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_fields = {
        "model_name",
        "model_type",
        "feature_order",
        "decision_threshold",
    }
    missing_fields = required_fields.difference(metadata)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Model metadata is missing required fields: {missing}")

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model, metadata
