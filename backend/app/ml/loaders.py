import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from app.schemas.disease_config import DiseaseConfig


ML_ROOT_DIR = Path(__file__).resolve().parent

# Global in-memory cache mapping: file_path -> (mtime, loaded_model_instance)
_MODEL_CACHE: Dict[str, Tuple[float, Any]] = {}


class ModelNotFoundError(FileNotFoundError):
    """Raised when a specified model artifact file is missing on disk."""
    pass


class UnsupportedModelFormatError(ValueError):
    """Raised when an artifact file has an unsupported file extension."""
    pass


class ModelLoadError(RuntimeError):
    """Raised when model deserialization/loading fails due to corruption or incompatibility."""
    pass


def clear_model_cache() -> None:
    """Clears the in-memory loaded model cache."""
    _MODEL_CACHE.clear()


def _patch_unpickled_model(model: Any) -> Any:
    """
    Patches unpickled scikit-learn models for forward-compatibility across minor versions
    (e.g., SimpleImputer missing _fill_dtype when loaded in scikit-learn 1.8+).
    """
    try:
        if hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                _patch_unpickled_model(step)
        if hasattr(model, "steps"):
            for _, step in model.steps:
                _patch_unpickled_model(step)
        if hasattr(model, "transformers_"):
            for _, trans, _ in model.transformers_:
                _patch_unpickled_model(trans)
        if hasattr(model, "statistics_") and not hasattr(model, "_fill_dtype"):
            model._fill_dtype = getattr(model.statistics_, "dtype", np.float64)
    except Exception:
        pass
    return model


def _load_joblib_or_pickle_artifact(path: Path) -> Any:
    """Loads a .joblib or .pkl/.pickle artifact safely using joblib / pickle."""
    try:
        import joblib
        model = joblib.load(str(path))
        return _patch_unpickled_model(model)
    except Exception as e_joblib:
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
                return _patch_unpickled_model(model)
        except Exception as e_pickle:
            raise ModelLoadError(
                f"Failed to load artifact '{path.name}' with joblib ({e_joblib}) "
                f"or standard pickle ({e_pickle})."
            ) from e_joblib


def _load_keras_artifact(path: Path) -> Any:
    """Loads a .keras or .h5 deep learning model artifact for inference."""
    if "KERAS_BACKEND" not in os.environ:
        os.environ["KERAS_BACKEND"] = "torch"

    try:
        import keras
        return keras.models.load_model(str(path), compile=False)
    except Exception as e_keras:
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(str(path), compile=False)
        except Exception as e_tf:
            raise ModelLoadError(
                f"Failed to load Keras artifact '{path.name}'. "
                f"Keras error: {e_keras}. TensorFlow error: {e_tf}"
            ) from e_keras


def load_model_artifact(
    artifact_path: Union[str, Path],
    use_cache: bool = True
) -> Any:
    """
    Unified model artifact loader.
    Supports .joblib, .pkl, .pickle, .keras, .h5 formats.
    Caches loaded model instances in memory by file path and modification time.
    """
    path = Path(artifact_path).resolve()

    if not path.exists() or not path.is_file():
        raise ModelNotFoundError(f"Model artifact not found at: {path}")

    # Check cache
    cache_key = str(path)
    current_mtime = path.stat().st_mtime

    if use_cache and cache_key in _MODEL_CACHE:
        cached_mtime, cached_model = _MODEL_CACHE[cache_key]
        if cached_mtime == current_mtime:
            return cached_model

    ext = path.suffix.lower()

    if ext == ".joblib":
        model = _load_joblib_or_pickle_artifact(path)
    elif ext in [".pkl", ".pickle"]:
        model = _load_joblib_or_pickle_artifact(path)
    elif ext in [".keras", ".h5"]:
        model = _load_keras_artifact(path)
    else:
        raise UnsupportedModelFormatError(
            f"Unsupported model artifact format '{ext}'. "
            f"Supported extensions: ['.joblib', '.pkl', '.pickle', '.keras', '.h5']"
        )

    if use_cache:
        _MODEL_CACHE[cache_key] = (current_mtime, model)

    return model


def load_model_for_disease(
    config: DiseaseConfig,
    use_cache: bool = True
) -> Any:
    """
    Resolves the artifact path for a given DiseaseConfig and loads the model.
    """
    if not config.model_dir:
        raise ValueError(f"model_dir is missing on DiseaseConfig for '{config.id}'")

    artifact_path = Path(config.model_dir) / config.artifact_filename
    return load_model_artifact(artifact_path, use_cache=use_cache)


def load_disease_config_from_file(
    config_path: Union[str, Path],
    validate_artifact_exists: bool = True
) -> DiseaseConfig:
    """
    Loads and validates a disease configuration from a JSON file.
    Resolves the directory of model artifacts.
    """
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Disease configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Set model_dir to the directory containing the config file
    model_dir = path.parent
    data["model_dir"] = str(model_dir)

    config = DiseaseConfig(**data)

    if validate_artifact_exists:
        artifact_path = model_dir / config.artifact_filename
        if not artifact_path.exists():
            raise ModelNotFoundError(
                f"Model artifact '{config.artifact_filename}' for disease '{config.id}' "
                f"was not found in directory: {model_dir}"
            )

    return config


def discover_and_load_disease_configs(
    base_dir: Optional[Union[str, Path]] = None,
    validate_artifacts: bool = True
) -> Dict[str, DiseaseConfig]:
    """
    Recursively discovers all `config.json` files within `tabular_models`
    and `image_models` directories and loads them into a dictionary keyed by disease ID.
    """
    root = Path(base_dir).resolve() if base_dir else ML_ROOT_DIR
    configs: Dict[str, DiseaseConfig] = {}

    search_dirs = [root / "tabular_models", root / "image_models"]

    for parent_dir in search_dirs:
        if not parent_dir.exists():
            continue

        for config_file in parent_dir.glob("**/config.json"):
            try:
                config = load_disease_config_from_file(
                    config_file,
                    validate_artifact_exists=validate_artifacts
                )
                configs[config.id] = config
            except Exception as e:
                print(f"[Warning] Failed to load disease config from {config_file}: {e}")

    return configs
