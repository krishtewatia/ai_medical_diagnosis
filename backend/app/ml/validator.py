import io
import numbers
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

from app.schemas.disease_config import (
    DiseaseCategory,
    DiseaseConfig,
    FeatureDataType,
    TabularFeatureSpec,
)


class PredictionValidationError(ValueError):
    """
    Raised when prediction request inputs violate required fields,
    expected data types, boundary constraints, or file upload requirements.
    """
    def __init__(self, detail: str, field_errors: Optional[Dict[str, str]] = None):
        super().__init__(detail)
        self.detail = detail
        self.field_errors = field_errors or {}


class DiseaseInputValidator:
    """
    Validation engine ensuring disease prediction requests strictly adhere
    to declarative DiseaseConfig constraints BEFORE reaching the predictor or model.
    """

    @classmethod
    def validate_tabular_inputs(
        cls,
        config: DiseaseConfig,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validates tabular input data against the disease's TabularFeatureSpec list.
        
        Checks:
        1. Input structure is a non-null dictionary.
        2. All required features are present and non-null.
        3. Feature data types (int, float, categorical) are valid and convertible.
        4. Categorical features conform to allowed_values enumerations.
        5. Numeric features adhere to min_value and max_value boundary limits.
        
        Returns a sanitized, properly type-cast dictionary of inputs.
        """
        if not isinstance(inputs, dict):
            raise PredictionValidationError(f"Tabular inputs must be a JSON object, got {type(inputs).__name__}.")

        if not config.tabular_features:
            return dict(inputs)

        sanitized: Dict[str, Any] = {}
        field_errors: Dict[str, str] = {}

        # Feature alias mappings for clinical terms
        aliases = {
            "cp": "chest_pain_type",
            "trestbps": "resting_bp",
            "chol": "cholestoral",
            "fbs": "fasting_blood_sugar",
            "thalach": "max_hr",
            "ca": "num_major_vessels",
        }

        # 1. Check required fields
        for spec in config.tabular_features:
            val = inputs.get(spec.name)
            if val is None:
                # Check for alias in inputs
                for alias_k, canonical in aliases.items():
                    if canonical == spec.name and alias_k in inputs:
                        val = inputs.get(alias_k)
                        break

            if val is None:
                if spec.required:
                    field_errors[spec.name] = f"Missing required feature '{spec.name}' ({spec.display_name})."
                continue

            # 2. Type validation and conversion
            converted_val, err = cls._validate_and_cast_type(spec, val)
            if err:
                field_errors[spec.name] = err
                continue

            # 3. Categorical allowed_values constraint check
            if spec.allowed_values is not None:
                if converted_val not in spec.allowed_values and str(converted_val) not in [str(x) for x in spec.allowed_values]:
                    field_errors[spec.name] = (
                        f"Value '{val}' for categorical feature '{spec.name}' is not allowed. "
                        f"Permitted values: {spec.allowed_values}"
                    )
                    continue

            # 4. Numeric boundary range checks (min_value & max_value)
            if isinstance(converted_val, (int, float)) and not isinstance(converted_val, bool):
                if spec.min_value is not None and converted_val < spec.min_value:
                    field_errors[spec.name] = (
                        f"Value {converted_val} for feature '{spec.name}' is below the minimum allowed limit of {spec.min_value}."
                    )
                    continue

                if spec.max_value is not None and converted_val > spec.max_value:
                    field_errors[spec.name] = (
                        f"Value {converted_val} for feature '{spec.name}' exceeds the maximum allowed limit of {spec.max_value}."
                    )
                    continue

            sanitized[spec.name] = converted_val

        if field_errors:
            first_err = next(iter(field_errors.values()))
            summary = f"Validation failed for disease '{config.id}': {first_err}"
            raise PredictionValidationError(summary, field_errors=field_errors)

        return sanitized

    @classmethod
    def _validate_and_cast_type(
        cls,
        spec: TabularFeatureSpec,
        val: Any
    ) -> tuple[Optional[Any], Optional[str]]:
        """Validates and coerces a single feature value according to its spec."""
        # Reject booleans when integer/float is expected (Python booleans are instances of int)
        if isinstance(val, bool) and spec.data_type in [FeatureDataType.INTEGER, FeatureDataType.FLOAT, "int", "float"]:
            return None, f"Feature '{spec.name}' expects a numeric value, but received boolean '{val}'."

        if spec.data_type in [FeatureDataType.INTEGER, "int"]:
            try:
                if isinstance(val, float):
                    if not val.is_integer():
                        return None, f"Feature '{spec.name}' expects an integer, got float with decimal '{val}'."
                    return int(val), None
                if isinstance(val, str):
                    stripped = val.strip()
                    if not (stripped.isdigit() or (stripped.startswith('-') and stripped[1:].isdigit())):
                        return None, f"Feature '{spec.name}' must be an integer, got non-numeric string '{val}'."
                int_val = int(val)
                return int_val, None
            except (ValueError, TypeError):
                return None, f"Feature '{spec.name}' must be an integer, got '{val}'."

        elif spec.data_type in [FeatureDataType.FLOAT, "float"]:
            try:
                float_val = float(val)
                return float_val, None
            except (ValueError, TypeError):
                return None, f"Feature '{spec.name}' must be a numeric float, got '{val}'."

        elif spec.data_type in [FeatureDataType.CATEGORICAL, "categorical"]:
            if spec.allowed_values:
                for av in spec.allowed_values:
                    if isinstance(av, int) and not isinstance(av, bool):
                        try:
                            if int(val) == av:
                                return av, None
                        except (ValueError, TypeError):
                            pass
                    elif isinstance(av, float):
                        try:
                            if float(val) == av:
                                return av, None
                        except (ValueError, TypeError):
                            pass
                    elif str(val) == str(av):
                        return av, None
            return val, None

        return val, None

    @classmethod
    def validate_image_input(
        cls,
        config: DiseaseConfig,
        image_bytes: bytes,
        filename: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> bytes:
        """
        Validates uploaded image file payload:
        1. Non-empty byte content.
        2. Size boundary check against ImageInputSpec.max_size_bytes.
        3. File extension and MIME type allowlist.
        4. Actual image decodability and corruption detection via PIL.
        """
        if not image_bytes or len(image_bytes) == 0:
            raise PredictionValidationError("Uploaded image file is empty (0 bytes).")

        spec = config.image_spec
        if not spec:
            raise PredictionValidationError(
                f"Disease '{config.id}' is configured as IMAGE but lacks image_spec."
            )

        # 1. Maximum file size check
        if len(image_bytes) > spec.max_size_bytes:
            raise PredictionValidationError(
                f"Image size ({len(image_bytes)} bytes) exceeds the maximum allowed limit of {spec.max_size_bytes} bytes."
            )

        # 2. Filename extension validation
        if filename:
            safe_filename = Path(filename).name
            ext = safe_filename.rsplit(".", 1)[-1].lower() if "." in safe_filename else ""
            if ext == "jpeg":
                ext = "jpg"
            if spec.allowed_formats and ext and ext not in spec.allowed_formats:
                raise PredictionValidationError(
                    f"Unsupported file extension '.{ext}'. Allowed formats: {spec.allowed_formats}"
                )

        # 3. Content-Type MIME validation
        if content_type and spec.allowed_formats:
            fmt = content_type.split("/")[-1].lower()
            if fmt == "jpeg":
                fmt = "jpg"
            if fmt not in spec.allowed_formats:
                raise PredictionValidationError(
                    f"Unsupported MIME type '{content_type}'. Allowed formats: {spec.allowed_formats}"
                )

        # 4. Binary decodability & corruption check
        try:
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.verify()
        except Exception as e:
            raise PredictionValidationError(
                f"Uploaded file is corrupt or not a valid readable image: {str(e)}"
            )

        return image_bytes
