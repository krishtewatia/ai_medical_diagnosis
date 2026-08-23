import React, { useState } from "react";
import type { DiseaseResponse, PredictionResponse, TabularFeatureSpec } from "../types";
import { predictionService } from "../services/predictionService";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  RotateCcw,
} from "lucide-react";

interface DiseaseFormProps {
  config: DiseaseResponse;
  onSuccess: (result: PredictionResponse) => void;
}

export const DiseaseForm: React.FC<DiseaseFormProps> = ({
  config,
  onSuccess,
}) => {
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [globalError, setGlobalError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Validate a single field
  const validateField = (field: TabularFeatureSpec, value: any): string | null => {
    if (field.required && (value === undefined || value === null || value === "")) {
      return `${field.display_name} is required.`;
    }

    if (value === "" || value === undefined || value === null) {
      return null;
    }

    if (field.data_type === "int" || field.data_type === "float") {
      const num = Number(value);
      if (isNaN(num)) {
        return `${field.display_name} must be a valid number.`;
      }
      if (field.data_type === "int" && !Number.isInteger(num)) {
        return `${field.display_name} must be an integer.`;
      }
      if (field.min_value !== undefined && field.min_value !== null && num < field.min_value) {
        return `${field.display_name} cannot be less than ${field.min_value}${field.unit ? ` ${field.unit}` : ""}.`;
      }
      if (field.max_value !== undefined && field.max_value !== null && num > field.max_value) {
        return `${field.display_name} cannot exceed ${field.max_value}${field.unit ? ` ${field.unit}` : ""}.`;
      }
    }

    if (field.allowed_values && field.allowed_values.length > 0) {
      const isAllowed = field.allowed_values.some((v) => String(v) === String(value));
      if (!isAllowed) {
        return `Please select a valid option for ${field.display_name}.`;
      }
    }

    return null;
  };

  const handleInputChange = (field: TabularFeatureSpec, rawValue: string) => {
    let parsedValue: any = rawValue;

    if (rawValue === "") {
      parsedValue = "";
    } else if (field.data_type === "int") {
      const parsed = parseInt(rawValue, 10);
      parsedValue = isNaN(parsed) ? rawValue : parsed;
    } else if (field.data_type === "float") {
      const parsed = parseFloat(rawValue);
      parsedValue = isNaN(parsed) ? rawValue : parsed;
    } else if (field.data_type === "categorical") {
      // Check if allowed values are numeric
      const numericVal = Number(rawValue);
      if (!isNaN(numericVal) && field.allowed_values?.some((v) => typeof v === "number")) {
        parsedValue = numericVal;
      }
    }

    setFormData((prev) => ({
      ...prev,
      [field.name]: parsedValue,
    }));

    // Revalidate field
    const error = validateField(field, parsedValue);
    setFieldErrors((prev) => ({
      ...prev,
      [field.name]: error || "",
    }));
  };

  const handleReset = () => {
    setFormData({});
    setFieldErrors({});
    setGlobalError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setGlobalError(null);

    // Full validation pass across all required fields
    const errors: Record<string, string> = {};
    for (const field of config.required_fields) {
      const val = formData[field.name];
      const err = validateField(field, val);
      if (err) {
        errors[field.name] = err;
      }
    }

    if (Object.keys(errors).length > 0) {
      setFieldErrors(errors);
      setGlobalError("Please correct the highlighted validation errors before submitting.");
      return;
    }

    setSubmitting(true);
    try {
      // Send exactly the backend field names and values
      const payload = {
        disease_id: config.id,
        inputs: formData,
      };

      const result = await predictionService.predictTabular(payload);
      onSuccess(result);
    } catch (err: any) {
      setGlobalError(err?.message || "Prediction request failed. Please check your inputs.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="dynamic-disease-form-wrapper">
      {globalError && (
        <div className="error-alert" role="alert">
          <AlertCircle size={18} />
          <span>{globalError}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} className="modular-screening-form" noValidate>
        <div className="form-fields-grid">
          {config.required_fields.map((field) => {
            const currentValue = formData[field.name] ?? "";
            const fieldError = fieldErrors[field.name];

            return (
              <div key={field.name} className="dynamic-field-group">
                <div className="field-label-row">
                  <label htmlFor={`field-${field.name}`} className="form-label">
                    {field.display_name}
                    {field.required && <span className="required-star">*</span>}
                  </label>
                  {field.unit && <span className="field-unit-tag">{field.unit}</span>}
                </div>

                {/* Categorical Select Dropdown */}
                {field.allowed_values && field.allowed_values.length > 0 ? (
                  <select
                    id={`field-${field.name}`}
                    value={currentValue}
                    disabled={submitting}
                    onChange={(e) => handleInputChange(field, e.target.value)}
                    className={`form-input ${fieldError ? "form-input-error" : ""}`}
                  >
                    <option value="">Select option...</option>
                    {field.allowed_values.map((opt) => {
                      const optStr = String(opt);
                      const label = field.value_descriptions?.[optStr] || optStr;
                      return (
                        <option key={optStr} value={optStr}>
                          {label} ({optStr})
                        </option>
                      );
                    })}
                  </select>
                ) : (
                  /* Numeric Input (Float / Int) */
                  <div className="input-range-wrapper">
                    <input
                      id={`field-${field.name}`}
                      type="number"
                      step={field.data_type === "float" ? "any" : "1"}
                      min={field.min_value !== null ? field.min_value : undefined}
                      max={field.max_value !== null ? field.max_value : undefined}
                      placeholder={
                        field.min_value !== null && field.max_value !== null
                          ? `${field.min_value} - ${field.max_value}`
                          : "Enter value"
                      }
                      value={currentValue}
                      disabled={submitting}
                      onChange={(e) => handleInputChange(field, e.target.value)}
                      className={`form-input ${fieldError ? "form-input-error" : ""}`}
                    />
                    {field.min_value !== null && field.max_value !== null && !fieldError && (
                      <span className="field-range-hint">
                        Allowed: {field.min_value} – {field.max_value} {field.unit || ""}
                      </span>
                    )}
                  </div>
                )}

                {/* Field-level error indicator */}
                {fieldError && (
                  <span className="field-validation-error-text">
                    {fieldError}
                  </span>
                )}
              </div>
            );
          })}
        </div>

        {/* Action Controls */}
        <div className="form-submit-row">
          <button
            type="button"
            onClick={handleReset}
            disabled={submitting}
            className="btn btn-secondary"
          >
            <RotateCcw size={16} />
            <span>Reset Fields</span>
          </button>

          <button
            type="submit"
            disabled={submitting}
            className="btn btn-primary btn-lg submit-screening-btn"
          >
            {submitting ? (
              <>
                <Activity size={18} className="spinner-icon pulse-animation" />
                <span>Running Screening...</span>
              </>
            ) : (
              <>
                <CheckCircle2 size={18} />
                <span>Submit {config.display_name}</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};
