import React, { useState, useEffect } from "react";
import type { DiseaseResponse, MedicalProfile, PredictionResponse, TabularFeatureSpec } from "../types";
import { predictionService } from "../services/predictionService";
import { profileService } from "../services/profileService";
import {
  Activity,
  AlertCircle,
  Sparkles,
  UserCheck,
  CheckCircle2,
  RotateCcw,
} from "lucide-react";

interface DynamicTabularFormProps {
  disease: DiseaseResponse;
  onSuccess: (result: PredictionResponse) => void;
}

export const DynamicTabularForm: React.FC<DynamicTabularFormProps> = ({
  disease,
  onSuccess,
}) => {
  const [formData, setFormData] = useState<Record<string, any>>({});
  const [profile, setProfile] = useState<MedicalProfile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [profileLoaded, setProfileLoaded] = useState(false);

  // Attempt to load medical profile for smart autofill
  useEffect(() => {
    const loadProfile = async () => {
      try {
        const data = await profileService.getProfile();
        setProfile(data);
      } catch {
        // User may not have created a profile yet; silent fallback
      } finally {
        setProfileLoaded(true);
      }
    };
    loadProfile();
  }, []);

  const handleInputChange = (name: string, value: string, spec: TabularFeatureSpec) => {
    let parsedValue: any = value;
    if (value === "") {
      parsedValue = "";
    } else if (spec.data_type === "int") {
      parsedValue = parseInt(value, 10);
    } else if (spec.data_type === "float") {
      parsedValue = parseFloat(value);
    }

    setFormData((prev) => ({
      ...prev,
      [name]: parsedValue,
    }));
  };

  const handleAutofillFromProfile = () => {
    if (!profile) return;
    const updates: Record<string, any> = {};

    // 1. Calculate Age from date_of_birth
    if (profile.date_of_birth) {
      const birthYear = new Date(profile.date_of_birth).getFullYear();
      const currentYear = new Date().getFullYear();
      const calculatedAge = currentYear - birthYear;
      if (calculatedAge > 0) {
        updates["Age"] = calculatedAge;
        updates["age"] = calculatedAge;
      }
    }

    // 2. Calculate BMI from height & weight
    if (profile.height_cm && profile.weight_kg) {
      const heightM = profile.height_cm / 100;
      const bmi = parseFloat((profile.weight_kg / (heightM * heightM)).toFixed(1));
      updates["BMI"] = bmi;
      updates["bmi"] = bmi;
    }

    // 3. Map Gender/Sex
    if (profile.gender) {
      const g = profile.gender.toLowerCase();
      const sexVal = g === "male" ? 1 : 0;
      updates["Sex"] = sexVal;
      updates["sex"] = sexVal;
    }

    setFormData((prev) => ({
      ...prev,
      ...updates,
    }));
  };

  const handleReset = () => {
    setFormData({});
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    // Client-side validation for required fields
    for (const field of disease.required_fields) {
      const val = formData[field.name];
      if (field.required && (val === undefined || val === "" || isNaN(val))) {
        setError(`Please enter a valid value for '${field.display_name}'.`);
        return;
      }
      if (typeof val === "number") {
        if (field.min_value !== undefined && field.min_value !== null && val < field.min_value) {
          setError(`'${field.display_name}' cannot be less than ${field.min_value}.`);
          return;
        }
        if (field.max_value !== undefined && field.max_value !== null && val > field.max_value) {
          setError(`'${field.display_name}' cannot exceed ${field.max_value}.`);
          return;
        }
      }
    }

    setLoading(true);
    try {
      const result = await predictionService.predictTabular({
        disease_id: disease.id,
        inputs: formData,
      });
      onSuccess(result);
    } catch (err: any) {
      setError(err?.message || "Prediction execution failed. Please verify input parameters.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dynamic-form-wrapper">
      {/* Autofill from Profile Banner */}
      {profileLoaded && profile && (
        <div className="profile-autofill-banner">
          <div className="banner-text">
            <UserCheck size={18} color="#10b981" />
            <span>
              Connected medical profile for <strong>{profile.gender}, DOB: {profile.date_of_birth}</strong>
            </span>
          </div>
          <button
            type="button"
            onClick={handleAutofillFromProfile}
            className="btn btn-secondary btn-sm"
          >
            <Sparkles size={14} color="#38bdf8" />
            <span>Autofill Profile Biometrics</span>
          </button>
        </div>
      )}

      {error && (
        <div className="error-alert">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} className="modular-screening-form">
        <div className="form-fields-grid">
          {disease.required_fields.map((field) => {
            const currentValue = formData[field.name] ?? "";

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
                    onChange={(e) => handleInputChange(field.name, e.target.value, field)}
                    required={field.required}
                    className="form-input"
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
                  /* Numeric / Text Input */
                  <div className="input-range-wrapper">
                    <input
                      id={`field-${field.name}`}
                      type="number"
                      step={field.data_type === "float" ? "any" : "1"}
                      min={field.min_value !== null ? field.min_value : undefined}
                      max={field.max_value !== null ? field.max_value : undefined}
                      placeholder={
                        field.min_value !== null && field.max_value !== null
                          ? `e.g. ${field.min_value} - ${field.max_value}`
                          : "Enter value"
                      }
                      value={currentValue}
                      onChange={(e) => handleInputChange(field.name, e.target.value, field)}
                      required={field.required}
                      className="form-input"
                    />
                    {field.min_value !== null && field.max_value !== null && (
                      <span className="field-range-hint">
                        Range: {field.min_value} – {field.max_value} {field.unit || ""}
                      </span>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Form Action Controls */}
        <div className="form-submit-row">
          <button
            type="button"
            onClick={handleReset}
            disabled={loading}
            className="btn btn-secondary"
          >
            <RotateCcw size={16} />
            <span>Reset Form</span>
          </button>

          <button
            type="submit"
            disabled={loading}
            className="btn btn-primary btn-lg submit-screening-btn"
          >
            {loading ? (
              <>
                <Activity size={18} className="spinner-icon pulse-animation" />
                <span>Running ML Inference...</span>
              </>
            ) : (
              <>
                <CheckCircle2 size={18} />
                <span>Execute Clinical Screening</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};
