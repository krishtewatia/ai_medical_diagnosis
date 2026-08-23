import React, { useEffect, useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import { diseaseService } from "../services/diseaseService";
import type { DiseaseResponse } from "../types";
import { DiseaseForm } from "../components/DiseaseForm";
import { ImageUploadForm } from "../components/ImageUploadForm";
import {
  Activity,
  ArrowLeft,
  ShieldAlert,
  HeartPulse,
  Brain,
  Sparkles,
  ImageIcon,
} from "lucide-react";

export const PredictionPage: React.FC = () => {
  const { diseaseId, disease: diseaseParam } = useParams<{ diseaseId?: string; disease?: string }>();
  const activeDiseaseId = diseaseId || diseaseParam;
  const navigate = useNavigate();

  const [disease, setDisease] = useState<DiseaseResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDisease = async () => {
    if (!activeDiseaseId) return;
    try {
      setLoading(true);
      setError(null);
      const data = await diseaseService.getDiseaseById(activeDiseaseId);
      setDisease(data);
    } catch (err: any) {
      setError(err?.message || `Failed to load configuration for '${activeDiseaseId}'.`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDisease();
  }, [activeDiseaseId]);

  if (loading) {
    return (
      <div className="container page-container">
        <div className="loading-state">
          <Activity className="spinner-icon pulse-animation" size={36} />
          <p>Querying dynamic disease specification from GET /diseases/{activeDiseaseId}...</p>
        </div>
      </div>
    );
  }

  if (error || !disease) {
    return (
      <div className="container page-container">
        <div className="error-alert">
          <ShieldAlert size={20} />
          <span>{error || `Disease module '${activeDiseaseId}' not found.`}</span>
        </div>
        <Link to="/diseases" className="btn btn-secondary" style={{ marginTop: "1rem" }}>
          <ArrowLeft size={16} />
          <span>Return to Catalog</span>
        </Link>
      </div>
    );
  }

  const isImageScreening = disease.category === "image" || disease.input_type === "image_upload";

  return (
    <div className="container page-container">
      {/* Header */}
      <div className="page-header">
        <div>
          <Link to="/dashboard" className="back-link">
            <ArrowLeft size={16} />
            <span>Back to Dashboard</span>
          </Link>
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginTop: "0.5rem" }}>
            <div className="disease-card-icon" style={{ width: "40px", height: "40px" }}>
              {disease.id === "diabetes" ? (
                <HeartPulse size={22} color="#38bdf8" />
              ) : disease.id === "heart_disease" ? (
                <Activity size={22} color="#f43f5e" />
              ) : (
                <Brain size={22} color="#8b5cf6" />
              )}
            </div>
            <div>
              <h1 className="page-title">{disease.display_name}</h1>
              <p className="page-subtitle">{disease.description}</p>
            </div>
          </div>
        </div>

        {/* Model Spec Badges */}
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
          <span className={`badge ${isImageScreening ? "badge-emerald" : "badge-cyan"}`}>
            {disease.category}
          </span>
          <span className="badge badge-purple">
            {disease.model_info?.model_type || "Model"} ({disease.model_info?.version || "v1"})
          </span>
          <span className="badge badge-rose">
            {isImageScreening
              ? `${disease.image_spec?.target_dimensions.join("×") || "224×224"} Tensor`
              : `${disease.required_fields.length} Biometric Features`}
          </span>
        </div>
      </div>

      {/* Dynamic Screening Form Container */}
      <div className="screening-workflow-card glass-card">
        <div className="workflow-card-header">
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            {isImageScreening ? (
              <ImageIcon size={18} color="#10b981" />
            ) : (
              <Sparkles size={18} color="#38bdf8" />
            )}
            <h2 style={{ fontSize: "1.125rem", fontWeight: 700 }}>
              {isImageScreening
                ? "Medical Imaging Screening Upload"
                : "Dynamic Clinical Input Parameters"}
            </h2>
          </div>
          <span style={{ fontSize: "0.8125rem", color: "var(--text-dim)" }}>
            {isImageScreening
              ? disease.id === "brain_tumor"
                ? "Axial brain MRI (.png, .jpg, .jpeg)"
                : "Frontal chest radiograph (.png, .jpg, .jpeg)"
              : `${disease.required_fields.length} dynamic fields configured`}
          </span>
        </div>

        {isImageScreening ? (
          <ImageUploadForm
            config={disease}
            onSuccess={(result) => navigate("/result", { state: { result } })}
          />
        ) : (
          <DiseaseForm
            config={disease}
            onSuccess={(result) => navigate("/result", { state: { result } })}
          />
        )}
      </div>
    </div>
  );
};
