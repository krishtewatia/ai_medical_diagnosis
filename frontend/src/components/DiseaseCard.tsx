import React from "react";
import { Link } from "react-router-dom";
import type { DiseaseResponse } from "../types";
import {
  Activity,
  ArrowRight,
  Brain,
  Cpu,
  FileSpreadsheet,
  HeartPulse,
  ImageIcon,
  Sparkles,
} from "lucide-react";

interface DiseaseCardProps {
  disease: DiseaseResponse;
}

export const DiseaseCard: React.FC<DiseaseCardProps> = ({ disease }) => {
  // Select icon based on disease id or category
  const renderIcon = () => {
    switch (disease.id) {
      case "diabetes":
        return <HeartPulse size={24} color="#38bdf8" />;
      case "heart_disease":
        return <Activity size={24} color="#f43f5e" />;
      default:
        return disease.category === "image" ? (
          <Brain size={24} color="#8b5cf6" />
        ) : (
          <Sparkles size={24} color="#06b6d4" />
        );
    }
  };

  return (
    <div className="disease-card glass-card">
      <div className="disease-card-header">
        <div className="disease-card-icon">{renderIcon()}</div>
        <div className="badge-group">
          <span className={`badge ${disease.category === "tabular" ? "badge-cyan" : "badge-emerald"}`}>
            {disease.category === "tabular" ? (
              <FileSpreadsheet size={12} style={{ marginRight: "3px" }} />
            ) : (
              <ImageIcon size={12} style={{ marginRight: "3px" }} />
            )}
            {disease.category}
          </span>
          {disease.model_info?.version && (
            <span className="badge badge-purple" style={{ fontSize: "0.6875rem" }}>
              <Cpu size={10} style={{ marginRight: "3px" }} />
              {disease.model_info.version}
            </span>
          )}
        </div>
      </div>

      <h3 className="disease-card-title">{disease.display_name}</h3>
      <p className="disease-card-description">{disease.description}</p>

      {/* Dynamic Specifications Box */}
      <div className="disease-specs-box">
        <div className="spec-row">
          <span className="spec-label">Algorithm:</span>
          <span className="spec-value">{disease.model_info?.model_type || "Classifier"}</span>
        </div>
        <div className="spec-row">
          <span className="spec-label">Input Specification:</span>
          <span className="spec-value">
            {disease.category === "tabular"
              ? `${disease.required_fields.length} Biometric Features`
              : "Medical Imaging Scan"}
          </span>
        </div>
        {disease.model_info?.threshold !== undefined && (
          <div className="spec-row">
            <span className="spec-label">Decision Threshold:</span>
            <span className="spec-value">{(disease.model_info.threshold * 100).toFixed(0)}%</span>
          </div>
        )}
      </div>

      {/* Dynamic Route: /predict/:diseaseId */}
      <Link to={`/predict/${disease.id}`} className="btn btn-primary btn-block">
        <span>Launch Screening</span>
        <ArrowRight size={16} />
      </Link>
    </div>
  );
};
