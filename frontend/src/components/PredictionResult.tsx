import React from "react";
import { Link, useNavigate } from "react-router-dom";
import type { PredictionResponse } from "../types";
import { formatDate } from "../utils/formatters";
import {
  AlertTriangle,
  CheckCircle2,
  Cpu,
  FileText,
  History,
  Home,
  Info,
  Layers,
  Printer,
  RotateCcw,
  ShieldAlert,
  Sparkles,
} from "lucide-react";

interface PredictionResultProps {
  result: PredictionResponse;
  predictionId?: string;
  onPerformAnother?: () => void;
}

export const PredictionResult: React.FC<PredictionResultProps> = ({
  result,
  predictionId,
  onPerformAnother,
}) => {
  const navigate = useNavigate();
  const isPositive = result.is_positive;
  const hasProbability = result.probability !== undefined && result.probability !== null;
  const probPercent = hasProbability ? ((result.probability as number) * 100).toFixed(2) : null;

  const handlePrint = () => {
    window.print();
  };

  const handleNewScreening = () => {
    if (onPerformAnother) {
      onPerformAnother();
    } else {
      navigate(`/predict/${result.disease_id}`);
    }
  };

  return (
    <div className="prediction-result-card glass-card">
      {/* Telemetry Header */}
      <div className="result-card-header">
        <div className="header-badge-row">
          <span className="badge badge-purple">
            <Cpu size={12} style={{ marginRight: "4px" }} />
            {result.model_type} ({result.model_version})
          </span>
          <span className="badge badge-cyan">
            <Layers size={12} style={{ marginRight: "4px" }} />
            AI Screening Model
          </span>
          <span className="result-timestamp">{formatDate(result.timestamp)}</span>
        </div>

        <h2 className="result-disease-title">{result.disease_display_name} — Screening Result</h2>
      </div>

      {/* Outcome Classification Banner */}
      <div className={`outcome-banner ${isPositive ? "outcome-positive" : "outcome-negative"}`}>
        <div className="outcome-icon-badge">
          {isPositive ? (
            <AlertTriangle size={36} className="outcome-icon" />
          ) : (
            <CheckCircle2 size={36} className="outcome-icon" />
          )}
        </div>
        <div className="outcome-text-block">
          <div className="outcome-classification">{result.prediction_label}</div>
          <div className="outcome-subtext">
            {isPositive
              ? "Elevated statistical risk pattern identified across submitted clinical indicators."
              : "No significant screening risk patterns identified based on submitted parameters."}
          </div>
        </div>
      </div>

      {/* Model Probability Gauge */}
      {hasProbability ? (
        <div className="probability-gauge-box">
          <div className="gauge-label-row">
            <span className="gauge-title">Model Screening Probability</span>
            <span className="gauge-percentage">{probPercent}%</span>
          </div>

          <div className="gauge-track">
            <div
              className={`gauge-fill ${isPositive ? "fill-positive" : "fill-negative"}`}
              style={{ width: `${Math.min(100, Math.max(0, (result.probability as number) * 100))}%` }}
            />
          </div>

          <div className="gauge-footer-row">
            <span>Low Risk (0%)</span>
            {result.decision_threshold !== undefined && result.decision_threshold !== null && (
              <span className="threshold-marker">
                Decision Threshold: {(result.decision_threshold * 100).toFixed(0)}%
              </span>
            )}
            <span>High Risk (100%)</span>
          </div>
        </div>
      ) : (
        <div className="probability-gauge-box" style={{ padding: "0.875rem 1.25rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", color: "var(--text-dim)", fontSize: "0.875rem" }}>
            <Info size={16} />
            <span>Calibrated probability score not applicable for this model classification architecture.</span>
          </div>
        </div>
      )}

      {/* AI Decision Support Explanation */}
      {result.explanation && (
        <div className="explanation-section">
          <div className="section-subtitle-row">
            <Sparkles size={16} color="#38bdf8" />
            <span className="section-label">AI Decision Support Narrative</span>
          </div>
          <p className="explanation-paragraph">{result.explanation}</p>
        </div>
      )}

      {/* Purpose & Limitations Note */}
      {result.clinical_purpose && (
        <div style={{ marginBottom: "1.5rem", padding: "1rem 1.25rem", background: "rgba(15, 23, 42, 0.4)", borderRadius: "var(--radius-md)", border: "1px solid var(--border-glass)" }}>
          <div style={{ fontSize: "0.75rem", color: "var(--text-dim)", textTransform: "uppercase", fontWeight: 700, letterSpacing: "0.05em", marginBottom: "0.25rem" }}>
            Clinical Scope & Limitations
          </div>
          <div style={{ fontSize: "0.875rem", color: "var(--text-muted)" }}>
            {result.clinical_purpose} {result.limitations ? `— ${result.limitations}` : ""}
          </div>
        </div>
      )}

      {/* Medical Disclaimer Banner */}
      <div className="result-disclaimer-note">
        <ShieldAlert size={18} className="disclaimer-mini-icon" />
        <span>
          {result.disclaimer ||
            "Screening result, not medical diagnosis. This automated assessment is intended solely for clinical decision-support and educational purposes. Always consult a qualified medical professional for diagnosis and treatment."}
        </span>
      </div>

      {/* Navigation & Action Controls */}
      <div className="result-actions-row">
        <Link to="/dashboard" className="btn btn-secondary">
          <Home size={16} />
          <span>Dashboard</span>
        </Link>

        <button type="button" onClick={handleNewScreening} className="btn btn-secondary">
          <RotateCcw size={16} />
          <span>New Screening</span>
        </button>

        {predictionId && (
          <Link to={`/reports/${predictionId}`} className="btn btn-primary">
            <FileText size={16} />
            <span>View Clinical Report</span>
          </Link>
        )}

        <button type="button" onClick={handlePrint} className="btn btn-secondary">
          <Printer size={16} />
          <span>Print Summary</span>
        </button>

        <Link to="/history" className="btn btn-secondary">
          <History size={16} />
          <span>View in History</span>
        </Link>
      </div>
    </div>
  );
};
