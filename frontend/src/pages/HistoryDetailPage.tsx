import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { historyService } from "../services/historyService";
import type { PredictionHistoryItem } from "../types";
import { formatDate, formatFeatureName } from "../utils/formatters";
import {
  Activity,
  ArrowLeft,
  ShieldAlert,
  Printer,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  Calendar,
  Layers,
  Cpu,
  FileText,
} from "lucide-react";

export const HistoryDetailPage: React.FC = () => {
  const { predictionId } = useParams<{ predictionId: string }>();
  const [record, setRecord] = useState<PredictionHistoryItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchRecord = async () => {
      if (!predictionId) return;
      try {
        setLoading(true);
        const data = await historyService.getPredictionById(predictionId);
        setRecord(data);
      } catch (err: any) {
        setError(err?.message || "Failed to load historical prediction audit.");
      } finally {
        setLoading(false);
      }
    };
    fetchRecord();
  }, [predictionId]);

  const handlePrint = () => {
    window.print();
  };

  if (loading) {
    return (
      <div className="container page-container">
        <div className="loading-state">
          <Activity className="spinner-icon pulse-animation" size={36} />
          <p>Retrieving immutable prediction record from database...</p>
        </div>
      </div>
    );
  }

  if (error || !record) {
    return (
      <div className="container page-container">
        <div className="error-alert">
          <ShieldAlert size={20} />
          <span>{error || "Prediction record not found."}</span>
        </div>
        <Link to="/history" className="btn btn-secondary" style={{ marginTop: "1rem" }}>
          <ArrowLeft size={16} />
          <span>Return to Prediction History</span>
        </Link>
      </div>
    );
  }

  const isPos = record.result.is_positive;
  const prob = record.result.probability ?? 0.5;
  const probPercent = (prob * 100).toFixed(1);

  return (
    <div className="container page-container">
      {/* Back and Action Navigation */}
      <div className="page-header">
        <div>
          <Link to="/history" className="back-link">
            <ArrowLeft size={16} />
            <span>Back to Prediction History</span>
          </Link>
          <h1 className="page-title">{record.disease_display_name} — Clinical Audit Report</h1>
          <p className="page-subtitle">
            <Calendar size={14} style={{ display: "inline", marginRight: "4px", verticalAlign: "middle" }} />
            Screening evaluated on {formatDate(record.created_at)}
          </p>
        </div>

        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
          <Link to={`/reports/${record.id}`} className="btn btn-primary">
            <FileText size={16} />
            <span>View PDF Report</span>
          </Link>
          <button onClick={handlePrint} className="btn btn-secondary">
            <Printer size={16} />
            <span>Print Summary</span>
          </button>
          <Link to={`/predict/${record.disease}`} className="btn btn-secondary">
            <Sparkles size={16} />
            <span>New Screening</span>
          </Link>
        </div>
      </div>

      {/* Main Report Card */}
      <div className="prediction-result-card glass-card">
        {/* Telemetry Header */}
        <div className="result-card-header">
          <div className="header-badge-row">
            <span className="badge badge-purple">
              <Cpu size={12} style={{ marginRight: "4px" }} />
              {record.model.model_type} ({record.model.version})
            </span>
            <span className="badge badge-cyan">
              <Layers size={12} style={{ marginRight: "4px" }} />
              {record.input_type}
            </span>
            <span className="result-timestamp" style={{ fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>
              Audit ID: {record.id}
            </span>
          </div>
        </div>

        {/* Outcome Banner */}
        <div className={`outcome-banner ${isPos ? "outcome-positive" : "outcome-negative"}`}>
          <div className="outcome-icon-badge">
            {isPos ? (
              <AlertTriangle size={32} className="outcome-icon" />
            ) : (
              <CheckCircle2 size={32} className="outcome-icon" />
            )}
          </div>
          <div className="outcome-text-block">
            <div className="outcome-classification">{record.result.prediction}</div>
            <div className="outcome-subtext">
              {isPos
                ? "Elevated statistical risk pattern identified across screening biometrics."
                : "No abnormal clinical risk patterns detected based on submitted parameters."}
            </div>
          </div>
        </div>

        {/* Calibrated Probability Gauge */}
        {record.result.probability !== undefined && (
          <div className="probability-gauge-box">
            <div className="gauge-label-row">
              <span className="gauge-title">Calibrated Risk Probability</span>
              <span className="gauge-percentage">{probPercent}%</span>
            </div>

            <div className="gauge-track">
              <div
                className={`gauge-fill ${isPos ? "fill-positive" : "fill-negative"}`}
                style={{ width: `${Math.min(100, Math.max(0, prob * 100))}%` }}
              />
            </div>

            <div className="gauge-footer-row">
              <span>Low Risk (0%)</span>
              {record.model.threshold !== undefined && (
                <span className="threshold-marker">
                  Decision Threshold: {(record.model.threshold * 100).toFixed(0)}%
                </span>
              )}
              <span>High Risk (100%)</span>
            </div>
          </div>
        )}

        {/* Clinical Narrative Explanation */}
        {record.explanation && (
          <div className="explanation-section">
            <div className="section-subtitle-row">
              <Sparkles size={16} color="#38bdf8" />
              <span className="section-label">AI Decision Support Narrative</span>
            </div>
            <p className="explanation-paragraph">{record.explanation}</p>
          </div>
        )}

        {/* Biometric Snapshot */}
        {record.input_data && Object.keys(record.input_data).length > 0 && (
          <div className="inputs-snapshot-section">
            <h4 className="snapshot-title">Evaluated Biometric Features Snapshot</h4>
            <div className="snapshot-grid">
              {Object.entries(record.input_data).map(([key, val]) => (
                <div key={key} className="snapshot-item">
                  <span className="snapshot-key">{formatFeatureName(key)}</span>
                  <span className="snapshot-val">{String(val)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Clinical Disclaimer */}
        <div className="result-disclaimer-note">
          <ShieldAlert size={16} className="disclaimer-mini-icon" />
          <span>
            This document is a computer-generated screening report from AI Medical Diagnosis Assistant. It is intended solely for clinical decision support and patient record-keeping. It does not replace comprehensive medical diagnosis by a licensed clinician.
          </span>
        </div>
      </div>
    </div>
  );
};
