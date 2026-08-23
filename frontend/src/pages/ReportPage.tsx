import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { reportService } from "../services/reportService";
import type { MedicalReportResponse } from "../types";
import { formatDate, formatFeatureName } from "../utils/formatters";
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Calendar,
  CheckCircle2,
  Cpu,
  Download,
  ExternalLink,
  FileText,
  Printer,
  ShieldAlert,
  Sparkles,
  User,
} from "lucide-react";

export const ReportPage: React.FC = () => {
  const { reportId, predictionId } = useParams<{ reportId?: string; predictionId?: string }>();
  const activeId = predictionId || reportId;

  const [report, setReport] = useState<MedicalReportResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchReport = async () => {
      if (!activeId) return;
      try {
        setLoading(true);
        setError(null);
        const data = await reportService.getReport(activeId);
        setReport(data);
      } catch (err: any) {
        setError(err?.message || "Failed to retrieve clinical screening report.");
      } finally {
        setLoading(false);
      }
    };

    fetchReport();
  }, [activeId]);

  const handleDownload = async () => {
    if (!report) return;
    try {
      setDownloading(true);
      const filename = `${report.disease}_${report.report_id}.pdf`;
      await reportService.downloadReportPdf(report.prediction_id, filename);
    } catch (err: any) {
      alert(`Download failed: ${err?.message || "Unable to download PDF"}`);
    } finally {
      setDownloading(false);
    }
  };

  const handlePrint = () => {
    window.print();
  };

  if (loading) {
    return (
      <div className="container page-container">
        <div className="loading-state">
          <Activity className="spinner-icon pulse-animation" size={36} />
          <p>Generating and retrieving clinical screening report #{activeId}...</p>
        </div>
      </div>
    );
  }

  if (error || !report) {
    return (
      <div className="container page-container">
        <div className="error-alert">
          <ShieldAlert size={20} />
          <span>{error || "Requested medical report could not be found or you do not have permission to view it."}</span>
        </div>
        <div style={{ marginTop: "1.5rem" }}>
          <Link to="/history" className="btn btn-secondary">
            <ArrowLeft size={16} />
            <span>Return to History</span>
          </Link>
        </div>
      </div>
    );
  }

  const isPos = report.is_positive;
  const prob = report.probability ?? 0.5;
  const probPercent = (prob * 100).toFixed(1);

  return (
    <div className="container page-container">
      {/* Navigation & Header */}
      <div className="page-header" style={{ marginBottom: "2rem" }}>
        <div>
          <Link to="/history" className="back-link">
            <ArrowLeft size={16} />
            <span>Back to Prediction History</span>
          </Link>
          <h1 className="page-title">{report.disease_display_name} — Clinical Report</h1>
          <p className="page-subtitle">
            <Calendar size={14} style={{ display: "inline", marginRight: "4px", verticalAlign: "middle" }} />
            Document ID: <strong style={{ color: "var(--accent-cyan)", fontFamily: "var(--font-mono)" }}>{report.report_id}</strong> • Generated on {formatDate(report.created_at)}
          </p>
        </div>

        {/* Action Controls */}
        <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "center" }}>
          <button
            onClick={handleDownload}
            disabled={downloading}
            className="btn btn-primary"
            title="Download PDF report to device"
          >
            {downloading ? (
              <Activity className="spinner-icon pulse-animation" size={16} />
            ) : (
              <Download size={16} />
            )}
            <span>{downloading ? "Downloading..." : "Download PDF"}</span>
          </button>

          {report.download_url && (
            <a
              href={report.download_url}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-secondary"
              title="Open PDF in new browser tab"
            >
              <ExternalLink size={16} />
              <span>Open PDF</span>
            </a>
          )}

          <button onClick={handlePrint} className="btn btn-secondary" title="Print document">
            <Printer size={16} />
            <span>Print</span>
          </button>
        </div>
      </div>

      {/* Primary KPI Grid */}
      <div className="stats-grid" style={{ marginBottom: "2rem" }}>
        {/* Patient / Session Info */}
        <div className="stat-card glass-card">
          <div className="stat-icon-wrapper" style={{ background: "rgba(56, 189, 248, 0.1)" }}>
            <User size={20} color="#38bdf8" />
          </div>
          <div>
            <div className="stat-label">Patient / Practitioner</div>
            <div className="stat-value" style={{ fontSize: "1.125rem", wordBreak: "break-all" }}>
              {report.user_name || "User"}
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-dim)", marginTop: "2px" }}>
              {report.user_email || "Authenticated Account"}
            </div>
          </div>
        </div>

        {/* Modality & Engine */}
        <div className="stat-card glass-card">
          <div className="stat-icon-wrapper" style={{ background: "rgba(139, 92, 246, 0.1)" }}>
            <Cpu size={20} color="#8b5cf6" />
          </div>
          <div>
            <div className="stat-label">Model Architecture</div>
            <div className="stat-value" style={{ fontSize: "1.125rem" }}>
              {report.model.model_type}
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-dim)", marginTop: "2px" }}>
              Version: {report.model.version} • {report.input_type.toUpperCase()}
            </div>
          </div>
        </div>

        {/* Screening Outcome Status */}
        <div className="stat-card glass-card">
          <div className="stat-icon-wrapper" style={{ background: isPos ? "rgba(244, 63, 94, 0.1)" : "rgba(16, 185, 129, 0.1)" }}>
            {isPos ? <AlertTriangle size={20} color="#f43f5e" /> : <CheckCircle2 size={20} color="#10b981" />}
          </div>
          <div>
            <div className="stat-label">Screening Finding</div>
            <div className="stat-value" style={{ fontSize: "1.125rem", color: isPos ? "#fda4af" : "#86efac" }}>
              {report.prediction}
            </div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-dim)", marginTop: "2px" }}>
              {report.probability !== undefined && report.probability !== null
                ? `Probability: ${probPercent}%`
                : "Categorical Assessment"}
            </div>
          </div>
        </div>
      </div>

      {/* Main Report Body */}
      <div className="prediction-result-card glass-card">
        {/* Outcome Banner */}
        <div className={`outcome-banner ${isPos ? "outcome-positive" : "outcome-negative"}`}>
          <div className="outcome-icon-badge">
            {isPos ? <AlertTriangle size={36} className="outcome-icon" /> : <CheckCircle2 size={36} className="outcome-icon" />}
          </div>
          <div className="outcome-text-block">
            <div className="outcome-classification">{report.prediction}</div>
            <div className="outcome-subtext">
              {isPos
                ? "Elevated clinical risk pattern identified across evaluated screening parameters."
                : "No abnormal clinical risk patterns detected based on submitted biomarkers."}
            </div>
          </div>
        </div>

        {/* Probability Gauge */}
        {report.probability !== undefined && report.probability !== null && (
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
              {report.model.threshold !== undefined && (
                <span className="threshold-marker">
                  Decision Threshold: {(report.model.threshold * 100).toFixed(0)}%
                </span>
              )}
              <span>High Risk (100%)</span>
            </div>
          </div>
        )}

        {/* AI Decision Support Narrative */}
        {report.explanation && (
          <div className="explanation-section">
            <div className="section-subtitle-row">
              <Sparkles size={16} color="#38bdf8" />
              <span className="section-label">AI Decision-Support Narrative</span>
            </div>
            <p className="explanation-paragraph">{report.explanation}</p>
          </div>
        )}

        {/* Evaluated Inputs Section */}
        {report.input_summary && Object.keys(report.input_summary).length > 0 && (
          <div className="inputs-snapshot-section">
            <h4 className="snapshot-title">Evaluated Clinical Parameters & Scan Telemetry</h4>
            <div className="snapshot-grid">
              {Object.entries(report.input_summary).map(([key, val]) => (
                <div key={key} className="snapshot-item">
                  <span className="snapshot-key">{formatFeatureName(key)}</span>
                  <span className="snapshot-val">{String(val)}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Embedded PDF Document Preview Container */}
        {report.download_url && (
          <div style={{ marginTop: "2rem", marginBottom: "1.5rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontWeight: 700, fontSize: "0.9375rem" }}>
                <FileText size={18} color="#38bdf8" />
                <span>Generated PDF Document Preview</span>
              </div>
              <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>
                {report.storage_key}
              </span>
            </div>

            <div style={{ width: "100%", height: "520px", borderRadius: "var(--radius-md)", overflow: "hidden", border: "1px solid var(--border-glass)", background: "#0f172a" }}>
              <iframe
                src={report.download_url}
                title="Clinical PDF Screening Report"
                width="100%"
                height="100%"
                style={{ border: "none" }}
              />
            </div>
          </div>
        )}

        {/* Non-Diagnostic Clinical Disclaimer */}
        <div className="result-disclaimer-note">
          <ShieldAlert size={18} className="disclaimer-mini-icon" />
          <span>
            {report.disclaimer ||
              "This automated screening report is generated by artificial intelligence decision-support models. It is intended solely for preliminary triaging and record-keeping. It does NOT constitute a confirmed medical diagnosis. Always consult a licensed medical practitioner."}
          </span>
        </div>
      </div>
    </div>
  );
};
