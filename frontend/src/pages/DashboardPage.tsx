import React, { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { diseaseService } from "../services/diseaseService";
import { historyService } from "../services/historyService";
import type { DiseaseResponse, PredictionHistoryItem } from "../types";
import { DiseaseCard } from "../components/DiseaseCard";
import { formatDate, formatProbability } from "../utils/formatters";
import {
  Activity,
  AlertCircle,
  ArrowRight,
  Calendar,
  FileText,
  Grid,
  History,
  LogOut,
  RefreshCw,
  Sparkles,
  User,
  Zap,
} from "lucide-react";

export const DashboardPage: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const [diseases, setDiseases] = useState<DiseaseResponse[]>([]);
  const [recentPredictions, setRecentPredictions] = useState<PredictionHistoryItem[]>([]);
  const [totalHistoryCount, setTotalHistoryCount] = useState<number>(0);
  
  const [loadingDiseases, setLoadingDiseases] = useState(true);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [diseaseError, setDiseaseError] = useState<string | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);

  const fetchDashboardData = async () => {
    // 1. Fetch Dynamic Active Diseases
    setLoadingDiseases(true);
    setDiseaseError(null);
    try {
      const diseaseList = await diseaseService.getAllDiseases();
      setDiseases(diseaseList);
    } catch (err: any) {
      setDiseaseError(err?.message || "Failed to load clinical disease models.");
    } finally {
      setLoadingDiseases(false);
    }

    // 2. Fetch Recent Predictions (limit 4)
    setLoadingHistory(true);
    setHistoryError(null);
    try {
      const historyRes = await historyService.getHistory({ limit: 4 });
      setRecentPredictions(historyRes.items);
      setTotalHistoryCount(historyRes.total);
    } catch (err: any) {
      setHistoryError(err?.message || "Failed to retrieve recent prediction history.");
    } finally {
      setLoadingHistory(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <div className="container page-container">
      {/* 1. Welcome & Practitioner Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">
            Welcome back, {user?.name || "Medical Practitioner"}!
          </h1>
          <p className="page-subtitle">
            Authenticated clinical workstation session active for <strong>{user?.email}</strong>
          </p>
        </div>

        {/* Global Quick Action Buttons */}
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
          <button
            onClick={fetchDashboardData}
            disabled={loadingDiseases || loadingHistory}
            className="btn btn-secondary btn-sm"
            title="Refresh dashboard feeds"
          >
            <RefreshCw
              size={14}
              className={loadingDiseases || loadingHistory ? "spin-animation" : ""}
            />
            <span>Refresh</span>
          </button>
          <button
            onClick={handleLogout}
            className="btn btn-secondary btn-sm"
            title="Sign out of current workstation session"
          >
            <LogOut size={14} />
            <span>Sign Out</span>
          </button>
        </div>
      </div>

      {/* 2. Workstation Quick Action Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "1rem", marginBottom: "2.5rem" }}>
        <Link
          to="/diseases"
          className="glass-card"
          style={{ padding: "1.25rem", textDecoration: "none", color: "inherit", display: "flex", alignItems: "center", gap: "1rem" }}
        >
          <div style={{ padding: "0.75rem", borderRadius: "var(--radius-md)", background: "rgba(56, 189, 248, 0.15)", border: "1px solid rgba(56, 189, 248, 0.3)" }}>
            <Zap size={22} color="#38bdf8" />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: "0.9375rem" }}>Start Screening</div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              {loadingDiseases ? "Querying..." : `${diseases.length} models ready`}
            </div>
          </div>
        </Link>

        <Link
          to="/history"
          className="glass-card"
          style={{ padding: "1.25rem", textDecoration: "none", color: "inherit", display: "flex", alignItems: "center", gap: "1rem" }}
        >
          <div style={{ padding: "0.75rem", borderRadius: "var(--radius-md)", background: "rgba(16, 185, 129, 0.15)", border: "1px solid rgba(16, 185, 129, 0.3)" }}>
            <History size={22} color="#10b981" />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: "0.9375rem" }}>Prediction History</div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
              {loadingHistory ? "Querying..." : `${totalHistoryCount} total evaluations`}
            </div>
          </div>
        </Link>

        <Link
          to="/profile"
          className="glass-card"
          style={{ padding: "1.25rem", textDecoration: "none", color: "inherit", display: "flex", alignItems: "center", gap: "1rem" }}
        >
          <div style={{ padding: "0.75rem", borderRadius: "var(--radius-md)", background: "rgba(139, 92, 246, 0.15)", border: "1px solid rgba(139, 92, 246, 0.3)" }}>
            <User size={22} color="#8b5cf6" />
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: "0.9375rem" }}>Medical Profile</div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Manage baseline vitals</div>
          </div>
        </Link>
      </div>

      {/* 3. Available AI Screening Modules Section */}
      <section className="modules-section" style={{ marginBottom: "3rem" }}>
        <div className="section-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
          <div>
            <h2 className="section-title" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <Sparkles size={20} color="#38bdf8" />
              <span>Available AI Screening Modules</span>
            </h2>
            <p className="section-subtitle">
              Dynamically discovered from FastAPI model registry
            </p>
          </div>

          <Link to="/diseases" className="btn btn-secondary btn-sm">
            <Grid size={14} />
            <span>View Catalog</span>
          </Link>
        </div>

        {diseaseError && (
          <div className="error-alert">
            <AlertCircle size={18} />
            <span>{diseaseError}</span>
          </div>
        )}

        {loadingDiseases ? (
          <div className="loading-state">
            <Activity className="spinner-icon pulse-animation" size={36} />
            <p>Querying dynamic disease registry...</p>
          </div>
        ) : diseases.length === 0 ? (
          <div className="glass-card" style={{ padding: "3rem", textAlign: "center" }}>
            <Activity size={40} color="var(--text-dim)" style={{ marginBottom: "1rem" }} />
            <h3 style={{ fontSize: "1.125rem", fontWeight: 700, marginBottom: "0.5rem" }}>No Active Disease Models</h3>
            <p style={{ color: "var(--text-muted)", fontSize: "0.875rem" }}>
              The backend returned an empty disease list.
            </p>
          </div>
        ) : (
          <div className="disease-cards-grid">
            {diseases.map((disease) => (
              <DiseaseCard key={disease.id} disease={disease} />
            ))}
          </div>
        )}
      </section>

      {/* 4. Recent Predictions Feed */}
      <section className="recent-predictions-section">
        <div className="section-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
          <div>
            <h2 className="section-title" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <History size={20} color="#10b981" />
              <span>Recent Screenings & Audit Log</span>
            </h2>
            <p className="section-subtitle">
              Your latest diagnostic predictions recorded in MongoDB
            </p>
          </div>

          {totalHistoryCount > 0 && (
            <Link to="/history" className="btn btn-secondary btn-sm">
              <span>View All History ({totalHistoryCount})</span>
              <ArrowRight size={14} />
            </Link>
          )}
        </div>

        {historyError && (
          <div className="error-alert">
            <AlertCircle size={18} />
            <span>{historyError}</span>
          </div>
        )}

        {loadingHistory ? (
          <div className="loading-state">
            <Activity className="spinner-icon pulse-animation" size={32} />
            <p>Retrieving recent prediction logs...</p>
          </div>
        ) : recentPredictions.length === 0 ? (
          <div className="glass-card" style={{ padding: "2.5rem 1.5rem", textAlign: "center" }}>
            <History size={36} color="var(--text-dim)" style={{ marginBottom: "0.75rem" }} />
            <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "0.375rem" }}>
              No Screenings Performed Yet
            </h3>
            <p style={{ color: "var(--text-muted)", fontSize: "0.8125rem", maxWidth: "420px", margin: "0 auto 1.5rem" }}>
              Launch your first AI screening above to view your persistent history log and clinical telemetry here.
            </p>
            <Link to="/diseases" className="btn btn-primary btn-sm">
              <Sparkles size={14} />
              <span>Launch First Screening</span>
            </Link>
          </div>
        ) : (
          <div className="history-list">
            {recentPredictions.map((item) => {
              const isPos = item.result.is_positive;
              return (
                <div key={item.id} className="history-card glass-card">
                  <div className="history-card-main">
                    <div className="history-card-header">
                      <span className="history-disease-badge">{item.disease_display_name}</span>
                      <span className="badge badge-purple" style={{ fontSize: "0.6875rem" }}>
                        {item.model.model_type} ({item.model.version})
                      </span>
                      <span className="history-date">
                        <Calendar size={13} style={{ display: "inline", marginRight: "4px", verticalAlign: "middle" }} />
                        {formatDate(item.created_at)}
                      </span>
                    </div>

                    <div className="history-result-summary">
                      <span className={`result-indicator ${isPos ? "result-positive" : "result-negative"}`}>
                        {item.result.prediction}
                      </span>
                      {item.result.probability !== undefined && (
                        <span className="result-probability">
                          Calibrated Risk: <strong>{formatProbability(item.result.probability)}</strong>
                        </span>
                      )}
                      {item.explanation && (
                        <span style={{ fontSize: "0.8125rem", color: "var(--text-dim)", maxWidth: "300px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          — {item.explanation}
                        </span>
                      )}
                    </div>
                  </div>

                  <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                    <Link
                      to={`/reports/${item.id}`}
                      className="btn btn-primary btn-sm"
                      title="View & Download clinical PDF report"
                    >
                      <FileText size={13} />
                      <span>Report</span>
                    </Link>

                    <Link
                      to={`/history/${item.id}`}
                      className="btn btn-secondary btn-sm"
                      title="View complete clinical audit log"
                    >
                      <span>Audit</span>
                      <ArrowRight size={13} />
                    </Link>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
};
