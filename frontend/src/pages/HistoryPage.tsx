import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { historyService } from "../services/historyService";
import { diseaseService } from "../services/diseaseService";
import type { DiseaseResponse, PredictionHistoryItem } from "../types";
import { formatDate, formatProbability } from "../utils/formatters";
import {
  History,
  Activity,
  AlertCircle,
  ArrowRight,
  Filter,
  ChevronLeft,
  ChevronRight,
  Sparkles,
  Calendar,
  FileText,
} from "lucide-react";

export const HistoryPage: React.FC = () => {
  const [items, setItems] = useState<PredictionHistoryItem[]>([]);
  const [diseases, setDiseases] = useState<DiseaseResponse[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [diseaseFilter, setDiseaseFilter] = useState<string>("");
  const [page, setPage] = useState(0);
  const [pageSize] = useState(10);
  const [error, setError] = useState<string | null>(null);

  // Load available disease options for the filter
  useEffect(() => {
    const loadDiseaseFilters = async () => {
      try {
        const list = await diseaseService.getAllDiseases();
        setDiseases(list);
      } catch (err) {
        console.error("Failed to load disease filter list:", err);
      }
    };
    loadDiseaseFilters();
  }, []);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await historyService.getHistory({
        disease: diseaseFilter || undefined,
        skip: page * pageSize,
        limit: pageSize,
      });
      setItems(res.items);
      setTotal(res.total);
    } catch (err: any) {
      setError(err?.message || "Failed to retrieve historical predictions.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, [diseaseFilter, page]);

  const totalPages = Math.ceil(total / pageSize);

  const handleFilterChange = (val: string) => {
    setDiseaseFilter(val);
    setPage(0);
  };

  return (
    <div className="container page-container">
      {/* Header & Controls */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Prediction History & Audit Log</h1>
          <p className="page-subtitle">
            Immutable clinical screening records isolated to your account ({total} total evaluations)
          </p>
        </div>

        {/* Dynamic Disease Filter */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", background: "rgba(15, 23, 42, 0.8)", padding: "0.25rem 0.75rem", borderRadius: "var(--radius-md)", border: "1px solid var(--border-subtle)" }}>
            <Filter size={16} color="var(--text-dim)" />
            <select
              className="form-input"
              style={{ width: "auto", border: "none", background: "transparent", padding: "0.375rem 0.5rem" }}
              value={diseaseFilter}
              onChange={(e) => handleFilterChange(e.target.value)}
            >
              <option value="">All Clinical Modules ({total})</option>
              {diseases.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.display_name}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {error && (
        <div className="error-alert">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {loading ? (
        <div className="loading-state">
          <Activity className="spinner-icon pulse-animation" size={36} />
          <p>Querying tenant-isolated prediction audit logs...</p>
        </div>
      ) : items.length === 0 ? (
        <div className="glass-card" style={{ padding: "4rem 2rem", textAlign: "center" }}>
          <div style={{ display: "inline-flex", padding: "1.25rem", borderRadius: "50%", background: "rgba(56, 189, 248, 0.1)", marginBottom: "1.5rem" }}>
            <History size={48} color="#38bdf8" />
          </div>
          <h3 style={{ fontSize: "1.375rem", fontWeight: 700, marginBottom: "0.5rem" }}>No Historical Records Located</h3>
          <p style={{ color: "var(--text-muted)", maxWidth: "480px", margin: "0 auto 2rem", lineHeight: 1.6 }}>
            {diseaseFilter
              ? `No historical predictions recorded for module '${diseaseFilter}'.`
              : "You have not performed any AI medical diagnostic screenings yet."}
          </p>
          <Link to="/diseases" className="btn btn-primary btn-lg">
            <Sparkles size={18} />
            <span>Launch Clinical Screening</span>
          </Link>
        </div>
      ) : (
        <>
          <div className="history-list">
            {items.map((item) => {
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
                        <span style={{ fontSize: "0.8125rem", color: "var(--text-dim)", maxWidth: "360px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                          — {item.explanation}
                        </span>
                      )}
                    </div>
                  </div>

                  <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
                    <Link
                      to={`/reports/${item.id}`}
                      className="btn btn-primary btn-sm"
                      title="View and download clinical PDF report"
                    >
                      <FileText size={14} />
                      <span>PDF Report</span>
                    </Link>

                    <Link
                      to={`/history/${item.id}`}
                      className="btn btn-secondary btn-sm"
                      title="View complete clinical audit record"
                    >
                      <span>Audit</span>
                      <ArrowRight size={14} />
                    </Link>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Pagination Controls */}
          {totalPages > 1 && (
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "2rem", paddingTop: "1.25rem", borderTop: "1px solid var(--border-glass)" }}>
              <span style={{ fontSize: "0.875rem", color: "var(--text-dim)" }}>
                Showing page <strong>{page + 1}</strong> of <strong>{totalPages}</strong> ({total} total)
              </span>

              <div style={{ display: "flex", gap: "0.5rem" }}>
                <button
                  type="button"
                  disabled={page === 0}
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  className="btn btn-secondary btn-sm"
                >
                  <ChevronLeft size={16} />
                  <span>Previous</span>
                </button>

                <button
                  type="button"
                  disabled={page >= totalPages - 1}
                  onClick={() => setPage((p) => p + 1)}
                  className="btn btn-secondary btn-sm"
                >
                  <span>Next</span>
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};
