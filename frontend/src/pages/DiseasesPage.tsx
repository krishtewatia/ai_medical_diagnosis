import React, { useEffect, useState } from "react";
import { diseaseService } from "../services/diseaseService";
import type { DiseaseCategory, DiseaseResponse } from "../types";
import { DiseaseCard } from "../components/DiseaseCard";
import {
  Activity,
  HeartPulse,
  Brain,
  Shield,
  Filter,
} from "lucide-react";

export const DiseasesPage: React.FC = () => {
  const [diseases, setDiseases] = useState<DiseaseResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [categoryFilter, setCategoryFilter] = useState<"all" | DiseaseCategory>("all");

  useEffect(() => {
    const loadDiseases = async () => {
      try {
        setLoading(true);
        const data = await diseaseService.getAllDiseases();
        setDiseases(data);
      } catch (err: any) {
        setError(err?.message || "Failed to discover disease modules.");
      } finally {
        setLoading(false);
      }
    };

    loadDiseases();
  }, []);

  const filtered = diseases.filter((d) => {
    if (categoryFilter === "all") return true;
    return d.category === categoryFilter;
  });

  return (
    <div className="container page-container">
      <div className="page-header">
        <div>
          <h1 className="page-title">Disease Screening Catalog</h1>
          <p className="page-subtitle">
            Explore certified AI screening modules backed by calibrated machine learning models
          </p>
        </div>

        {/* Category Filter Pills */}
        <div className="category-filter-pills">
          <button
            onClick={() => setCategoryFilter("all")}
            className={`pill-btn ${categoryFilter === "all" ? "pill-btn-active" : ""}`}
          >
            <Filter size={14} />
            <span>All Modules ({diseases.length})</span>
          </button>
          <button
            onClick={() => setCategoryFilter("tabular")}
            className={`pill-btn ${categoryFilter === "tabular" ? "pill-btn-active" : ""}`}
          >
            <HeartPulse size={14} />
            <span>Tabular Biometrics</span>
          </button>
          <button
            onClick={() => setCategoryFilter("image")}
            className={`pill-btn ${categoryFilter === "image" ? "pill-btn-active" : ""}`}
          >
            <Brain size={14} />
            <span>Medical Imaging</span>
          </button>
        </div>
      </div>

      {loading ? (
        <div className="loading-state">
          <Activity className="spinner-icon pulse-animation" size={36} />
          <p>Querying dynamic disease registry...</p>
        </div>
      ) : error ? (
        <div className="error-alert">
          <Shield size={18} />
          <span>{error}</span>
        </div>
      ) : (
        <div className="disease-cards-grid">
          {filtered.map((disease) => (
            <DiseaseCard key={disease.id} disease={disease} />
          ))}
        </div>
      )}
    </div>
  );
};
