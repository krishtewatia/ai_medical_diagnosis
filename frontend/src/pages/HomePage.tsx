import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { diseaseService } from "../services/diseaseService";
import type { DiseaseResponse } from "../types";
import {
  Activity,
  ArrowRight,
  ShieldCheck,
  Zap,
  Layers,
  HeartPulse,
  Brain,
  Stethoscope,
  Sparkles,
} from "lucide-react";

export const HomePage: React.FC = () => {
  const [diseases, setDiseases] = useState<DiseaseResponse[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchDiseases = async () => {
      try {
        const data = await diseaseService.getAllDiseases();
        setDiseases(data);
      } catch (err) {
        console.error("Failed to load diseases:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchDiseases();
  }, []);

  return (
    <div className="container home-page-container">
      {/* Hero Section */}
      <section className="hero-banner glass-card">
        <div className="hero-badge">
          <Sparkles size={14} className="hero-sparkle" />
          <span>Next-Generation Multi-Disease AI Diagnostics</span>
        </div>

        <h1 className="hero-headline">
          Precision AI Screening for{" "}
          <span className="text-gradient">Early Disease Detection</span>
        </h1>

        <p className="hero-description">
          AI Medical Diagnosis Assistant empowers clinical practitioners and patients with instant, machine-learning-driven risk assessments across tabular biometrics and deep learning imaging models.
        </p>

        <div className="hero-actions">
          <Link to="/diseases" className="btn btn-primary btn-lg">
            <span>Explore Disease Modules</span>
            <ArrowRight size={18} />
          </Link>
          <Link to="/profile" className="btn btn-secondary btn-lg">
            <Stethoscope size={18} />
            <span>Manage Medical Profile</span>
          </Link>
        </div>

        {/* Highlight Feature Badges */}
        <div className="hero-highlights-grid">
          <div className="highlight-item">
            <ShieldCheck size={20} className="highlight-icon icon-emerald" />
            <div>
              <strong>Tenant Isolated</strong>
              <span>HIPAA-aligned user scoping</span>
            </div>
          </div>

          <div className="highlight-item">
            <Zap size={20} className="highlight-icon icon-cyan" />
            <div>
              <strong>Sub-Second Inference</strong>
              <span>Calibrated risk confidence</span>
            </div>
          </div>

          <div className="highlight-item">
            <Layers size={20} className="highlight-icon icon-purple" />
            <div>
              <strong>Modular Engine</strong>
              <span>Dynamic form discovery</span>
            </div>
          </div>
        </div>
      </section>

      {/* Available Modules Section */}
      <section className="modules-section">
        <div className="section-header">
          <div>
            <h2 className="section-title">Active Clinical Modules</h2>
            <p className="section-subtitle">
              Dynamically discovered from our configuration-driven AI engine
            </p>
          </div>
          <Link to="/diseases" className="btn btn-secondary">
            <span>View All</span>
            <ArrowRight size={16} />
          </Link>
        </div>

        {loading ? (
          <div className="loading-state">
            <Activity className="spinner-icon pulse-animation" size={32} />
            <p>Discovering available disease models...</p>
          </div>
        ) : (
          <div className="disease-cards-grid">
            {diseases.map((d) => (
              <div key={d.id} className="disease-card glass-card">
                <div className="disease-card-header">
                  <div className="disease-card-icon">
                    {d.id === "diabetes" ? (
                      <HeartPulse size={24} color="#38bdf8" />
                    ) : d.id === "heart_disease" ? (
                      <Activity size={24} color="#f43f5e" />
                    ) : (
                      <Brain size={24} color="#8b5cf6" />
                    )}
                  </div>
                  <span className={`badge ${d.category === "tabular" ? "badge-cyan" : "badge-emerald"}`}>
                    {d.category}
                  </span>
                </div>

                <h3 className="disease-card-title">{d.display_name}</h3>
                <p className="disease-card-description">{d.description}</p>

                <div className="disease-card-meta">
                  <div className="meta-tag">
                    <span className="meta-label">Model:</span>
                    <span className="meta-val">{d.model_info?.model_type || "ML Classifier"}</span>
                  </div>
                  <div className="meta-tag">
                    <span className="meta-label">Inputs:</span>
                    <span className="meta-val">{d.required_fields.length} features</span>
                  </div>
                </div>

                <Link to={`/predict/${d.id}`} className="btn btn-primary btn-block">
                  <span>Start Screening</span>
                  <ArrowRight size={16} />
                </Link>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
};
