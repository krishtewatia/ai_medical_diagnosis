import React from "react";
import { Link } from "react-router-dom";
import { AlertTriangle, Home } from "lucide-react";

export const NotFoundPage: React.FC = () => {
  return (
    <div className="container page-container" style={{ textAlign: "center", padding: "4rem 1.5rem" }}>
      <div className="glass-card" style={{ maxWidth: "540px", margin: "0 auto", padding: "3rem 2rem" }}>
        <div style={{ display: "inline-flex", padding: "1rem", borderRadius: "50%", background: "rgba(245, 158, 11, 0.15)", marginBottom: "1.5rem", border: "1px solid rgba(245, 158, 11, 0.3)" }}>
          <AlertTriangle size={40} color="#f59e0b" />
        </div>
        <h1 style={{ fontSize: "2rem", fontWeight: 800, marginBottom: "0.5rem" }}>404 - Page Not Found</h1>
        <p style={{ color: "var(--text-muted)", marginBottom: "2rem", lineHeight: 1.6 }}>
          The clinical route or resource you requested could not be located.
        </p>
        <Link to="/" className="btn btn-primary btn-lg">
          <Home size={18} />
          <span>Return to Dashboard</span>
        </Link>
      </div>
    </div>
  );
};
