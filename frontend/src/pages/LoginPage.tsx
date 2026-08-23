import React, { useState } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { LogIn, Mail, Lock, AlertCircle } from "lucide-react";

export const LoginPage: React.FC = () => {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Determine redirection target (default to /dashboard)
  const from = (location.state as any)?.from?.pathname || "/dashboard";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    const trimmedEmail = email.trim();
    if (!trimmedEmail || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(trimmedEmail)) {
      setError("Please enter a valid email address.");
      return;
    }

    if (!password) {
      setError("Please enter your password.");
      return;
    }

    setSubmitting(true);
    try {
      await login(trimmedEmail, password);
      // On success, redirect to dashboard or original protected destination
      navigate(from, { replace: true });
    } catch (err: any) {
      setError(err?.message || "Invalid email or password.");
      setPassword(""); // Clear password field on error
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-page-container container">
      <div className="auth-card glass-card">
        <div className="auth-header">
          <div className="auth-icon-badge">
            <LogIn size={24} color="#38bdf8" />
          </div>
          <h1 className="auth-title">Welcome Back</h1>
          <p className="auth-subtitle">Sign in to your clinical diagnostic workstation</p>
        </div>

        {error && (
          <div className="error-alert" role="alert">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="auth-form" noValidate>
          <div className="form-group">
            <label className="form-label" htmlFor="email">
              Email Address
            </label>
            <div className="input-with-icon">
              <Mail size={18} className="input-icon" />
              <input
                id="email"
                type="email"
                required
                disabled={submitting}
                className="form-input with-left-icon"
                placeholder="doctor@hospital.org"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="password">
              Password
            </label>
            <div className="input-with-icon">
              <Lock size={18} className="input-icon" />
              <input
                id="password"
                type="password"
                required
                disabled={submitting}
                className="form-input with-left-icon"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={submitting}
            className="btn btn-primary btn-block btn-lg"
          >
            {submitting ? (
              <span>Signing In...</span>
            ) : (
              <>
                <LogIn size={18} />
                <span>Sign In to Dashboard</span>
              </>
            )}
          </button>
        </form>

        <div className="auth-footer">
          <span>Don't have an account?</span>{" "}
          <Link to="/register" className="auth-link">
            Create an account
          </Link>
        </div>
      </div>
    </div>
  );
};
