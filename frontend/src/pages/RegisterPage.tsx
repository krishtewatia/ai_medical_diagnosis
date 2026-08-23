import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import { UserPlus, Mail, Lock, User, AlertCircle, CheckCircle2 } from "lucide-react";

export const RegisterPage: React.FC = () => {
  const { register } = useAuth();
  const navigate = useNavigate();

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    // Client-side validation
    const trimmedName = name.trim();
    const trimmedEmail = email.trim();

    if (!trimmedName) {
      setError("Please enter your full name.");
      return;
    }

    if (!trimmedEmail || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(trimmedEmail)) {
      setError("Please enter a valid email address.");
      return;
    }

    if (password.length < 6) {
      setError("Password must be at least 6 characters long.");
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match. Please re-enter.");
      return;
    }

    setSubmitting(true);
    try {
      await register(trimmedName, trimmedEmail, password);
      // Successful registration establishes session and redirects to dashboard
      navigate("/dashboard", { replace: true });
    } catch (err: any) {
      setError(err?.message || "Registration failed. Please verify your details.");
      setPassword("");
      setConfirmPassword("");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-page-container container">
      <div className="auth-card glass-card">
        <div className="auth-header">
          <div className="auth-icon-badge">
            <UserPlus size={24} color="#10b981" />
          </div>
          <h1 className="auth-title">Create Account</h1>
          <p className="auth-subtitle">Register to begin conducting AI medical screenings</p>
        </div>

        {error && (
          <div className="error-alert" role="alert">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="auth-form" noValidate>
          <div className="form-group">
            <label className="form-label" htmlFor="name">
              Full Name
            </label>
            <div className="input-with-icon">
              <User size={18} className="input-icon" />
              <input
                id="name"
                type="text"
                required
                disabled={submitting}
                className="form-input with-left-icon"
                placeholder="Dr. Sarah Johnson"
                value={name}
                onChange={(e) => setName(e.target.value)}
                autoComplete="name"
              />
            </div>
          </div>

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
                placeholder="Minimum 6 characters"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="new-password"
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="confirmPassword">
              Confirm Password
            </label>
            <div className="input-with-icon">
              <Lock size={18} className="input-icon" />
              <input
                id="confirmPassword"
                type="password"
                required
                disabled={submitting}
                className="form-input with-left-icon"
                placeholder="Re-enter your password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                autoComplete="new-password"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={submitting}
            className="btn btn-primary btn-block btn-lg"
          >
            {submitting ? (
              <span>Creating Account...</span>
            ) : (
              <>
                <CheckCircle2 size={18} />
                <span>Complete Registration</span>
              </>
            )}
          </button>
        </form>

        <div className="auth-footer">
          <span>Already have an account?</span>{" "}
          <Link to="/login" className="auth-link">
            Sign in
          </Link>
        </div>
      </div>
    </div>
  );
};
