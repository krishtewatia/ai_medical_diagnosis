import React from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import {
  Activity,
  User as UserIcon,
  LogOut,
  LogIn,
  UserPlus,
  History,
  Grid,
  HeartPulse,
} from "lucide-react";

export const Navbar: React.FC = () => {
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  const navLinkClasses = ({ isActive }: { isActive: boolean }) =>
    `nav-link ${isActive ? "nav-link-active" : ""}`;

  return (
    <header className="navbar-wrapper">
      <div className="container navbar-container">
        {/* Brand Logo */}
        <Link to="/" className="brand-logo">
          <div className="brand-icon-wrapper">
            <HeartPulse className="brand-icon" size={24} />
          </div>
          <div className="brand-text">
            <span className="brand-title">AI Medical Diagnosis</span>
            <span className="brand-subtitle">Clinical Decision Support</span>
          </div>
        </Link>

        {/* Primary Navigation Links */}
        <nav className="nav-links">
          <NavLink to="/" className={navLinkClasses} end>
            <span>Home</span>
          </NavLink>
          {isAuthenticated && (
            <NavLink to="/dashboard" className={navLinkClasses}>
              <Activity size={18} />
              <span>Dashboard</span>
            </NavLink>
          )}
          <NavLink to="/diseases" className={navLinkClasses}>
            <Grid size={18} />
            <span>Diseases</span>
          </NavLink>
          {isAuthenticated && (
            <>
              <NavLink to="/history" className={navLinkClasses}>
                <History size={18} />
                <span>History</span>
              </NavLink>
              <NavLink to="/profile" className={navLinkClasses}>
                <UserIcon size={18} />
                <span>Profile</span>
              </NavLink>
            </>
          )}
        </nav>

        {/* User / Authentication Actions */}
        <div className="nav-actions">
          {isAuthenticated && user ? (
            <div className="user-profile-menu">
              <div className="user-info-badge">
                <div className="user-avatar">
                  {user.name.charAt(0).toUpperCase()}
                </div>
                <div className="user-details">
                  <span className="user-name">{user.name}</span>
                  <span className="user-email">{user.email}</span>
                </div>
              </div>
              <button
                onClick={handleLogout}
                className="btn btn-secondary btn-icon-only"
                title="Log out of account"
                aria-label="Logout"
              >
                <LogOut size={18} />
              </button>
            </div>
          ) : (
            <div className="auth-buttons">
              <Link to="/login" className="btn btn-secondary">
                <LogIn size={16} />
                <span>Sign In</span>
              </Link>
              <Link to="/register" className="btn btn-primary">
                <UserPlus size={16} />
                <span>Register</span>
              </Link>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
