import React, { useState, useEffect } from "react";
import { Link, NavLink, useNavigate, useLocation } from "react-router-dom";
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
  Menu,
  X,
} from "lucide-react";

export const Navbar: React.FC = () => {
  const { user, isAuthenticated, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Close mobile menu whenever the route changes
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);

  const handleLogout = () => {
    setMobileMenuOpen(false);
    logout();
    navigate("/login");
  };

  const navLinkClasses = ({ isActive }: { isActive: boolean }) =>
    `nav-link ${isActive ? "nav-link-active" : ""}`;

  const mobileNavLinkClasses = ({ isActive }: { isActive: boolean }) =>
    `mobile-nav-link ${isActive ? "mobile-nav-link-active" : ""}`;

  return (
    <header className="navbar-wrapper">
      <div className="container navbar-container">
        {/* Brand Logo */}
        <Link to="/" className="brand-logo" onClick={() => setMobileMenuOpen(false)}>
          <div className="brand-icon-wrapper">
            <HeartPulse className="brand-icon" size={24} />
          </div>
          <div className="brand-text">
            <span className="brand-title">AI Medical Diagnosis</span>
            <span className="brand-subtitle">Clinical Decision Support</span>
          </div>
        </Link>

        {/* Desktop Navigation Links */}
        <nav className="nav-links desktop-nav">
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

        {/* Desktop User / Authentication Actions */}
        <div className="nav-actions desktop-actions">
          {isAuthenticated && user ? (
            <div className="user-profile-menu">
              <Link to="/profile" className="user-info-badge-link" title="View Profile">
                <div className="user-info-badge">
                  <div className="user-avatar">
                    {user.name.charAt(0).toUpperCase()}
                  </div>
                  <div className="user-details">
                    <span className="user-name">{user.name}</span>
                    <span className="user-email">{user.email}</span>
                  </div>
                </div>
              </Link>
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

        {/* Mobile Hamburger Toggle Button */}
        <button
          type="button"
          className="mobile-toggle-btn"
          onClick={() => setMobileMenuOpen((prev) => !prev)}
          aria-label={mobileMenuOpen ? "Close navigation menu" : "Open navigation menu"}
          aria-expanded={mobileMenuOpen}
        >
          {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Mobile Navigation Drawer / Dropdown */}
      {mobileMenuOpen && (
        <div className="mobile-nav-drawer" role="dialog" aria-modal="true">
          <div className="mobile-drawer-content container">
            <nav className="mobile-nav-list">
              <NavLink to="/" className={mobileNavLinkClasses} end>
                <HeartPulse size={18} />
                <span>Home</span>
              </NavLink>
              {isAuthenticated && (
                <NavLink to="/dashboard" className={mobileNavLinkClasses}>
                  <Activity size={18} />
                  <span>Dashboard</span>
                </NavLink>
              )}
              <NavLink to="/diseases" className={mobileNavLinkClasses}>
                <Grid size={18} />
                <span>Disease Modules</span>
              </NavLink>
              {isAuthenticated && (
                <>
                  <NavLink to="/history" className={mobileNavLinkClasses}>
                    <History size={18} />
                    <span>Prediction History</span>
                  </NavLink>
                  <NavLink to="/profile" className={mobileNavLinkClasses}>
                    <UserIcon size={18} />
                    <span>Clinical Profile</span>
                  </NavLink>
                </>
              )}
            </nav>

            <div className="mobile-drawer-divider" />

            {/* Mobile Auth / Profile Section */}
            <div className="mobile-auth-section">
              {isAuthenticated && user ? (
                <div className="mobile-user-section">
                  <div className="mobile-user-card">
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
                    className="btn btn-secondary btn-block mobile-logout-btn"
                  >
                    <LogOut size={18} />
                    <span>Sign Out</span>
                  </button>
                </div>
              ) : (
                <div className="mobile-auth-buttons">
                  <Link to="/login" className="btn btn-secondary btn-block">
                    <LogIn size={18} />
                    <span>Sign In</span>
                  </Link>
                  <Link to="/register" className="btn btn-primary btn-block">
                    <UserPlus size={18} />
                    <span>Register Account</span>
                  </Link>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </header>
  );
};
