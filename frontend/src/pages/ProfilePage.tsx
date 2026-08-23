import React, { useEffect, useState } from "react";
import { profileService } from "../services/profileService";
import type { MedicalProfile, MedicalProfileCreate, MedicalProfileUpdate } from "../types";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Edit3,
  Heart,
  Phone,
  RotateCcw,
  Save,
  ShieldCheck,
  Sparkles,
  User,
  X,
} from "lucide-react";

export const ProfilePage: React.FC = () => {
  const [profile, setProfile] = useState<MedicalProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // Form Fields
  const [dateOfBirth, setDateOfBirth] = useState("");
  const [gender, setGender] = useState("male");
  const [bloodType, setBloodType] = useState("");
  const [heightCm, setHeightCm] = useState<string>("");
  const [weightKg, setWeightKg] = useState<string>("");
  const [allergies, setAllergies] = useState<string>("");
  const [chronicConditions, setChronicConditions] = useState<string>("");
  const [currentMedications, setCurrentMedications] = useState<string>("");
  const [smokingStatus, setSmokingStatus] = useState("");
  const [alcoholConsumption, setAlcoholConsumption] = useState("");
  const [emergencyName, setEmergencyName] = useState("");
  const [emergencyRelation, setEmergencyRelation] = useState("");
  const [emergencyPhone, setEmergencyPhone] = useState("");

  const populateForm = (p: MedicalProfile) => {
    setDateOfBirth(p.date_of_birth || "");
    setGender(p.gender || "male");
    setBloodType(p.blood_type || "");
    setHeightCm(p.height_cm !== null && p.height_cm !== undefined ? String(p.height_cm) : "");
    setWeightKg(p.weight_kg !== null && p.weight_kg !== undefined ? String(p.weight_kg) : "");
    setAllergies(p.allergies ? p.allergies.join(", ") : "");
    setChronicConditions(p.chronic_conditions ? p.chronic_conditions.join(", ") : "");
    setCurrentMedications(p.current_medications ? p.current_medications.join(", ") : "");
    setSmokingStatus(p.smoking_status || "");
    setAlcoholConsumption(p.alcohol_consumption || "");
    setEmergencyName(p.emergency_contact?.name || "");
    setEmergencyRelation(p.emergency_contact?.relationship || "");
    setEmergencyPhone(p.emergency_contact?.phone || "");
  };

  const fetchProfile = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await profileService.getProfile();
      setProfile(data);
      populateForm(data);
    } catch (err: any) {
      if (err.status === 404) {
        setProfile(null);
      } else {
        setError(err?.message || "Failed to load medical profile from server.");
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProfile();
  }, []);

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccessMessage(null);

    // Client validation
    if (!dateOfBirth) {
      setError("Date of birth is required.");
      return;
    }

    const today = new Date().toISOString().split("T")[0];
    if (dateOfBirth >= today) {
      setError("Date of birth must be a date in the past.");
      return;
    }

    if (heightCm) {
      const h = parseFloat(heightCm);
      if (isNaN(h) || h < 30.0 || h > 300.0) {
        setError("Height must be between 30.0 cm and 300.0 cm.");
        return;
      }
    }

    if (weightKg) {
      const w = parseFloat(weightKg);
      if (isNaN(w) || w < 1.0 || w > 500.0) {
        setError("Weight must be between 1.0 kg and 500.0 kg.");
        return;
      }
    }

    const parseList = (str: string) =>
      str
        .split(",")
        .map((s) => s.trim())
        .filter((s) => s.length > 0);

    // Validate emergency contact if partially filled
    let emergencyContactPayload: { name: string; relationship: string; phone: string } | undefined = undefined;
    const hasAnyEmergencyField = emergencyName.trim() || emergencyRelation.trim() || emergencyPhone.trim();
    if (hasAnyEmergencyField) {
      if (!emergencyName.trim() || emergencyName.trim().length < 2) {
        setError("Emergency contact name must be at least 2 characters.");
        return;
      }
      if (!emergencyRelation.trim() || emergencyRelation.trim().length < 2) {
        setError("Emergency contact relationship must be at least 2 characters.");
        return;
      }
      if (!emergencyPhone.trim() || emergencyPhone.trim().length < 5) {
        setError("Emergency contact phone must be at least 5 digits/characters.");
        return;
      }
      emergencyContactPayload = {
        name: emergencyName.trim(),
        relationship: emergencyRelation.trim(),
        phone: emergencyPhone.trim(),
      };
    }

    setSaving(true);
    try {
      if (profile) {
        // Update existing profile with PATCH
        const updatePayload: MedicalProfileUpdate = {
          date_of_birth: dateOfBirth,
          gender: gender as any,
          blood_type: bloodType ? (bloodType as any) : undefined,
          height_cm: heightCm ? parseFloat(heightCm) : undefined,
          weight_kg: weightKg ? parseFloat(weightKg) : undefined,
          allergies: parseList(allergies),
          chronic_conditions: parseList(chronicConditions),
          current_medications: parseList(currentMedications),
          smoking_status: smokingStatus ? (smokingStatus as any) : undefined,
          alcohol_consumption: alcoholConsumption ? (alcoholConsumption as any) : undefined,
          emergency_contact: emergencyContactPayload,
        };

        const updated = await profileService.updateProfile(updatePayload);
        setProfile(updated);
        populateForm(updated);
        setSuccessMessage("Medical profile updated successfully.");
      } else {
        // Create new profile with POST
        const createPayload: MedicalProfileCreate = {
          date_of_birth: dateOfBirth,
          gender: gender as any,
          blood_type: bloodType ? (bloodType as any) : undefined,
          height_cm: heightCm ? parseFloat(heightCm) : undefined,
          weight_kg: weightKg ? parseFloat(weightKg) : undefined,
          allergies: parseList(allergies),
          chronic_conditions: parseList(chronicConditions),
          current_medications: parseList(currentMedications),
          smoking_status: smokingStatus ? (smokingStatus as any) : undefined,
          alcohol_consumption: alcoholConsumption ? (alcoholConsumption as any) : undefined,
          emergency_contact: emergencyContactPayload,
        };

        const created = await profileService.createProfile(createPayload);
        setProfile(created);
        populateForm(created);
        setSuccessMessage("Medical baseline profile created successfully.");
      }

      setIsEditing(false);
    } catch (err: any) {
      setError(err?.message || "Failed to save medical profile. Please check validation rules.");
    } finally {
      setSaving(false);
    }
  };

  // Calculations for summary display
  const calculateAge = (dob?: string) => {
    if (!dob) return null;
    const birth = new Date(dob);
    const now = new Date();
    let age = now.getFullYear() - birth.getFullYear();
    const m = now.getMonth() - birth.getMonth();
    if (m < 0 || (m === 0 && now.getDate() < birth.getDate())) {
      age--;
    }
    return age >= 0 ? age : null;
  };

  const calculateBMI = (h?: number, w?: number) => {
    if (!h || !w || h <= 0) return null;
    const hm = h / 100;
    const bmi = parseFloat((w / (hm * hm)).toFixed(1));
    let category = "Normal weight";
    if (bmi < 18.5) category = "Underweight";
    else if (bmi >= 25 && bmi < 30) category = "Overweight";
    else if (bmi >= 30) category = "Obese";
    return { bmi, category };
  };

  const age = profile ? calculateAge(profile.date_of_birth) : null;
  const bmiInfo = profile ? calculateBMI(profile.height_cm, profile.weight_kg) : null;

  return (
    <div className="container page-container">
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Personal Medical Profile</h1>
          <p className="page-subtitle">
            Encrypted baseline biometrics used exclusively for clinical risk profiling and screening autofill
          </p>
        </div>

        {profile && !isEditing && (
          <button
            onClick={() => setIsEditing(true)}
            className="btn btn-secondary"
          >
            <Edit3 size={16} />
            <span>Edit Profile</span>
          </button>
        )}
      </div>

      {successMessage && (
        <div className="success-banner glass-card" style={{ padding: "0.875rem 1.25rem", marginBottom: "1.5rem", display: "flex", alignItems: "center", gap: "0.5rem", border: "1px solid rgba(16, 185, 129, 0.3)", color: "var(--accent-emerald)" }}>
          <CheckCircle2 size={18} />
          <span>{successMessage}</span>
        </div>
      )}

      {error && (
        <div className="error-alert" role="alert">
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      {loading ? (
        <div className="loading-state">
          <Activity className="spinner-icon pulse-animation" size={36} />
          <p>Retrieving encrypted clinical baseline...</p>
        </div>
      ) : isEditing || !profile ? (
        /* Create / Edit Form */
        <div className="glass-card" style={{ padding: "2.5rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.75rem", paddingBottom: "1rem", borderBottom: "1px solid var(--border-glass)" }}>
            <h2 style={{ fontSize: "1.25rem", fontWeight: 700 }}>
              {profile ? "Edit Medical Baseline Profile" : "Create Clinical Baseline Profile"}
            </h2>
            {profile && (
              <button
                type="button"
                onClick={() => setIsEditing(false)}
                className="btn btn-secondary btn-icon-only"
                title="Cancel editing"
              >
                <X size={18} />
              </button>
            )}
          </div>

          <form onSubmit={handleSave} noValidate>
            <div className="form-fields-grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
              <div className="dynamic-field-group">
                <label className="form-label" htmlFor="dob">
                  Date of Birth <span className="required-star">*</span>
                </label>
                <input
                  id="dob"
                  type="date"
                  required
                  disabled={saving}
                  className="form-input"
                  value={dateOfBirth}
                  onChange={(e) => setDateOfBirth(e.target.value)}
                />
              </div>

              <div className="dynamic-field-group">
                <label className="form-label" htmlFor="gender">
                  Biological Gender <span className="required-star">*</span>
                </label>
                <select
                  id="gender"
                  className="form-input"
                  value={gender}
                  disabled={saving}
                  onChange={(e) => setGender(e.target.value)}
                  required
                >
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="other">Other</option>
                </select>
              </div>

              <div className="dynamic-field-group">
                <label className="form-label" htmlFor="bloodType">
                  Blood Group
                </label>
                <select
                  id="bloodType"
                  className="form-input"
                  value={bloodType}
                  disabled={saving}
                  onChange={(e) => setBloodType(e.target.value)}
                >
                  <option value="">Unknown / Not Set</option>
                  <option value="A+">A+</option>
                  <option value="A-">A-</option>
                  <option value="B+">B+</option>
                  <option value="B-">B-</option>
                  <option value="AB+">AB+</option>
                  <option value="AB-">AB-</option>
                  <option value="O+">O+</option>
                  <option value="O-">O-</option>
                  <option value="unknown">Unknown</option>
                </select>
              </div>

              <div className="dynamic-field-group">
                <label className="form-label" htmlFor="height">
                  Height (cm)
                </label>
                <input
                  id="height"
                  type="number"
                  step="0.1"
                  min="30"
                  max="300"
                  placeholder="e.g. 175"
                  disabled={saving}
                  className="form-input"
                  value={heightCm}
                  onChange={(e) => setHeightCm(e.target.value)}
                />
              </div>

              <div className="dynamic-field-group">
                <label className="form-label" htmlFor="weight">
                  Weight (kg)
                </label>
                <input
                  id="weight"
                  type="number"
                  step="0.1"
                  min="1"
                  max="500"
                  placeholder="e.g. 70"
                  disabled={saving}
                  className="form-input"
                  value={weightKg}
                  onChange={(e) => setWeightKg(e.target.value)}
                />
              </div>

              <div className="dynamic-field-group">
                <label className="form-label" htmlFor="smoking">
                  Smoking Status
                </label>
                <select
                  id="smoking"
                  className="form-input"
                  value={smokingStatus}
                  disabled={saving}
                  onChange={(e) => setSmokingStatus(e.target.value)}
                >
                  <option value="">Not Specified</option>
                  <option value="never">Never</option>
                  <option value="former">Former Smoker</option>
                  <option value="current">Current Smoker</option>
                </select>
              </div>

              <div className="dynamic-field-group">
                <label className="form-label" htmlFor="alcohol">
                  Alcohol Consumption
                </label>
                <select
                  id="alcohol"
                  className="form-input"
                  value={alcoholConsumption}
                  disabled={saving}
                  onChange={(e) => setAlcoholConsumption(e.target.value)}
                >
                  <option value="">Not Specified</option>
                  <option value="none">None</option>
                  <option value="occasional">Occasional</option>
                  <option value="moderate">Moderate</option>
                  <option value="frequent">Frequent</option>
                </select>
              </div>
            </div>

            {/* Medical History Tags */}
            <div style={{ marginTop: "1.5rem", display: "flex", flexDirection: "column", gap: "1rem" }}>
              <div className="dynamic-field-group">
                <label className="form-label" htmlFor="allergies">
                  Allergies (comma-separated)
                </label>
                <input
                  id="allergies"
                  type="text"
                  placeholder="e.g. Penicillin, Peanuts, Latex"
                  disabled={saving}
                  className="form-input"
                  value={allergies}
                  onChange={(e) => setAllergies(e.target.value)}
                />
              </div>

              <div className="dynamic-field-group">
                <label className="form-label" htmlFor="conditions">
                  Chronic Conditions (comma-separated)
                </label>
                <input
                  id="conditions"
                  type="text"
                  placeholder="e.g. Hypertension, Asthma, Type 2 Diabetes"
                  disabled={saving}
                  className="form-input"
                  value={chronicConditions}
                  onChange={(e) => setChronicConditions(e.target.value)}
                />
              </div>

              <div className="dynamic-field-group">
                <label className="form-label" htmlFor="meds">
                  Current Medications (comma-separated)
                </label>
                <input
                  id="meds"
                  type="text"
                  placeholder="e.g. Metformin 500mg, Lisinopril 10mg"
                  disabled={saving}
                  className="form-input"
                  value={currentMedications}
                  onChange={(e) => setCurrentMedications(e.target.value)}
                />
              </div>
            </div>

            {/* Emergency Contact */}
            <div style={{ marginTop: "2rem", paddingTop: "1.5rem", borderTop: "1px solid var(--border-glass)" }}>
              <h3 style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "1rem", color: "var(--accent-cyan)", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <Phone size={16} />
                <span>Emergency Contact (Optional)</span>
              </h3>

              <div className="form-fields-grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))" }}>
                <div className="dynamic-field-group">
                  <label className="form-label" htmlFor="emName">Contact Name</label>
                  <input
                    id="emName"
                    type="text"
                    placeholder="e.g. Jane Doe"
                    disabled={saving}
                    className="form-input"
                    value={emergencyName}
                    onChange={(e) => setEmergencyName(e.target.value)}
                  />
                </div>

                <div className="dynamic-field-group">
                  <label className="form-label" htmlFor="emRel">Relationship</label>
                  <input
                    id="emRel"
                    type="text"
                    placeholder="e.g. Spouse, Sibling"
                    disabled={saving}
                    className="form-input"
                    value={emergencyRelation}
                    onChange={(e) => setEmergencyRelation(e.target.value)}
                  />
                </div>

                <div className="dynamic-field-group">
                  <label className="form-label" htmlFor="emPhone">Emergency Phone</label>
                  <input
                    id="emPhone"
                    type="tel"
                    placeholder="e.g. +1 555-0199"
                    disabled={saving}
                    className="form-input"
                    value={emergencyPhone}
                    onChange={(e) => setEmergencyPhone(e.target.value)}
                  />
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="form-submit-row">
              {profile && (
                <button
                  type="button"
                  onClick={() => {
                    populateForm(profile);
                    setIsEditing(false);
                  }}
                  disabled={saving}
                  className="btn btn-secondary"
                >
                  <RotateCcw size={16} />
                  <span>Cancel</span>
                </button>
              )}

              <button
                type="submit"
                disabled={saving}
                className="btn btn-primary btn-lg"
              >
                {saving ? (
                  <>
                    <Activity size={18} className="spinner-icon pulse-animation" />
                    <span>Saving Profile...</span>
                  </>
                ) : (
                  <>
                    <Save size={18} />
                    <span>{profile ? "Update Medical Profile" : "Create Medical Profile"}</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      ) : (
        /* Summary View Mode */
        <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
          {/* Key Vitals Card */}
          <div className="glass-card" style={{ padding: "2rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1.5rem" }}>
              <div className="user-avatar" style={{ width: "48px", height: "48px", fontSize: "1.25rem" }}>
                <User size={24} />
              </div>
              <div>
                <h2 style={{ fontSize: "1.375rem", fontWeight: 700 }}>Clinical Baseline Parameters</h2>
                <span className="badge badge-emerald">
                  <ShieldCheck size={12} style={{ marginRight: "3px" }} />
                  Encrypted & Isolated to User Account
                </span>
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "1.25rem" }}>
              <div className="snapshot-item">
                <span className="snapshot-key">Age / Date of Birth</span>
                <span className="snapshot-val">{age !== null ? `${age} yrs` : "—"} ({profile.date_of_birth})</span>
              </div>

              <div className="snapshot-item">
                <span className="snapshot-key">Biological Gender</span>
                <span className="snapshot-val" style={{ textTransform: "capitalize" }}>{profile.gender}</span>
              </div>

              <div className="snapshot-item">
                <span className="snapshot-key">Blood Group</span>
                <span className="snapshot-val">{profile.blood_type || "Not recorded"}</span>
              </div>

              <div className="snapshot-item">
                <span className="snapshot-key">Height & Weight</span>
                <span className="snapshot-val">
                  {profile.height_cm ? `${profile.height_cm} cm` : "—"} / {profile.weight_kg ? `${profile.weight_kg} kg` : "—"}
                </span>
              </div>

              {bmiInfo && (
                <div className="snapshot-item">
                  <span className="snapshot-key">Body Mass Index (BMI)</span>
                  <span className="snapshot-val" style={{ color: "var(--accent-cyan)" }}>
                    {bmiInfo.bmi} ({bmiInfo.category})
                  </span>
                </div>
              )}

              <div className="snapshot-item">
                <span className="snapshot-key">Lifestyle Habits</span>
                <span className="snapshot-val" style={{ textTransform: "capitalize" }}>
                  Smoking: {profile.smoking_status || "none"}, Alcohol: {profile.alcohol_consumption || "none"}
                </span>
              </div>
            </div>
          </div>

          {/* Clinical Details Grid */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1.5rem" }}>
            {/* Allergies & Conditions */}
            <div className="glass-card" style={{ padding: "1.75rem" }}>
              <h3 style={{ fontSize: "1.125rem", fontWeight: 700, marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <Heart size={18} color="#f43f5e" />
                <span>Allergies & Chronic Conditions</span>
              </h3>

              <div style={{ marginBottom: "1.25rem" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-dim)", display: "block", marginBottom: "0.375rem" }}>Known Allergies</span>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.375rem" }}>
                  {profile.allergies && profile.allergies.length > 0 ? (
                    profile.allergies.map((a) => (
                      <span key={a} className="badge badge-rose">{a}</span>
                    ))
                  ) : (
                    <span style={{ fontSize: "0.875rem", color: "var(--text-muted)" }}>No documented allergies</span>
                  )}
                </div>
              </div>

              <div>
                <span style={{ fontSize: "0.75rem", color: "var(--text-dim)", display: "block", marginBottom: "0.375rem" }}>Chronic Conditions</span>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.375rem" }}>
                  {profile.chronic_conditions && profile.chronic_conditions.length > 0 ? (
                    profile.chronic_conditions.map((c) => (
                      <span key={c} className="badge badge-purple">{c}</span>
                    ))
                  ) : (
                    <span style={{ fontSize: "0.875rem", color: "var(--text-muted)" }}>No chronic conditions reported</span>
                  )}
                </div>
              </div>
            </div>

            {/* Medications & Emergency Contact */}
            <div className="glass-card" style={{ padding: "1.75rem" }}>
              <h3 style={{ fontSize: "1.125rem", fontWeight: 700, marginBottom: "1rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <Sparkles size={18} color="#38bdf8" />
                <span>Medications & Emergency Contact</span>
              </h3>

              <div style={{ marginBottom: "1.25rem" }}>
                <span style={{ fontSize: "0.75rem", color: "var(--text-dim)", display: "block", marginBottom: "0.375rem" }}>Current Medications</span>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "0.375rem" }}>
                  {profile.current_medications && profile.current_medications.length > 0 ? (
                    profile.current_medications.map((m) => (
                      <span key={m} className="badge badge-cyan">{m}</span>
                    ))
                  ) : (
                    <span style={{ fontSize: "0.875rem", color: "var(--text-muted)" }}>No current medications</span>
                  )}
                </div>
              </div>

              <div>
                <span style={{ fontSize: "0.75rem", color: "var(--text-dim)", display: "block", marginBottom: "0.375rem" }}>Emergency Contact</span>
                {profile.emergency_contact?.name ? (
                  <div style={{ background: "rgba(15, 23, 42, 0.6)", padding: "0.75rem 1rem", borderRadius: "var(--radius-sm)", border: "1px solid var(--border-glass)" }}>
                    <div style={{ fontWeight: 700, fontSize: "0.9375rem" }}>{profile.emergency_contact.name} ({profile.emergency_contact.relationship || "Contact"})</div>
                    <div style={{ fontSize: "0.8125rem", color: "var(--text-highlight)", marginTop: "0.125rem" }}>{profile.emergency_contact.phone}</div>
                  </div>
                ) : (
                  <span style={{ fontSize: "0.875rem", color: "var(--text-muted)" }}>No emergency contact registered</span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
