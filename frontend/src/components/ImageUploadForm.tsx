import React, { useState, useRef } from "react";
import type { DiseaseResponse, PredictionResponse } from "../types";
import { predictionService } from "../services/predictionService";
import {
  Activity,
  AlertCircle,
  Brain,
  CheckCircle2,
  FileImage,
  RotateCcw,
  Upload,
  X,
} from "lucide-react";

interface ImageUploadFormProps {
  config: DiseaseResponse;
  onSuccess: (result: PredictionResponse) => void;
}

export const ImageUploadForm: React.FC<ImageUploadFormProps> = ({
  config,
  onSuccess,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const maxSizeBytes = config.image_spec?.max_size_bytes || 15728640; // 15MB default
  const allowedFormats = config.image_spec?.allowed_formats || ["png", "jpg", "jpeg"];

  const isBrainMri = config.id === "brain_tumor";

  const handleFileValidation = (file: File): boolean => {
    setError(null);

    // Format validation
    const ext = file.name.split(".").pop()?.toLowerCase();
    if (!ext || !allowedFormats.includes(ext)) {
      setError(`Invalid file format '.${ext}'. Please upload a ${allowedFormats.join(", ").toUpperCase()} image.`);
      return false;
    }

    // Size validation
    if (file.size > maxSizeBytes) {
      const maxMb = (maxSizeBytes / (1024 * 1024)).toFixed(0);
      setError(`File size exceeds the maximum limit of ${maxMb} MB.`);
      return false;
    }

    return true;
  };

  const handleFileSelect = (file: File) => {
    if (handleFileValidation(file)) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!selectedFile) {
      setError(`Please select a ${isBrainMri ? "brain MRI scan" : "medical image"} to evaluate.`);
      return;
    }

    setSubmitting(true);
    try {
      const result = await predictionService.predictImage(config.id, selectedFile);
      onSuccess(result);
    } catch (err: any) {
      setError(err?.message || "Medical image inference failed. Please verify the image file and retry.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="image-upload-form-wrapper">
      {error && (
        <div className="error-alert" role="alert" style={{ marginBottom: "1.5rem" }}>
          <AlertCircle size={18} />
          <span>{error}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} noValidate>
        {/* Dropzone Area */}
        {!selectedFile ? (
          <div
            className={`file-dropzone ${dragActive ? "dropzone-active" : ""}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".png,.jpg,.jpeg"
              style={{ display: "none" }}
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  handleFileSelect(e.target.files[0]);
                }
              }}
            />

            <div className="dropzone-content">
              <div className="dropzone-icon-circle">
                {isBrainMri ? (
                  <Brain size={32} color="#8b5cf6" />
                ) : (
                  <Upload size={32} color="#38bdf8" />
                )}
              </div>
              <h3 className="dropzone-title">
                {isBrainMri
                  ? "Upload Axial Brain MRI Scan"
                  : "Upload Chest Radiograph (X-Ray)"}
              </h3>
              <p className="dropzone-subtitle">
                Drag and drop your DICOM/PNG/JPEG medical scan, or click to browse
              </p>
              <div className="dropzone-meta-tags">
                <span className="badge badge-purple">
                  Formats: {allowedFormats.join(", ").toUpperCase()}
                </span>
                <span className="badge badge-cyan">
                  Max Size: {(maxSizeBytes / (1024 * 1024)).toFixed(0)} MB
                </span>
                <span className="badge badge-emerald">
                  Target Tensor: 224×224 RGB
                </span>
              </div>
            </div>
          </div>
        ) : (
          /* Preview Selected Image */
          <div className="image-preview-card glass-card" style={{ padding: "1.75rem", marginBottom: "1.5rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <FileImage size={20} color="#38bdf8" />
                <span style={{ fontWeight: 700, fontSize: "0.9375rem" }}>
                  Selected {isBrainMri ? "MRI Scan" : "Radiograph Scan"}
                </span>
              </div>
              <button
                type="button"
                onClick={handleReset}
                disabled={submitting}
                className="btn btn-secondary btn-sm"
                title="Remove image and select another"
              >
                <X size={14} />
                <span>Remove Scan</span>
              </button>
            </div>

            <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap", alignItems: "center" }}>
              {previewUrl && (
                <div style={{ width: "160px", height: "160px", borderRadius: "var(--radius-md)", overflow: "hidden", border: "1px solid var(--border-glass)", background: "#000", display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <img
                    src={previewUrl}
                    alt="Scan Preview"
                    style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
                  />
                </div>
              )}

              <div style={{ flex: 1, minWidth: "220px" }}>
                <div style={{ fontWeight: 700, fontSize: "1rem", marginBottom: "0.25rem", wordBreak: "break-all" }}>
                  {selectedFile.name}
                </div>
                <div style={{ color: "var(--text-muted)", fontSize: "0.8125rem", marginBottom: "1rem" }}>
                  Size: {(selectedFile.size / 1024).toFixed(1)} KB • Type: {selectedFile.type || "image/png"}
                </div>
                <div style={{ padding: "0.75rem", background: "rgba(56, 189, 248, 0.08)", borderRadius: "var(--radius-sm)", border: "1px solid rgba(56, 189, 248, 0.2)", fontSize: "0.75rem", color: "var(--text-highlight)" }}>
                  Image will be resized to 224×224 and normalized via {config.model_info?.model_type || "CNN"} for clinical decision support.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Action Controls */}
        <div className="form-submit-row">
          <button
            type="button"
            onClick={handleReset}
            disabled={submitting || !selectedFile}
            className="btn btn-secondary"
          >
            <RotateCcw size={16} />
            <span>Reset</span>
          </button>

          <button
            type="submit"
            disabled={submitting || !selectedFile}
            className="btn btn-primary btn-lg submit-screening-btn"
          >
            {submitting ? (
              <>
                <Activity size={18} className="spinner-icon pulse-animation" />
                <span>Running {config.model_info?.model_type || "Model"} Screening...</span>
              </>
            ) : (
              <>
                <CheckCircle2 size={18} />
                <span>Analyze Scan with {config.model_info?.model_type || "CNN"}</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};
