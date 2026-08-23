// ==========================================
// User & Authentication Types
// ==========================================

export interface User {
  id: string;
  name: string;
  email: string;
}

export interface TokenResponse {
  access_token: string;
  token_type?: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  name: string;
  email: string;
  password: string;
}

// ==========================================
// Medical Profile Types
// ==========================================

export interface MedicalProfile {
  id: string;
  user_id: string;
  date_of_birth: string;
  gender: string;
  blood_type?: string;
  height_cm?: number;
  weight_kg?: number;
  allergies: string[];
  chronic_conditions: string[];
  current_medications: string[];
  smoking_status?: string;
  alcohol_consumption?: string;
  emergency_contact?: {
    name?: string;
    relationship?: string;
    phone?: string;
  };
  created_at: string;
  updated_at: string;
}

export type MedicalProfileCreate = Omit<MedicalProfile, "id" | "user_id" | "created_at" | "updated_at">;
export type MedicalProfileUpdate = Partial<MedicalProfileCreate>;

// ==========================================
// Disease Discovery & Configuration Types
// ==========================================

export type DiseaseCategory = "tabular" | "image";
export type InputType = "form" | "image_upload";
export type FeatureDataType = "int" | "float" | "categorical";

export interface TabularFeatureSpec {
  name: string;
  display_name: string;
  data_type: FeatureDataType;
  required: boolean;
  min_value?: number;
  max_value?: number;
  unit?: string;
  allowed_values?: (string | number)[];
  value_descriptions?: Record<string, string>;
}

export interface ImageInputSpec {
  allowed_formats: string[];
  max_size_bytes: number;
  target_dimensions: [number, number];
  channels: number;
}

export interface DiseaseModelInfo {
  version: string;
  framework: string;
  model_type: string;
  threshold?: number;
  supports_probability: boolean;
}

export interface DiseaseSafetyInfo {
  clinical_purpose: string;
  is_diagnostic_tool: boolean;
  disclaimer: string;
}

export interface DiseaseResponse {
  id: string;
  display_name: string;
  category: DiseaseCategory;
  input_type: InputType;
  description: string;
  is_active: boolean;
  required_fields: TabularFeatureSpec[];
  image_spec?: ImageInputSpec;
  positive_label: string;
  negative_label: string;
  supports_probability: boolean;
  metrics: Record<string, number>;
  model_info?: DiseaseModelInfo;
  safety_info?: DiseaseSafetyInfo;
}

// ==========================================
// Prediction Types (Exact Backend Alignment)
// ==========================================

export interface PredictionRequest {
  disease_id: string;
  inputs: Record<string, any>;
}

export interface PredictionResponse {
  disease_id: string;
  disease_display_name: string;
  prediction_label: string;
  is_positive: boolean;
  probability?: number;
  decision_threshold?: number;
  model_version: string;
  model_type: string;
  explanation?: string;
  clinical_purpose?: string;
  disclaimer?: string;
  limitations?: string;
  metadata?: Record<string, any>;
  timestamp: string;
}

// ==========================================
// Prediction History Types
// ==========================================

export interface PredictionHistoryModelMeta {
  version: string;
  model_type: string;
  threshold?: number;
}

export interface PredictionHistoryResult {
  prediction: string;
  is_positive: boolean;
  probability?: number;
  confidence?: number;
}

export interface PredictionHistoryItem {
  id: string;
  user_id: string;
  disease: string;
  disease_display_name: string;
  input_type: string;
  model: PredictionHistoryModelMeta;
  input_data: Record<string, any>;
  result: PredictionHistoryResult;
  explanation?: string;
  metadata?: Record<string, any>;
  created_at: string;
}

export interface PredictionHistoryListResponse {
  items: PredictionHistoryItem[];
  total: number;
  limit: number;
  skip: number;
}

// ==========================================
// Medical Report Types
// ==========================================

export interface ReportModelInfo {
  model_type: string;
  version: string;
  threshold?: number;
}

export interface MedicalReportResponse {
  report_id: string;
  prediction_id: string;
  user_id: string;
  user_name?: string;
  user_email?: string;
  disease: string;
  disease_display_name: string;
  input_type: string;
  prediction: string;
  is_positive: boolean;
  probability?: number;
  model: ReportModelInfo;
  prediction_date: string;
  input_summary: Record<string, any>;
  explanation?: string;
  disclaimer: string;
  storage_key: string;
  download_url?: string;
  created_at: string;
}
