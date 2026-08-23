import { api, getAuthToken } from "./api";
import type { MedicalReportResponse } from "../types";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const reportService = {
  /**
   * Fetches or generates a clinical PDF screening report for a specific prediction.
   */
  getReport: async (predictionId: string): Promise<MedicalReportResponse> => {
    return api.get<MedicalReportResponse>(`/reports/${predictionId}`);
  },

  /**
   * Downloads the clinical PDF report directly as a file.
   */
  downloadReportPdf: async (predictionId: string, customFilename?: string): Promise<void> => {
    const token = getAuthToken();
    const url = `${API_BASE_URL}/reports/${predictionId}/download`;

    const headers: Record<string, string> = {};
    if (token) {
      headers["Authorization"] = `Bearer ${token}`;
    }

    const res = await fetch(url, {
      method: "GET",
      headers,
    });

    if (!res.ok) {
      let errorMsg = `Failed to download report (HTTP ${res.status})`;
      try {
        const errorData = await res.json();
        if (errorData?.detail) {
          errorMsg = errorData.detail;
        }
      } catch {
        // Not JSON
      }
      throw new Error(errorMsg);
    }

    const blob = await res.blob();
    const downloadUrl = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = downloadUrl;
    a.download = customFilename || `Clinical_Report_${predictionId.slice(-8)}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.URL.revokeObjectURL(downloadUrl);
  },
};
