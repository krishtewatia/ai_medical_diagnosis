import { api } from "./api";
import type { PredictionRequest, PredictionResponse } from "../types";

export const predictionService = {
  /**
   * Executes tabular disease screening inference.
   */
  predictTabular: async (payload: PredictionRequest): Promise<PredictionResponse> => {
    return api.post<PredictionResponse>("/predictions", payload);
  },

  /**
   * Executes medical image screening inference via multipart file upload.
   */
  predictImage: async (diseaseId: string, imageFile: File): Promise<PredictionResponse> => {
    const formData = new FormData();
    formData.append("disease_id", diseaseId);
    formData.append("file", imageFile);

    return api.post<PredictionResponse>("/predictions/image", formData);
  },
};
