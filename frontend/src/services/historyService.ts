import { api } from "./api";
import type { PredictionHistoryItem, PredictionHistoryListResponse } from "../types";

export const historyService = {
  /**
   * Retrieves paginated prediction history for authenticated user, with optional disease filter.
   */
  getHistory: async (params?: {
    disease?: string;
    skip?: number;
    limit?: number;
  }): Promise<PredictionHistoryListResponse> => {
    const query = new URLSearchParams();
    if (params?.disease) query.set("disease", params.disease);
    if (params?.skip !== undefined) query.set("skip", params.skip.toString());
    if (params?.limit !== undefined) query.set("limit", params.limit.toString());

    const qs = query.toString();
    return api.get<PredictionHistoryListResponse>(`/history${qs ? `?${qs}` : ""}`);
  },

  /**
   * Retrieves a single historical prediction record by ID.
   */
  getPredictionById: async (predictionId: string): Promise<PredictionHistoryItem> => {
    return api.get<PredictionHistoryItem>(`/history/${predictionId}`);
  },
};
