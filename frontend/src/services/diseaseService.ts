import { api } from "./api";
import type { DiseaseResponse } from "../types";

export const diseaseService = {
  /**
   * Retrieves all actively registered disease modules.
   */
  getAllDiseases: async (): Promise<DiseaseResponse[]> => {
    return api.get<DiseaseResponse[]>("/diseases");
  },

  /**
   * Retrieves full declarative metadata and input specifications for a specific disease module.
   */
  getDiseaseById: async (diseaseId: string): Promise<DiseaseResponse> => {
    return api.get<DiseaseResponse>(`/diseases/${diseaseId}`);
  },
};
