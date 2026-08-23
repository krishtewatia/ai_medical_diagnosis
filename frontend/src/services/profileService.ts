import { api } from "./api";
import type { MedicalProfile, MedicalProfileCreate, MedicalProfileUpdate } from "../types";

export const profileService = {
  /**
   * Retrieves the medical profile for the authenticated user.
   */
  getProfile: async (): Promise<MedicalProfile> => {
    return api.get<MedicalProfile>("/profile");
  },

  /**
   * Creates a new medical profile for the authenticated user.
   */
  createProfile: async (payload: MedicalProfileCreate): Promise<MedicalProfile> => {
    return api.post<MedicalProfile>("/profile", payload);
  },

  /**
   * Partially updates an existing medical profile.
   */
  updateProfile: async (payload: MedicalProfileUpdate): Promise<MedicalProfile> => {
    return api.patch<MedicalProfile>("/profile", payload);
  },
};
