import { api, getAuthToken, setAuthToken, removeAuthToken } from "./api";
import type { LoginRequest, RegisterRequest, TokenResponse, User } from "../types";

export const authService = {
  /**
   * Registers a new user account with FastAPI backend.
   */
  register: async (payload: RegisterRequest): Promise<User> => {
    return api.post<User>("/auth/register", payload);
  },

  /**
   * Authenticates user credentials, acquires and stores JWT bearer token.
   */
  login: async (credentials: LoginRequest): Promise<TokenResponse> => {
    const data = await api.post<TokenResponse>("/auth/login", credentials);
    if (data.access_token) {
      setAuthToken(data.access_token);
    }
    return data;
  },

  /**
   * Retrieves the currently authenticated user's profile using JWT.
   */
  getMe: async (): Promise<User> => {
    return api.get<User>("/auth/me");
  },

  /**
   * Clears the stored JWT token.
   */
  logout: (): void => {
    removeAuthToken();
  },

  /**
   * Checks if an authentication token exists in storage.
   */
  isAuthenticated: (): boolean => {
    return !!getAuthToken();
  },

  /**
   * Retrieves the raw token string.
   */
  getToken: (): string | null => {
    return getAuthToken();
  },
};
