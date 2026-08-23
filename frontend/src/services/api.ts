/**
 * Centralized API Client Service
 * Handles base URL configuration, JWT token injection, content headers,
 * and unified error formatting.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  data: any;

  constructor(status: number, message: string, data?: any) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
  }
}

export function getAuthToken(): string | null {
  return localStorage.getItem("token");
}

export function setAuthToken(token: string): void {
  localStorage.setItem("token", token);
}

export function removeAuthToken(): void {
  localStorage.removeItem("token");
}

export async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint.startsWith("/") ? endpoint : `/${endpoint}`}`;
  const headers = new Headers(options.headers || {});

  // Inject JWT token if available
  const token = getAuthToken();
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }

  // Set default JSON Content-Type if payload is not FormData
  if (!(options.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  let response: Response;
  try {
    response = await fetch(url, {
      ...options,
      headers,
    });
  } catch (err: any) {
    throw new ApiError(
      0,
      `Network connection error: Unable to reach backend at ${API_BASE_URL}.`,
      err
    );
  }

  if (!response.ok) {
    let errorDetail = `Request failed with status ${response.status}`;
    let errorData = null;

    try {
      errorData = await response.json();
      if (errorData && typeof errorData.detail === "string") {
        errorDetail = errorData.detail;
      } else if (errorData && Array.isArray(errorData.detail)) {
        // FastAPI Pydantic validation error array
        errorDetail = errorData.detail
          .map((d: any) => d.msg || JSON.stringify(d))
          .join("; ");
      } else if (errorData && typeof errorData.message === "string") {
        errorDetail = errorData.message;
      }
    } catch {
      // Body is not JSON
    }

    // Auto-clear invalid token on 401 Unauthorized (unless requesting login itself)
    if (response.status === 401 && !endpoint.includes("/auth/login")) {
      removeAuthToken();
    }

    throw new ApiError(response.status, errorDetail, errorData);
  }

  // If response has no content (204)
  if (response.status === 204) {
    return {} as T;
  }

  return response.json() as Promise<T>;
}

export const api = {
  get: <T>(endpoint: string, headers?: HeadersInit) =>
    apiRequest<T>(endpoint, { method: "GET", headers }),

  post: <T>(endpoint: string, data?: any, headers?: HeadersInit) => {
    const isFormData = data instanceof FormData;
    return apiRequest<T>(endpoint, {
      method: "POST",
      body: isFormData ? data : JSON.stringify(data),
      headers,
    });
  },

  patch: <T>(endpoint: string, data?: any, headers?: HeadersInit) =>
    apiRequest<T>(endpoint, {
      method: "PATCH",
      body: JSON.stringify(data),
      headers,
    }),

  put: <T>(endpoint: string, data?: any, headers?: HeadersInit) =>
    apiRequest<T>(endpoint, {
      method: "PUT",
      body: JSON.stringify(data),
      headers,
    }),

  delete: <T>(endpoint: string, headers?: HeadersInit) =>
    apiRequest<T>(endpoint, { method: "DELETE", headers }),
};
