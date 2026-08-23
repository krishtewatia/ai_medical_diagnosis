import React, { createContext, useContext, useEffect, useState, useCallback } from "react";
import type { User } from "../types";
import { authService } from "../services/authService";

interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, pass: string) => Promise<User>;
  register: (name: string, email: string, pass: string) => Promise<User>;
  logout: () => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(authService.getToken());
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const fetchCurrentUser = useCallback(async () => {
    if (!authService.isAuthenticated()) {
      setUser(null);
      setToken(null);
      setIsLoading(false);
      return;
    }

    try {
      const currentUser = await authService.getMe();
      setUser(currentUser);
      setToken(authService.getToken());
    } catch {
      // Token invalid or expired
      authService.logout();
      setUser(null);
      setToken(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchCurrentUser();
  }, [fetchCurrentUser]);

  const login = async (email: string, pass: string): Promise<User> => {
    setIsLoading(true);
    try {
      await authService.login({ email, password: pass });
      const currentUser = await authService.getMe();
      setUser(currentUser);
      setToken(authService.getToken());
      return currentUser;
    } catch (err) {
      authService.logout();
      setUser(null);
      setToken(null);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const register = async (name: string, email: string, pass: string): Promise<User> => {
    setIsLoading(true);
    try {
      // 1. Create account
      await authService.register({ name, email, password: pass });
      // 2. Automatically log in to establish session
      await authService.login({ email, password: pass });
      // 3. Load user identity
      const currentUser = await authService.getMe();
      setUser(currentUser);
      setToken(authService.getToken());
      return currentUser;
    } catch (err) {
      authService.logout();
      setUser(null);
      setToken(null);
      throw err;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = () => {
    authService.logout();
    setUser(null);
    setToken(null);
  };

  const refreshUser = async () => {
    await fetchCurrentUser();
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user && !!token,
        isLoading,
        login,
        register,
        logout,
        refreshUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
};
