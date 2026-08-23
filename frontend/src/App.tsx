import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import { MainLayout } from "./layouts/MainLayout";
import { ProtectedRoute } from "./components/ProtectedRoute";

import { HomePage } from "./pages/HomePage";
import { LoginPage } from "./pages/LoginPage";
import { RegisterPage } from "./pages/RegisterPage";
import { DashboardPage } from "./pages/DashboardPage";
import { DiseasesPage } from "./pages/DiseasesPage";
import { PredictionPage } from "./pages/PredictionPage";
import { ResultPage } from "./pages/ResultPage";
import { HistoryPage } from "./pages/HistoryPage";
import { HistoryDetailPage } from "./pages/HistoryDetailPage";
import { ReportPage } from "./pages/ReportPage";
import { ProfilePage } from "./pages/ProfilePage";
import { NotFoundPage } from "./pages/NotFoundPage";

export function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/" element={<MainLayout />}>
            {/* Public Routes */}
            <Route index element={<HomePage />} />
            <Route path="login" element={<LoginPage />} />
            <Route path="register" element={<RegisterPage />} />

            {/* Protected Routes (Auth Guard) */}
            <Route element={<ProtectedRoute />}>
              <Route path="dashboard" element={<DashboardPage />} />
              <Route path="profile" element={<ProfilePage />} />
              <Route path="diseases" element={<DiseasesPage />} />
              <Route path="predict/:diseaseId" element={<PredictionPage />} />
              <Route path="predict/:disease" element={<PredictionPage />} />
              <Route path="result" element={<ResultPage />} />
              <Route path="result/:predictionId" element={<ResultPage />} />
              <Route path="history" element={<HistoryPage />} />
              <Route path="history/:predictionId" element={<HistoryDetailPage />} />
              <Route path="reports/:predictionId" element={<ReportPage />} />
              <Route path="reports/:reportId" element={<ReportPage />} />
            </Route>

            {/* 404 Catch-all Fallback */}
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
