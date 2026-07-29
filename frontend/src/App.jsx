import { useState } from "react";
import { BrowserRouter, Navigate, Outlet, Route, Routes } from "react-router-dom";
import { AuthProvider } from "./auth/AuthContext";
import ProtectedRoute from "./auth/ProtectedRoute";
import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";
import LoginPage from "./pages/LoginPage";
import IDSPage from "./pages/IDSPage";
import StatusPage from "./pages/StatusPage";
import EvaluatePage from "./pages/EvaluatePage";
import XaiGlobalPage from "./pages/XaiGlobalPage";
import XaiLocalPage from "./pages/XaiLocalPage";
import XaiQualityPage from "./pages/XaiQualityPage";
import PredictPage from "./pages/PredictPage";
import appBg from "./public/AI-network-intrusion-detection-1024x585.jpeg";

function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="relative min-h-screen">
      <div
        className="fixed inset-0 -z-10 bg-cover bg-center bg-no-repeat"
        style={{ backgroundImage: `url(${appBg})` }}
        aria-hidden="true"
      />
      <div className="fixed inset-0 -z-10 bg-slate-950/75" aria-hidden="true" />
      <Navbar />
      <Sidebar sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
      <main
        className={`min-h-screen overflow-y-auto p-6 pt-20 transition-all duration-300 ${
          sidebarOpen ? "ml-64" : "ml-14"
        }`}
      >
        <div className="mx-auto max-w-6xl space-y-6">
          <Outlet />
        </div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter
        future={{
          v7_startTransition: true,
          v7_relativeSplatPath: true,
        }}
      >
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route element={<ProtectedRoute />}>
            <Route element={<AppLayout />}>
              <Route path="/" element={<Navigate to="/ids" replace />} />
              <Route path="/ids" element={<IDSPage />} />
              <Route path="/status" element={<StatusPage />} />
              <Route path="/evaluate" element={<EvaluatePage />} />
              <Route path="/xai/global" element={<XaiGlobalPage />} />
              <Route path="/xai/local" element={<XaiLocalPage />} />
              <Route path="/xai/quality" element={<XaiQualityPage />} />
              <Route path="/predict" element={<PredictPage />} />
            </Route>
          </Route>
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
