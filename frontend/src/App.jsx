import { useState } from "react";
import { BrowserRouter, Navigate, Outlet, Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import Sidebar from "./components/Sidebar";
import StatusPage from "./pages/StatusPage";
import EvaluatePage from "./pages/EvaluatePage";
import XaiGlobalPage from "./pages/XaiGlobalPage";
import XaiLocalPage from "./pages/XaiLocalPage";
import XaiQualityPage from "./pages/XaiQualityPage";
import PredictPage from "./pages/PredictPage";
import ReportsPage from "./pages/ReportsPage";

function AppLayout() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [liveAlertsEnabled, setLiveAlertsEnabled] = useState(false);

  return (
    <div className="min-h-screen bg-page">
      <Navbar />
      <div className="flex min-h-screen">
        <Sidebar
          collapsed={sidebarCollapsed}
          onToggle={() => setSidebarCollapsed((v) => !v)}
          liveAlertsEnabled={liveAlertsEnabled}
          setLiveAlertsEnabled={setLiveAlertsEnabled}
        />
        <main
          className={`min-h-screen flex-1 overflow-y-auto p-6 pt-20 transition-all duration-300 ${
            sidebarCollapsed ? "ml-14" : "ml-64"
          }`}
        >
          <div className="mx-auto max-w-6xl space-y-6">
            <Outlet context={{ liveAlertsEnabled }} />
          </div>
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter
      future={{
        v7_startTransition: true,
        v7_relativeSplatPath: true,
      }}
    >
      <Routes>
        <Route element={<AppLayout />}>
          <Route path="/" element={<Navigate to="/status" replace />} />
          <Route path="/status" element={<StatusPage />} />
          <Route path="/evaluate" element={<EvaluatePage />} />
          <Route path="/xai/global" element={<XaiGlobalPage />} />
          <Route path="/xai/local" element={<XaiLocalPage />} />
          <Route path="/xai/quality" element={<XaiQualityPage />} />
          <Route path="/predict" element={<PredictPage />} />
          <Route path="/reports" element={<ReportsPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
