import { BrowserRouter, Navigate, Outlet, Route, Routes } from "react-router-dom";
import Navbar from "./components/Navbar";
import StatusPage from "./pages/StatusPage";
import EvaluatePage from "./pages/EvaluatePage";
import XaiGlobalPage from "./pages/XaiGlobalPage";
import XaiLocalPage from "./pages/XaiLocalPage";
import XaiQualityPage from "./pages/XaiQualityPage";
import PredictPage from "./pages/PredictPage";

function AppLayout() {
  return (
    <div className="min-h-screen bg-page">
      <Navbar />
      <main className="min-h-screen overflow-y-auto p-6 pt-28">
        <div className="mx-auto max-w-6xl space-y-6">
          <Outlet />
        </div>
      </main>
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
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
