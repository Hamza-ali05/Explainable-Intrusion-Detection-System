import { BrowserRouter, Routes, Route } from "react-router-dom";
import NavBar from "./components/NavBar";
import StatusPage from "./pages/StatusPage";
import EvaluatePage from "./pages/EvaluatePage";
import XaiGlobalPage from "./pages/XaiGlobalPage";
import XaiLocalPage from "./pages/XaiLocalPage";
import XaiQualityPage from "./pages/XaiQualityPage";
import PredictPage from "./pages/PredictPage";
import ReportsPage from "./pages/ReportsPage";

export default function App() {
  return (
    <BrowserRouter>
      <NavBar />
      <main className="mx-auto max-w-6xl px-4 py-6">
        <Routes>
          <Route path="/" element={<StatusPage />} />
          <Route path="/status" element={<StatusPage />} />
          <Route path="/evaluate" element={<EvaluatePage />} />
          <Route path="/xai/global" element={<XaiGlobalPage />} />
          <Route path="/xai/local" element={<XaiLocalPage />} />
          <Route path="/xai/quality" element={<XaiQualityPage />} />
          <Route path="/predict" element={<PredictPage />} />
          <Route path="/reports" element={<ReportsPage />} />
        </Routes>
      </main>
    </BrowserRouter>
  );
}
