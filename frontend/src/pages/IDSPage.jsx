import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Loader2 } from "lucide-react";
import {
  deleteIDSHistory,
  exportIDSPdf,
  getIDSHistory,
  getIDSFeatures,
  getIDSIpLookup,
  getIDSShapPlot,
  getPredictSamples,
  idsAnalyze,
  idsAnalyzeSingle,
} from "../api/client";
import Badge from "../components/Badge";
import ImageCard from "../components/ImageCard";
import { apiError } from "../utils/constants";

const CLASS_COLOURS = {
  BENIGN: "#4ade80",
  DDoS: "#f87171",
  DoS: "#fb923c",
  PortScan: "#fbbf24",
  BruteForce: "#c084fc",
  WebAttack: "#f472b6",
  Infiltration: "#38bdf8",
  Botnet: "#fb7185",
};

const SEVERITY_COLOURS = {
  Critical: "#ef4444",
  High: "#f59e0b",
  Medium: "#38bdf8",
  Low: "#64748b",
  None: "#4ade80",
};

const THREAT_CLASSES = [
  "BENIGN", "DDoS", "DoS", "PortScan", "BruteForce", "WebAttack", "Infiltration", "Botnet",
];

const PAGE_SIZE = 20;

// ── tiny helpers ──────────────────────────────────────────────────────────

function classColour(label) {
  return CLASS_COLOURS[label] || "#64748b";
}

function severityColour(sev) {
  return SEVERITY_COLOURS[sev] || "#64748b";
}

function pct(v) {
  return `${(v * 100).toFixed(1)}%`;
}

function timestamp() {
  return new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
}

function formatHistoryTime(ts) {
  if (!ts) return "—";
  const then = new Date(ts);
  const now = new Date();
  const diffMs = now - then;
  const dayMs = 24 * 60 * 60 * 1000;
  if (diffMs < dayMs) {
    const hours = Math.floor(diffMs / (60 * 60 * 1000));
    if (hours >= 1) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
    const minutes = Math.max(1, Math.floor(diffMs / (60 * 1000)));
    return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  }
  return then.toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// ── sub-components ────────────────────────────────────────────────────────

function SummaryBar({ data }) {
  const chartData = Object.entries(data.summary)
    .filter(([, v]) => v > 0)
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count);

  return (
    <div className="bg-slate-800 rounded-2xl border border-slate-700 shadow-lg p-5 flex flex-wrap items-center gap-6">
      <div>
        <p className="text-2xl font-bold text-sky-400">{data.total_rows}</p>
        <p className="text-xs text-slate-400">Flows analysed</p>
      </div>
      {chartData.length > 0 && (
        <div className="h-32 min-w-[240px] flex-1">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 16 }}>
              <XAxis type="number" hide />
              <YAxis type="category" dataKey="label" width={90} tick={{ fontSize: 11, fill: "#94a3b8" }} />
              <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, color: "#f1f5f9" }} />
              <Bar dataKey="count">
                {chartData.map((e) => (
                  <Cell key={e.label} fill={classColour(e.label)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function SeverityChips({ results }) {
  const counts = {};
  results.forEach((r) => {
    counts[r.severity] = (counts[r.severity] || 0) + 1;
  });
  const chips = Object.entries(counts).filter(([, v]) => v > 0);
  if (chips.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-2">
      {chips.map(([sev, count]) => (
        <span
          key={sev}
          className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold"
          style={{ backgroundColor: severityColour(sev) + "22", color: severityColour(sev) }}
        >
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: severityColour(sev) }} />
          {sev}: {count}
        </span>
      ))}
    </div>
  );
}

function ProbChart({ probabilities }) {
  const chartData = Object.entries(probabilities)
    .map(([label, value]) => ({ label, value: Number(value) }))
    .sort((a, b) => b.value - a.value);

  return (
    <div className="h-48 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 16 }}>
          <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 10, fill: "#94a3b8" }} />
          <YAxis type="category" dataKey="label" width={85} tick={{ fontSize: 10, fill: "#94a3b8" }} />
          <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, color: "#f1f5f9" }} />
          <Bar dataKey="value">
            {chartData.map((e) => (
              <Cell key={e.label} fill={classColour(e.label)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ShapChart({ shapTop15 }) {
  if (!shapTop15 || shapTop15.length === 0) return null;
  const data = [...shapTop15].reverse();

  return (
    <div style={{ height: Math.max(200, data.length * 24) }} className="w-full">
      <p className="text-sm font-medium text-slate-300 mb-2">Top features (SHAP)</p>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
          <XAxis type="number" tick={{ fontSize: 10, fill: "#94a3b8" }} />
          <YAxis type="category" dataKey="feature" width={160} tick={{ fontSize: 10, fill: "#94a3b8" }} />
          <Tooltip contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, color: "#f1f5f9" }} />
          <Bar dataKey="shap_value">
            {data.map((e, idx) => (
              <Cell key={idx} fill={e.direction === "positive" ? "#38bdf8" : "#ef4444"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function DetailPanel({ row, onClose }) {
  const [shapImage, setShapImage] = useState(null);
  const [shapLoading, setShapLoading] = useState(false);

  const handleViewShap = async () => {
    setShapLoading(true);
    try {
      const data = await getIDSShapPlot(row.features);
      setShapImage(data.image ?? null);
    } finally {
      setShapLoading(false);
    }
  };

  const handleExportPdf = async () => {
    const blob = await exportIDSPdf(row);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ids_report_${Date.now()}.pdf`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <tr>
      <td colSpan={5} className="p-0">
        <div className="bg-slate-900 rounded-xl p-4 border border-slate-700 m-2">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div>
                <p className="text-xs font-medium text-slate-400 mb-1">Narrative</p>
                <p className="text-sm text-slate-300 leading-relaxed">{row.narrative}</p>
              </div>
              <div>
                <p className="text-xs font-medium text-slate-400 mb-1">Probability distribution</p>
                <ProbChart probabilities={row.probabilities} />
              </div>
            </div>
            <div className="space-y-4">
              <ShapChart shapTop15={row.shap_top15} />
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={handleViewShap}
                  className="rounded-lg bg-slate-700 px-3 py-1.5 text-xs text-slate-100 transition hover:bg-slate-600"
                >
                  View SHAP Plot
                </button>
                <button
                  type="button"
                  onClick={handleExportPdf}
                  className="rounded-lg bg-emerald-700 px-3 py-1.5 text-xs text-white transition hover:bg-emerald-600"
                >
                  Export PDF
                </button>
              </div>
              {(shapLoading || shapImage) && (
                <ImageCard
                  title="SHAP Waterfall Plot"
                  base64={shapImage}
                  loading={shapLoading}
                  className="bg-slate-800"
                />
              )}
              <div>
                <p className="text-xs font-medium text-slate-400 mb-1">Mitigation steps</p>
                <ol className="space-y-1">
                  {row.mitigation.map((step, i) => (
                    <li key={i} className="text-sm text-slate-300">
                      <span className="text-sky-400 font-semibold mr-2">{i + 1}.</span>
                      {step}
                    </li>
                  ))}
                </ol>
              </div>
            </div>
          </div>
          <hr className="border-slate-700 my-3" />
          <button
            type="button"
            onClick={onClose}
            className="text-xs text-slate-400 hover:text-slate-200 transition"
          >
            Close
          </button>
        </div>
      </td>
    </tr>
  );
}

// ── main page ─────────────────────────────────────────────────────────────

export default function IDSPage() {
  const fileInputRef = useRef(null);

  const [tab, setTab] = useState("upload");
  const [file, setFile] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const [features, setFeatures] = useState(null);
  const [formValues, setFormValues] = useState({});
  const [sampleType, setSampleType] = useState("");
  const [samples, setSamples] = useState({});
  const [lookupIp, setLookupIp] = useState("");
  const [ipLookup, setIpLookup] = useState(null);

  const [expandedRow, setExpandedRow] = useState(null);
  const [page, setPage] = useState(0);
  const [history, setHistory] = useState([]);

  const fetchHistory = useCallback(async () => {
    try {
      const data = await getIDSHistory();
      setHistory(data.sessions ?? []);
    } catch {
      setHistory([]);
    }
  }, []);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const ensureFeaturesLoaded = useCallback(async () => {
    if (features) return features;
    const data = await getIDSFeatures();
    setFeatures(data.features);
    const init = {};
    data.features.forEach((f) => {
      init[f] = 0;
    });
    setFormValues(init);
    return data.features;
  }, [features]);

  useEffect(() => {
    if (tab === "manual" && !features) {
      ensureFeaturesLoaded().catch((err) => setError(apiError(err)));
    }
  }, [tab, features, ensureFeaturesLoaded]);

  const loadSamples = useCallback(async () => {
    if (Object.keys(samples).length > 0) return;
    try {
      const data = await getPredictSamples();
      setSamples(data);
    } catch {
      /* ignore — user can still type manually */
    }
  }, [samples]);

  const acceptFile = useCallback((f) => {
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".csv")) return;
    setFile(f);
    setResult(null);
    setPage(0);
  }, []);

  const handleUploadAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await idsAnalyze(file);
      setResult(data);
      setPage(0);
      setExpandedRow(null);
      fetchHistory();
    } catch (err) {
      setError(apiError(err));
    } finally {
      setLoading(false);
    }
  };

  const handleManualAnalyze = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await idsAnalyzeSingle(formValues);
      setResult({ total_rows: 1, summary: { [data.predicted_class]: 1 }, results: [data] });
      setPage(0);
      setExpandedRow(null);
      fetchHistory();
    } catch (err) {
      setError(apiError(err));
    } finally {
      setLoading(false);
    }
  };

  const handleSampleFill = (cls) => {
    setSampleType(cls);
    if (samples[cls]) {
      const next = {};
      if (features) features.forEach((f) => { next[f] = 0; });
      Object.entries(samples[cls]).forEach(([k, v]) => { next[k] = v; });
      setFormValues(next);
    }
  };

  const handleIpLookup = async () => {
    if (!lookupIp.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await getIDSIpLookup(lookupIp.trim());
      setIpLookup(data);
    } catch (err) {
      setError(apiError(err));
      setIpLookup(null);
    } finally {
      setLoading(false);
    }
  };

  const handleUseTemplate = async () => {
    if (!ipLookup?.suggested_class) return;
    setTab("manual");
    try {
      const currentFeatures = await ensureFeaturesLoaded();
      let nextSamples = samples;
      if (Object.keys(nextSamples).length === 0) {
        nextSamples = await getPredictSamples();
        setSamples(nextSamples);
      }
      const next = {};
      currentFeatures.forEach((f) => {
        next[f] = 0;
      });
      Object.entries(nextSamples[ipLookup.suggested_class] || {}).forEach(([k, v]) => {
        next[k] = v;
      });
      setSampleType(ipLookup.suggested_class);
      setFormValues(next);
    } catch (err) {
      setError(apiError(err));
    }
  };

  const rows = result?.results ?? [];
  const pageRows = useMemo(() => rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE), [rows, page]);
  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));

  const handleDownload = () => {
    if (!rows.length) return;
    const header = "row_index,predicted_class,confidence,severity\n";
    const body = rows.map((r) => `${r.row_index},${r.predicted_class},${r.confidence},${r.severity}`).join("\n");
    const blob = new Blob([header + body], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ids_results_${timestamp()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDeleteHistory = async (sessionId) => {
    try {
      const data = await deleteIDSHistory(sessionId);
      if (data.deleted) {
        setHistory((prev) => prev.filter((row) => row.id !== sessionId));
      }
    } catch (err) {
      setError(apiError(err));
    }
  };

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {/* Section A — Analysis Input */}
      <div className="bg-slate-800 rounded-2xl border border-slate-700 shadow-lg p-6">
        <p className="text-lg font-semibold text-slate-100 mb-4">Intrusion Detection</p>

        <div className="flex gap-2 mb-4">
          {["upload", "manual", "ip-lookup"].map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                tab === t
                  ? "bg-sky-600 text-white"
                  : "bg-slate-700 text-slate-300 hover:bg-slate-600"
              }`}
            >
              {t === "upload"
                ? "Upload CSV"
                : t === "manual"
                  ? "Manual Entry"
                  : "IP Lookup"}
            </button>
          ))}
        </div>

        {tab === "upload" && (
          <div className="space-y-3">
            <div
              role="button"
              tabIndex={0}
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click(); }}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => { e.preventDefault(); setDragOver(false); acceptFile(e.dataTransfer.files?.[0]); }}
              className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
                dragOver ? "border-sky-500 bg-sky-500/10" : "border-slate-600 hover:border-sky-500"
              }`}
            >
              <p className="text-sm text-slate-300">Drag and drop a CSV file here, or click to upload</p>
              <p className="text-xs text-slate-500 mt-2">CSV format. First 500 rows will be analysed.</p>
              {file && <p className="text-sm text-slate-300 mt-3 font-medium">{file.name}</p>}
            </div>
            <input ref={fileInputRef} type="file" accept=".csv" className="hidden" onChange={(e) => acceptFile(e.target.files?.[0])} />
            <button
              type="button"
              onClick={handleUploadAnalyze}
              disabled={!file || loading}
              className="bg-sky-500 hover:bg-sky-400 text-white font-medium rounded-lg px-6 py-2 text-sm transition disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-2"
            >
              {loading && <Loader2 className="h-4 w-4 animate-spin" />}
              Analyse
            </button>
          </div>
        )}

        {tab === "manual" && features && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-end gap-3">
              <button type="button" onClick={loadSamples} className="bg-slate-700 hover:bg-slate-600 text-slate-100 rounded-lg px-4 py-2 text-sm transition">
                Sample values
              </button>
              {Object.keys(samples).length > 0 && (
                <select
                  value={sampleType}
                  onChange={(e) => handleSampleFill(e.target.value)}
                  className="bg-slate-700 border border-slate-600 text-slate-100 rounded-lg px-3 py-2 text-sm"
                >
                  <option value="">Select attack type</option>
                  {THREAT_CLASSES.map((c) => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>
              )}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
              {features.map((f) => (
                <div key={f}>
                  <label className="text-xs text-slate-400 mb-1 block">{f}</label>
                  <input
                    type="number"
                    value={formValues[f] ?? 0}
                    onChange={(e) => setFormValues((prev) => ({ ...prev, [f]: Number(e.target.value) || 0 }))}
                    className="w-full bg-slate-700 border border-slate-600 text-slate-100 rounded-lg px-3 py-1.5 text-sm focus:ring-2 focus:ring-sky-500"
                  />
                </div>
              ))}
            </div>

            <button
              type="button"
              onClick={handleManualAnalyze}
              disabled={loading}
              className="bg-sky-500 hover:bg-sky-400 text-white font-medium rounded-lg px-6 py-2 text-sm transition disabled:opacity-50 disabled:cursor-not-allowed inline-flex items-center gap-2"
            >
              {loading && <Loader2 className="h-4 w-4 animate-spin" />}
              Analyse
            </button>
          </div>
        )}

        {tab === "ip-lookup" && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center gap-3">
              <input
                type="text"
                value={lookupIp}
                onChange={(e) => setLookupIp(e.target.value)}
                placeholder="Enter a public IP address (e.g. 8.8.8.8)"
                className="w-full max-w-sm rounded-lg border border-slate-600 bg-slate-700 px-4 py-2 text-sm text-slate-100 focus:ring-2 focus:ring-sky-500"
              />
              <button
                type="button"
                onClick={handleIpLookup}
                disabled={loading || !lookupIp.trim()}
                className="inline-flex items-center gap-2 rounded-lg bg-sky-500 px-6 py-2 text-sm font-medium text-white transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading && <Loader2 className="h-4 w-4 animate-spin" />}
                Look up
              </button>
            </div>

            {ipLookup && (
              <div className="rounded-xl border border-slate-700 bg-slate-900 p-4">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <tbody>
                      {[
                        ["IP Address", ipLookup.ip],
                        ["Country", ipLookup.country ?? "—"],
                        ["Organisation / ISP", ipLookup.org ?? "—"],
                        ["ASN", ipLookup.asn ?? "—"],
                        ["Is VPN", ipLookup.is_vpn ? "Yes" : "No"],
                        ["Is Tor Exit Node", ipLookup.is_tor ? "Yes" : "No"],
                        ["Is Datacenter/Cloud", ipLookup.is_datacenter ? "Yes" : "No"],
                      ].map(([label, value]) => (
                        <tr key={label} className="border-b border-slate-700/60">
                          <td className="py-2 pr-4 text-slate-400">{label}</td>
                          <td className="py-2 text-slate-200">{value}</td>
                        </tr>
                      ))}
                      <tr className="border-b border-slate-700/60">
                        <td className="py-2 pr-4 text-slate-400">Threat Score</td>
                        <td className="py-2">
                          <div className="flex items-center gap-2">
                            <div className="h-2 w-24 rounded-full bg-slate-700">
                              <div
                                className="h-2 rounded-full bg-sky-500"
                                style={{ width: `${ipLookup.threat_score}%` }}
                              />
                            </div>
                            <span className="text-xs text-slate-300">{ipLookup.threat_score}%</span>
                          </div>
                        </td>
                      </tr>
                      <tr>
                        <td className="py-2 pr-4 text-slate-400">Suggested Class</td>
                        <td className="py-2">
                          <Badge type={String(ipLookup.suggested_class).toLowerCase()}>
                            {ipLookup.suggested_class}
                          </Badge>
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>

                <button
                  type="button"
                  onClick={handleUseTemplate}
                  className="mt-4 rounded-lg bg-slate-700 px-4 py-2 text-sm text-slate-100 transition hover:bg-slate-600"
                >
                  Use as analysis template
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Section B — Results */}
      {result && (
        <>
          <SummaryBar data={result} />
          <SeverityChips results={rows} />

          <div className="bg-slate-800 rounded-2xl border border-slate-700 shadow-lg p-5">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
              <p className="text-sm text-slate-400">
                Showing rows {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, rows.length)} of {rows.length}
              </p>
              <button
                type="button"
                onClick={handleDownload}
                className="bg-slate-700 hover:bg-slate-600 text-slate-100 rounded-lg px-4 py-2 text-sm transition"
              >
                Download CSV
              </button>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead>
                  <tr className="text-xs text-slate-400 border-b border-slate-700">
                    <th className="py-2 pr-4">Row</th>
                    <th className="py-2 pr-4">Prediction</th>
                    <th className="py-2 pr-4">Confidence</th>
                    <th className="py-2 pr-4">Severity</th>
                    <th className="py-2">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {pageRows.map((r) => (
                    <>
                      <tr key={r.row_index} className="border-b border-slate-700/50 hover:bg-slate-700/30">
                        <td className="py-2 pr-4 text-slate-300">{r.row_index + 1}</td>
                        <td className="py-2 pr-4">
                          <Badge type={r.predicted_class.toLowerCase()}>{r.predicted_class}</Badge>
                        </td>
                        <td className="py-2 pr-4 text-slate-300">{pct(r.confidence)}</td>
                        <td className="py-2 pr-4">
                          <Badge type={r.severity.toLowerCase()}>{r.severity}</Badge>
                        </td>
                        <td className="py-2">
                          <button
                            type="button"
                            onClick={() => setExpandedRow(expandedRow === r.row_index ? null : r.row_index)}
                            className="text-xs text-sky-400 hover:text-sky-300 transition"
                          >
                            {expandedRow === r.row_index ? "Hide" : "Details"}
                          </button>
                        </td>
                      </tr>
                      {expandedRow === r.row_index && (
                        <DetailPanel
                          key={`detail-${r.row_index}`}
                          row={r}
                          onClose={() => setExpandedRow(null)}
                        />
                      )}
                    </>
                  ))}
                </tbody>
              </table>
            </div>

            {totalPages > 1 && (
              <div className="flex items-center gap-3 mt-4">
                <button
                  type="button"
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={page === 0}
                  className="rounded-lg bg-slate-700 px-3 py-1.5 text-sm text-slate-200 transition hover:bg-slate-600 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Prev
                </button>
                <span className="text-sm text-slate-400">
                  Page {page + 1} of {totalPages}
                </span>
                <button
                  type="button"
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={page >= totalPages - 1}
                  className="rounded-lg bg-slate-700 px-3 py-1.5 text-sm text-slate-200 transition hover:bg-slate-600 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            )}
          </div>
        </>
      )}

      <div className="bg-slate-800 rounded-2xl border border-slate-700 shadow-lg p-5">
        <p className="mb-4 text-lg font-semibold text-slate-100">Recent analyses</p>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-700 text-xs text-slate-400">
                <th className="py-2 pr-4">Time</th>
                <th className="py-2 pr-4">Rows</th>
                <th className="py-2 pr-4">Benign</th>
                <th className="py-2 pr-4">Threats</th>
                <th className="py-2 pr-4">Top Threat</th>
                <th className="py-2 pr-4">Model</th>
                <th className="py-2 pr-4">Input Method</th>
                <th className="py-2">Actions</th>
              </tr>
            </thead>
            <tbody>
              {history.map((session) => (
                <tr key={session.id} className="border-b border-slate-700/50">
                  <td className="py-2 pr-4 text-slate-300">{formatHistoryTime(session.session_timestamp)}</td>
                  <td className="py-2 pr-4 text-slate-300">{session.total_rows}</td>
                  <td className="py-2 pr-4 text-slate-300">{session.benign_count}</td>
                  <td className="py-2 pr-4 text-slate-300">{session.threat_count}</td>
                  <td className="py-2 pr-4">
                    {session.top_threat && session.top_threat !== "None" ? (
                      <Badge type={String(session.top_threat).toLowerCase()}>{session.top_threat}</Badge>
                    ) : (
                      <span className="text-slate-400">None</span>
                    )}
                  </td>
                  <td className="py-2 pr-4 text-slate-300">{session.model_used}</td>
                  <td className="py-2 pr-4 text-slate-300">{session.input_method}</td>
                  <td className="py-2">
                    <button
                      type="button"
                      onClick={() => handleDeleteHistory(session.id)}
                      className="rounded-lg bg-red-500/10 px-3 py-1 text-xs text-red-400 transition hover:bg-red-500/20 hover:text-red-300"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
              {history.length === 0 && (
                <tr>
                  <td colSpan={8} className="py-6 text-center text-sm text-slate-400">
                    No analysis sessions recorded yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
