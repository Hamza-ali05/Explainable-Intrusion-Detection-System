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
import { Loader2, RefreshCw } from "lucide-react";
import {
  getPredictSamples,
  predictFile,
  predictSingle,
  refreshPredictSample,
} from "../api/client";
import Badge from "../components/Badge";
import Card from "../components/Card";
import DataTable from "../components/DataTable";
import ErrorCard from "../components/ErrorCard";
import PageSkeleton from "../components/PageSkeleton";
import Select from "../components/Select";
import { MODELS, THREAT_CLASSES, apiError } from "../utils/constants";

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

const PAGE_SIZE = 50;

function classColour(label) {
  return CLASS_COLOURS[label] || "#64748b";
}

function badgeType(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/\s+/g, "");
}

function probabilityChartData(probabilities) {
  if (!probabilities) return [];
  return Object.entries(probabilities)
    .map(([label, value]) => ({
      label,
      value: Number(value),
    }))
    .sort((a, b) => b.value - a.value);
}

function distributionChartData(labelCounts) {
  if (!labelCounts) return [];
  return Object.entries(labelCounts)
    .filter(([, count]) => count > 0)
    .map(([label, count]) => ({ label, count }))
    .sort((a, b) => b.count - a.count);
}

export default function PredictPage() {
  const fileInputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [batchModel, setBatchModel] = useState("");
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchResult, setBatchResult] = useState(null);
  const [batchPage, setBatchPage] = useState(0);
  const [dragOver, setDragOver] = useState(false);

  const [samples, setSamples] = useState({});
  const [sampleType, setSampleType] = useState("");
  const [formValues, setFormValues] = useState({});
  const [singleModel, setSingleModel] = useState("");
  const [singleLoading, setSingleLoading] = useState(false);
  const [singleResult, setSingleResult] = useState(null);
  const [error, setError] = useState(null);
  const [samplesLoading, setSamplesLoading] = useState(true);
  const [refreshingSample, setRefreshingSample] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const data = await getPredictSamples();
        setSamples(data);
      } catch (err) {
        setError({ message: apiError(err) });
        setSamples({});
      } finally {
        setSamplesLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (!sampleType || !samples[sampleType]) {
      setFormValues({});
      return;
    }
    setFormValues({ ...samples[sampleType] });
    setSingleResult(null);
  }, [sampleType, samples]);

  const acceptFile = useCallback((nextFile) => {
    if (!nextFile) return;
    if (!nextFile.name.toLowerCase().endsWith(".csv")) return;
    setFile(nextFile);
    setBatchResult(null);
    setBatchPage(0);
  }, []);

  const handleBatchPredict = async () => {
    if (!file || !batchModel) return;
    setBatchLoading(true);
    setBatchPage(0);
    setError(null);
    try {
      const data = await predictFile(file, batchModel);
      setBatchResult(data);
    } catch (err) {
      setBatchResult(null);
      setError({ message: apiError(err) });
    } finally {
      setBatchLoading(false);
    }
  };

  const handleRefreshSample = async () => {
    if (!sampleType) return;
    setRefreshingSample(true);
    setError(null);
    try {
      const data = await refreshPredictSample(sampleType);
      setFormValues({ ...data.features });
      setSingleResult(null);
    } catch (err) {
      setError({ message: apiError(err) });
    } finally {
      setRefreshingSample(false);
    }
  };

  const handleSinglePredict = async () => {
    if (!singleModel || !sampleType) return;
    setSingleLoading(true);
    setError(null);
    try {
      const data = await predictSingle(singleModel, formValues);
      setSingleResult(data);
    } catch (err) {
      setSingleResult(null);
      setError({ message: apiError(err) });
    } finally {
      setSingleLoading(false);
    }
  };

  const featureKeys = useMemo(() => Object.keys(formValues), [formValues]);

  const allRows = useMemo(() => {
    const rows = batchResult?.predictions ?? [];
    return rows.map((label, index) => ({
      "Row Index": index,
      "Predicted Class": (
        <Badge type={badgeType(label)}>{label}</Badge>
      ),
    }));
  }, [batchResult]);

  const pageRows = useMemo(() => {
    const start = batchPage * PAGE_SIZE;
    return allRows.slice(start, start + PAGE_SIZE);
  }, [allRows, batchPage]);

  const totalPages = allRows.length
    ? Math.ceil(allRows.length / PAGE_SIZE)
    : 0;

  const batchChartData = useMemo(
    () => distributionChartData(batchResult?.label_counts),
    [batchResult],
  );

  const singleChartData = useMemo(
    () => probabilityChartData(singleResult?.probabilities),
    [singleResult],
  );

  const handleDownload = () => {
    if (!batchResult?.predictions) return;
    const header = "row_index,predicted_label\n";
    const body = batchResult.predictions
      .map((label, i) => `${i},${label}`)
      .join("\n");
    const blob = new Blob([header + body], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <ErrorCard error={error} />

      <Card title="Batch Prediction from CSV">
        <div className="space-y-4">
          <div
            role="button"
            tabIndex={0}
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
            }}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              acceptFile(e.dataTransfer.files?.[0] ?? null);
            }}
            className={`cursor-pointer rounded-xl border-2 border-dashed px-6 py-10 text-center transition ${
              dragOver
                ? "border-sky-400 bg-sky-500/10"
                : "border-slate-600 bg-slate-900/40 hover:border-slate-500"
            }`}
          >
            <p className="text-sm text-slate-300">
              Drag and drop a CSV file here, or click to upload
            </p>
            <p className="mt-1 text-xs text-slate-500">Accepts .csv only</p>
            {file && (
              <p className="mt-3 text-sm font-medium text-sky-400">{file.name}</p>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => acceptFile(e.target.files?.[0] ?? null)}
          />

          <Select
            id="batch-model"
            label="Model"
            value={batchModel}
            onChange={(e) => setBatchModel(e.target.value)}
            options={MODELS}
            className="min-w-[200px]"
          />

          <button
            type="button"
            onClick={handleBatchPredict}
            disabled={!file || !batchModel || batchLoading}
            className="inline-flex items-center gap-2 rounded-lg bg-sky-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {batchLoading && <Loader2 className="h-4 w-4 animate-spin" />}
            Run Prediction
          </button>
        </div>
      </Card>

      <Card title="Test with Sample Values">
        <div className="space-y-4">
          {samplesLoading && <PageSkeleton />}

          {!samplesLoading && (
          <>
          <div className="flex flex-wrap items-end gap-3">
            <Select
              id="sample-type"
              label="Sample Attack Type"
              value={sampleType}
              onChange={(e) => setSampleType(e.target.value)}
              options={THREAT_CLASSES}
              className="min-w-[200px]"
            />

            <button
              type="button"
              onClick={handleRefreshSample}
              disabled={!sampleType || refreshingSample}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-600 bg-slate-800 px-4 py-2 text-sm font-medium text-slate-100 transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {refreshingSample ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              Refresh sample values
            </button>
          </div>

          {sampleType && featureKeys.length > 0 && (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
              {featureKeys.map((key) => {
                const raw = formValues[key];
                const isNumber =
                  typeof raw === "number" ||
                  (raw !== "" && raw != null && !Number.isNaN(Number(raw)));
                return (
                  <div key={key}>
                    <label
                      htmlFor={`feature-${key}`}
                      className="mb-1 block text-xs text-slate-400"
                    >
                      {key}
                    </label>
                    <input
                      id={`feature-${key}`}
                      type={isNumber ? "number" : "text"}
                      value={raw ?? ""}
                      onChange={(e) => {
                        const next = e.target.value;
                        setFormValues((prev) => ({
                          ...prev,
                          [key]: isNumber && next !== "" ? Number(next) : next,
                        }));
                      }}
                      className="w-full rounded-lg border border-slate-600 bg-slate-700 px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-sky-500"
                    />
                  </div>
                );
              })}
            </div>
          )}

          <Select
            id="single-model"
            label="Model"
            value={singleModel}
            onChange={(e) => setSingleModel(e.target.value)}
            options={MODELS}
            className="min-w-[200px]"
          />

          <button
            type="button"
            onClick={handleSinglePredict}
            disabled={!singleModel || !sampleType || singleLoading}
            className="inline-flex items-center gap-2 rounded-lg bg-sky-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {singleLoading && <Loader2 className="h-4 w-4 animate-spin" />}
            Predict Single Row
          </button>

          {singleResult && (
            <Card title="Prediction Result">
              <div className="mb-4 flex flex-wrap items-center gap-3">
                <span className="text-sm text-slate-400">Predicted class</span>
                <Badge type={badgeType(singleResult.predicted_class)}>
                  {singleResult.predicted_class}
                </Badge>
                <span
                  className="text-2xl font-bold"
                  style={{ color: classColour(singleResult.predicted_class) }}
                >
                  {singleResult.predicted_class}
                </span>
              </div>

              {singleChartData.length > 0 && (
                <div className="h-56 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={singleChartData} layout="vertical" margin={{ left: 8, right: 16 }}>
                      <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11, fill: "#94a3b8" }} />
                      <YAxis
                        type="category"
                        dataKey="label"
                        width={90}
                        tick={{ fontSize: 11, fill: "#94a3b8" }}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#1e293b",
                          border: "1px solid #334155",
                          borderRadius: 8,
                          color: "#f1f5f9",
                        }}
                      />
                      <Bar dataKey="value">
                        {singleChartData.map((entry) => (
                          <Cell
                            key={entry.label}
                            fill={classColour(entry.label)}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </Card>
          )}
          </>
          )}
        </div>
      </Card>

      {batchResult && (
        <Card title="Prediction Results">
          <div className="space-y-6">
            <div className="flex flex-wrap items-start gap-8">
              <div>
                <div className="text-2xl font-bold text-sky-400">
                  {batchResult.total_rows}
                </div>
                <div className="text-xs text-slate-400">Total rows</div>
              </div>

              {batchChartData.length > 0 && (
                <div className="h-40 min-w-[260px] flex-1">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={batchChartData} layout="vertical" margin={{ left: 8, right: 16 }}>
                      <XAxis type="number" hide />
                      <YAxis
                        type="category"
                        dataKey="label"
                        width={90}
                        tick={{ fontSize: 11, fill: "#94a3b8" }}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#1e293b",
                          border: "1px solid #334155",
                          borderRadius: 8,
                          color: "#f1f5f9",
                        }}
                      />
                      <Bar dataKey="count">
                        {batchChartData.map((entry) => (
                          <Cell
                            key={entry.label}
                            fill={classColour(entry.label)}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            <div className="overflow-x-auto">
              <DataTable data={pageRows} />
            </div>

            {totalPages > 1 && (
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => setBatchPage((p) => Math.max(0, p - 1))}
                  disabled={batchPage === 0}
                  className="rounded-lg bg-slate-700 px-3 py-1.5 text-sm text-slate-200 transition hover:bg-slate-600 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Previous
                </button>
                <span className="text-sm text-slate-400">
                  Page {batchPage + 1} of {totalPages}
                </span>
                <button
                  type="button"
                  onClick={() => setBatchPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={batchPage >= totalPages - 1}
                  className="rounded-lg bg-slate-700 px-3 py-1.5 text-sm text-slate-200 transition hover:bg-slate-600 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Next
                </button>
              </div>
            )}

            <button
              type="button"
              onClick={handleDownload}
              className="rounded-lg bg-slate-700 px-4 py-2 text-sm text-slate-100 transition hover:bg-slate-600"
            >
              Download CSV
            </button>
          </div>
        </Card>
      )}
    </div>
  );
}
