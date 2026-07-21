import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { predictFile } from "../api/client";
import LoadingSpinner from "../components/LoadingSpinner";
import { MODELS } from "../utils/constants";

const PAGE_SIZE = 20;

export default function PredictPage() {
  const [file, setFile] = useState(null);
  const [model, setModel] = useState("xgboost");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [page, setPage] = useState(0);

  const chartData = useMemo(() => {
    if (!result?.label_counts) return [];
    return Object.entries(result.label_counts)
      .filter(([, count]) => count > 0)
      .map(([label, count]) => ({ label, count }));
  }, [result]);

  const displayPredictions = useMemo(
    () => (result?.predictions ?? []).slice(0, 100),
    [result],
  );

  const pageRows = useMemo(() => {
    const start = page * PAGE_SIZE;
    return displayPredictions.slice(start, start + PAGE_SIZE);
  }, [displayPredictions, page]);

  const totalPages = result
    ? Math.ceil(displayPredictions.length / PAGE_SIZE)
    : 0;

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setPage(0);
    try {
      const data = await predictFile(file, model);
      setResult(data);
    } catch (err) {
      alert(err.response?.data?.detail ?? err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!result?.predictions) return;
    const header = "row_index,predicted_label\n";
    const body = result.predictions
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
    <div>
      <h1 className="mb-4 text-lg font-medium">Predict</h1>

      <div className="mb-4 space-y-4">
        <div>
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            className="text-sm"
          />
          {file && <span className="ml-2 text-sm text-gray-600">{file.name}</span>}
        </div>

        <div>
          <label className="mr-2 text-sm">Model</label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="rounded border border-gray-400 px-2 py-1 text-sm"
          >
            {MODELS.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
        </div>

        <button
          type="button"
          onClick={handlePredict}
          disabled={!file || loading}
          className="rounded border border-gray-400 px-4 py-1 text-sm hover:bg-gray-50 disabled:opacity-50"
        >
          Run Prediction
        </button>
      </div>

      {loading && <LoadingSpinner />}

      {result && !loading && (
        <div className="space-y-6">
          <div className="flex flex-wrap items-start gap-8">
            <div>
              <div className="text-2xl font-semibold">{result.total_rows}</div>
              <div className="text-xs text-gray-600">Total Rows</div>
            </div>
            {chartData.length > 0 && (
              <div className="h-32 flex-1 min-w-[200px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} layout="vertical">
                    <XAxis type="number" hide />
                    <YAxis type="category" dataKey="label" width={80} tick={{ fontSize: 11 }} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#64748b" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="border-b border-gray-300 text-left">
                <th className="py-2 pr-4">Row Index</th>
                <th className="py-2">Predicted Class</th>
              </tr>
            </thead>
            <tbody>
              {pageRows.map((label, i) => (
                <tr key={page * PAGE_SIZE + i} className="border-b border-gray-200">
                  <td className="py-2 pr-4">{page * PAGE_SIZE + i}</td>
                  <td className="py-2">{label}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {totalPages > 1 && (
            <div className="flex items-center gap-2 text-sm">
              <button
                type="button"
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
                className="rounded border border-gray-400 px-2 py-0.5 disabled:opacity-50"
              >
                Prev
              </button>
              <span>
                Page {page + 1} of {totalPages}
              </span>
              <button
                type="button"
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={page >= totalPages - 1}
                className="rounded border border-gray-400 px-2 py-0.5 disabled:opacity-50"
              >
                Next
              </button>
            </div>
          )}

          <button
            type="button"
            onClick={handleDownload}
            className="rounded border border-gray-400 px-4 py-1 text-sm hover:bg-gray-50"
          >
            Download CSV
          </button>
        </div>
      )}
    </div>
  );
}
