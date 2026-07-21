import { useEffect, useMemo, useState } from "react";
import { getEvaluateDetail, getLeaderboard, getTrainingCurve } from "../api/client";
import LoadingSpinner from "../components/LoadingSpinner";
import { MODELS, formatNumber, modelLabel, toImageSrc } from "../utils/constants";

const COLUMNS = [
  { key: "model", label: "Model", format: (v) => modelLabel(v) },
  { key: "accuracy", label: "Accuracy", format: formatNumber },
  { key: "macro_f1", label: "Macro F1", format: formatNumber },
  { key: "weighted_f1", label: "Weighted F1", format: formatNumber },
  { key: "mcc", label: "MCC", format: formatNumber },
  { key: "pr_auc", label: "PR-AUC", format: formatNumber },
];

export default function EvaluatePage() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState("xgboost");
  const [detail, setDetail] = useState(null);
  const [trainingCurve, setTrainingCurve] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [sortKey, setSortKey] = useState("macro_f1");
  const [sortAsc, setSortAsc] = useState(false);

  useEffect(() => {
    getLeaderboard()
      .then(setRows)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedModel) return;
    setDetailLoading(true);
    Promise.all([
      getEvaluateDetail(selectedModel),
      getTrainingCurve(selectedModel),
    ])
      .then(([d, curve]) => {
        setDetail(d);
        setTrainingCurve(curve.image);
      })
      .finally(() => setDetailLoading(false));
  }, [selectedModel]);

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const av = a[sortKey] ?? "";
      const bv = b[sortKey] ?? "";
      if (typeof av === "number" && typeof bv === "number") {
        return sortAsc ? av - bv : bv - av;
      }
      return sortAsc
        ? String(av).localeCompare(String(bv))
        : String(bv).localeCompare(String(av));
    });
    return copy;
  }, [rows, sortKey, sortAsc]);

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  const confusionImg =
    detail?.images?.confusion_matrix ??
    detail?.images?.confusion_matrix_normalized ??
    Object.values(detail?.images ?? {}).find((v) => v);
  const rocImg =
    detail?.images?.roc_curve ??
    detail?.images?.roc ??
    detail?.images?.roc_ovr;

  if (loading) return <LoadingSpinner />;

  return (
    <div>
      <h1 className="mb-4 text-lg font-medium">Model Evaluation</h1>

      <table className="mb-8 w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-gray-300 text-left">
            {COLUMNS.map((col) => (
              <th
                key={col.key}
                className="cursor-pointer py-2 pr-4 hover:underline"
                onClick={() => handleSort(col.key)}
              >
                {col.label}
                {sortKey === col.key ? (sortAsc ? " ↑" : " ↓") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={row.model} className="border-b border-gray-200">
              {COLUMNS.map((col) => (
                <td key={col.key} className="py-2 pr-4">
                  {col.format(row[col.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      <div className="mb-4">
        <label className="mr-2 text-sm">Select Model</label>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="rounded border border-gray-400 px-2 py-1 text-sm"
        >
          {MODELS.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      {detailLoading && <LoadingSpinner />}

      {detail && !detailLoading && (
        <div className="space-y-6">
          <div className="flex gap-8 text-center">
            <div>
              <div className="text-2xl font-semibold">
                {formatNumber(detail.metrics?.accuracy)}
              </div>
              <div className="text-xs text-gray-600">Accuracy</div>
            </div>
            <div>
              <div className="text-2xl font-semibold">
                {formatNumber(detail.metrics?.f1?.macro)}
              </div>
              <div className="text-xs text-gray-600">Macro F1</div>
            </div>
            <div>
              <div className="text-2xl font-semibold">
                {formatNumber(detail.metrics?.mcc)}
              </div>
              <div className="text-xs text-gray-600">MCC</div>
            </div>
          </div>

          {confusionImg && (
            <img
              src={toImageSrc(confusionImg)}
              alt="Confusion matrix"
              className="w-full border border-gray-200"
            />
          )}

          {rocImg && (
            <img
              src={toImageSrc(rocImg)}
              alt="ROC curve"
              className="w-full border border-gray-200"
            />
          )}

          {trainingCurve && (
            <div>
              <h2 className="mb-2 text-sm font-medium">Training Curve</h2>
              <img
                src={toImageSrc(trainingCurve)}
                alt="Training curve"
                className="w-full border border-gray-200"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
