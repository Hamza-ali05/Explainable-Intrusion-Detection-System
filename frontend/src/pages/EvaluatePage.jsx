import { useEffect, useMemo, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { getEvaluateDetail, getLeaderboard, getTrainingCurve } from "../api/client";
import Card from "../components/Card";
import DataTable from "../components/DataTable";
import ErrorCard from "../components/ErrorCard";
import ImageCard from "../components/ImageCard";
import PageSkeleton from "../components/PageSkeleton";
import PageTitle from "../components/PageTitle";
import Select from "../components/Select";
import {
  MODELS,
  THREAT_CLASSES,
  apiError,
  formatNumber,
  modelLabel,
} from "../utils/constants";

const COLUMNS = [
  { key: "model", label: "Model" },
  { key: "accuracy", label: "Accuracy" },
  { key: "macro_f1", label: "Macro F1" },
  { key: "weighted_f1", label: "Weighted F1" },
  { key: "mcc", label: "MCC" },
  { key: "pr_auc", label: "PR-AUC" },
];

function findImage(images, needle) {
  if (!images) return null;
  const lower = needle.toLowerCase();

  if (Array.isArray(images)) {
    const hit = images.find((img) => {
      const name = String(img.filename || img.name || img.path || img.key || "");
      return name.toLowerCase().includes(lower);
    });
    return hit?.base64 ?? hit?.data ?? hit?.image ?? hit?.png ?? null;
  }

  const entry = Object.entries(images).find(([key, value]) => {
    if (!value) return false;
    return key.toLowerCase().includes(lower);
  });
  return entry ? entry[1] : null;
}

function classificationReportRows(detail) {
  if (Array.isArray(detail?.classification_report)) {
    return detail.classification_report.map((row) => {
      const out = {};
      for (const [key, value] of Object.entries(row)) {
        out[key] = typeof value === "number" ? formatNumber(value) : value;
      }
      return out;
    });
  }

  const metrics = detail?.metrics;
  if (!metrics) return [];

  const rows = THREAT_CLASSES.map((cls) => ({
    Class: cls,
    Precision: formatNumber(metrics.precision?.per_class?.[cls]),
    Recall: formatNumber(metrics.recall?.per_class?.[cls]),
    F1: formatNumber(metrics.f1?.per_class?.[cls]),
  }));

  rows.push(
    {
      Class: "macro",
      Precision: formatNumber(metrics.precision?.macro),
      Recall: formatNumber(metrics.recall?.macro),
      F1: formatNumber(metrics.f1?.macro),
    },
    {
      Class: "weighted",
      Precision: formatNumber(metrics.precision?.weighted),
      Recall: formatNumber(metrics.recall?.weighted),
      F1: formatNumber(metrics.f1?.weighted),
    },
  );

  return rows;
}

export default function EvaluatePage() {
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState("");
  const [detail, setDetail] = useState(null);
  const [trainingCurve, setTrainingCurve] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [sortKey, setSortKey] = useState("macro_f1");
  const [sortAsc, setSortAsc] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const data = await getLeaderboard();
        setRows(data);
      } catch (err) {
        setError({ message: apiError(err) });
        setRows([]);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (!selectedModel) {
      setDetail(null);
      setTrainingCurve(null);
      return undefined;
    }

    let cancelled = false;
    setDetailLoading(true);

    Promise.all([getEvaluateDetail(selectedModel), getTrainingCurve(selectedModel)])
      .then(([d, curve]) => {
        if (cancelled) return;
        setDetail(d);
        setTrainingCurve(curve?.image ?? null);
      })
      .catch((err) => {
        if (cancelled) return;
        setDetail(null);
        setTrainingCurve(null);
        setError({ message: apiError(err) });
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });

    return () => {
      cancelled = true;
    };
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
      setSortAsc((v) => !v);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  const confusionPng = findImage(detail?.images, "confusion");
  const rocPng = findImage(detail?.images, "roc");
  const reportRows = classificationReportRows(detail);

  return (
    <div className="space-y-6">
      <PageTitle>Evaluate</PageTitle>
      <ErrorCard error={error} />

      {loading ? (
        <PageSkeleton />
      ) : (
      <Card title="Model Leaderboard">
        <div className="overflow-x-auto">
          <table className="ui-table">
            <thead>
              <tr>
                {COLUMNS.map((col) => (
                  <th
                    key={col.key}
                    className="cursor-pointer select-none hover:text-sky-400"
                    onClick={() => handleSort(col.key)}
                  >
                    <span className="inline-flex items-center gap-1">
                      {col.label}
                      {sortKey === col.key &&
                        (sortAsc ? <ChevronUp size={14} /> : <ChevronDown size={14} />)}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.length === 0 ? (
                <tr>
                  <td colSpan={COLUMNS.length} className="px-4 py-3 text-slate-500">
                    No data.
                  </td>
                </tr>
              ) : (
                sorted.map((row) => (
                  <tr key={row.model} className="even:bg-slate-800/50">
                    <td>{modelLabel(row.model)}</td>
                    <td>{formatNumber(row.accuracy)}</td>
                    <td>{formatNumber(row.macro_f1)}</td>
                    <td>{formatNumber(row.weighted_f1)}</td>
                    <td>{formatNumber(row.mcc)}</td>
                    <td>{formatNumber(row.pr_auc)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>
      )}

      <Card title="Model detail">
        <div className="space-y-6">
          <Select
            id="evaluate-model"
            label="Model"
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            options={MODELS}
            className="min-w-[200px]"
          />

          {!selectedModel && (
            <p className="text-sm text-slate-500">Select a model.</p>
          )}

          {selectedModel && detailLoading && <PageSkeleton />}
        </div>
      </Card>

      {selectedModel && detail && !detailLoading && (
        <>
          <Card title="Metrics">
            <div className="flex flex-wrap gap-8">
              {[
                ["Accuracy", detail.metrics?.accuracy],
                ["Macro F1", detail.metrics?.f1?.macro],
                ["MCC", detail.metrics?.mcc],
              ].map(([label, value]) => (
                <div key={label} className="min-w-[7rem]">
                  <div className="text-3xl font-bold text-sky-400">
                    {formatNumber(value)}
                  </div>
                  <div className="text-xs text-slate-400">{label}</div>
                </div>
              ))}
            </div>
          </Card>
          <ImageCard title="Confusion Matrix" base64={confusionPng} />
          <ImageCard title="ROC Curve" base64={rocPng} />

          <Card title="Classification report">
            <div className="overflow-x-auto">
              <DataTable data={reportRows} />
            </div>
          </Card>

          {trainingCurve != null && (
            <ImageCard
              title="Training Curve / Feature Importance"
              base64={trainingCurve}
            />
          )}
        </>
      )}
    </div>
  );
}
