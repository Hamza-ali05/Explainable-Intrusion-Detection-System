import { useEffect, useMemo, useState } from "react";
import { getXaiQuality } from "../api/client";
import Badge from "../components/Badge";
import Card from "../components/Card";
import DataTable from "../components/DataTable";
import ErrorCard from "../components/ErrorCard";
import ImageCard from "../components/ImageCard";
import PageSkeleton from "../components/PageSkeleton";
import { apiError, formatNumber, modelLabel } from "../utils/constants";

const TABS = [
  { key: "fidelity", label: "Fidelity" },
  { key: "stability", label: "Stability" },
  { key: "comprehensibility", label: "Comprehensibility" },
];

function formatTop10Features(value) {
  if (value == null) return "—";
  if (Array.isArray(value)) return value.join(", ");
  return String(value);
}

function badgeType(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/\s+/g, "");
}

function buildTableData(tab, rows) {
  if (!rows?.length) return [];

  if (tab === "fidelity") {
    return rows.map((row) => ({
      Model: modelLabel(row.model),
      "k Ablated": row.k_ablated,
      "Baseline Accuracy": formatNumber(row.baseline_accuracy),
      "Ablated Accuracy": formatNumber(row.ablated_accuracy),
      "Accuracy Drop": formatNumber(row.accuracy_drop),
      "Baseline F1": formatNumber(row.baseline_f1),
      "Ablated F1": formatNumber(row.ablated_f1),
      "F1 Drop": formatNumber(row.f1_drop),
    }));
  }

  if (tab === "stability") {
    return rows.map((row) => ({
      "Instance Index": row.instance_idx,
      "True Class": row.true_class,
      "LIME Jaccard": formatNumber(row.lime_jaccard_mean),
      "LIME Spearman": formatNumber(row.lime_spearman_mean),
      "SHAP Jaccard": formatNumber(row.shap_jaccard_mean),
      "SHAP Spearman": formatNumber(row.shap_spearman_mean),
    }));
  }

  return rows.map((row) => ({
    Model: modelLabel(row.model),
    "Threat Class": (
      <Badge type={badgeType(row.threat_class)}>{row.threat_class}</Badge>
    ),
    "IoC Alignment Score": (
      <div className="flex items-center gap-2">
        <div className="w-24 bg-slate-700 rounded-full h-2">
          <div
            className="bg-sky-500 h-2 rounded-full"
            style={{ width: `${((Number(row.ioc_alignment_score) || 0) * 100).toFixed(0)}%` }}
          />
        </div>
        <span className="text-xs text-slate-300">
          {((Number(row.ioc_alignment_score) || 0) * 100).toFixed(0)}%
        </span>
      </div>
    ),
    "Top 10 Features": formatTop10Features(row.top10_features),
  }));
}

const PLOT_TITLES = {
  fidelity: "Fidelity Plot",
  stability: "Stability Plot",
  comprehensibility: "IoC Alignment Heatmap",
};

const DATA_TITLES = {
  fidelity: "Fidelity Data",
  stability: "Stability Data",
  comprehensibility: "Comprehensibility Data",
};

export default function XaiQualityPage() {
  const [tab, setTab] = useState("fidelity");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    getXaiQuality(tab)
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .catch((err) => {
        if (!cancelled) {
          setData(null);
          setError({ message: apiError(err) });
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [tab]);

  const tableData = useMemo(
    () => buildTableData(tab, data?.rows),
    [tab, data?.rows],
  );

  return (
    <div className="space-y-6">
      <ErrorCard error={error} />

      <div className="rounded-2xl border border-slate-700 bg-slate-800 px-6 py-4 shadow-lg">
        <div className="flex gap-6">
          {TABS.map((t) => (
            <button
              key={t.key}
              type="button"
              onClick={() => setTab(t.key)}
              className={
                tab === t.key
                  ? "border-b-2 border-sky-400 pb-2 font-medium text-sky-400"
                  : "pb-2 text-slate-400 hover:text-slate-200"
              }
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      <ImageCard
        title={PLOT_TITLES[tab]}
        base64={data?.image ?? null}
        loading={loading}
      />

      <Card title={DATA_TITLES[tab]}>
        {loading ? (
          <PageSkeleton />
        ) : (
          <div className="overflow-x-auto">
            <DataTable data={tableData} />
          </div>
        )}
      </Card>
    </div>
  );
}
