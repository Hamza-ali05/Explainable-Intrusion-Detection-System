import { useEffect, useState } from "react";
import { getXaiQuality } from "../api/client";
import LoadingSpinner from "../components/LoadingSpinner";
import { formatNumber, modelLabel, toImageSrc } from "../utils/constants";

const TABS = [
  { key: "fidelity", label: "Fidelity" },
  { key: "stability", label: "Stability" },
  { key: "comprehensibility", label: "Comprehensibility" },
];

export default function XaiQualityPage() {
  const [tab, setTab] = useState("fidelity");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    getXaiQuality(tab)
      .then(setData)
      .finally(() => setLoading(false));
  }, [tab]);

  return (
    <div>
      <h1 className="mb-4 text-lg font-medium">XAI Quality</h1>

      <div className="mb-6 flex gap-4 border-b border-gray-300">
        {TABS.map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setTab(t.key)}
            className={`pb-2 text-sm ${
              tab === t.key ? "border-b-2 border-gray-900 font-medium" : "text-gray-600"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {loading && <LoadingSpinner />}

      {data && !loading && (
        <div>
          {data.image && (
            <img
              src={toImageSrc(data.image)}
              alt={tab}
              className="mb-6 w-full border border-gray-200"
            />
          )}

          {tab === "fidelity" && (
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-gray-300 text-left">
                  <th className="py-2 pr-4">Model</th>
                  <th className="py-2 pr-4">k Ablated</th>
                  <th className="py-2 pr-4">Accuracy Drop</th>
                  <th className="py-2">F1 Drop</th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((row, i) => (
                  <tr key={i} className="border-b border-gray-200">
                    <td className="py-2 pr-4">{modelLabel(row.model)}</td>
                    <td className="py-2 pr-4">{row.k_ablated}</td>
                    <td className="py-2 pr-4">{formatNumber(row.accuracy_drop)}</td>
                    <td className="py-2">{formatNumber(row.f1_drop)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {tab === "stability" && (
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-gray-300 text-left">
                  <th className="py-2 pr-4">Instance Index</th>
                  <th className="py-2 pr-4">True Class</th>
                  <th className="py-2 pr-4">LIME Jaccard</th>
                  <th className="py-2 pr-4">LIME Spearman</th>
                  <th className="py-2 pr-4">SHAP Jaccard</th>
                  <th className="py-2">SHAP Spearman</th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((row, i) => (
                  <tr key={i} className="border-b border-gray-200">
                    <td className="py-2 pr-4">{row.instance_idx}</td>
                    <td className="py-2 pr-4">{row.true_class}</td>
                    <td className="py-2 pr-4">{formatNumber(row.lime_jaccard_mean)}</td>
                    <td className="py-2 pr-4">{formatNumber(row.lime_spearman_mean)}</td>
                    <td className="py-2 pr-4">{formatNumber(row.shap_jaccard_mean)}</td>
                    <td className="py-2">{formatNumber(row.shap_spearman_mean)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {tab === "comprehensibility" && (
            <table className="w-full border-collapse text-sm">
              <thead>
                <tr className="border-b border-gray-300 text-left">
                  <th className="py-2 pr-4">Model</th>
                  <th className="py-2 pr-4">Threat Class</th>
                  <th className="py-2 pr-4">IoC Alignment Score</th>
                  <th className="py-2">Top 10 Features</th>
                </tr>
              </thead>
              <tbody>
                {data.rows.map((row, i) => (
                  <tr key={i} className="border-b border-gray-200">
                    <td className="py-2 pr-4">{modelLabel(row.model)}</td>
                    <td className="py-2 pr-4">{row.threat_class}</td>
                    <td className="py-2 pr-4">{formatNumber(row.ioc_alignment_score)}</td>
                    <td className="py-2 text-xs">{row.top10_features}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
