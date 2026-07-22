import { useEffect, useState } from "react";
import { getXaiLocalImage, getXaiLocalInstances } from "../api/client";
import Badge from "../components/Badge";
import Card from "../components/Card";
import ErrorCard from "../components/ErrorCard";
import ImageCard from "../components/ImageCard";
import PageSkeleton from "../components/PageSkeleton";
import PageTitle from "../components/PageTitle";
import Select from "../components/Select";
import { EXPLAINERS, apiError } from "../utils/constants";

function badgeType(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/\s+/g, "");
}

function instanceKey(inst) {
  return `${inst.instance_id}-${inst.kind}-${inst.true_label}`;
}

export default function XaiLocalPage() {
  const [instances, setInstances] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState(null);
  const [explainer, setExplainer] = useState("");
  const [image, setImage] = useState(null);
  const [imageLoading, setImageLoading] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const data = await getXaiLocalInstances();
        setInstances(data?.instances ?? []);
      } catch (err) {
        setError({ message: apiError(err) });
        setInstances([]);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (!selected || !explainer) {
      setImage(null);
      setImageLoading(false);
      return undefined;
    }

    let cancelled = false;
    setImageLoading(true);

    getXaiLocalImage({
      explainer,
      true_class: selected.true_label,
      outcome: selected.kind,
      instance_idx: selected.instance_id,
    })
      .then((data) => {
        if (!cancelled) setImage(data?.image ?? null);
      })
      .catch((err) => {
        if (!cancelled) {
          setImage(null);
          setError({ message: apiError(err) });
        }
      })
      .finally(() => {
        if (!cancelled) setImageLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selected, explainer]);

  const explainerLabel =
    EXPLAINERS.find((e) => e.value === explainer)?.label ?? explainer;

  return (
    <div className="space-y-6">
      <PageTitle>XAI Local</PageTitle>
      <ErrorCard error={error} />

      <Card title="Instances">
        {loading ? (
          <PageSkeleton />
        ) : instances.length === 0 ? (
          <p className="text-sm text-slate-500">No data.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="ui-table">
              <thead>
                <tr>
                  {["Instance Index", "True Class", "Predicted Class", "Outcome"].map(
                    (col) => (
                      <th key={col}>{col}</th>
                    ),
                  )}
                </tr>
              </thead>
              <tbody>
                {instances.map((inst) => {
                  const active =
                    selected && instanceKey(selected) === instanceKey(inst);
                  return (
                    <tr
                      key={instanceKey(inst)}
                      onClick={() => setSelected(inst)}
                      className={`cursor-pointer transition-colors even:bg-slate-800/50 ${
                        active ? "bg-slate-700" : "hover:bg-slate-700/40"
                      }`}
                    >
                      <td className="font-mono text-xs">{inst.instance_id}</td>
                      <td>
                        <Badge type={badgeType(inst.true_label)}>
                          {inst.true_label}
                        </Badge>
                      </td>
                      <td>
                        <Badge type={badgeType(inst.predicted_label)}>
                          {inst.predicted_label}
                        </Badge>
                      </td>
                      <td>
                        <Badge type={badgeType(inst.kind)}>{inst.kind}</Badge>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {selected && (
        <Card title="Explanation">
          <div className="space-y-4">
            <Select
              id="local-explainer"
              label="Explainer"
              value={explainer}
              onChange={(e) => setExplainer(e.target.value)}
              options={EXPLAINERS}
              className="min-w-[200px]"
            />

            {explainer && (
              <>
                <ImageCard
                  title={explainerLabel}
                  base64={image}
                  loading={imageLoading}
                />
                <p className="mt-2 text-sm text-slate-400">
                  True: {selected.true_label} | Predicted:{" "}
                  {selected.predicted_label} | Outcome: {selected.kind}
                </p>
              </>
            )}
          </div>
        </Card>
      )}
    </div>
  );
}
