import { useEffect, useState } from "react";
import { getXaiLocalImage, getXaiLocalInstances } from "../api/client";
import LoadingSpinner from "../components/LoadingSpinner";
import { EXPLAINERS, toImageSrc } from "../utils/constants";

export default function XaiLocalPage() {
  const [instances, setInstances] = useState([]);
  const [selected, setSelected] = useState(null);
  const [explainer, setExplainer] = useState("xgb_shap");
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(true);
  const [imageLoading, setImageLoading] = useState(false);

  useEffect(() => {
    getXaiLocalInstances()
      .then((data) => {
        const list = data.instances ?? [];
        setInstances(list);
        if (list.length > 0) setSelected(list[0]);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selected || !explainer) return;
    setImageLoading(true);
    getXaiLocalImage({
      explainer,
      true_class: selected.true_label,
      outcome: selected.kind,
      instance_idx: selected.instance_id,
    })
      .then((data) => setImage(data.image))
      .catch(() => setImage(null))
      .finally(() => setImageLoading(false));
  }, [selected, explainer]);

  if (loading) return <LoadingSpinner />;

  return (
    <div>
      <h1 className="mb-4 text-lg font-medium">XAI Local</h1>

      <table className="mb-6 w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-gray-300 text-left">
            <th className="py-2 pr-4">Instance Index</th>
            <th className="py-2 pr-4">True Class</th>
            <th className="py-2 pr-4">Predicted Class</th>
            <th className="py-2">Outcome</th>
          </tr>
        </thead>
        <tbody>
          {instances.map((inst) => (
            <tr
              key={inst.instance_id}
              onClick={() => setSelected(inst)}
              className={`cursor-pointer border-b border-gray-200 hover:bg-gray-50 ${
                selected?.instance_id === inst.instance_id ? "bg-gray-100" : ""
              }`}
            >
              <td className="py-2 pr-4">{inst.instance_id}</td>
              <td className="py-2 pr-4">{inst.true_label}</td>
              <td className="py-2 pr-4">{inst.predicted_label}</td>
              <td className="py-2 capitalize">{inst.kind}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {selected && (
        <div>
          <div className="mb-4">
            <label className="mr-2 text-sm">Explainer</label>
            <select
              value={explainer}
              onChange={(e) => setExplainer(e.target.value)}
              className="rounded border border-gray-400 px-2 py-1 text-sm"
            >
              {EXPLAINERS.map((ex) => (
                <option key={ex.value} value={ex.value}>
                  {ex.label}
                </option>
              ))}
            </select>
          </div>

          {imageLoading && <LoadingSpinner />}

          {image && !imageLoading && (
            <img
              src={toImageSrc(image)}
              alt="Local explanation"
              className="mb-4 w-full border border-gray-200"
            />
          )}

          <p className="text-sm">
            True: {selected.true_label} | Predicted: {selected.predicted_label} |
            Outcome: {selected.kind}
          </p>
        </div>
      )}
    </div>
  );
}
