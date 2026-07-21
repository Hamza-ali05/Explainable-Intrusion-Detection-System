import { useEffect, useState } from "react";
import { getXaiGlobal } from "../api/client";
import LoadingSpinner from "../components/LoadingSpinner";
import { MODELS, PLOT_TYPES, toImageSrc } from "../utils/constants";

export default function XaiGlobalPage() {
  const [model, setModel] = useState("xgboost");
  const [plotType, setPlotType] = useState("beeswarm");
  const [selectedImage, setSelectedImage] = useState(null);
  const [stackedImages, setStackedImages] = useState({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!model || !plotType) return;
    setLoading(true);
    getXaiGlobal(model, plotType)
      .then((data) => setSelectedImage(data.image))
      .catch(() => setSelectedImage(null))
      .finally(() => setLoading(false));
  }, [model, plotType]);

  useEffect(() => {
    if (!model) return;
    Promise.all(
      PLOT_TYPES.map((pt) =>
        getXaiGlobal(model, pt.value)
          .then((data) => [pt.value, data.image])
          .catch(() => [pt.value, null]),
      ),
    ).then((results) => {
      const map = {};
      results.forEach(([key, img]) => {
        map[key] = img;
      });
      setStackedImages(map);
    });
  }, [model]);

  return (
    <div>
      <h1 className="mb-4 text-lg font-medium">XAI Global</h1>

      <div className="mb-6 flex flex-wrap gap-4">
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
        <div>
          <label className="mr-2 text-sm">Plot Type</label>
          <select
            value={plotType}
            onChange={(e) => setPlotType(e.target.value)}
            className="rounded border border-gray-400 px-2 py-1 text-sm"
          >
            {PLOT_TYPES.map((pt) => (
              <option key={pt.value} value={pt.value}>
                {pt.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {loading && <LoadingSpinner />}

      {selectedImage && !loading && (
        <img
          src={toImageSrc(selectedImage)}
          alt={`${model} ${plotType}`}
          className="mb-8 w-full border border-gray-200"
        />
      )}

      <div className="space-y-6">
        {PLOT_TYPES.map((pt) =>
          stackedImages[pt.value] ? (
            <div key={pt.value}>
              <h2 className="mb-2 text-sm font-medium">{pt.label}</h2>
              <img
                src={toImageSrc(stackedImages[pt.value])}
                alt={pt.label}
                className="w-full border border-gray-200"
              />
            </div>
          ) : null,
        )}
      </div>
    </div>
  );
}
