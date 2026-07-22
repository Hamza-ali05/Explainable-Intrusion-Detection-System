import { useEffect, useState } from "react";
import { getXaiGlobal } from "../api/client";
import ErrorCard from "../components/ErrorCard";
import ImageCard from "../components/ImageCard";
import PageTitle from "../components/PageTitle";
import Select from "../components/Select";
import { MODELS, PLOT_TYPES, apiError, modelLabel } from "../utils/constants";

export default function XaiGlobalPage() {
  const [model, setModel] = useState("");
  const [plotType, setPlotType] = useState("");
  const [error, setError] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [selectedLoading, setSelectedLoading] = useState(false);
  const [allImages, setAllImages] = useState({
    beeswarm: null,
    bar: null,
    dependence: null,
  });
  const [allLoading, setAllLoading] = useState(false);

  useEffect(() => {
    if (!model || !plotType) {
      setSelectedImage(null);
      setSelectedLoading(false);
      return undefined;
    }

    let cancelled = false;
    setSelectedLoading(true);

    getXaiGlobal(model, plotType)
      .then((data) => {
        if (!cancelled) setSelectedImage(data?.image ?? null);
      })
      .catch((err) => {
        if (!cancelled) {
          setSelectedImage(null);
          setError({ message: apiError(err) });
        }
      })
      .finally(() => {
        if (!cancelled) setSelectedLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [model, plotType]);

  useEffect(() => {
    if (!model) {
      setAllImages({ beeswarm: null, bar: null, dependence: null });
      setAllLoading(false);
      return undefined;
    }

    let cancelled = false;
    setAllLoading(true);

    Promise.all(
      PLOT_TYPES.map((pt) =>
        getXaiGlobal(model, pt.value)
          .then((data) => [pt.value, data?.image ?? null])
          .catch(() => [pt.value, null]),
      ),
    )
      .then((results) => {
        if (cancelled) return;
        const next = { beeswarm: null, bar: null, dependence: null };
        results.forEach(([key, img]) => {
          next[key] = img;
        });
        setAllImages(next);
      })
      .finally(() => {
        if (!cancelled) setAllLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [model]);

  const plotLabel =
    PLOT_TYPES.find((p) => p.value === plotType)?.label ?? plotType;
  const selectedTitle = `${modelLabel(model)} — ${plotLabel}`;

  return (
    <div className="space-y-6">
      <PageTitle>XAI Global</PageTitle>
      <ErrorCard error={error} />

      <div className="flex flex-wrap gap-4">
        <Select
          id="global-model"
          label="Model"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          options={MODELS}
          className="min-w-[180px]"
        />
        <Select
          id="global-plot"
          label="Plot type"
          value={plotType}
          onChange={(e) => setPlotType(e.target.value)}
          options={PLOT_TYPES}
          className="min-w-[180px]"
        />
      </div>

      {model && plotType && (
        <ImageCard
          title={selectedTitle}
          base64={selectedImage}
          loading={selectedLoading}
        />
      )}

      {model && (
        <div className="space-y-6">
          {PLOT_TYPES.map((pt) => (
            <ImageCard
              key={pt.value}
              title={pt.label}
              base64={allImages[pt.value]}
              loading={allLoading}
            />
          ))}
        </div>
      )}
    </div>
  );
}
