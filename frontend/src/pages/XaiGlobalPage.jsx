import { useEffect, useState } from "react";
import { getXaiGlobal } from "../api/client";
import ErrorCard from "../components/ErrorCard";
import ImageCard from "../components/ImageCard";
import Select from "../components/Select";
import { MODELS, PLOT_TYPES, apiError, modelLabel } from "../utils/constants";

export default function XaiGlobalPage() {
  const [model, setModel] = useState(MODELS[0]?.value ?? "");
  const [plotType, setPlotType] = useState(PLOT_TYPES[0]?.value ?? "");
  const [error, setError] = useState(null);
  const [allImages, setAllImages] = useState({
    beeswarm: null,
    bar: null,
    dependence: null,
  });
  const [allLoading, setAllLoading] = useState(false);

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

  return (
    <div className="space-y-6">
      <ErrorCard error={error} />

      <div className="flex flex-wrap gap-4">
        <Select
          id="global-model"
          label="Model"
          value={model}
          onChange={(e) => {
            setError(null);
            setModel(e.target.value);
          }}
          options={MODELS}
          className="min-w-[180px]"
        />
        <Select
          id="global-plot"
          label="Plot type"
          value={plotType}
          onChange={(e) => {
            setError(null);
            setPlotType(e.target.value);
          }}
          options={PLOT_TYPES}
          className="min-w-[180px]"
        />
      </div>

      {model && (
        <div className="grid w-full grid-cols-1 gap-4 sm:grid-cols-2 sm:gap-x-4 sm:gap-y-5">
          {PLOT_TYPES.map((pt, index) => {
            const title = `${modelLabel(model)} — ${pt.label}`;
            const isSelected = plotType === pt.value;
            const alignClass =
              index % 2 === 0 ? "sm:justify-self-start" : "sm:justify-self-end";
            return (
              <ImageCard
                key={pt.value}
                title={title}
                base64={allImages[pt.value]}
                loading={allLoading}
                className={`w-full max-w-md ${alignClass} ${
                  isSelected
                    ? "border-sky-400/60 ring-2 ring-sky-400/40"
                    : ""
                }`}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
