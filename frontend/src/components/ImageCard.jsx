import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { Maximize2, Minimize2 } from "lucide-react";
import Card from "./Card";
import { toImageSrc } from "../utils/constants";

export default function ImageCard({
  title,
  base64,
  loading = false,
  className = "",
  imageClassName = "",
  expandable = true,
}) {
  const [expanded, setExpanded] = useState(false);
  const imageSrc = base64 != null ? toImageSrc(base64) : null;
  const canExpand = expandable && !loading && imageSrc != null;

  useEffect(() => {
    if (!expanded) return undefined;

    const onKeyDown = (event) => {
      if (event.key === "Escape") setExpanded(false);
    };

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [expanded]);

  const imageContent = loading ? (
    <div
      className={`h-40 w-full animate-pulse rounded-xl border border-slate-700 bg-slate-700/60 ${imageClassName}`}
    />
  ) : imageSrc == null ? (
    <p className="text-sm text-slate-500">Image not available.</p>
  ) : (
    <img
      src={imageSrc}
      className={`ui-card-image w-full max-h-80 object-contain ${imageClassName}`}
      alt={title}
    />
  );

  const expandAction = canExpand ? (
    <button
      type="button"
      onClick={() => setExpanded(true)}
      className="rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-700 hover:text-sky-400"
      aria-label={`Expand ${title}`}
      title="Expand"
    >
      <Maximize2 size={16} />
    </button>
  ) : null;

  const fullscreenOverlay =
    expanded && imageSrc != null
      ? createPortal(
          <div
            className="fixed inset-0 z-[100] flex flex-col bg-page/95 backdrop-blur-md"
            role="dialog"
            aria-modal="true"
            aria-label={title}
          >
            <div className="flex items-center justify-between border-b border-slate-700 px-4 py-3 sm:px-6">
              <h2 className="text-base font-medium text-slate-100 sm:text-lg">{title}</h2>
              <button
                type="button"
                onClick={() => setExpanded(false)}
                className="inline-flex items-center gap-2 rounded-lg border border-slate-600 bg-slate-800 px-3 py-2 text-sm text-slate-200 transition hover:border-sky-500 hover:text-sky-400"
                aria-label={`Collapse ${title}`}
              >
                <Minimize2 size={16} />
                Collapse
              </button>
            </div>
            <div className="flex flex-1 items-center justify-center overflow-auto p-4 sm:p-8">
              <img
                src={imageSrc}
                className="max-h-[calc(100vh-5rem)] max-w-full object-contain"
                alt={title}
              />
            </div>
          </div>,
          document.body,
        )
      : null;

  return (
    <>
      <Card
        title={title}
        className={className}
        contentClassName="px-2 pb-4 sm:px-3"
        actions={expandAction}
      >
        {imageContent}
      </Card>
      {fullscreenOverlay}
    </>
  );
}
