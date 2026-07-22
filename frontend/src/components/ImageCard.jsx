import Card from "./Card";
import { toImageSrc } from "../utils/constants";

export default function ImageCard({ title, base64, loading = false }) {
  return (
    <Card title={title}>
      {loading && (
        <div className="h-48 w-full animate-pulse rounded-xl border border-slate-700 bg-slate-700/60" />
      )}
      {!loading && base64 == null && (
        <p className="text-sm text-slate-500">Image not available.</p>
      )}
      {!loading && base64 != null && (
        <img
          src={toImageSrc(base64)}
          className="ui-card-image"
          alt={title}
        />
      )}
    </Card>
  );
}
