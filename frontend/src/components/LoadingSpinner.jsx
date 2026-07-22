export default function LoadingSpinner({ label = "Loading…" }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-400">
      <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-slate-500 border-t-sky-400" />
      {label}
    </div>
  );
}
