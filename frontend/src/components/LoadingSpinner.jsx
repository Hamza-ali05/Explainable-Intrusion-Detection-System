export default function LoadingSpinner({ label = "Loading…" }) {
  return (
    <div className="flex items-center gap-2 text-sm text-gray-600">
      <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-gray-400 border-t-transparent" />
      {label}
    </div>
  );
}
