export default function PageSkeleton({ className = "" }) {
  return (
    <div
      className={`h-32 animate-pulse rounded-xl bg-slate-700 ${className}`}
      role="status"
      aria-label="Loading"
    />
  );
}
