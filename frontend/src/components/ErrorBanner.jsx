export default function ErrorBanner({ message }) {
  if (!message) return null;
  return (
    <div className="mb-4 rounded-xl border border-red-400/40 bg-red-400/10 px-4 py-3 text-sm text-red-300">
      {message}
    </div>
  );
}
