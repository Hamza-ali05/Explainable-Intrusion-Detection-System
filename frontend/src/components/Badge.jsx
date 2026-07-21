export default function Badge({ done }) {
  return (
    <span
      className={`inline-block rounded px-2 py-0.5 text-xs ${
        done ? "bg-green-100 text-green-800" : "bg-gray-200 text-gray-600"
      }`}
    >
      {done ? "Done" : "Pending"}
    </span>
  );
}
