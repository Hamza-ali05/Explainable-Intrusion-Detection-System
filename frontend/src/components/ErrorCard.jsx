import Card from "./Card";

export default function ErrorCard({ error }) {
  if (!error) return null;
  const message = error?.message ?? String(error);
  return (
    <Card title="Error">
      <p className="text-sm text-red-400">{message}</p>
    </Card>
  );
}
