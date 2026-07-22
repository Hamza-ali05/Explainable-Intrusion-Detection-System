const TYPE_STYLES = {
  done: "bg-green-900 text-green-400",
  pending: "bg-slate-700 text-slate-400",
  benign: "bg-green-900 text-green-400",
  ddos: "bg-red-900 text-red-400",
  dos: "bg-orange-900 text-orange-400",
  portscan: "bg-amber-900 text-amber-400",
  bruteforce: "bg-purple-900 text-purple-400",
  webattack: "bg-pink-900 text-pink-400",
  infiltration: "bg-sky-900 text-sky-400",
  botnet: "bg-rose-900 text-rose-400",
  correct: "bg-green-900 text-green-400",
  wrong: "bg-red-900 text-red-400",
  critical: "bg-red-900 text-red-400",
  high: "bg-amber-900 text-amber-400",
  medium: "bg-sky-900 text-sky-400",
  low: "bg-slate-700 text-slate-400",
};

const TYPE_LABELS = {
  done: "Done",
  pending: "Pending",
  benign: "Benign",
  ddos: "DDoS",
  dos: "DoS",
  portscan: "PortScan",
  bruteforce: "BruteForce",
  webattack: "WebAttack",
  infiltration: "Infiltration",
  botnet: "Botnet",
  correct: "Correct",
  wrong: "Wrong",
  critical: "Critical",
  high: "High",
  medium: "Medium",
  low: "Low",
};

export default function Badge({ type, children }) {
  const style = TYPE_STYLES[type] || TYPE_STYLES.pending;
  const label = children ?? TYPE_LABELS[type] ?? type;

  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${style}`}>
      {label}
    </span>
  );
}
