import { Link, useLocation } from "react-router-dom";

const LINKS = [
  { to: "/status", label: "Status" },
  { to: "/evaluate", label: "Evaluate" },
  { to: "/xai/global", label: "XAI Global" },
  { to: "/xai/local", label: "XAI Local" },
  { to: "/xai/quality", label: "XAI Quality" },
  { to: "/predict", label: "Predict" },
  { to: "/reports", label: "Reports" },
];

export default function NavBar() {
  const { pathname } = useLocation();

  return (
    <nav className="border-b border-gray-300 px-4 py-3">
      <ul className="flex flex-wrap gap-4 text-sm">
        {LINKS.map(({ to, label }) => (
          <li key={to}>
            <Link
              to={to}
              className={pathname === to ? "underline" : "hover:underline"}
            >
              {label}
            </Link>
          </li>
        ))}
      </ul>
    </nav>
  );
}
