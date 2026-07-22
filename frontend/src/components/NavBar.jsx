import { NavLink } from "react-router-dom";
import { NAV_LINKS } from "../utils/nav";

export default function Navbar() {
  return (
    <nav className="fixed top-4 left-1/2 z-50 mx-auto flex -translate-x-1/2 items-center gap-6 rounded-full border border-slate-700 bg-slate-800/90 px-6 py-3 shadow-xl backdrop-blur">
      {NAV_LINKS.map(({ to, label }) => (
        <NavLink
          key={to}
          to={to}
          className={({ isActive }) =>
            isActive
              ? "text-sm font-semibold text-sky-400"
              : "text-sm text-slate-400 transition-colors hover:text-slate-100"
          }
        >
          {label}
        </NavLink>
      ))}
    </nav>
  );
}
