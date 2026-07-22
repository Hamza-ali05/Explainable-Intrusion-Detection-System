import { NavLink } from "react-router-dom";
import { NAV_LINKS } from "../utils/nav";

export default function Navbar() {
  return (
    <header className="fixed left-0 right-0 top-0 z-50 px-4 pt-4 sm:px-6">
      <nav
        className="flex w-full items-center gap-4 overflow-hidden rounded-full border border-violet-500/35 bg-gradient-to-r from-indigo-950/95 via-slate-800/95 to-violet-900/90 px-5 py-3.5 shadow-[0_12px_40px_rgba(76,29,149,0.35),inset_0_1px_0_rgba(255,255,255,0.06)] backdrop-blur-xl sm:gap-6 sm:px-8 sm:py-4"
        aria-label="Main navigation"
      >
        <div className="shrink-0 border-r border-violet-400/25 pr-5 sm:pr-6">
          <p className="whitespace-nowrap bg-gradient-to-r from-cyan-300 via-sky-400 to-violet-300 bg-clip-text text-lg font-bold tracking-tight text-transparent sm:text-xl">
            XAI Cybersecurity
          </p>
          <p className="hidden whitespace-nowrap text-xs font-medium text-violet-200/70 sm:block">
            Explainable threat classification
          </p>
        </div>

        <ul className="flex min-w-0 flex-1 flex-nowrap items-center gap-1.5 overflow-x-auto sm:gap-2">
          {NAV_LINKS.map(({ to, label }) => (
            <li key={to} className="shrink-0">
              <NavLink
                to={to}
                className={({ isActive }) =>
                  [
                    "inline-flex items-center whitespace-nowrap rounded-full px-3.5 py-2.5 text-sm font-medium transition-all duration-200 sm:px-5 sm:py-3 sm:text-base",
                    isActive
                      ? "bg-gradient-to-r from-sky-500 to-cyan-400 text-slate-950 shadow-[0_0_18px_rgba(56,189,248,0.45)]"
                      : "text-violet-100/80 hover:bg-violet-500/20 hover:text-white",
                  ].join(" ")
                }
              >
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
}
