import { ChevronLeft, ChevronRight } from "lucide-react";

export default function Sidebar({
  collapsed,
  onToggle,
  liveAlertsEnabled,
  setLiveAlertsEnabled,
}) {
  return (
    <aside
      className={`fixed left-0 top-16 z-40 flex h-[calc(100vh-4rem)] flex-col overflow-hidden border-r border-slate-700 bg-slate-900 transition-all duration-300 ${
        collapsed ? "w-14" : "w-64"
      }`}
    >
      <div className={`flex py-2 ${collapsed ? "justify-center px-2" : "justify-end px-3"}`}>
        <button
          type="button"
          onClick={onToggle}
          className="rounded-lg p-2 text-slate-400 transition hover:bg-slate-800 hover:text-slate-100"
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-expanded={!collapsed}
        >
          {collapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
        </button>
      </div>

      <div
        className={`flex min-h-0 flex-1 flex-col overflow-hidden px-5 pb-6 transition-opacity duration-300 ${
          collapsed ? "pointer-events-none opacity-0" : "opacity-100"
        }`}
      >
        <div className="flex min-h-[50%] flex-1 flex-col justify-center border-b border-slate-700/80 py-8">
          <p className="text-2xl font-semibold leading-tight text-sky-400">
            XAI Cybersecurity
          </p>
          <p className="mt-5 text-sm leading-relaxed text-slate-300">
            Explainable AI for network threat classification
          </p>
          <p className="mt-4 text-sm italic leading-relaxed text-slate-500">
            Explainable AI (XAI) for Cybersecurity Threat Classification
          </p>
        </div>

        <div className="mt-auto pt-6">
          <label className="flex items-center gap-2 text-xs text-slate-400">
            <input
              type="checkbox"
              checked={!!liveAlertsEnabled}
              onChange={(e) => setLiveAlertsEnabled?.(e.target.checked)}
              className="rounded border-slate-600 bg-slate-800 text-sky-500 focus:ring-sky-500"
            />
            Live account alerts
          </label>
        </div>
      </div>
    </aside>
  );
}
