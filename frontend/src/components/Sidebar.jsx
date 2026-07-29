import { useContext, useEffect, useState } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { NavLink, useNavigate } from "react-router-dom";
import { signOut } from "firebase/auth";
import { AuthContext } from "../auth/AuthContext";
import { getGoogleAlerts } from "../api/client";
import Badge from "./Badge";
import { auth } from "../firebase";
import { NAV_LINKS } from "../utils/nav";

export default function Sidebar({ sidebarOpen, setSidebarOpen }) {
  const { user } = useContext(AuthContext);
  const navigate = useNavigate();
  const [alertsEnabled, setAlertsEnabled] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [alertsError, setAlertsError] = useState(false);

  useEffect(() => {
    if (!alertsEnabled || !sidebarOpen) {
      return;
    }

    let cancelled = false;

    async function loadAlerts() {
      setAlertsError(false);
      try {
        const data = await getGoogleAlerts();
        if (!cancelled) {
          setAlerts(Array.isArray(data) ? data : []);
        }
      } catch {
        if (!cancelled) {
          setAlerts([]);
          setAlertsError(true);
        }
      }
    }

    loadAlerts();
    return () => {
      cancelled = true;
    };
  }, [alertsEnabled, sidebarOpen]);

  const handleSignOut = async () => {
    await signOut(auth);
    navigate("/login");
  };

  return (
    <aside
      className={`fixed left-0 top-0 z-40 flex h-full flex-col border-r border-slate-700/80 bg-slate-900/85 backdrop-blur-sm pt-20 transition-all duration-300 ease-in-out ${
        sidebarOpen ? "w-64" : "w-14"
      }`}
    >
      <div className="flex items-center justify-end px-3 py-2">
        <button
          type="button"
          onClick={() => setSidebarOpen((prev) => !prev)}
          className="rounded-md p-1 text-slate-400 transition hover:bg-slate-800 hover:text-slate-200"
          aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
        >
          {sidebarOpen ? (
            <ChevronLeft className="h-4 w-4" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
        </button>
      </div>

      <div
        className={`overflow-hidden transition-all duration-300 ${
          sidebarOpen ? "opacity-100" : "opacity-0"
        }`}
      >
        {sidebarOpen && (
          <>
            <div className="px-2">
              {NAV_LINKS.map(({ to, label }) => (
                <NavLink
                  key={to}
                  to={to}
                  className={({ isActive }) =>
                    `flex items-center gap-2 rounded-lg px-3 py-2 text-sm transition-colors ${
                      isActive
                        ? "bg-slate-700 font-medium text-sky-400"
                        : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                    }`
                  }
                >
                  <span>{label}</span>
                </NavLink>
              ))}
            </div>

            <hr className="mx-4 my-3 border-slate-700" />

            <div className="px-4">
              <p className="text-xs uppercase tracking-wider text-slate-500">
                Live Alerts
              </p>
              <div className="mt-1 flex items-center justify-between">
                <p className="text-sm text-slate-300">Gmail Account Alerts</p>
                <button
                  type="button"
                  onClick={() => setAlertsEnabled((prev) => !prev)}
                  className={`relative h-5 w-10 rounded-full ${
                    alertsEnabled ? "bg-sky-600" : "bg-slate-700"
                  }`}
                  aria-label="Toggle Gmail alerts"
                >
                  <span
                    className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${
                      alertsEnabled ? "left-5" : "left-0.5"
                    }`}
                  />
                </button>
              </div>

              {alertsEnabled && (
                <div className="mt-3">
                  {alerts.map((alert, index) => (
                    <div
                      key={`${alert.type}-${index}`}
                      className="mb-2 rounded-lg border border-slate-700 bg-slate-800 p-3"
                    >
                      <div className="flex items-center gap-2">
                        <Badge type={String(alert.severity).toLowerCase()}>
                          {alert.severity}
                        </Badge>
                        <p className="text-sm text-slate-200">{alert.type}</p>
                      </div>
                      <p className="mt-0.5 text-xs text-slate-400">
                        {alert.detail}
                      </p>
                    </div>
                  ))}
                  {alertsError && (
                    <p className="mt-2 text-xs text-red-400">
                      Could not load alerts.
                    </p>
                  )}
                </div>
              )}
            </div>

            <div className="mt-auto px-4 pb-4">
              {user && (
                <button
                  type="button"
                  onClick={handleSignOut}
                  className="text-xs text-red-400 transition hover:text-red-300"
                >
                  Sign out
                </button>
              )}
            </div>
          </>
        )}
      </div>
    </aside>
  );
}
