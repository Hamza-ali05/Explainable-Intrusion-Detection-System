import { useCallback, useEffect, useRef, useState } from "react";
import { useOutletContext } from "react-router-dom";
import { Loader2 } from "lucide-react";
import {
  getGoogleAlerts,
  getPipelineLog,
  getPipelineState,
  getStatus,
  runPipeline,
} from "../api/client";
import Badge from "../components/Badge";
import Card from "../components/Card";
import DataTable from "../components/DataTable";
import ErrorCard from "../components/ErrorCard";
import PageSkeleton from "../components/PageSkeleton";
import PageTitle from "../components/PageTitle";
import { apiError, formatTimestamp } from "../utils/constants";

export default function StatusPage() {
  const { liveAlertsEnabled } = useOutletContext() || {};

  const [steps, setSteps] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [loadingStep, setLoadingStep] = useState(null);
  const [polling, setPolling] = useState(false);
  const [logLines, setLogLines] = useState([]);
  const [alerts, setAlerts] = useState(null);
  const [alertsError, setAlertsError] = useState(false);
  const logRef = useRef(null);

  const fetchStatus = useCallback(async () => {
    const data = await getStatus();
    setSteps(data);
    return data;
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const [statusRows, state] = await Promise.all([getStatus(), getPipelineState()]);
        setSteps(statusRows);
        if (state.running) {
          setLoadingStep(state.step || "all");
          setPolling(true);
        }
      } catch (err) {
        setError({ message: apiError(err) });
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (!polling) return undefined;

    const tick = async () => {
      try {
        const [{ lines }, statusRows, state] = await Promise.all([
          getPipelineLog(),
          getStatus(),
          getPipelineState(),
        ]);
        setLogLines((lines || []).slice(-50));
        setSteps(statusRows);

        const allDone = statusRows.length > 0 && statusRows.every((s) => s.done);
        if (allDone || !state.running) {
          setPolling(false);
          setLoadingStep(null);
        }
      } catch {
        /* keep polling */
      }
    };

    tick();
    const id = setInterval(tick, 3000);
    return () => clearInterval(id);
  }, [polling]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines]);

  useEffect(() => {
    if (!liveAlertsEnabled) {
      setAlerts(null);
      setAlertsError(false);
      return undefined;
    }

    let cancelled = false;
    (async () => {
      try {
        const data = await getGoogleAlerts();
        if (!cancelled) {
          setAlerts(data?.alerts || []);
          setAlertsError(false);
        }
      } catch (err) {
        if (!cancelled) {
          setAlerts(null);
          setAlertsError(true);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [liveAlertsEnabled]);

  const handleRun = async (step) => {
    setLoadingStep(step);
    setPolling(true);
    setError(null);
    try {
      await runPipeline(step);
      await fetchStatus();
    } catch (err) {
      setPolling(false);
      setLoadingStep(null);
      setError({ message: apiError(err) });
    }
  };

  const statusTableData = steps.map((row) => {
    const isLoading = loadingStep === row.step || (loadingStep === "all" && !row.done);
    return {
      Step: row.step,
      Status: (
        <Badge type={row.done ? "done" : "pending"}>
          {row.done ? "Done" : "Pending"}
        </Badge>
      ),
      Timestamp: formatTimestamp(row.timestamp),
      Action: row.done ? (
        "—"
      ) : isLoading ? (
        <Loader2 className="h-4 w-4 animate-spin text-sky-400" aria-label="Running" />
      ) : (
        <button
          type="button"
          onClick={() => handleRun(row.step)}
          disabled={loadingStep !== null}
          className="rounded-lg bg-sky-500 px-3 py-1 text-xs font-medium text-white transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Run
        </button>
      ),
    };
  });

  const alertsTableData =
    alerts?.map((alert) => ({
      Severity: (
        <Badge type={String(alert.severity || "").toLowerCase()}>
          {alert.severity}
        </Badge>
      ),
      Type: alert.type,
      Timestamp: formatTimestamp(alert.timestamp),
      Detail: alert.detail,
    })) || [];

  return (
    <div className="space-y-6">
      <PageTitle>Status</PageTitle>
      <ErrorCard error={error} />

      {loading ? (
        <PageSkeleton />
      ) : (
        <>
      <Card
        title="Pipeline Status"
        actions={
          <button
            type="button"
            onClick={() => handleRun("all")}
            disabled={loadingStep !== null}
            className="rounded-lg bg-sky-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loadingStep === "all" ? (
              <span className="inline-flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                Running
              </span>
            ) : (
              "Run Full Pipeline"
            )}
          </button>
        }
      >
        <div className="overflow-x-auto">
          <DataTable data={statusTableData} />
        </div>
      </Card>

      <Card title="Pipeline Log">
        <pre
          ref={logRef}
          className="max-h-96 overflow-y-auto rounded-xl bg-slate-900 p-4 font-mono text-xs text-slate-300"
        >
          {logLines.length > 0 ? logLines.join("\n") : "No log output yet."}
        </pre>
      </Card>

      {liveAlertsEnabled && (
        <Card title="Live Account Alerts">
          {alertsError ? (
            <p className="text-sm text-slate-500">Could not load alerts.</p>
          ) : (
            <div className="overflow-x-auto">
              <DataTable data={alertsTableData} />
            </div>
          )}
        </Card>
      )}
        </>
      )}
    </div>
  );
}
