import { useCallback, useEffect, useState } from "react";
import { Loader2 } from "lucide-react";
import {
  getPipelineState,
  getStatus,
  runPipeline,
} from "../api/client";
import Badge from "../components/Badge";
import Card from "../components/Card";
import DataTable from "../components/DataTable";
import ErrorCard from "../components/ErrorCard";
import PageSkeleton from "../components/PageSkeleton";
import { apiError, formatPipelineStep, formatTimestamp } from "../utils/constants";

export default function StatusPage() {
  const [steps, setSteps] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [loadingStep, setLoadingStep] = useState(null);
  const [polling, setPolling] = useState(false);

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
        const [statusRows, state] = await Promise.all([getStatus(), getPipelineState()]);
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

  const statusTableData = steps.map((row) => ({
    Step: formatPipelineStep(row.step),
    Status: (
      <Badge type={row.done ? "done" : "pending"}>
        {row.done ? "Done" : "Pending"}
      </Badge>
    ),
    Timestamp: formatTimestamp(row.timestamp),
  }));

  return (
    <div className="space-y-6">
      <ErrorCard error={error} />

      {loading ? (
        <PageSkeleton />
      ) : (
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
      )}
    </div>
  );
}
