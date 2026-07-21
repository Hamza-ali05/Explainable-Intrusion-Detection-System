import { useCallback, useEffect, useRef, useState } from "react";
import { getPipelineLog, getStatus, runPipeline } from "../api/client";
import Badge from "../components/Badge";
import LoadingSpinner from "../components/LoadingSpinner";
import { formatTimestamp } from "../utils/constants";

export default function StatusPage() {
  const [steps, setSteps] = useState([]);
  const [loading, setLoading] = useState(true);
  const [runningStep, setRunningStep] = useState(null);
  const [polling, setPolling] = useState(false);
  const [logLines, setLogLines] = useState([]);
  const logRef = useRef(null);

  const fetchStatus = useCallback(async () => {
    const data = await getStatus();
    setSteps(data);
    return data;
  }, []);

  useEffect(() => {
    fetchStatus().finally(() => setLoading(false));
  }, [fetchStatus]);

  useEffect(() => {
    if (!polling) return undefined;

    const pollLog = async () => {
      try {
        const { lines } = await getPipelineLog();
        setLogLines(lines.slice(-50));
      } catch {
        /* ignore */
      }
    };

    pollLog();
    const id = setInterval(pollLog, 3000);
    return () => clearInterval(id);
  }, [polling]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logLines]);

  useEffect(() => {
    if (!polling || !runningStep) return undefined;

    const checkDone = async () => {
      const data = await fetchStatus();
      if (runningStep === "all") {
        if (data.every((s) => s.done)) {
          setPolling(false);
          setRunningStep(null);
        }
      } else {
        const row = data.find((s) => s.step === runningStep);
        if (row?.done) {
          setPolling(false);
          setRunningStep(null);
        }
      }
    };

    const id = setInterval(checkDone, 3000);
    return () => clearInterval(id);
  }, [polling, runningStep, fetchStatus]);

  const handleRun = async (step) => {
    setRunningStep(step);
    setPolling(true);
    try {
      await runPipeline(step);
      await fetchStatus();
    } catch (err) {
      setPolling(false);
      setRunningStep(null);
      alert(err.response?.data?.detail ?? err.message);
    }
  };

  if (loading) return <LoadingSpinner />;

  return (
    <div>
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-lg font-medium">Pipeline Status</h1>
        <button
          type="button"
          onClick={() => handleRun("all")}
          disabled={runningStep !== null}
          className="rounded border border-gray-400 px-3 py-1 text-sm hover:bg-gray-50 disabled:opacity-50"
        >
          Run Full Pipeline
        </button>
      </div>

      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-gray-300 text-left">
            <th className="py-2 pr-4">Step</th>
            <th className="py-2 pr-4">Status</th>
            <th className="py-2 pr-4">Timestamp</th>
            <th className="py-2">Action</th>
          </tr>
        </thead>
        <tbody>
          {steps.map((row) => (
            <tr key={row.step} className="border-b border-gray-200">
              <td className="py-2 pr-4 font-mono">{row.step}</td>
              <td className="py-2 pr-4">
                {runningStep === row.step || (runningStep === "all" && !row.done) ? (
                  <LoadingSpinner label="Running…" />
                ) : (
                  <Badge done={row.done} />
                )}
              </td>
              <td className="py-2 pr-4">{formatTimestamp(row.timestamp)}</td>
              <td className="py-2">
                {!row.done && (
                  <button
                    type="button"
                    onClick={() => handleRun(row.step)}
                    disabled={runningStep !== null}
                    className="rounded border border-gray-400 px-2 py-0.5 text-xs hover:bg-gray-50 disabled:opacity-50"
                  >
                    Run
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {(polling || logLines.length > 0) && (
        <div className="mt-6">
          <h2 className="mb-2 text-sm font-medium">Pipeline Log</h2>
          <pre
            ref={logRef}
            className="max-h-64 overflow-auto rounded border border-gray-300 bg-gray-50 p-3 font-mono text-xs"
          >
            {logLines.join("\n") || "Waiting for log output…"}
          </pre>
        </div>
      )}
    </div>
  );
}
