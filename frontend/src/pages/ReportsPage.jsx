import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import { getReportFile, getReports } from "../api/client";
import Card from "../components/Card";
import DataTable from "../components/DataTable";
import ErrorCard from "../components/ErrorCard";
import PageSkeleton from "../components/PageSkeleton";
import PageTitle from "../components/PageTitle";
import JsonTable from "../components/JsonTable";
import { FOLDER_ORDER, apiError, toImageSrc } from "../utils/constants";

function subfolderFromPath(path) {
  const rel = path.replace(/^reports\/?/, "");
  const parts = rel.split("/");
  if (parts.length <= 1) return "root";
  return parts.slice(0, -1).join("/");
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];

  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const cells = line.split(",").map((c) => c.trim());
    const row = {};
    headers.forEach((header, index) => {
      row[header] = cells[index] ?? "";
    });
    return row;
  });
}

function parseJson(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function FilePreview({ file, content, loading, error }) {
  if (!file) return null;

  const filename = file.path.split("/").pop();

  return (
    <Card title={filename}>
      <ErrorCard error={error} />

      {loading && <PageSkeleton />}

      {!loading && content && (
        <div className="max-h-[70vh] overflow-auto">
          {content.type === "image" || file.path.endsWith(".png") ? (
            <img
              src={toImageSrc(content.content)}
              alt={filename}
              className="ui-card-image"
            />
          ) : file.path.endsWith(".md") ? (
            <div className="prose prose-invert prose-sm max-w-none">
              <ReactMarkdown>{content.content}</ReactMarkdown>
            </div>
          ) : file.path.endsWith(".csv") ? (
            <div className="overflow-x-auto">
              <DataTable data={parseCsv(content.content)} />
            </div>
          ) : file.path.endsWith(".json") ? (
            <JsonTable data={parseJson(content.content)} />
          ) : (
            <pre className="whitespace-pre-wrap font-mono text-xs text-slate-300">
              {content.content}
            </pre>
          )}
        </div>
      )}

      {!loading && !content && !error && (
        <p className="text-sm text-slate-500">Could not load file.</p>
      )}
    </Card>
  );
}

export default function ReportsPage() {
  const [files, setFiles] = useState([]);
  const [selected, setSelected] = useState(null);
  const [content, setContent] = useState(null);
  const [loadingList, setLoadingList] = useState(true);
  const [loadingFile, setLoadingFile] = useState(false);
  const [error, setError] = useState(null);
  const [fileError, setFileError] = useState(null);

  useEffect(() => {
    (async () => {
      try {
        const data = await getReports();
        setFiles(data);
      } catch (err) {
        setError({ message: apiError(err) });
        setFiles([]);
      } finally {
        setLoadingList(false);
      }
    })();
  }, []);

  const grouped = useMemo(() => {
    const groups = {};
    for (const file of files) {
      const folder = subfolderFromPath(file.path);
      if (!groups[folder]) groups[folder] = [];
      groups[folder].push(file);
    }
    for (const folder of Object.keys(groups)) {
      groups[folder].sort((a, b) => a.path.localeCompare(b.path));
    }
    return groups;
  }, [files]);

  const sortedFolders = useMemo(() => {
    return Object.keys(grouped).sort((a, b) => {
      const ai = FOLDER_ORDER.indexOf(a.split("/")[0]);
      const bi = FOLDER_ORDER.indexOf(b.split("/")[0]);
      return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi) || a.localeCompare(b);
    });
  }, [grouped]);

  const handleSelect = async (file) => {
    setSelected(file);
    setContent(null);
    setFileError(null);
    setLoadingFile(true);
    try {
      const data = await getReportFile(file.path);
      setContent(data);
    } catch (err) {
      setContent(null);
      setFileError({ message: apiError(err) });
    } finally {
      setLoadingFile(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageTitle>Reports</PageTitle>
      <ErrorCard error={error} />

      {loadingList ? (
        <PageSkeleton />
      ) : (
        <>
          {sortedFolders.length === 0 && (
            <p className="text-sm text-slate-500">No report files found.</p>
          )}

          {sortedFolders.map((folder) => (
            <Card key={folder} title={folder} defaultOpen={false}>
              <ul className="space-y-1">
                {grouped[folder].map((file) => {
                  const name = file.path.split("/").pop();
                  const active = selected?.path === file.path;
                  return (
                    <li key={file.path}>
                      <button
                        type="button"
                        onClick={() => handleSelect(file)}
                        className={`text-left text-sm transition hover:text-sky-400 ${
                          active ? "font-medium text-sky-400" : "text-slate-300"
                        }`}
                      >
                        {name}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </Card>
          ))}
        </>
      )}

      <FilePreview
        file={selected}
        content={content}
        loading={loadingFile}
        error={fileError}
      />
    </div>
  );
}
