import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import { getReportFile, getReports } from "../api/client";
import LoadingSpinner from "../components/LoadingSpinner";
import { FOLDER_ORDER, toImageSrc } from "../utils/constants";

function normalizeFolder(path) {
  const rel = path.replace(/^reports\//, "");
  if (rel.startsWith("xai/global")) return "xai/global";
  if (rel.startsWith("xai/local")) return "xai/local";
  if (rel.startsWith("xai/quality")) return "xai/quality";
  if (rel.startsWith("evaluate/")) return "evaluate";
  if (rel.startsWith("train/")) return "train";
  if (rel.startsWith("eda/")) return "eda";
  const top = rel.split("/")[0];
  return top || "root";
}

export default function ReportsPage() {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selected, setSelected] = useState(null);
  const [content, setContent] = useState(null);
  const [contentLoading, setContentLoading] = useState(false);

  useEffect(() => {
    getReports()
      .then(setFiles)
      .finally(() => setLoading(false));
  }, []);

  const grouped = useMemo(() => {
    const groups = {};
    for (const file of files) {
      const folder = normalizeFolder(file.path);
      if (!groups[folder]) groups[folder] = [];
      groups[folder].push(file);
    }
    return groups;
  }, [files]);

  const sortedFolders = useMemo(() => {
    const keys = Object.keys(grouped);
    return keys.sort((a, b) => {
      const ai = FOLDER_ORDER.indexOf(a.split("/")[0]);
      const bi = FOLDER_ORDER.indexOf(b.split("/")[0]);
      return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi) || a.localeCompare(b);
    });
  }, [grouped]);

  const handleSelect = async (file) => {
    setSelected(file);
    setContentLoading(true);
    try {
      const data = await getReportFile(file.path);
      setContent(data);
    } catch (err) {
      setContent({ type: "text", content: err.message });
    } finally {
      setContentLoading(false);
    }
  };

  if (loading) return <LoadingSpinner />;

  return (
    <div>
      <h1 className="mb-4 text-lg font-medium">Reports</h1>

      <div className="grid gap-6 md:grid-cols-2">
        <div className="space-y-4">
          {sortedFolders.map((folder) => (
            <div key={folder}>
              <h2 className="mb-1 text-sm font-medium">{folder}/</h2>
              <ul className="space-y-1 text-sm">
                {grouped[folder].map((file) => (
                  <li key={file.path}>
                    <button
                      type="button"
                      onClick={() => handleSelect(file)}
                      className={`text-left hover:underline ${
                        selected?.path === file.path ? "font-medium underline" : ""
                      }`}
                    >
                      {file.path.split("/").pop()}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div>
          {!selected && (
            <p className="text-sm text-gray-500">Select a file to view.</p>
          )}
          {contentLoading && <LoadingSpinner />}
          {content && !contentLoading && selected && (
            <div className="max-h-[70vh] overflow-auto rounded border border-gray-300 p-4">
              {content.type === "image" ? (
                <img
                  src={toImageSrc(content.content)}
                  alt={selected.path}
                  className="w-full"
                />
              ) : selected.path.endsWith(".md") ? (
                <div className="prose prose-sm max-w-none">
                  <ReactMarkdown>{content.content}</ReactMarkdown>
                </div>
              ) : (
                <pre className="whitespace-pre-wrap font-mono text-xs">
                  {content.content}
                </pre>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
