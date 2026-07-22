import { isValidElement } from "react";

/**
 * Render arbitrary JSON as a table (never raw JSON text).
 * - Arrays of objects → columned table
 * - Flat objects → key/value table
 * - Nested values → nested tables
 */
export default function JsonTable({ data, nested = false }) {
  if (data == null) {
    return <p className="text-sm text-slate-400">—</p>;
  }

  const wrapperClass = nested ? "" : "overflow-x-auto";

  if (Array.isArray(data)) {
    if (data.length === 0) {
      return <p className="text-sm text-slate-400">Empty array</p>;
    }
    if (data.every((item) => item !== null && typeof item === "object" && !Array.isArray(item))) {
      const columns = Array.from(
        data.reduce((set, row) => {
          Object.keys(row).forEach((k) => set.add(k));
          return set;
        }, new Set()),
      );
      return (
        <div className={wrapperClass}>
          <table className="ui-table">
            <thead>
              <tr>
                {columns.map((col) => (
                  <th key={col}>{col}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, i) => (
                <tr key={i} className="even:bg-slate-800/50">
                  {columns.map((col) => (
                    <td key={col}>{formatTableCell(row[col])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }

    return (
      <div className={wrapperClass}>
        <table className="ui-table">
          <thead>
            <tr>
              <th>Index</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item, i) => (
              <tr key={i} className="even:bg-slate-800/50">
                <td>{i}</td>
                <td>{formatTableCell(item)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  if (typeof data === "object") {
    const entries = Object.entries(data);
    return (
      <div className={wrapperClass}>
        <table className="ui-table">
          <thead>
            <tr>
              <th>Key</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([key, value]) => (
              <tr key={key} className="even:bg-slate-800/50">
                <td className="text-slate-200">{key}</td>
                <td>{formatTableCell(value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  return (
    <div className={wrapperClass}>
      <table className="ui-table">
        <thead>
          <tr>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          <tr className="even:bg-slate-800/50">
            <td>{String(data)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

export function formatTableCell(value, depth = 0) {
  if (value == null) return "—";
  if (isValidElement(value)) return value;
  if (typeof value === "object") {
    if (depth >= 4) return "…";
    return <JsonTable data={value} nested />;
  }
  return String(value);
}
