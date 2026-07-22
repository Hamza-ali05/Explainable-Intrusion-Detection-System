import { formatTableCell } from "./JsonTable";

export default function DataTable({ data, className = "" }) {
  if (!data || data.length === 0) {
    return <p className="text-sm text-slate-500">No data.</p>;
  }

  const columns = Object.keys(data[0]);

  return (
    <div className={`overflow-x-auto ${className}`}>
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
