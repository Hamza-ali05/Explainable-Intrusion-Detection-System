export default function Select({
  id,
  label,
  value,
  onChange,
  options = [],
  placeholder,
  className = "",
}) {
  const resolvedPlaceholder =
    placeholder ?? (label ? `Select ${label.toLowerCase()}` : "Select");

  return (
    <div className={`flex flex-col gap-1 ${className}`}>
      {label && (
        <label htmlFor={id} className="ui-label">
          {label}
        </label>
      )}
      <select
        id={id}
        value={value}
        onChange={onChange}
        className="ui-select"
      >
        <option value="">{resolvedPlaceholder}</option>
        {options.map((opt) => {
          const val = typeof opt === "string" ? opt : opt.value;
          const text = typeof opt === "string" ? opt : opt.label;
          return (
            <option key={val} value={val}>
              {text}
            </option>
          );
        })}
      </select>
    </div>
  );
}
