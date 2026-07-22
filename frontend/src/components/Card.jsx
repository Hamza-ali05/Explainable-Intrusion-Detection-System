import { useState } from "react";
import { ChevronDown } from "lucide-react";

export default function Card({
  title,
  children,
  defaultOpen = true,
  className = "",
  actions = null,
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div
      className={`overflow-hidden rounded-2xl border border-slate-700 bg-slate-800 shadow-lg ${className}`}
    >
      <div
        role="button"
        tabIndex={0}
        onClick={() => setOpen((v) => !v)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setOpen((v) => !v);
          }
        }}
        className="flex cursor-pointer select-none items-center justify-between px-6 py-4"
        aria-expanded={open}
      >
        <h2 className="ui-section-heading">{title}</h2>
        <div className="flex items-center gap-3">
          {actions && (
            <div
              className="flex items-center gap-2"
              onClick={(e) => e.stopPropagation()}
              onKeyDown={(e) => e.stopPropagation()}
            >
              {actions}
            </div>
          )}
          <ChevronDown
            size={18}
            className={`text-slate-400 transition-transform duration-300 ${open ? "rotate-180" : ""}`}
          />
        </div>
      </div>

      <div
        className={`grid transition-[grid-template-rows] duration-300 ease-in-out ${
          open ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
        }`}
      >
        <div className="overflow-hidden">
          <div className="px-6 pb-6">{children}</div>
        </div>
      </div>
    </div>
  );
}
