export const NAV_LINKS = [
  { to: "/ids", label: "IDS" },
  { to: "/status", label: "Status" },
  { to: "/evaluate", label: "Evaluate" },
  { to: "/xai/global", label: "XAI Global" },
  { to: "/xai/local", label: "XAI Local" },
  { to: "/xai/quality", label: "XAI Quality" },
  { to: "/predict", label: "Predict" },
];

export function isNavActive(pathname, to) {
  return pathname === to || (to === "/ids" && pathname === "/");
}
