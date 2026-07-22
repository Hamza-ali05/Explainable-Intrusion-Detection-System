export const NAV_LINKS = [
  { to: "/status", label: "Status" },
  { to: "/evaluate", label: "Evaluate" },
  { to: "/xai/global", label: "XAI Global" },
  { to: "/xai/local", label: "XAI Local" },
  { to: "/xai/quality", label: "XAI Quality" },
  { to: "/predict", label: "Predict" },
  { to: "/reports", label: "Reports" },
];

export function isNavActive(pathname, to) {
  return pathname === to || (to === "/status" && pathname === "/");
}
