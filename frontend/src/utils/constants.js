export const MODELS = [
  { value: "decision_tree", label: "Decision Tree" },
  { value: "random_forest", label: "Random Forest" },
  { value: "xgboost", label: "XGBoost" },
];

export const PLOT_TYPES = [
  { value: "beeswarm", label: "Beeswarm" },
  { value: "bar", label: "Bar" },
  { value: "dependence", label: "Dependence" },
];

export const EXPLAINERS = [
  { value: "xgb_shap", label: "XGBoost SHAP" },
  { value: "xgb_lime", label: "XGBoost LIME" },
  { value: "dt_shap", label: "Decision Tree SHAP" },
];

export const modelLabel = (value) =>
  MODELS.find((m) => m.value === value)?.label ?? value;

export const toImageSrc = (base64) =>
  base64 ? `data:image/png;base64,${base64}` : null;

export const formatTimestamp = (ts) => {
  if (!ts) return "—";
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
};

export const formatNumber = (n) => {
  if (n == null || Number.isNaN(n)) return "—";
  return Number(n).toFixed(4);
};

export const groupReportsByFolder = (files) => {
  const groups = {};
  for (const file of files) {
    const parts = file.path.replace(/^reports\//, "").split("/");
    const folder = parts.length > 1 ? parts[0] : "root";
    if (!groups[folder]) groups[folder] = [];
    groups[folder].push(file);
  }
  return groups;
};

export const FOLDER_ORDER = ["evaluate", "xai", "train", "eda", "root"];
