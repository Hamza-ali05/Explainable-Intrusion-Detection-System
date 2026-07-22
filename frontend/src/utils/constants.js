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

export const THREAT_CLASSES = [
  "BENIGN",
  "DDoS",
  "DoS",
  "PortScan",
  "BruteForce",
  "WebAttack",
  "Infiltration",
  "Botnet",
];

export const modelLabel = (value) =>
  MODELS.find((m) => m.value === value)?.label ?? value;

export const toImageSrc = (base64) => {
  if (!base64) return null;
  const raw = String(base64).trim();
  if (raw.startsWith("data:image")) return raw;
  return `data:image/png;base64,${raw}`;
};

export const formatTimestamp = (ts) => {
  if (!ts) return "—";
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return ts;
  }
};

export const formatNumber = (n, digits = 4) => {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(digits);
};

export const apiError = (err) =>
  err?.response?.data?.detail ?? err?.message ?? "Request failed";

export const FOLDER_ORDER = ["evaluate", "xai", "train", "eda", "root"];

export const IMAGE_LABELS = {
  confusion_matrix: "Confusion matrix",
  confusion_matrix_row_normalized: "Confusion matrix (row-normalized)",
  roc_curves: "ROC curves",
  pr_curves: "PR curves",
};
