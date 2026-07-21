import axios from "axios";

const api = axios.create({
  baseURL: "/api",
});

export default api;

export const getStatus = () => api.get("/status").then((r) => r.data);
export const runPipeline = (step) => api.post("/pipeline/run", { step }).then((r) => r.data);
export const getPipelineLog = () => api.get("/pipeline/log").then((r) => r.data);

export const getLeaderboard = () => api.get("/evaluate/leaderboard").then((r) => r.data);
export const getEvaluateDetail = (model) =>
  api.get("/evaluate/detail", { params: { model } }).then((r) => r.data);
export const getTrainingCurve = (model) =>
  api.get("/evaluate/training-curve", { params: { model } }).then((r) => r.data);

export const getXaiGlobal = (model, plot_type) =>
  api.get("/xai/global", { params: { model, plot_type } }).then((r) => r.data);
export const getXaiLocalInstances = () => api.get("/xai/local/instances").then((r) => r.data);
export const getXaiLocalImage = (params) =>
  api.get("/xai/local/image", { params }).then((r) => r.data);
export const getXaiQuality = (metric) =>
  api.get("/xai/quality", { params: { metric } }).then((r) => r.data);

export const predictFile = (file, model) => {
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);
  return api.post("/predict", form).then((r) => r.data);
};

export const getReports = () => api.get("/reports").then((r) => r.data);
export const getReportFile = (path) =>
  api.get("/reports/file", { params: { path } }).then((r) => r.data);
