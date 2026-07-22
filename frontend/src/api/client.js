import axios from "axios";

const api = axios.create({
  baseURL: "/api",
});

export default api;

export const getStatus = () => api.get("/status").then((r) => r.data);
export const getPipelineState = () => api.get("/pipeline/state").then((r) => r.data);
export const runPipeline = (step, force = false) =>
  api.post("/pipeline/run", { step, force }).then((r) => r.data);
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

export const getPredictSamples = () => api.get("/predict/samples").then((r) => r.data);

export const refreshPredictSample = (threatClass) =>
  api
    .get("/predict/samples/refresh", { params: { threat_class: threatClass } })
    .then((r) => r.data);

export const predictSingle = (model, features) =>
  api.post("/predict/single", { model, features }).then((r) => r.data);

export const predictFile = (file, model) => {
  const form = new FormData();
  form.append("file", file);
  form.append("model", model);
  return api.post("/predict", form).then((r) => r.data);
};

export const predictDemo = (model) => {
  const form = new FormData();
  form.append("model", model);
  return api.post("/predict/demo", form).then((r) => r.data);
};

export const explainRow = (model, row) =>
  api.post("/predict/explain", { model, row }).then((r) => r.data);

export const getGoogleAlerts = () => api.get("/google/alerts").then((r) => r.data);
