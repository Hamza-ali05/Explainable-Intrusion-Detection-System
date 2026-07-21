"""Prompt 9 API smoke tests. Run after: uvicorn api.main:app --reload --host 127.0.0.1 --port 8000"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parents[1]
PASS = 0
FAIL = 0


def wait_for_server(timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req("GET", "/api/health")
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    print("ERROR: API not reachable. Start uvicorn first.")
    sys.exit(1)


def req(method: str, path: str, data: dict | None = None, expect: int = 200) -> dict | list:
    global PASS, FAIL
    url = f"{BASE}{path}"
    body = json.dumps(data).encode() if data is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body else {},
    )
    label = f"{method} {path}"
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:
            status = resp.status
            raw = resp.read().decode()
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read().decode()
        if status != expect:
            print(f"FAIL  {label}  expected {expect}, got {status}")
            FAIL += 1
            return {}
        print(f"PASS  {label}  HTTP {status}")
        PASS += 1
        return json.loads(raw) if raw else {}
    except urllib.error.URLError as exc:
        print(f"FAIL  {label}  {exc}")
        FAIL += 1
        return {}

    if status != expect:
        print(f"FAIL  {label}  expected {expect}, got {status}")
        FAIL += 1
        return {}

    print(f"PASS  {label}  HTTP {status}")
    PASS += 1
    return json.loads(raw) if raw else {}


def req_multipart(path: str, file_path: Path, model: str) -> dict:
    global PASS, FAIL
    import uuid

    boundary = uuid.uuid4().hex
    file_bytes = file_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode() + file_bytes + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"{model}\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    label = f"POST {path}"
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:
            raw = resp.read().decode()
            print(f"PASS  {label}  HTTP {resp.status}")
            PASS += 1
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        print(f"FAIL  {label}  HTTP {exc.code}: {exc.read().decode()[:200]}")
        FAIL += 1
        return {}


def main() -> None:
    print(f"Waiting for API at {BASE} ...")
    wait_for_server()

    status = req("GET", "/api/status")
    if isinstance(status, list):
        print(f"       steps: {len(status)}, done: {sum(1 for s in status if s.get('done'))}")

    req("POST", "/api/pipeline/run", {"step": "predict"})
    req("POST", "/api/pipeline/run", {"step": "status"}, expect=400)

    log = req("GET", "/api/pipeline/log")
    if isinstance(log, dict):
        print(f"       log lines: {len(log.get('lines', []))}")

    lb = req("GET", "/api/evaluate/leaderboard")
    if isinstance(lb, list):
        print(f"       models: {len(lb)}")

    detail = req("GET", "/api/evaluate/detail?model=xgboost")
    if isinstance(detail, dict):
        print(f"       images: {len(detail.get('images', {}))}")

    curve = req("GET", "/api/evaluate/training-curve?model=xgboost")
    if isinstance(curve, dict):
        print(f"       training curve image: {bool(curve.get('image'))}")

    beeswarm = req("GET", "/api/xai/global?model=xgboost&plot_type=beeswarm")
    if isinstance(beeswarm, dict):
        print(f"       beeswarm image: {bool(beeswarm.get('image'))}")

    instances = req("GET", "/api/xai/local/instances")
    if isinstance(instances, dict):
        print(f"       instances: {len(instances.get('instances', []))}")

    img = req(
        "GET",
        "/api/xai/local/image?explainer=xgb_shap&true_class=BENIGN&outcome=correct&instance_idx=800461",
    )
    if isinstance(img, dict):
        print(f"       local image: {bool(img.get('image'))}")

    for metric in ("fidelity", "stability", "comprehensibility"):
        q = req("GET", f"/api/xai/quality?metric={metric}")
        if isinstance(q, dict):
            print(f"       {metric} rows: {len(q.get('rows', []))}")

    pred = req_multipart("/api/predict", ROOT / "data/manual/example_flows.csv", "xgboost")
    if pred:
        print(f"       predictions: {pred.get('total_rows')}")

    reports = req("GET", "/api/reports")
    if isinstance(reports, list):
        print(f"       report files: {len(reports)}")

    card = req("GET", "/api/reports/file?path=reports/model_card.md")
    if isinstance(card, dict):
        print(f"       model_card chars: {len(card.get('content', ''))}")

    req("GET", "/api/reports/file?path=reports/does_not_exist.md", expect=404)

    print(f"\nResults: {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
