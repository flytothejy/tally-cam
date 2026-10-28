"""tally-cam 벤치마크 채점.

정답: 라벨러 CSV (route, t_gate_sec, 끝에 '# reviewed_ranges: [...]' 있으면 그 구간만 골든셋)
예측: 사건 목록 — count.py 결과 JSON({"events":[{"t","line"}]}) 또는 CSV(route,t)
방향: route/line 이름의 '-' 앞부분 (seoul-2 → seoul)

지표 (방향별, 그리고 합계)
  사건 단위: 같은 방향 ±tol초 1:1 매칭(헝가리안) → 재현율·정밀도·F1·시각 오차
  구간 대수: 골든 구간을 W초 창으로 잘라 창마다 |예측 대수 − 정답 대수| 평균(MAE)과 부호 있는 평균(bias)
  총 대수 오차: (예측 합 − 정답 합) / 정답 합

  python3 bench/score.py --gt eval/gt/x.csv --pred run.json
  python3 bench/score.py --dataset bench/dataset.json --method yolo11s-bytetrack-v0   # bench/preds/<method>/<clip>.json|csv
"""
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
direction = lambda r: str(r).split("-")[0]


def read_gt(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    ranges = None
    for l in lines:
        if l.startswith("# reviewed_ranges:"):
            ranges = json.loads(l.split(":", 1)[1])
    rows = list(csv.DictReader(l for l in lines if l and not l.startswith("#")))
    ev = [(direction(r["route"]), float(r["t_gate_sec"])) for r in rows]
    return ev, ranges


def read_pred(path):
    p = Path(path)
    if p.suffix == ".json":
        d = json.loads(p.read_text(encoding="utf-8"))
        return [(direction(e["line"]), float(e["t"])) for e in d["events"]], d
    rows = list(csv.DictReader(l for l in p.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")))
    return [(direction(r["route"]), float(r.get("t") or r["t_gate_sec"])) for r in rows], {}


def in_ranges(t, ranges):
    return ranges is None or any(a <= t <= b for a, b in ranges)


def match(g, p, tol):
    if not len(g) or not len(p):
        return []
    C = np.abs(np.array(g)[:, None] - np.array(p)[None, :])
    ri, ci = linear_sum_assignment(np.where(C <= tol, C, 1e6))
    return [(i, j, C[i, j]) for i, j in zip(ri, ci) if C[i, j] <= tol]


def score_clip(gt_path, pred_path, tol=1.0, windows=(10, 60), duration=None):
    gt, ranges = read_gt(gt_path)
    pred, meta = read_pred(pred_path)
    if ranges is None:
        dur = duration or meta.get("video_seconds") or max([t for _, t in gt + pred] + [0]) + 0.01
        ranges = [[0.0, float(dur)]]
    gt = [(d, t) for d, t in gt if in_ranges(t, ranges)]
    pred = [(d, t) for d, t in pred if in_ranges(t, ranges)]
    out = {"gold_seconds": round(sum(b - a for a, b in ranges), 1), "dirs": {}}
    for d in sorted({x for x, _ in gt} | {x for x, _ in pred}):
        g = sorted(t for x, t in gt if x == d); p = sorted(t for x, t in pred if x == d)
        m = match(g, p, tol); tp = len(m)
        r = {"gt": len(g), "pred": len(p), "tp": tp, "fn": len(g) - tp, "fp": len(p) - tp,
             "recall": tp / len(g) if g else None, "precision": tp / len(p) if p else None,
             "dt_mean": float(np.mean([c for *_, c in m])) if m else None,
             "count_err": (len(p) - len(g)) / len(g) if g else None}
        r["f1"] = 2 * r["recall"] * r["precision"] / (r["recall"] + r["precision"]) if r["recall"] and r["precision"] else 0.0
        for W in windows:
            errs = []
            for a, b in ranges:
                k = int((b - a) // W)
                for i in range(k):
                    lo, hi = a + i * W, a + (i + 1) * W
                    errs.append(sum(lo <= t < hi for t in p) - sum(lo <= t < hi for t in g))
            r[f"win{W}_mae"] = float(np.mean(np.abs(errs))) if errs else None
            r[f"win{W}_bias"] = float(np.mean(errs)) if errs else None
            r[f"win{W}_n"] = len(errs)
        out["dirs"][d] = r
    out["infer_ms"] = meta.get("infer_ms_avg")
    return out


def total(rs):
    """방향·클립을 합친 사건 단위 지표."""
    gt = sum(r["gt"] for r in rs); pred = sum(r["pred"] for r in rs); tp = sum(r["tp"] for r in rs)
    rec = tp / gt if gt else None; pre = tp / pred if pred else None
    return {"gt": gt, "pred": pred, "tp": tp, "fn": gt - tp, "fp": pred - tp, "recall": rec, "precision": pre,
            "f1": 2 * rec * pre / (rec + pre) if rec and pre else 0.0, "count_err": (pred - gt) / gt if gt else None}


def fmt(x, pct=True):
    return "—" if x is None else (f"{x * 100:.1f}" if pct else f"{x:.2f}")


def table(title, rows):
    print(f"\n{title}")
    print(f"{'':24} {'정답':>5} {'예측':>5} {'놓침':>5} {'헛셈':>5} {'재현율':>7} {'정밀도':>7} {'F1':>6} {'대수오차%':>9} {'10초MAE':>8}")
    for name, r in rows:
        print(f"{name:24} {r['gt']:>5} {r['pred']:>5} {r['fn']:>5} {r['fp']:>5} {fmt(r['recall']):>7} {fmt(r['precision']):>7} {fmt(r['f1']):>6} {fmt(r['count_err']):>9} {fmt(r.get('win10_mae'), False):>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt"); ap.add_argument("--pred")
    ap.add_argument("--dataset"); ap.add_argument("--method")
    ap.add_argument("--split", default="", help="dev 또는 test만")
    ap.add_argument("--tol", type=float, default=None)
    ap.add_argument("--json", default="", help="결과 저장 경로")
    a = ap.parse_args()
    if a.gt:
        s = score_clip(a.gt, a.pred, a.tol or 1.0)
        table(f"{Path(a.pred).name} vs {Path(a.gt).name} (골든 {s['gold_seconds']}초)", list(s["dirs"].items()) + [("합계", total(list(s["dirs"].values())))])
        if a.json: Path(a.json).write_text(json.dumps(s, ensure_ascii=False, indent=1))
        return
    ds = json.loads(Path(a.dataset).read_text(encoding="utf-8")); tol = a.tol or ds.get("tolerance_sec", 1.0)
    pdir = ROOT / "bench" / "preds" / a.method
    per, groups, skipped = {}, defaultdict(list), []
    for c in ds["clips"]:
        if a.split and c["split"] != a.split: continue
        pp = next((pdir / f"{c['id']}{ext}" for ext in (".json", ".csv") if (pdir / f"{c['id']}{ext}").exists()), None)
        if not c.get("gt") or not pp:
            skipped.append(c["id"] + (" (정답 없음)" if not c.get("gt") else " (예측 없음)")); continue
        s = score_clip(ROOT / c["gt"], pp, tol); per[c["id"]] = s
        for r in s["dirs"].values():
            for key in (f"split={c['split']}", f"light={c['light']}", f"density={c['density']}", "전체"):
                groups[key].append(r)
    table(f"방법: {a.method} · 클립별(방향 합계)", [(k, total(list(v["dirs"].values()))) for k, v in per.items()])
    table("조건별", [(k, total(v)) for k, v in sorted(groups.items())])
    if skipped: print("\n건너뜀:", ", ".join(skipped))
    if a.json: Path(a.json).write_text(json.dumps({"method": a.method, "clips": per, "groups": {k: total(v) for k, v in groups.items()}}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
