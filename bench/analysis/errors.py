"""남은 오류를 사례별로 분류한다.
놓침(FN): 검출 없음 / 저신뢰 검출만 / 검출은 있는데 셈 안 됨 / 시각 어긋남(1~2.5초 안 예측 남음)
헛셈(FP): 시각 어긋남 / 중복(같은 정답 근처 두 번) / 반대 방향 정답 근처 / 근거 없음
  python3 bench/analysis/errors.py --method A7-rfdetrS-c0.25 --cache-tag rfdetrS [--clip ID] [--list]
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from score import read_gt, read_pred, in_ranges, match, direction, ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True); ap.add_argument("--cache-tag", required=True)
ap.add_argument("--clip"); ap.add_argument("--conf", type=float, default=0.25); ap.add_argument("--list", action="store_true")
ap.add_argument("--json")
a = ap.parse_args()
ds = json.loads((ROOT / "bench/dataset.json").read_text())
report = {}
for c in ds["clips"]:
    if c["split"] != "dev" or not c["gt"] or (a.clip and c["id"] != a.clip): continue
    rows = [r for r in csv.DictReader(l for l in (ROOT / c["gt"]).read_text().splitlines() if l and not l.startswith("#"))]
    _, ranges = read_gt(ROOT / c["gt"])
    pred, meta = read_pred(ROOT / f"bench/preds/{a.method}/{c['id']}.json")
    pev = meta["events"]
    z = np.load(ROOT / f"bench/cache/{c['id']}_{a.cache_tag}.npz"); D = z["det"]; fps = float(z["fps"])
    byf = {}
    for r in D: byf.setdefault(int(r[0]), []).append(r)
    def det_at(t, x, y):
        f = int(round(t * fps)); best = 0.0
        for ff in range(f - 3, f + 4):
            for r in byf.get(ff, []):
                x1, y1, x2, y2, cf = r[1:6]
                if x1 - 6 <= x <= x2 + 6 and y1 - 12 <= y <= y2 + 12: best = max(best, float(cf))
        return best
    cats = {}; items = []
    gts = [r for r in rows if in_ranges(float(r["t_gate_sec"]), ranges)]
    pv = [e for e in pev if in_ranges(float(e["t"]), ranges)]
    for d in sorted({direction(r["route"]) for r in gts} | {direction(e["line"]) for e in pv}):
        G = [r for r in gts if direction(r["route"]) == d]; P = [e for e in pv if direction(e["line"]) == d]
        gt_t = [float(r["t_gate_sec"]) for r in G]; p_t = [float(e["t"]) for e in P]
        m = match(gt_t, p_t, 1.0); mg = {i for i, _, _ in m}; mp = {j for _, j, _ in m}
        for i, r in enumerate(G):
            if i in mg: continue
            t = gt_t[i]; x, y = float(r["x"]), float(r["y"])
            un = [p_t[j] for j in range(len(P)) if j not in mp and abs(p_t[j] - t) <= 2.5]
            cf = det_at(t, x, y)
            k = "FN:시각어긋남" if un else ("FN:검출없음" if cf == 0 else ("FN:저신뢰검출만" if cf < a.conf else "FN:검출있는데안셈"))
            cats[k] = cats.get(k, 0) + 1
            items.append((k, d, round(t, 2), round(float(r["gate_pos"]), 2), int(x), int(y), round(cf, 2)))
        for j, e in enumerate(P):
            if j in mp: continue
            t = p_t[j]
            near_un = [gt_t[i] for i in range(len(G)) if i not in mg and abs(gt_t[i] - t) <= 2.5]
            near_m = [gt_t[i] for i in range(len(G)) if i in mg and abs(gt_t[i] - t) <= 1.0]
            other = [r for r in gts if direction(r["route"]) != d and abs(float(r["t_gate_sec"]) - t) <= 1.0]
            k = "FP:시각어긋남" if near_un else ("FP:중복" if near_m else ("FP:반대방향근처" if other else "FP:근거없음"))
            cats[k] = cats.get(k, 0) + 1
            pt = e.get("pt") or [0, 0]
            items.append((k, d, round(t, 2), None, int(pt[0]), int(pt[1]), e.get("cls")))
    report[c["id"]] = {"cats": cats, "items": items}
    print(f"\n== {c['id']}  " + "  ".join(f"{k} {v}" for k, v in sorted(cats.items())))
    if a.list:
        for it in sorted(items, key=lambda x: (x[0], x[2])): print("  ", *it)
if a.json: Path(a.json).write_text(json.dumps(report, ensure_ascii=False, indent=1))
