"""뒤 절반만 채점(앞 절반은 라벨 예산·학습에 쓴 경우).
  python3 bench/analysis/score_half.py --dataset bench/dataset_mot.json --split mot --method X [--method Y ...]
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from score import score_clip, total

ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="bench/dataset.json"); ap.add_argument("--split", default="dev")
ap.add_argument("--method", action="append", required=True); ap.add_argument("--json"); a = ap.parse_args()
ds = json.loads((ROOT / a.dataset).read_text()); tmp = ROOT / "bench/cache/half_gt"; tmp.mkdir(parents=True, exist_ok=True); res = {}
for M in a.method:
    rs, per = [], {}
    for c in ds["clips"]:
        if c["split"] != a.split or not c["gt"]: continue
        p = ROOT / f"bench/preds/{M}/{c['id']}.json"
        if not p.exists(): continue
        dur = json.loads(p.read_text()).get("video_seconds")
        txt = [l for l in (ROOT / c["gt"]).read_text().splitlines() if not l.startswith("# reviewed_ranges")]
        g = tmp / f"{c['id']}.csv"; g.write_text("\n".join(txt) + f"\n# reviewed_ranges: [[{dur / 2},{dur}]]\n")
        r = list(score_clip(g, p)["dirs"].values()); rs += r; t = total(r); per[c["id"]] = round(t["f1"] * 100, 1)
    t = total(rs); res[M] = {"f1": round(t["f1"] * 100, 1), "recall": round((t["recall"] or 0) * 100, 1), "precision": round((t["precision"] or 0) * 100, 1), "gt": t["gt"], "per": per}
    print(f"{M:34} F1 {res[M]['f1']:5.1f}  재현율 {res[M]['recall']:5.1f}  정밀도 {res[M]['precision']:5.1f}  (정답 {t['gt']})  {per}")
if a.json: Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=1))
