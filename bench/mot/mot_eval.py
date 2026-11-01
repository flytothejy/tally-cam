"""MOT17 기준점 만들기: (1) 표준 추적 지표(MOTA·IDF1)로 우리 추적을 논문과 비교, (2) 검출 바꾸기로 손실 나누기.

검출 소스(모두 같은 ByteTrack·같은 세기 규칙):
  rfdetrS  : 우리 검출기(COCO 사전학습 RF-DETR small, 학습 없음)
  public   : MOT17 공식 FRCNN 공개 검출(det.txt) — 논문의 'public detection' 설정에 쓰는 것
  oracle   : 정답 상자(가시도 ≥ --vis) = 검출이 완벽할 때의 상한
  python3 bench/mot/mot_eval.py --root /home/claude/datasets/MOT17/train
"""
import argparse, csv, json, subprocess, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
np.asfarray = lambda x, dtype=float: np.asarray(x, dtype=dtype)   # motmetrics 1.4 + NumPy 2 호환
import motmetrics as mm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from score import score_clip, total

DISTRACT = {2, 7, 8, 12}   # 정지한 사람·반사·마네킹 등: 여기 겹친 예측은 오검출로 치지 않음(MOT 규칙)


def load_gt(d):
    G = defaultdict(list)
    for r in csv.reader(open(d / "gt/gt.txt")):
        G[int(r[0])].append((int(r[1]), *map(float, r[2:6]), int(r[6]), int(r[7]), float(r[8])))
    return G


def write_cache(path, rows, F, fps):
    np.savez_compressed(path, det=np.array(rows, np.float32).reshape(-1, 7), frames=F, fps=fps, det_ms=np.zeros(1, np.float32), model="ref", imgsz=0, h=0)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--vis", type=float, default=0.25)
    ap.add_argument("--src", default="rfdetrS,public,oracle"); a = ap.parse_args()
    ds = json.loads((ROOT / "bench/dataset_mot.json").read_text()); tmp = ROOT / "bench/cache/mot_ref"; tmp.mkdir(parents=True, exist_ok=True)
    out = {}
    for src in a.src.split(","):
        accs, names, cnt = [], [], []
        for c in ds["clips"]:
            d = Path(a.root) / f"{c['id']}-FRCNN"; G = load_gt(d); z = np.load(ROOT / f"bench/cache/{c['id']}_rfdetrS.npz"); F, fps = int(z["frames"]), float(z["fps"])
            if src == "rfdetrS": cache = ROOT / f"bench/cache/{c['id']}_rfdetrS.npz"
            else:
                cache = tmp / f"{c['id']}_{src}.npz"; rows = []
                if src == "public":
                    for r in csv.reader(open(d / "det/det.txt")):
                        f, x, y, w, h, s = int(r[0]) - 1, *map(float, r[2:7]); rows.append((f, x, y, x + w, y + h, min(max(s, 0.0), 1.0), 0))
                else:
                    for f, L in G.items():
                        for i, x, y, w, h, keep, cls, v in L:
                            if keep == 1 and cls == 1 and v >= a.vis: rows.append((f - 1, x, y, x + w, y + h, 0.99, 0))
                rows.sort(); write_cache(cache, rows, F, fps)
            pred = ROOT / f"bench/preds/MOT-ref-{src}/{c['id']}.json"; dump = tmp / f"{c['id']}_{src}_trk.npz"
            subprocess.run([sys.executable, str(ROOT / "bench/methods/track_count.py"), "--cache", str(cache), "--lines", str(ROOT / c["gates"]), "--out", str(pred),
                            "--agnostic", "--conf", "0.25" if src != "public" else "0.0", "--anchor", "bottom", "--dump", str(dump)], check=True, capture_output=True)
            cnt += list(score_clip(ROOT / c["gt"], pred)["dirs"].values())
            T = np.load(dump)["trk"]; byf = defaultdict(list)
            for r in T: byf[int(r[0]) + 1].append(r)
            acc = mm.MOTAccumulator(auto_id=False)
            for f in range(1, F + 1):
                gl = [g for g in G.get(f, []) if g[5] == 1 and g[6] == 1]; dl = [g for g in G.get(f, []) if g[6] in DISTRACT]
                pr = [r for r in byf.get(f, []) if not any(iou(r[2:6], (g[1], g[2], g[1] + g[3], g[2] + g[4])) > 0.5 for g in dl)]
                gb = np.array([[g[1], g[2], g[3], g[4]] for g in gl]).reshape(-1, 4); pb = np.array([[r[2], r[3], r[4] - r[2], r[5] - r[3]] for r in pr]).reshape(-1, 4)
                acc.update([g[0] for g in gl], [int(r[1]) for r in pr], mm.distances.iou_matrix(gb, pb, max_iou=0.5), frameid=f)
            accs.append(acc); names.append(c["id"])
        mh = mm.metrics.create(); s = mh.compute_many(accs, names=names, metrics=["mota", "idf1", "num_switches", "recall", "precision"], generate_overall=True)
        static = [i for i, c in enumerate(ds["clips"]) if c["view"] == "static"]
        t = total(cnt); out[src] = {"MOTA": round(float(s.loc["OVERALL", "mota"]) * 100, 1), "IDF1": round(float(s.loc["OVERALL", "idf1"]) * 100, 1),
                                    "IDsw": int(s.loc["OVERALL", "num_switches"]), "det_recall": round(float(s.loc["OVERALL", "recall"]) * 100, 1),
                                    "count_F1": round(t["f1"] * 100, 1), "count_recall": round(t["recall"] * 100, 1), "count_precision": round(t["precision"] * 100, 1)}
        print(src, out[src], flush=True)
        print(s.round(3).to_string(), flush=True)
    (ROOT / "bench/RESULTS_mot_ref.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
