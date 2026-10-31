"""트랙 B: 시공간 띠에서 직접 세기(규칙 기반). 검출기·추적기 없음.

띠(가로=시간, 세로=기준선 위 위치)에서 배경과 다른 밝기 덩어리를 찾아 하나를 차 한 대로 센다.
  배경: 위치마다 시간 중앙값 (--bg median) 또는 하위 분위수 (--bg pQ, 정체 대비)
  덩어리: 배경 차이 > thresh → 가우시안 → 닫힘(세로 kh × 가로 kw) → 연결 성분, 넓이 ≥ area
  사건 시각: 덩어리의 시작(--when start) / 중심(center) / 끝(end)

  python3 bench/methods/strip_count.py --video v.mp4 --lines l.json --out preds/B0/clip.json [옵션]
"""
import argparse, json, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from strips import build


def blobs(strip, fps, thresh=40, kh=5, kw=9, area=60, bg="median", when="center", min_dur=0.0):
    img = strip.astype(np.float32)
    if bg == "median":
        base = np.median(img, axis=1, keepdims=True)
    else:
        base = np.percentile(img, float(bg[1:]), axis=1, keepdims=True)
    d = np.abs(img - base).max(axis=2)
    m = (cv2.GaussianBlur(d, (5, 5), 0) > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((kh, kw), np.uint8))
    n, lab, st, cent = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < area or w / fps < min_dur: continue
        t = {"start": x, "center": cent[i][0], "end": x + w - 1}[when] / fps
        out.append((round(float(t), 3), round(float(cent[i][1]) / img.shape[0], 3), int(a)))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--lines", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=""); ap.add_argument("--thresh", type=float, default=40)
    ap.add_argument("--kh", type=int, default=5); ap.add_argument("--kw", type=int, default=9); ap.add_argument("--area", type=int, default=60)
    ap.add_argument("--bg", default="median"); ap.add_argument("--when", default="center"); ap.add_argument("--min-dur", type=float, default=0.0)
    a = ap.parse_args()
    t0 = time.perf_counter()
    strips, fps, lines = build(a.video, a.lines, a.cache or f"bench/cache/strips_{Path(a.out).stem}.npz")
    ev = []
    for L in lines:
        for t, y, ar in blobs(strips[L["name"]], fps, a.thresh, a.kh, a.kw, a.area, a.bg, a.when, a.min_dur):
            ev.append({"t": t, "line": L["name"], "y": y, "area": ar})
    F = next(iter(strips.values())).shape[1]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"method": vars(a), "video_seconds": round(F / fps, 2), "events": sorted(ev, key=lambda e: e["t"])}))
    print(len(ev), "events")


if __name__ == "__main__":
    main()
