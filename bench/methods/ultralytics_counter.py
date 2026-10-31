"""오픈소스 비교 대상: Ultralytics solutions.ObjectCounter (기본 설정 그대로: 모델 기본 imgsz·conf, 기본 추적기).
기준선마다 카운터 하나. 카운터 수가 늘어난 프레임 시각을 사건으로 기록.

  python3 bench/methods/ultralytics_counter.py --video x.mp4 --lines counter/lines/x.json --model yolo11s.pt --out preds/ultralytics-counter/clip.json
"""
import argparse, json, time
from pathlib import Path
import cv2
from ultralytics import solutions

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--lines", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", default="yolo11s.pt")
ap.add_argument("--tuned", action="store_true", help="imgsz 960 · conf 0.10 · bytetrack (우리 기존 카운터와 같은 조건)")
a = ap.parse_args()
lines = json.loads(Path(a.lines).read_text(encoding="utf-8"))
counters = {L["name"]: solutions.ObjectCounter(model=a.model, region=[tuple(L["p1"]), tuple(L["p2"])], classes=[2, 3, 5, 7], show=False, verbose=False,
                                             **({"imgsz": 960, "conf": 0.10, "tracker": "bytetrack.yaml"} if a.tuned else {})) for L in lines}
cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS)
events, ms, f, prev = [], [], 0, {k: 0 for k in counters}
while True:
    ok, img = cap.read()
    if not ok: break
    t0 = time.perf_counter()
    for name, c in counters.items():
        c(img)
        n = c.in_count + c.out_count
        for _ in range(n - prev[name]): events.append({"t": round(f / fps, 3), "line": name})
        prev[name] = n
    ms.append((time.perf_counter() - t0) * 1000); f += 1
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({"method": "ultralytics solutions.ObjectCounter " + ("(imgsz 960, conf 0.10, bytetrack)" if a.tuned else "(defaults: imgsz 640, conf 0.25, tracktrack)"), "video_seconds": round(f / fps, 2),
                                   "infer_ms_avg": round(sum(ms) / max(len(ms), 1), 1), "events": events}))
print(len(events), "events", round(sum(ms) / max(len(ms), 1)), "ms/frame")
