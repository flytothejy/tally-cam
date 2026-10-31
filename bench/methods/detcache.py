"""검출 결과 캐시: 프레임마다 차량 박스(xyxy, conf, cls)를 낮은 기준(0.05)으로 저장.
추적·세기 실험은 이 캐시로 반복해 검출을 다시 돌리지 않는다.

  python3 bench/methods/detcache.py --video samples/x.mp4 --out bench/cache/x_yolo11s_960.npz
"""
import argparse, time
import cv2, numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--model", default="yolo11s.pt"); ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--conf", type=float, default=0.05)
a = ap.parse_args()
m = YOLO(a.model); cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS)
rows, ms = [], []
f = 0
while True:
    ok, img = cap.read()
    if not ok: break
    t0 = time.perf_counter()
    r = m.predict(img, imgsz=a.imgsz, conf=a.conf, classes=[2, 3, 5, 7], verbose=False)[0]
    ms.append((time.perf_counter() - t0) * 1000)
    for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
        rows.append((f, x1, y1, x2, y2, c, k))
    f += 1
np.savez_compressed(a.out, det=np.array(rows, np.float32), frames=f, fps=fps, det_ms=np.array(ms, np.float32),
                    model=a.model, imgsz=a.imgsz, h=img.shape[0] if f else 0)
print(f"{f} frames, {len(rows)} boxes, {np.mean(ms):.0f} ms/frame")
