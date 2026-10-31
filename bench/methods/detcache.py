"""검출 결과 캐시: 프레임마다 차량 박스(xyxy, conf, cls)를 낮은 기준(0.05)으로 저장.
추적·세기 실험은 이 캐시로 반복해 검출을 다시 돌리지 않는다.

  python3 bench/methods/detcache.py --video x.mp4 --out bench/cache/x_yolo11s_960.npz
  옵션: --backend yolo|rfdetr, --model (yolo 가중치 또는 rfdetr 크기 small|medium|nano)
        --roi counter/lines/x.json  : 기준선 주변만 잘라 검출(작은 차가 커짐), 박스는 전체 화면 좌표로 되돌림
        --max-seconds S            : 앞 S초만
캐시의 cls는 COCO 80클래스 번호로 통일(2 car, 3 motorcycle, 5 bus, 7 truck).
"""
import argparse, json, time
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--backend", default="yolo"); ap.add_argument("--model", default="yolo11s.pt")
ap.add_argument("--imgsz", type=int, default=960); ap.add_argument("--conf", type=float, default=0.05)
ap.add_argument("--roi", default=""); ap.add_argument("--roi-margin", type=int, default=60)
ap.add_argument("--max-seconds", type=float, default=0)
a = ap.parse_args()

VEH = {2, 3, 5, 7}
if a.backend == "yolo":
    from ultralytics import YOLO
    m = YOLO(a.model)
    def detect(img):
        r = m.predict(img, imgsz=a.imgsz, conf=a.conf, classes=sorted(VEH), verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
else:
    import rfdetr
    from PIL import Image
    m = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}[a.model]()
    try: m.optimize_for_inference()
    except Exception as e: print("optimize 실패:", e)
    COCO91_TO_80 = {3: 2, 4: 3, 6: 5, 8: 7}  # rfdetr는 COCO 원래 번호(91)를 씀
    def detect(img):
        d = m.predict(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), threshold=a.conf)
        keep = np.array([c in COCO91_TO_80 for c in d.class_id], bool) if len(d) else np.zeros(0, bool)
        cls = np.array([COCO91_TO_80.get(int(c), -1) for c in d.class_id])[keep] if len(d) else np.zeros(0)
        return d.xyxy[keep], d.confidence[keep], cls

roi = None
cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(3)), int(cap.get(4))
if a.roi:
    pts = np.array([p for L in json.load(open(a.roi, encoding="utf-8")) for p in (L["p1"], L["p2"])], float)
    x0, y0 = np.maximum(pts.min(0) - a.roi_margin, 0).astype(int); x1, y1 = np.minimum(pts.max(0) + a.roi_margin, [W, H]).astype(int)
    roi = (x0, y0, x1, y1); print("ROI", roi)
rows, ms, f = [], [], 0
while True:
    ok, img = cap.read()
    if not ok or (a.max_seconds and f >= a.max_seconds * fps): break
    t0 = time.perf_counter()
    if roi:
        xy, cf, cl = detect(np.ascontiguousarray(img[roi[1]:roi[3], roi[0]:roi[2]]))
        if len(xy): xy = xy + np.array([roi[0], roi[1], roi[0], roi[1]])
    else:
        xy, cf, cl = detect(img)
    ms.append((time.perf_counter() - t0) * 1000)
    for (x1_, y1_, x2_, y2_), c, k in zip(xy.tolist(), cf.tolist(), cl.tolist()):
        rows.append((f, x1_, y1_, x2_, y2_, c, k))
    f += 1
np.savez_compressed(a.out, det=np.array(rows, np.float32).reshape(-1, 7), frames=f, fps=fps, det_ms=np.array(ms, np.float32),
                    model=f"{a.backend}:{a.model}", imgsz=a.imgsz, h=H)
print(f"{f} frames, {len(rows)} boxes, {np.mean(ms):.0f} ms/frame")
