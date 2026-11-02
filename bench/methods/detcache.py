"""검출 결과 캐시: 프레임마다 차량 박스(xyxy, conf, cls)를 낮은 기준(0.05)으로 저장.
추적·세기 실험은 이 캐시로 반복해 검출을 다시 돌리지 않는다.

  python3 bench/methods/detcache.py --video x.mp4 --out bench/cache/x_yolo11s_960.npz
  옵션: --backend yolo|rfdetr, --model (yolo 가중치 또는 rfdetr 크기 small|medium|nano)
        --roi counter/lines/x.json  : 기준선 주변만 잘라 검출(작은 차가 커짐), 박스는 전체 화면 좌표로 되돌림
        --max-seconds S            : 앞 S초만
캐시의 cls는 COCO 80클래스 번호로 통일(2 car, 3 motorcycle, 5 bus, 7 truck, 0 person). --target person 이면 사람만.
"""
import argparse, json, time
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--backend", default="yolo"); ap.add_argument("--model", default="yolo11s.pt")
ap.add_argument("--imgsz", type=int, default=960); ap.add_argument("--conf", type=float, default=0.05)
ap.add_argument("--roi", default=""); ap.add_argument("--roi-margin", type=int, default=60)
ap.add_argument("--max-seconds", type=float, default=0)
ap.add_argument("--stride", type=int, default=1, help="N프레임마다 한 번만 검출(나머지는 건너뜀, 프레임 번호는 그대로)")
ap.add_argument("--from-seconds", type=float, default=0, help="이 시각 전 프레임은 검출하지 않음(프레임 번호는 그대로)")
ap.add_argument("--prompt", default="", help="gdino/owlv2: 셀 대상을 글자로(예: 'car. truck. bus.') — 대상별 사전학습 검출기 없이")
ap.add_argument("--target", default="vehicle", help="vehicle(2,3,5,7) | person(0) — 무엇을 셀지")
ap.add_argument("--weights", default="", help="미세조정한 RF-DETR 가중치(.pth). 주면 모든 검출을 차(cls 2)로 봄")
a = ap.parse_args()

VEH = {"vehicle": {2, 3, 5, 7}, "person": {0}, "all": {0, 2, 3, 5, 7}}[a.target]
if a.backend == "yolo":
    from ultralytics import YOLO
    m = YOLO(a.model)
    def detect(img):
        r = m.predict(img, imgsz=a.imgsz, conf=a.conf, classes=sorted(VEH), verbose=False)[0]
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
elif a.backend in ("gdino", "owlv2"):   # 2단계: 글자로 지정하는 검출기(대상 클래스 학습 없음). 모든 검출을 cls 2로 저장
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, Owlv2Processor, Owlv2ForObjectDetection
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    phrases = [t.strip() for t in a.prompt.split(".") if t.strip()]
    if a.backend == "gdino":
        mid = a.model if "/" in a.model else "IDEA-Research/grounding-dino-tiny"
        proc = AutoProcessor.from_pretrained(mid); net = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(dev).eval()
        def detect(img):
            im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); inp = proc(images=im, text=". ".join(phrases) + ".", return_tensors="pt").to(dev)
            with torch.no_grad(): o = net(**inp)
            r = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=a.conf, text_threshold=a.conf, target_sizes=[im.size[::-1]])[0]
            return r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), np.full(len(r["scores"]), 2)
    else:
        mid = a.model if "/" in a.model else "google/owlv2-base-patch16-ensemble"
        proc = Owlv2Processor.from_pretrained(mid); net = Owlv2ForObjectDetection.from_pretrained(mid).to(dev).eval()
        def detect(img):
            im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); inp = proc(text=[[f"a photo of a {t}" for t in phrases]], images=im, return_tensors="pt").to(dev)
            with torch.no_grad(): o = net(**inp)
            S = max(im.size); r = proc.post_process_grounded_object_detection(o, threshold=a.conf, target_sizes=torch.tensor([[S, S]], device=dev))[0]
            return r["boxes"].cpu().numpy(), r["scores"].cpu().numpy(), np.full(len(r["scores"]), 2)
else:
    import rfdetr
    from PIL import Image
    M = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}[a.model]
    m = M(pretrain_weights=a.weights) if a.weights else M()
    try: m.optimize_for_inference()
    except Exception as e: print("optimize 실패:", e)
    COCO91_TO_80 = {"vehicle": {3: 2, 4: 3, 6: 5, 8: 7}, "person": {1: 0}, "all": {1: 0, 3: 2, 4: 3, 6: 5, 8: 7}}[a.target]  # rfdetr는 COCO 원래 번호(91)를 씀
    def detect(img):
        d = m.predict(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), threshold=a.conf)
        if a.weights: return d.xyxy, d.confidence, np.full(len(d), 2)   # 미세조정 모델은 vehicle 한 클래스
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
    if f < a.from_seconds * fps or f % a.stride:
        if not cap.grab(): break
        f += 1; continue
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
