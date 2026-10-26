"""tally-cam 차량 카운터 (프로토타입).

YOLO 검출 + ByteTrack 추적으로, 차로별 카운팅 선분을 넘는 차량을
선분(차로)별로 센다. 소스는 파일 또는 HLS URL.

  python3 count.py --source clip.mp4 --model yolo11n.pt --stride 2 --out run.json
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def side(p1, p2, q):
    """선분 p1→p2 기준 점 q가 어느 쪽인지 (+1 / -1)."""
    return 1 if (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0]) > 0 else -1


def crosses(a, b, p1, p2):
    """이동 선분 a→b가 카운팅 선분 p1-p2를 가로지르는지."""
    return side(p1, p2, a) != side(p1, p2, b) and side(a, b, p1) != side(a, b, p2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=1, help="N프레임마다 1프레임 처리")
    p.add_argument("--lines", default="counter/lines/dongtan-jct.json", help="차로별 카운팅 선분 정의 JSON")
    p.add_argument("--max-seconds", type=float, default=0, help="0이면 끝까지")
    p.add_argument("--out", default="run.json")
    p.add_argument("--annotate", default="", help="주석 영상 저장 경로 (선택)")
    return p.parse_args()


def main():
    a = parse_args()
    model = YOLO(a.model)
    cap = cv2.VideoCapture(a.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    lines = json.loads(Path(a.lines).read_text(encoding="utf-8"))

    writer = None
    if a.annotate:
        writer = cv2.VideoWriter(a.annotate, cv2.VideoWriter_fourcc(*"mp4v"), fps / a.stride, (w, h))

    last_pt = {}                     # track id -> 마지막 중심점
    counted = set()                  # (line, track id)
    counts = defaultdict(int)        # line name -> count
    events = []
    infer_ms = []
    frame_idx = processed = 0
    t_wall = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_video = frame_idx / fps
        if a.max_seconds and t_video > a.max_seconds:
            break
        if frame_idx % a.stride:
            frame_idx += 1
            continue

        t0 = time.perf_counter()
        res = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=VEHICLE_CLASSES,
                          imgsz=a.imgsz, conf=a.conf, verbose=False)[0]
        infer_ms.append((time.perf_counter() - t0) * 1000)
        processed += 1

        boxes = res.boxes
        if boxes is not None and boxes.id is not None:
            for (x1, y1, x2, y2), tid, cls in zip(boxes.xyxy.tolist(), boxes.id.int().tolist(), boxes.cls.int().tolist()):
                cur = ((x1 + x2) / 2, (y1 + y2) / 2)
                prev = last_pt.get(tid)
                last_pt[tid] = cur
                if prev is None:
                    continue
                for ln in lines:
                    key = (ln["name"], tid)
                    if key in counted or not crosses(prev, cur, ln["p1"], ln["p2"]):
                        continue
                    counted.add(key)
                    counts[ln["name"]] += 1
                    events.append({"t": round(t_video, 2), "id": tid, "line": ln["name"],
                                   "side": side(ln["p1"], ln["p2"], cur),
                                   "cls": CLASS_NAMES.get(cls, str(cls))})

        if writer is not None:
            vis = res.plot(line_width=1, font_size=8)
            for i, ln in enumerate(lines):
                cv2.line(vis, tuple(ln["p1"]), tuple(ln["p2"]), (0, 200, 255), 2)
                cv2.putText(vis, f"{ln['name']} {counts[ln['name']]}", (10, 30 + 28 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            writer.write(vis)
        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    wall = time.time() - t_wall
    summary = {
        "source": Path(a.source).name if "://" not in a.source else "stream",
        "model": a.model, "imgsz": a.imgsz, "stride": a.stride, "conf": a.conf,
        "video_seconds": round(frame_idx / fps, 1),
        "processed_frames": processed,
        "counts": dict(counts), "total": sum(counts.values()),
        "infer_ms_avg": round(sum(infer_ms) / max(len(infer_ms), 1), 1),
        "realtime_factor": round((frame_idx / fps) / wall, 2) if wall else None,
        "events": events,
    }
    Path(a.out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
