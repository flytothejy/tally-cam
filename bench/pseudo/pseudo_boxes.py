"""사람 점 라벨 → 검출기 학습용 의사 상자.

1) 검출 캐시(RF-DETR, conf≥0.05)를 낮은 기준으로 오프라인 추적(ByteTrack). 트랙 상자는 칼만 추정값이 아니라 원래 검출 상자를 씀(idx로 연결).
2) 점마다 그 순간 점을 덮는 트랙을 찾음(바닥 중앙이 가장 가까운 것) = 사람이 확인한 차.
3) 그 트랙을 앞뒤로 따라가며 상자를 모음. 검출이 끊긴 짧은 구간(≤ --max-gap 프레임)은 앞뒤 상자를 선형 보간 → '어려운 예제'.
   안전장치: 크기 급변(면적 비 > --max-area-ratio)·위치 급변(프레임당 중심 이동 > 상자 대각선 × --max-jump)·긴 끊김이 나오면 거기서 자름(점 쪽만 남김).
4) 같은 프레임의 다른 차: 신뢰도 ≥ --hi 검출은 상자로 채우고, --lo~--hi 사이는 '무시 영역'(학습 이미지에서 흐리게 지움).
5) 프레임 고르기: 점에 묶인 상자가 있는 프레임을 --stride 간격으로, 보간 상자가 있는 프레임은 --hard-stride 간격으로.

  python3 bench/pseudo/pseudo_boxes.py --clip songpa-am-01 --out bench/pseudo/out            # 통계 + JSON
  python3 bench/pseudo/pseudo_boxes.py --all --out bench/pseudo/out --export data/pseudo_coco   # 이미지 + COCO까지
"""
import argparse, csv, json, math, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch, torchvision

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "bench/methods"))
from ultralytics.trackers.byte_tracker import BYTETracker
from track_count import Dets

VIDEO_DIRS = [ROOT, Path("/mnt/user-data/uploads/Project/tally-cam")]


def find_video(rel):
    for d in VIDEO_DIRS:
        if (d / rel).exists(): return d / rel
    raise FileNotFoundError(rel)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1e-6)


def track_offline(det, F, high=0.25, low=0.05):
    """프레임별 [(tid, 원래 검출 행)] — 검출이 붙은 프레임만."""
    args = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=high, track_low_thresh=low, new_track_thresh=high,
                           track_buffer=30, match_thresh=0.8, fuse_score=True)
    trk = BYTETracker(args); starts = np.searchsorted(det[:, 0], np.arange(F + 1)); tracks = {}
    for f in range(F):
        d = det[starts[f]:starts[f + 1]]
        if len(d):
            keep = torchvision.ops.nms(torch.tensor(d[:, 1:5]), torch.tensor(d[:, 5]), 0.6).numpy(); d = d[np.sort(keep)]
        out = trk.update(Dets(d[:, 1:5].astype(np.float32), d[:, 5].astype(np.float32), d[:, 6].astype(np.float32)), None)
        for r in (out if len(out) else []):
            tid, idx = int(r[4]), int(r[7])
            if 0 <= idx < len(d): tracks.setdefault(tid, {})[f] = d[idx]
    return tracks


def follow(tr, f_anchor, a):
    """점 프레임에서 앞뒤로 상자 모으기. 반환: {f: (box, 'obs'|'interp')}, 자른 이유들."""
    fs = sorted(tr); out = {}; cuts = []
    i0 = min(range(len(fs)), key=lambda i: abs(fs[i] - f_anchor))
    out[fs[i0]] = (tr[fs[i0]][1:5], "obs")
    for step in (1, -1):
        i = i0
        while 0 <= i + step < len(fs):
            fa, fb = fs[i], fs[i + step]; A, B = tr[fa][1:5], tr[fb][1:5]; gap = abs(fb - fa)
            recent = [out[k][0] for k in sorted(out, key=lambda k: abs(k - fa))[:5] if out[k][1] == "obs"]
            areaA = float(np.median([(r[2] - r[0]) * (r[3] - r[1]) for r in recent])); areaB = (B[2] - B[0]) * (B[3] - B[1])
            diag = math.hypot(A[2] - A[0], A[3] - A[1]); jump = math.dist(((A[0] + A[2]) / 2, (A[1] + A[3]) / 2), ((B[0] + B[2]) / 2, (B[1] + B[3]) / 2)) / gap
            if gap - 1 > a.max_gap: cuts.append("gap"); break
            if max(areaA, areaB) / max(min(areaA, areaB), 1) > a.max_area_ratio * (1 + 0.02 * gap): cuts.append("area"); break
            if jump > a.max_jump * diag: cuts.append("jump"); break
            for k in range(1, gap):   # 끊긴 프레임 보간
                w = k / gap; out[fa + step * k] = (A * (1 - w) + B * w, "interp")
            out[fb] = (B, "obs"); i += step
    return out, cuts


def process(c, a):
    z = np.load(ROOT / f"bench/cache/{c['id']}_{a.cache_tag}.npz"); det = z["det"]; F = int(z["frames"]); fps = float(z["fps"])
    tracks = track_offline(det, F)
    byf = {}
    for tid, tr in tracks.items():
        for f, row in tr.items(): byf.setdefault(f, []).append((tid, row))
    rows = [r for r in csv.DictReader(l for l in (ROOT / c["gt"]).read_text().splitlines() if l and not l.startswith("#"))]
    anchored = {}; stats = {"labels": len(rows), "no_track": 0, "dup_track": 0, "cuts": {}}
    used = set()
    for r in rows:
        f = int(r["frame"]); x, y = float(r["x"]), float(r["y"]); best = None
        for ff in (f, f - 1, f + 1, f - 2, f + 2):
            for tid, row in byf.get(ff, []):
                x1, y1, x2, y2 = row[1:5]; h = y2 - y1
                if x1 - 6 <= x <= x2 + 6 and y1 - 6 <= y <= y2 + 0.3 * h + 6:
                    dd = abs((x1 + x2) / 2 - x) + abs(y2 - y)
                    if best is None or dd < best[0]: best = (dd, tid, ff)
            if best: break
        if not best: stats["no_track"] += 1; continue
        if best[1] in used: stats["dup_track"] += 1; continue   # 한 트랙에 점 두 개 → 하나만
        used.add(best[1])
        boxes, cuts = follow(tracks[best[1]], best[2], a)
        for k in cuts: stats["cuts"][k] = stats["cuts"].get(k, 0) + 1
        anchored[best[1]] = {"event": r["event_id"], "boxes": boxes}
    # 프레임 고르기
    per_f = {}
    for tid, A in anchored.items():
        for f, (b, kind) in A["boxes"].items(): per_f.setdefault(f, []).append((tid, b, kind))
    pick = sorted({f for f in per_f if f % a.stride == 0} | {f for f, L in per_f.items() if any(k == "interp" for *_, k in L) and f % a.hard_stride == 0})
    det_by_f = {}; s0 = np.searchsorted(det[:, 0], np.arange(F + 1))
    images = []
    for f in pick:
        anc = [{"box": [round(float(v), 1) for v in b], "kind": k, "track": int(t)} for t, b, k in per_f[f]]
        others, ignore = [], []
        dfr = det[s0[f]:s0[f + 1]]
        if len(dfr): dfr = dfr[np.sort(torchvision.ops.nms(torch.tensor(dfr[:, 1:5]), torch.tensor(dfr[:, 5]), 0.5).numpy())]
        for row in dfr:
            b = row[1:5]
            if any(iou(b, x["box"]) > 0.3 for x in anc) or any(iou(b, x) > 0.5 for x in others): continue
            (others if row[5] >= a.hi else ignore if row[5] >= a.lo else []).append([round(float(v), 1) for v in b])
        images.append({"frame": f, "t": round(f / fps, 3), "anchored": anc, "others": others, "ignore": ignore})
    nb = sum(len(i["anchored"]) for i in images); ni = sum(sum(x["kind"] == "interp" for x in i["anchored"]) for i in images)
    stats.update({"tracks": len(anchored), "frames": len(images), "anchored_boxes": nb, "interp_boxes": ni,
                  "other_boxes": sum(len(i["others"]) for i in images), "ignore_boxes": sum(len(i["ignore"]) for i in images),
                  "all_interp_frames": sum(1 for A in anchored.values() for _, k in A["boxes"].values() if k == "interp")})
    return {"clip": c["id"], "video": c["video"], "fps": fps, "split": c["split"], "camera": c["camera"], "stats": stats, "images": images}


def export(res, outdir, quality=92):
    """프레임 JPEG(무시 영역은 흐리게) + COCO json. 클래스는 vehicle 하나."""
    import cv2
    outdir = Path(outdir); img_dir = outdir / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(find_video(res["video"]))); want = {i["frame"]: i for i in res["images"]}; f = 0; coco_imgs, anns = [], []
    while want:
        ok, im = cap.read()
        if not ok: break
        if f in want:
            it = want.pop(f); H, W = im.shape[:2]
            for x1, y1, x2, y2 in it["ignore"]:
                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))
                if x2 > x1 and y2 > y1: im[y1:y2, x1:x2] = cv2.GaussianBlur(im[y1:y2, x1:x2], (0, 0), 6)
            name = f"{res['clip']}_{f:05d}.jpg"; cv2.imwrite(str(img_dir / name), im, [cv2.IMWRITE_JPEG_QUALITY, quality])
            iid = len(coco_imgs) + 1; coco_imgs.append({"id": iid, "file_name": f"images/{name}", "width": W, "height": H, "clip": res["clip"], "frame": f})
            for x in it["anchored"] + [{"box": b, "kind": "auto"} for b in it["others"]]:
                x1, y1, x2, y2 = x["box"]; x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                if x2 - x1 < 3 or y2 - y1 < 3: continue
                anns.append({"id": len(anns) + 1, "image_id": iid, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0, "source": x["kind"]})
        f += 1
    return coco_imgs, anns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip"); ap.add_argument("--all", action="store_true"); ap.add_argument("--out", required=True); ap.add_argument("--export")
    ap.add_argument("--cache-tag", default="rfdetrS"); ap.add_argument("--max-gap", type=int, default=15)
    ap.add_argument("--max-area-ratio", type=float, default=1.6); ap.add_argument("--max-jump", type=float, default=0.25)
    ap.add_argument("--hi", type=float, default=0.4); ap.add_argument("--lo", type=float, default=0.15)
    ap.add_argument("--reuse", action="store_true", help="--out에 있는 JSON을 다시 씀(검출 캐시 없이 내보내기만; 클라우드용)")
    ap.add_argument("--stride", type=int, default=5); ap.add_argument("--hard-stride", type=int, default=2)
    a = ap.parse_args()
    ds = json.loads((ROOT / "bench/dataset.json").read_text()); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    clips = [c for c in ds["clips"] if c["gt"] and c["split"] != "test" and (a.all or c["id"] == a.clip)]   # 테스트 클립은 학습 데이터에 절대 넣지 않음
    for c in clips:
        jp = out / f"{c['id']}.json"
        if a.reuse and jp.exists(): res = json.loads(jp.read_text())
        else: res = process(c, a); jp.write_text(json.dumps(res, ensure_ascii=False))
        print(c["id"], json.dumps(res["stats"], ensure_ascii=False), flush=True)
        if a.export:
            ex = Path(a.export) / c["id"]; imgs, anns = export(res, ex)
            (ex / "_annotations.coco.json").write_text(json.dumps({"images": imgs, "annotations": anns, "categories": [{"id": 1, "name": "vehicle", "supercategory": "vehicle"}]}))
            print(f"  내보냄 {len(imgs)}장, 상자 {len(anns)}개 → {ex}", flush=True)


if __name__ == "__main__":
    main()
