"""캐시된 검출 → 추적(ByteTrack/BoT-SORT) → 기준선 통과 세기. 트랙 A 실험용.

옵션
  --tracker bytetrack|botsort, --high/--low/--new/--buffer: 추적 기준값
  --agnostic: 클래스 무시 NMS(버스가 car+truck 두 박스로 잡히는 것 방지)
  --anchor center|bottom: 통과 판정에 쓰는 점(박스 중심 / 바닥 중앙)
  --dedup S: 같은 기준선에서 S초 안, 통과 지점이 가까운(기준선 길이의 15%) 셈은 하나로
  --ext E: 기준선을 양 끝으로 E px 늘려 판정(끝 부근 놓침 방지)

  python3 bench/methods/track_count.py --cache c.npz --lines counter/lines/x.json --out preds/m/clip.json [옵션]
"""
import argparse, json, math, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch, torchvision
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT


class Dets:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls
        self.xywh = np.c_[(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2, xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]] if len(xyxy) else np.zeros((0, 4), np.float32)
    def __len__(self): return len(self.conf)
    def __getitem__(self, i): return Dets(self.xyxy[i], self.conf[i], self.cls[i])


def side(p1, p2, q): return 1 if (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0]) > 0 else -1
def cross_pt(a, b, p1, p2):
    if side(p1, p2, a) == side(p1, p2, b) or side(a, b, p1) == side(a, b, p2): return None
    d = (b[0] - a[0]) * (p2[1] - p1[1]) - (b[1] - a[1]) * (p2[0] - p1[0])
    if d == 0: return None
    s = ((p1[0] - a[0]) * (p2[1] - p1[1]) - (p1[1] - a[1]) * (p2[0] - p1[0])) / d
    return (a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True); ap.add_argument("--lines", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tracker", default="bytetrack"); ap.add_argument("--high", type=float, default=0.25); ap.add_argument("--low", type=float, default=0.1)
    ap.add_argument("--new", type=float, default=0.25); ap.add_argument("--buffer", type=int, default=30); ap.add_argument("--match", type=float, default=0.8)
    ap.add_argument("--conf", type=float, default=0.10, help="이보다 낮은 검출은 버림")
    ap.add_argument("--agnostic", action="store_true"); ap.add_argument("--anchor", default="center")
    ap.add_argument("--dedup", type=float, default=0.0); ap.add_argument("--ext", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=1, help="N프레임마다만 추적(검출을 N프레임마다 한 경우)")
    ap.add_argument("--oneway", action="store_true", help="기준선마다 다수 통과 방향을 정하고 반대로 지난 통과는 버림(반대 차로 차 섞임 방지)")
    ap.add_argument("--lost", type=float, default=0.0, help="선 앞에서 선을 향하다 사라진 트랙을 셈: 선까지 거리가 상자 높이×이 값 이하일 때(0=끔, --oneway 필요)")
    ap.add_argument("--dump", help="추적 결과(f, id, x1, y1, x2, y2, conf)를 npz로 저장(분석용)")
    a = ap.parse_args()
    z = np.load(a.cache, allow_pickle=True); det = z["det"]; F = int(z["frames"]); fps = float(z["fps"])
    lines = json.loads(Path(a.lines).read_text(encoding="utf-8"))
    for L in lines:
        p1, p2 = np.array(L["p1"], float), np.array(L["p2"], float); u = (p2 - p1) / np.linalg.norm(p2 - p1)
        L["q1"], L["q2"], L["len"] = tuple(p1 - u * a.ext), tuple(p2 + u * a.ext), float(np.linalg.norm(p2 - p1))
    args = SimpleNamespace(tracker_type=a.tracker, track_high_thresh=a.high, track_low_thresh=a.low, new_track_thresh=a.new,
                           track_buffer=a.buffer, match_thresh=a.match, fuse_score=True, gmc_method="none",
                           proximity_thresh=0.5, appearance_thresh=0.8, with_reid=False, model="auto")
    def run(major=None):
        trk = (BOTSORT if a.tracker == "botsort" else BYTETracker)(args)
        dd = det[det[:, 5] >= a.conf]
        starts = np.searchsorted(dd[:, 0], np.arange(F + 1))
        last, counted, events = {}, set(), []
        hist = {}   # tid → 최근 (f, cx, cy, h)
        dump = []
        t0 = time.perf_counter()
        for f in range(0, F, a.stride):
            d = dd[starts[f]:starts[f + 1]]
            if len(d) and a.agnostic:
                keep = torchvision.ops.nms(torch.tensor(d[:, 1:5]), torch.tensor(d[:, 5]), 0.6).numpy(); d = d[np.sort(keep)]
            D = Dets(d[:, 1:5].astype(np.float32), d[:, 5].astype(np.float32), d[:, 6].astype(np.float32))
            out = trk.update(D, None) if len(D) or True else []
            for x1, y1, x2, y2, tid, sc, cl, _ in (out if len(out) else []):
                tid = int(tid)
                if a.dump: dump.append((f, tid, x1, y1, x2, y2, sc))
                cen, bot = ((x1 + x2) / 2, (y1 + y2) / 2), ((x1 + x2) / 2, y2)
                cur = bot if a.anchor == "bottom" else cen
                prevs = last.get(tid); last[tid] = (cen, bot); prev = prevs and (prevs[1] if a.anchor == "bottom" else prevs[0])
                hh = hist.setdefault(tid, []); hh.append((f, cur[0], cur[1], y2 - y1)); del hh[:-8]
                if prev is None: continue
                for L in lines:
                    key = (L["name"], tid)
                    if key in counted: continue
                    if "classes" in L and int(cl) not in L["classes"]: continue   # 선마다 셀 대상(예: 사람만 [0], 차만 [2,3,5,7])
                    if "anchor" in L:   # 선마다 기준점(사람은 발, 차는 중심)
                        prev, cur = (prevs[1], bot) if L["anchor"] == "bottom" else (prevs[0], cen)
                    else:
                        cur = bot if a.anchor == "bottom" else cen; prev = prevs[1] if a.anchor == "bottom" else prevs[0]
                    cp = cross_pt(prev, cur, L["q1"], L["q2"])
                    if cp is None: continue
                    if major and side(L["q1"], L["q2"], cur) != major.get(L["name"], 0): continue   # 반대 방향 통과는 세지 않고, 이 트랙은 계속 지켜봄
                    counted.add(key)
                    t = f / fps
                    if a.dedup and any(e["line"] == L["name"] and t - e["t"] <= a.dedup and math.dist(cp, e["pt"]) <= 0.15 * L["len"] for e in events):
                        continue
                    events.append({"t": round(t, 3), "id": tid, "line": L["name"], "pt": [round(cp[0], 1), round(cp[1], 1)], "cls": int(cl), "sgn": side(L["q1"], L["q2"], cur)})
        if major and a.lost:   # 사라진 트랙 외삽: 마지막 위치가 선 앞(반대편)이고 선을 향해 움직였으면 닿는 시각에 셈
            for tid, hh in hist.items():
                if len(hh) < 4 or hh[-1][0] > F - 1 - 30: continue
                (f0, x0, y0, _), (f1, x1, y1, h1) = hh[0], hh[-1]
                vx, vy = (x1 - x0) / max(f1 - f0, 1), (y1 - y0) / max(f1 - f0, 1)
                for L in lines:
                    if (L["name"], tid) in counted or side(L["q1"], L["q2"], (x1, y1)) == major.get(L["name"], 0): continue
                    reach = a.lost * h1; tgt = (x1 + vx * reach / max(math.hypot(vx, vy), 1e-6), y1 + vy * reach / max(math.hypot(vx, vy), 1e-6))
                    cp = cross_pt((x1, y1), tgt, L["q1"], L["q2"])
                    if cp is None or side(L["q1"], L["q2"], tgt) != major[L["name"]]: continue
                    sp = math.hypot(vx, vy); dtf = math.dist((x1, y1), cp) / max(sp, 1e-6)
                    counted.add((L["name"], tid))
                    events.append({"t": round((f1 + dtf) / fps, 3), "id": tid, "line": L["name"], "pt": [round(cp[0], 1), round(cp[1], 1)], "cls": -1, "sgn": major[L["name"]], "lost": True})
            events.sort(key=lambda e: e["t"])
        return events, dump, t0
    events, dump, t0 = run()
    if a.oneway:   # 1차: 선마다 다수 통과 방향을 정함 → 2차: 그 방향 통과만 셈(방향을 미리 몰라도 됨)
        major = {}
        for L in lines:
            sg = [e["sgn"] for e in events if e["line"] == L["name"]]
            major[L["name"]] = 1 if sum(sg) >= 0 else -1
        events, dump, t0 = run(major)
    if a.dump: np.savez_compressed(a.dump, trk=np.array(dump, np.float32), fps=fps)
    track_ms = (time.perf_counter() - t0) * 1000 / max(F, 1)
    det_ms = float(np.mean(z["det_ms"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"method": vars(a), "video_seconds": round(F / fps, 2), "infer_ms_avg": round(det_ms + track_ms, 1),
                                       "det_ms": round(det_ms, 1), "track_ms": round(track_ms, 2), "events": events}, ensure_ascii=False))
    print(f"{len(events)} events, det {det_ms:.0f} + track {track_ms:.1f} ms/frame")


if __name__ == "__main__":
    main()
