"""검출·추적 없이 '움직임(광류)'으로 흐름을 찾아 기준선 제안 — 노트북처럼 검출이 느려 궤적이 끊기는 곳용.

1) 1초에 두 번쯤 바로 이웃한 두 프레임을 작게 줄여 광류(Farneback)를 구함 → 움직이는 픽셀의 방향·속도
2) 칸(8×8 화면 픽셀)마다 방향 8개 막대그래프로 쌓음(같은 칸을 다른 시간에 다른 방향 차가 지나도 섞이지 않게)
3) 칸마다 가장 많이 쌓인 방향으로 흐름장을 만들고, 그 흐름을 따라 흐르는 선(유선)을 그려 '가짜 궤적'으로 씀
4) 가짜 궤적을 tools/autogate.py의 흐름 묶기·선 긋기에 그대로 넣음(차 크기는 검출 몇 장에서 높이-위치 관계로 추정)

  python3 tools/flowgate.py --video x.mp4 --t0 0 --t1 150 --out gates.json [--dets cache.npz]
"""
import argparse, json, math
from pathlib import Path
import cv2
import numpy as np
import autogate

DIRS = 8


class FlowAcc:
    """이웃한 두 프레임을 받을 때마다 광류를 칸별 방향 막대그래프에 쌓음(라이브에서도 그대로 씀)."""
    def __init__(self, W, H, scale=0.5, mag_min=0.6, cell=8):
        self.W, self.H, self.scale, self.mag_min, self.cell = W, H, scale, mag_min, cell
        gw, gh = W // cell + 1, H // cell + 1
        self.hist = np.zeros((gh, gw, DIRS)); self.vsum = np.zeros((gh, gw, DIRS, 2)); self.n = 0
    def add(self, a, b):
        s = self.scale
        ga = cv2.cvtColor(cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        fl = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0) / s   # 원래 화면 픽셀/프레임
        ys, xs = np.mgrid[0:fl.shape[0], 0:fl.shape[1]]
        m = np.hypot(fl[..., 0], fl[..., 1]) > self.mag_min
        if m.any():
            X, Y = np.minimum((xs[m] / s).astype(int) // self.cell, self.hist.shape[1] - 1), np.minimum((ys[m] / s).astype(int) // self.cell, self.hist.shape[0] - 1)
            d = (np.round(np.arctan2(fl[..., 1][m], fl[..., 0][m]) / (2 * np.pi / DIRS)).astype(int)) % DIRS
            np.add.at(self.hist, (Y, X, d), 1); np.add.at(self.vsum, (Y, X, d, 0), fl[..., 0][m]); np.add.at(self.vsum, (Y, X, d, 1), fl[..., 1][m])
        self.n += 1
    def propose(self, fps, boxes=None):
        """boxes: 차 상자 (x1, y1, x2, y2) 몇십 개(크기 추정용, 없어도 됨)."""
        lines, _ = streamlines(self.hist, self.vsum, self.cell, self.n)
        hm = size_model(np.array([[0, b[0], b[1], b[2], b[3], 0.9, 2] for b in boxes], np.float32) if boxes is not None and len(boxes) else None, self.W, self.H)
        trk = to_tracks(lines, hm)
        if not len(trk): return []
        return [g for g in autogate.propose(trk, fps, self.W, self.H, min_tracks=8, flat=True, ylim=(0.35, 0.8)) if g["target"] == "vehicle"]   # 횡단보도는 광류로 구분 못 함


def flow_field(video, t0, t1, every=0.5):
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS); acc = FlowAcc(int(cap.get(3)), int(cap.get(4)))
    t = t0
    while t < t1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok1, a = cap.read(); ok2, b = cap.read()
        if not (ok1 and ok2): break
        acc.add(a, b); t += every
    return acc, fps


def streamlines(hist, vsum, cell, n_pairs, min_hits=0.06, steps=400):
    """칸마다 우세 방향(그 방향 표본이 전체 표본의 min_hits 이상)으로 흐름장 → 시작점마다 흐름을 따라가는 선."""
    gh, gw, _ = hist.shape; tot = hist.sum(2)
    dom = hist.argmax(2); cnt = np.take_along_axis(hist, dom[..., None], 2)[..., 0]
    ok = cnt >= max(3.0, min_hits * n_pairs * 4)   # 4 = 한 칸(8×8)을 반 크기로 줄였을 때 픽셀 수
    v = np.take_along_axis(vsum, dom[..., None, None].repeat(2, -1), 2)[..., 0, :] / np.maximum(cnt[..., None], 1)
    lines = []
    seeds = np.argwhere(ok)
    rng = np.random.default_rng(0); rng.shuffle(seeds)
    for (cy, cx) in seeds[: 1500]:
        d0 = dom[cy, cx]
        def walk(sign):   # 시작점에서 흐름 방향(+1)·거꾸로(-1)로 — 양쪽으로 다 그어야 선이 흐름 전체를 덮음(한쪽만 그으면 하류에 몰림)
            x, y = (cx + 0.5) * cell, (cy + 0.5) * cell; out = []
            for _ in range(steps):
                gx, gy = int(x // cell), int(y // cell)
                if not (0 <= gx < gw and 0 <= gy < gh) or not ok[gy, gx]: break
                if min(abs(dom[gy, gx] - d0), DIRS - abs(dom[gy, gx] - d0)) > 1: break   # 방향이 45° 넘게 꺾이면 멈춤(다른 흐름)
                vx, vy = v[gy, gx]; sp = math.hypot(vx, vy)
                if sp < 1e-3: break
                x += sign * vx / sp * 2.0; y += sign * vy / sp * 2.0; out.append((x, y))
            return out
        back = walk(-1)[::-1]; pts = back + [((cx + 0.5) * cell, (cy + 0.5) * cell)] + walk(1)
        if len(pts) > 15: lines.append((np.array(pts), float(np.median([math.hypot(*v[int(py // cell), int(px // cell)]) for px, py in pts[:: 5] if 0 <= int(py // cell) < gh and 0 <= int(px // cell) < gw]))))
    return lines, ok


def size_model(dets, W, H):
    """검출 상자(차)에서 화면 높이 y → 상자 높이 h 를 직선으로(원근). 없으면 화면 높이의 6%."""
    if dets is None or not len(dets): return lambda y: 0.06 * H
    v = dets[np.isin(dets[:, 6], (2, 5, 7)) & (dets[:, 5] > 0.4)]
    if len(v) < 20: return lambda y: 0.06 * H
    yc, h = (v[:, 2] + v[:, 4]) / 2, v[:, 4] - v[:, 2]
    a, b = np.polyfit(yc, h, 1); return lambda y: max(8.0, a * y + b)


def to_tracks(lines, hmodel):
    """유선 → autogate용 가짜 궤적 dump(f, id, x1, y1, x2, y2, conf, cls). 프레임 간격은 유선의 속도로."""
    rows = []
    for i, (pts, sp) in enumerate(lines):
        step = 2.0 / max(sp, 0.3)   # 유선 한 걸음(2픽셀)이 몇 프레임인지
        for j, (x, y) in enumerate(pts[:: 3]):
            h = hmodel(y); w = h * 1.3
            rows.append((j * 3 * step, i, x - w / 2, y - h / 2, x + w / 2, y + h / 2, 0.9, 2))
    return np.array(rows, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True); ap.add_argument("--t0", type=float, default=0); ap.add_argument("--t1", type=float, default=150)
    ap.add_argument("--dets", help="검출 캐시(npz) — 차 크기 추정용(몇 장이면 충분)"); ap.add_argument("--out", required=True)
    ap.add_argument("--debug", help="흐름장 그림 저장")
    a = ap.parse_args()
    acc, fps = flow_field(a.video, a.t0, a.t1)
    boxes = None
    if a.dets:
        z = np.load(a.dets); d = z["det"]; d = d[(d[:, 0] >= a.t0 * fps) & (d[:, 0] < a.t1 * fps) & np.isin(d[:, 6], (2, 5, 7)) & (d[:, 5] > 0.4)]; boxes = d[:, 1:5]
    gates = acc.propose(fps, boxes)
    lines, _ = streamlines(acc.hist, acc.vsum, acc.cell, acc.n)
    Path(a.out).write_text(json.dumps(gates, ensure_ascii=False, indent=1), encoding="utf-8")
    for g in gates: print(g["name"], g["label"], g["p1"], g["p2"], g.get("minor", ""))
    if a.debug:
        cap = cv2.VideoCapture(a.video); cap.set(cv2.CAP_PROP_POS_FRAMES, int(a.t0 * fps)); _, img = cap.read()
        import colorsys
        for pts, _ in lines:
            d = pts[-1] - pts[0]; c = [int(255 * v) for v in colorsys.hsv_to_rgb((math.degrees(math.atan2(d[1], d[0])) % 360) / 360, 1, 1)]
            cv2.polylines(img, [pts.astype(np.int32)], False, c, 1)
        for g in gates: cv2.line(img, tuple(g["p1"]), tuple(g["p2"]), (255, 255, 255), 3)
        cv2.imwrite(a.debug, img)


if __name__ == "__main__":
    main()
