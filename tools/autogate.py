"""궤적으로 기준선 자동 제안.

몇 분 동안의 추적 결과(track_count.py --dump)를 받아
  1) 차 궤적을 움직임 방향으로 묶고(앞·뒤 두 흐름, 옆길처럼 비스듬한 흐름은 따로)
  2) 흐름마다 진행 방향에 수직인 선을 긋는다 — 차가 잘 움직이고(대기 줄 피함), 크게 보이고,
     흐름의 궤적 대부분이 지나가며, 다른 흐름은 덜 지나가는 자리
  3) 사람 궤적 중 차가 다니는 영역을 차 흐름과 엇갈리게 가로지르는 무리를 횡단보도로 보고,
     걷는 방향에 수직인 선(양방향)을 긋는다.

  python3 bench/methods/track_count.py --cache <캐시> --lines <아무 기준선> --out /tmp/x.json --agnostic --conf 0.25 --stride 2 --dump trk.npz
  python3 tools/autogate.py --trk trk.npz --size 720x480 --t0 0 --t1 220 --out gates.json [--meta tools/labeler/clips/<id>/meta.json]

검증(11/03): 역삼 클립1 합계 F1 91.3(손으로 그은 선 92.6), 대구 클립2 93.1(92.5). 신호 한 주기(2~3분) 넘게 봐야 함 — 60초만 보면 한쪽 흐름을 놓침.
"""
import argparse, json, math
from pathlib import Path
import numpy as np

VEH = (2, 3, 5, 7)


def load_tracks(trk, fps, t0=0.0, t1=1e9, min_len=6):
    """dump(f, id, x1, y1, x2, y2, conf, cls) → 트랙 목록. 순간이동(추적 번호가 다른 물체로 넘어감)은 끊어서 버림."""
    T = trk[(trk[:, 0] >= t0 * fps) & (trk[:, 0] < t1 * fps)]
    T = T[np.lexsort((T[:, 0], T[:, 1]))]
    out = []
    for tid in np.unique(T[:, 1]):
        t = T[T[:, 1] == tid]
        cls = int(np.bincount(t[:, 7].astype(int), minlength=8).argmax())
        h = t[:, 5] - t[:, 3]; w = t[:, 4] - t[:, 2]
        x = (t[:, 2] + t[:, 4]) / 2; y = t[:, 5] if cls == 0 else (t[:, 3] + t[:, 5]) / 2
        step = np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t[:, 0]), 1)
        cuts = np.where(step > 0.5 * np.median(h))[0] + 1   # 한 프레임에 상자 높이 절반 넘게 뛰면 다른 물체
        for seg in np.split(np.arange(len(t)), cuts):
            if len(seg) < min_len: continue
            out.append({"cls": cls, "f": t[seg, 0], "x": x[seg], "y": y[seg], "h": float(np.median(h[seg])), "w": float(np.median(w[seg]))})
    return out


def moving(tr, k=1.5):
    d = math.hypot(tr["x"][-1] - tr["x"][0], tr["y"][-1] - tr["y"][0])
    return d >= max(30.0, k * tr["h"])


def heading(tr):
    v = np.array([tr["x"][-1] - tr["x"][0], tr["y"][-1] - tr["y"][0]]); return v / (np.linalg.norm(v) + 1e-9)


def axis_of(units):
    """방향 벡터들의 주축(각도 두 배 평균) — 앞뒤를 구분하지 않는 축."""
    a = np.arctan2(units[:, 1], units[:, 0]); m = math.atan2(np.sin(2 * a).sum(), np.cos(2 * a).sum()) / 2
    return np.array([math.cos(m), math.sin(m)])


def group_vehicles(trs, min_tracks, bw=12.0, tol=35.0):
    """진행 방향(각도) 분포의 봉우리마다 흐름 하나. 교차로처럼 여러 길이 비스듬히 만나도 길마다 나뉘게
    (봉우리 사이가 좁아 한 길이 둘로 갈리면, 뒤에서 '같은 차가 두 선을 지나는' 검사로 작은 쪽이 빠짐)."""
    if len(trs) < min_tracks: return []
    ang = np.array([math.degrees(math.atan2(*heading(t)[::-1])) % 360 for t in trs])
    grid = np.arange(0, 360, 2.0)
    dif = lambda a, b: np.abs((a - b + 180) % 360 - 180)
    dens = np.exp(-0.5 * (dif(grid[:, None], ang[None, :]) / bw) ** 2).sum(1)
    peaks = [i for i in range(len(grid)) if dens[i] >= dens[i - 1] and dens[i] > dens[(i + 1) % len(grid)]]
    peaks = sorted(peaks, key=lambda i: -dens[i])
    groups, used = [], np.zeros(len(trs), bool)
    for i in peaks:
        m = (~used) & (dif(ang, grid[i]) <= tol)
        if m.sum() < max(min_tracks, 0.04 * len(trs)): continue
        groups.append([t for t, k in zip(trs, m) if k]); used |= m
    return merge_adjacent(groups)


def occ(g, cell=8):
    o = set()
    for t in g:
        r = max(1, int(t["w"] / 2 / cell))
        for x, y in zip(t["x"].astype(int) // cell, t["y"].astype(int) // cell):
            o.update((x + i, y + j) for i in range(-r, r + 1) for j in range(-r, r + 1))
    return o


def merge_adjacent(groups, max_deg=60.0, min_overlap=0.3):
    """원근 때문에 한 길의 차선들이 부채꼴로 벌어져 방향 봉우리가 여럿 생김 → 방향이 60° 안이고 영역이 겹치는 흐름은 하나로."""
    groups = list(groups)
    while True:
        done = True
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                ui, uj = np.mean([heading(t) for t in groups[i]], 0), np.mean([heading(t) for t in groups[j]], 0)
                cosd = ui @ uj / (np.linalg.norm(ui) * np.linalg.norm(uj) + 1e-9)
                if cosd < math.cos(math.radians(max_deg)): continue
                A, B = occ(groups[i]), occ(groups[j])
                if len(A & B) / max(min(len(A), len(B)), 1) < min_overlap: continue
                groups[i] = groups[i] + groups[j]; del groups[j]; done = False; break
            if not done: break
        if done: return sorted(groups, key=len, reverse=True)


def interp_at(tr, u, s):
    """트랙이 u 방향 위치 s를 지나는 점(없으면 None)과 그때 속도(픽셀/프레임)."""
    p = tr["x"] * u[0] + tr["y"] * u[1]
    for i in range(len(p) - 1):
        if (p[i] - s) * (p[i + 1] - s) <= 0 and p[i] != p[i + 1]:
            r = (s - p[i]) / (p[i + 1] - p[i])
            x = tr["x"][i] + r * (tr["x"][i + 1] - tr["x"][i]); y = tr["y"][i] + r * (tr["y"][i + 1] - tr["y"][i])
            j0, j1 = max(i - 2, 0), min(i + 3, len(p) - 1)
            sp = math.hypot(tr["x"][j1] - tr["x"][j0], tr["y"][j1] - tr["y"][j0]) / max(tr["f"][j1] - tr["f"][j0], 1)
            return x, y, sp
    return None


def seg_crosses(tr, a, b):
    x, y = tr["x"], tr["y"]
    def side(px, py): return np.sign((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]))
    s = side(x, y); idx = np.where(s[:-1] * s[1:] < 0)[0]
    for i in idx:
        s1 = np.sign((x[i + 1] - x[i]) * (a[1] - y[i]) - (y[i + 1] - y[i]) * (a[0] - x[i]))
        s2 = np.sign((x[i + 1] - x[i]) * (b[1] - y[i]) - (y[i + 1] - y[i]) * (b[0] - x[i]))
        if s1 * s2 < 0: return True
    return False


def place_gate(g, others, W, H, margin_k=0.5, undirected=False, flat=False, ylim=(0.06, 0.94)):
    """흐름 g에 수직인 선 하나. 후보 위치마다 점수 = 지나는 비율 × 속도 × 크기^0.5 × (1 - 다른 흐름 비율)."""
    U = np.array([heading(t) for t in g])
    u = axis_of(U) if undirected else U.mean(0) / (np.linalg.norm(U.mean(0)) + 1e-9)
    n0 = np.array([-u[1], u[0]])
    sp_all = np.median([np.median(np.hypot(np.diff(t["x"]), np.diff(t["y"])) / np.maximum(np.diff(t["f"]), 1)) for t in g]) + 1e-6
    rot = lambda v, d: np.array([v[0] * math.cos(d) - v[1] * math.sin(d), v[0] * math.sin(d) + v[1] * math.cos(d)])
    dirs = [rot(n0, math.radians(d)) for d in (-45, -30, -15, 0, 15, 30, 45)] + [np.array([1.0, 0.0])]
    if flat: dirs = [rot(np.array([1.0, 0.0]), math.radians(d)) for d in (-20, -10, 0, 10, 20)] if abs(u[1]) >= abs(u[0]) else [rot(np.array([0.0, 1.0]), math.radians(d)) for d in (-20, -10, 0, 10, 20)]   # 원근: 길을 따라 내려다보면 같은 거리는 화면에서 거의 수평   # 선 방향 후보: 흐름에 수직 ± 45°, 그리고 수평(원근 때문에 사람이 긋는 선은 대개 수평)
    best = None
    for n in dirs:
        m = np.array([-n[1], n[0]])   # 선에 수직인 축 — 이 축을 따라 선을 옮겨 가며 봄
        if abs(m @ u) < 0.4: continue   # 흐름과 거의 나란한 선은 제외
        P = np.concatenate([t["x"] * m[0] + t["y"] * m[1] for t in g])
        for s in np.linspace(np.quantile(P, 0.1), np.quantile(P, 0.9), 25):
            hits = [(t, interp_at(t, m, s)) for t in g]; hits = [(t, r) for t, r in hits if r]
            if len(hits) < max(3, 0.3 * len(g)): continue
            pts = np.array([[r[0], r[1]] for _, r in hits]); off = pts @ n
            lo, hi = np.quantile(off, 0.03), np.quantile(off, 0.97)
            wmed = np.median([t["w"] for t, _ in hits]); hmed = np.median([t["h"] for t, _ in hits])
            lo -= margin_k * wmed; hi += margin_k * wmed
            c = pts.mean(0); a = c + n * (lo - c @ n); b = c + n * (hi - c @ n)
            a = np.clip(a, 0, [W - 1, H - 1]); b = np.clip(b, 0, [W - 1, H - 1])
            edge = min(c[1], H - c[1]) / H   # 화면 위·아래 끝은 상자가 잘려서 불리
            if not (ylim[0] * H <= c[1] <= ylim[1] * H): continue
            speed = min(1.0, np.median([r[2] for _, r in hits]) / sp_all)
            oth = sum(seg_crosses(t, a, b) for t in others) / max(len(others), 1) if others else 0.0
            ang = np.mean([abs(heading(t)[0] * n[1] - heading(t)[1] * n[0]) for t, _ in hits])   # 궤적과 선이 이루는 각의 sin — 비스듬히 스치면 낮음
            score = len(hits) / len(g) * speed * math.sqrt(hmed / H) * (1 - oth) * ang
            if best is None or score > best[0]:
                best = (score, a, b, len(hits), speed, oth)
    return best


def road_mask(vtrs, W, H, cell=8):
    m = np.zeros((H // cell + 1, W // cell + 1), bool)
    for t in vtrs:
        r = max(1, int(t["w"] / 2 / cell))
        for x, y in zip(t["x"].astype(int) // cell, t["y"].astype(int) // cell):
            m[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1] = True
    return m, cell


def flow_dir_at(vtrs, x, y, rad):
    """(x, y) 근처를 지나는 차들의 진행 축(앞뒤 구분 없음)."""
    U = [heading(t) for t in vtrs if np.hypot(t["x"] - x, t["y"] - y).min() < rad]
    return axis_of(np.array(U)) if U else np.zeros(2)


def crosswalk_groups(ptrs, vtrs, W, H, min_tracks):
    """차 영역을 차 흐름과 엇갈리게(각도 차 50° 이상) 지나가는 사람 무리 → 걷는 축으로 묶고 위치(축에 수직인 좌표)로 나눔."""
    if not ptrs or not vtrs: return []
    m, cell = road_mask(vtrs, W, H)
    cand = []
    for t in ptrs:
        xi = np.clip(t["x"].astype(int) // cell, 0, m.shape[1] - 1); yi = np.clip(t["y"].astype(int) // cell, 0, m.shape[0] - 1)
        on = m[yi, xi]
        if on.mean() < 0.3: continue
        fd = flow_dir_at(vtrs, t["x"][on].mean(), t["y"][on].mean(), 3 * t["h"])
        if abs(heading(t) @ fd) > math.cos(math.radians(50)): continue
        cand.append(t)
    if len(cand) < min_tracks: return []
    ax = axis_of(np.array([heading(t) for t in cand])); nrm = np.array([-ax[1], ax[0]])
    off = np.array([np.mean(t["x"] * nrm[0] + t["y"] * nrm[1]) for t in cand]); order = np.argsort(off)
    gap = 3 * np.median([t["h"] for t in cand])
    groups, cur = [], [cand[order[0]]]
    for i, j in zip(order[:-1], order[1:]):
        if off[j] - off[i] > gap: groups.append(cur); cur = []
        cur.append(cand[j])
    groups.append(cur)
    return [g for g in groups if len(g) >= min_tracks]


def propose(trk, fps, W, H, t0=0.0, t1=1e9, min_tracks=8, flat=False, ylim=(0.06, 0.94)):
    trs = [t for t in load_tracks(trk, fps, t0, t1) if moving(t)]
    vtrs = [t for t in trs if t["cls"] in VEH]; ptrs = [t for t in trs if t["cls"] == 0]
    gates = []
    vg = group_vehicles(vtrs, min_tracks)
    placed = []
    for g in sorted(vg, key=len, reverse=True):   # 큰 흐름부터
        others = [t for gg in vg if gg is not g for t in gg]
        b = place_gate(g, others, W, H, flat=flat, ylim=ylim)
        if not b: continue
        # 이 흐름의 차 대부분이 이미 놓인 다른 선도 지나가면(옆길에서 꺾여 들어와 큰 흐름에 합류) 같은 차를 두 번 세게 됨 → 버림
        dup = max([np.mean([seg_crosses(t, pa, pb) for t in g]) for _, pa, pb in placed] or [0.0])
        if dup > 0.5: continue
        # 큰 흐름의 1/4이 안 되는 흐름(옆길에서 꺾여 합류하는 차 등)은 '보조'로 표시 — 큰 흐름 선과 같은 차를 셀 수 있어 기본으로는 빼고, 라벨러에서 켤 수 있게
        minor = len(g) < 0.25 * max(len(gg) for gg in vg)
        placed.append((g, b[1], b[2]))
        i = len(placed); u = np.mean([heading(t) for t in g], 0)
        gates.append({"name": f"auto_v{i}", "label": f"차 흐름 {i} ({len(g)}대)" + (" · 보조" if minor else ""), "note": "자동 제안" + (" · 작은 흐름(합류·회전일 수 있음)" if minor else ""), **({"minor": True} if minor else {}),
                      "p1": [int(round(v)) for v in b[1]], "p2": [int(round(v)) for v in b[2]],
                      "target": "vehicle", "dir": "one", "auto": {"tracks": len(g), "hit": b[3], "speed": round(float(b[4]), 2), "other": round(float(b[5]), 2), "score": round(float(b[0]), 3), "u": [round(float(v), 2) for v in u]}})
    for i, g in enumerate(crosswalk_groups(ptrs, vtrs, W, H, max(3, min_tracks // 2))):
        b = place_gate(g, [], W, H, margin_k=1.0, undirected=True)
        if not b: continue
        gates.append({"name": f"auto_cw{i + 1}", "label": f"횡단보도 {i + 1} ({len(g)}명)", "note": "자동 제안 · 사람 발이 선을 밟는 순간",
                      "p1": [int(round(v)) for v in b[1]], "p2": [int(round(v)) for v in b[2]],
                      "target": "person", "dir": "both", "auto": {"tracks": len(g), "hit": b[3], "score": round(float(b[0]), 3)}})
    return gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trk", required=True, help="track_count.py --dump 결과")
    ap.add_argument("--size", required=True, help="영상 크기 WxH")
    ap.add_argument("--t0", type=float, default=0.0); ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--min-tracks", type=int, default=8)
    ap.add_argument("--out", help="기준선 JSON(counter/lines 형식)")
    ap.add_argument("--meta", help="라벨러 클립 meta.json — auto_lanes로 넣어 라벨러의 '자동 기준선' 버튼이 불러오게")
    a = ap.parse_args()
    z = np.load(a.trk); W, H = map(int, a.size.split("x"))
    g = propose(z["trk"], float(z["fps"]), W, H, a.t0, a.t1, a.min_tracks)
    if a.out: Path(a.out).write_text(json.dumps(g, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.meta:
        m = json.loads(Path(a.meta).read_text(encoding="utf-8"))
        m["auto_lanes"] = [{k: L[k] for k in ("name", "label", "note", "p1", "p2", "target", "dir", "minor") if k in L} for L in g]
        Path(a.meta).write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    for L in g: print(L["name"], L["label"], L["p1"], L["p2"], L["auto"])


if __name__ == "__main__":
    main()
