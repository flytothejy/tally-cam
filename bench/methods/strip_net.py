"""띠 위 점 검출기 (Track B 2단계): 사람 점 라벨(시각, 기준선 위치)로 시공간 띠에서 통과 사건을 직접 찾는다.

입력: 기준선 띠 (선 위치 × 프레임 × RGB) + 배경(행별 시간 중앙값)을 뺀 3채널 = 6채널
출력: 같은 크기의 히트맵. 라벨 점에 가우시안을 두고 CenterNet식 focal loss로 학습.
추론: 히트맵 극대점(시간 ±K 프레임, 위치 ±R 행 안에서 최대) 중 임계값 이상 → 사건.

평가: 카메라 하나를 빼고 학습 → 빠진 카메라 클립으로 예측(leave-one-camera-out). 임계값은 학습 클립에서 고름.
  python3 bench/methods/strip_net.py --out bench/preds/E1-stripnet [--iters 1500]
"""
import argparse, csv, json, math, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "bench")); sys.path.insert(0, str(ROOT / "bench/methods"))
from score import read_gt, match, direction
from strips import build

VIDEO_DIRS = [ROOT, Path("/mnt/user-data/uploads/Project/tally-cam")]
SIG_T, SIG_P = 4.0, 4.0     # 가우시안 폭(프레임, 행)


def find_video(rel):
    for d in VIDEO_DIRS:
        if (d / rel).exists(): return d / rel
    raise FileNotFoundError(rel)


def prep(strip):
    """(P, T, 3) uint8 → (6, P, T) float: 원본 + 배경 뺀 값."""
    x = strip.astype(np.float32) / 255.0
    bg = np.median(x[:, ::5], axis=1, keepdims=True)
    return np.concatenate([x, x - bg], 2).transpose(2, 0, 1).copy()


def load_clip(c):
    strips, fps, lines = build(find_video(c["video"]), ROOT / c["gates"], ROOT / f"bench/cache/strips_{c['id']}.npz")
    gt_path = ROOT / c["gt"]
    rows = [r for r in csv.DictReader(l for l in gt_path.read_text().splitlines() if l and not l.startswith("#"))]
    _, ranges = read_gt(gt_path)
    out = []
    for L in lines:
        s = strips[L["name"]]; P, T = s.shape[:2]
        pts = [(float(r["gate_pos"]) * (P - 1), float(r["t_gate_sec"]) * fps) for r in rows if r["route"] == L["name"]]
        mask = np.zeros(T, np.float32)
        for a, b in (ranges or [[0, T / fps]]): mask[int(a * fps):int(math.ceil(b * fps)) + 1] = 1
        out.append({"clip": c["id"], "line": L["name"], "x": prep(s), "pts": pts, "mask": mask, "fps": fps})
    return out


def heat(P, T, pts):
    h = np.zeros((P, T), np.float32)
    pp, tt = np.arange(P)[:, None], np.arange(T)[None, :]
    for p, t in pts:
        t0, t1 = max(0, int(t - 4 * SIG_T)), min(T, int(t + 4 * SIG_T) + 1)
        g = np.exp(-((pp - p) ** 2) / (2 * SIG_P ** 2) - ((tt[:, t0:t1] - t) ** 2) / (2 * SIG_T ** 2))
        h[:, t0:t1] = np.maximum(h[:, t0:t1], g)
    return h


class Net(nn.Module):
    def __init__(s, c=24):
        super().__init__()
        blk = lambda i, o, d=1: nn.Sequential(nn.Conv2d(i, o, 3, padding=d, dilation=d), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        s.e1 = nn.Sequential(blk(6, c), blk(c, c))
        s.e2 = nn.Sequential(nn.MaxPool2d(2), blk(c, 2 * c), blk(2 * c, 2 * c))
        s.e3 = nn.Sequential(nn.MaxPool2d(2), blk(2 * c, 2 * c, 2), blk(2 * c, 2 * c, 4), blk(2 * c, 2 * c, 8))
        s.d2 = nn.Sequential(blk(4 * c, 2 * c), blk(2 * c, c))
        s.d1 = nn.Sequential(blk(2 * c, c), nn.Conv2d(c, 1, 1))
        s.d1[-1].bias.data.fill_(-4.0)

    def forward(s, x):
        P, T = x.shape[-2:]; x = F.pad(x, (0, (-T) % 4, 0, (-P) % 4))
        a = s.e1(x); b = s.e2(a); c = s.e3(b)
        d = s.d2(torch.cat([F.interpolate(c, size=b.shape[-2:]), b], 1))
        o = s.d1(torch.cat([F.interpolate(d, size=a.shape[-2:]), a], 1))
        return o[..., :P, :T]


def focal(logit, y, m):
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4); pos = (y > 0.99).float()
    l = -(pos * (1 - p) ** 2 * torch.log(p) + (1 - pos) * (1 - y) ** 4 * p ** 2 * torch.log(1 - p)) * m
    return l.sum() / pos.mul(m).sum().clamp(min=1)


def sample(items, rng, CP=128, CT=256):
    it = items[rng.integers(len(items))]; x, P, T = it["x"], it["x"].shape[1], it["x"].shape[2]
    s = rng.uniform(0.8, 1.25)                 # 시간축 신축(속도 변화)
    ct = min(T - 1, int(CT * s)); t0 = rng.integers(0, T - ct); p0 = rng.integers(0, max(1, P - CP + 1)); cp = min(CP, P)
    xc = torch.from_numpy(x[:, p0:p0 + cp, t0:t0 + ct])[None]
    xc = F.interpolate(xc, size=(cp, CT), mode="bilinear", align_corners=False)[0]
    pts = [((p - p0), (t - t0) * CT / ct) for p, t in it["pts"] if p0 - 8 <= p < p0 + cp + 8 and t0 - 20 <= t < t0 + ct + 20]
    y = heat(cp, CT, pts); m = np.interp(np.linspace(t0, t0 + ct - 1, CT), np.arange(len(it["mask"])), it["mask"]).astype(np.float32)
    g = rng.uniform(0.7, 1.4); b = rng.uniform(-0.1, 0.1)           # 밝기·대비
    xc[:3] = (xc[:3] * g + b).clamp(0, 1); xc[3:] = xc[3:] * g
    if rng.random() < 0.5: xc = xc.flip(1); y = y[::-1].copy()      # 선 방향 뒤집기(위치축)
    if cp < CP:
        xc = F.pad(xc, (0, 0, 0, CP - cp)); y = np.pad(y, ((0, CP - cp), (0, 0)))
        mm = np.zeros((CP, CT), np.float32); mm[:cp] = m
    else: mm = np.broadcast_to(m, (CP, CT)).copy()
    return xc, torch.from_numpy(y), torch.from_numpy(mm)


def train(items, iters, seed=0, bs=8, log=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=iters, pct_start=0.1)
    net.train(); t0 = time.time()
    for i in range(iters):
        b = [sample(items, rng) for _ in range(bs)]
        x, y, m = (torch.stack(v) for v in zip(*b))
        loss = focal(net(x)[:, 0], y, m); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if log and (i % 200 == 0 or i == iters - 1): log(f"  iter {i} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
    return net.eval()


@torch.no_grad()
def predict(net, it, KT=9, KP=11):
    h = torch.sigmoid(net(torch.from_numpy(it["x"])[None]))[0, 0]
    mx = F.max_pool2d(h[None, None], (KP, KT), 1, (KP // 2, KT // 2))[0, 0]
    pk = ((h == mx) & (h > 0.05)).nonzero().numpy()
    return [(float(p[1]) / it["fps"], float(p[0]) / (h.shape[0] - 1), float(h[p[0], p[1]])) for p in pk]   # (t초, 위치, 점수)


def f1_at(dets, items, th):
    tp = n_gt = n_pr = 0
    for it in items:
        mask = it["mask"]; fps = it["fps"]
        g = [t / fps for _, t in it["pts"] if mask[min(int(t), len(mask) - 1)]]
        p = [t for t, _, s in dets[(it["clip"], it["line"])] if s >= th and mask[min(int(t * fps), len(mask) - 1)]]
        tp += len(match(sorted(g), sorted(p), 1.0)); n_gt += len(g); n_pr += len(p)
    r, pr = tp / max(n_gt, 1), tp / max(n_pr, 1)
    return 2 * r * pr / max(r + pr, 1e-9)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--split", default="dev"); a = ap.parse_args()
    torch.set_num_threads(2)
    ds = json.loads((ROOT / "bench/dataset.json").read_text())
    clips = [c for c in ds["clips"] if c["split"] == a.split and c["gt"]]
    data = {c["id"]: load_clip(c) for c in clips}
    cams = sorted({c["camera"] for c in clips}); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cam in cams:
        test = [c for c in clips if c["camera"] == cam]; trn = [it for c in clips if c["camera"] != cam for it in data[c["id"]]]
        print(f"== 빼는 카메라 {cam}: 학습 {len(trn)}개 띠 ({sum(len(i['pts']) for i in trn)}점)", flush=True)
        t0 = time.time(); net = train(trn, a.iters, log=lambda s: print(s, flush=True))
        dt = {(it["clip"], it["line"]): predict(net, it) for it in trn}
        th = max(np.arange(0.1, 0.9, 0.05), key=lambda v: f1_at(dt, trn, v))
        print(f"  임계값 {th:.2f} (학습 F1 {f1_at(dt, trn, th) * 100:.1f}), {time.time() - t0:.0f}s", flush=True)
        for c in test:
            evs = []; ms = []
            for it in data[c["id"]]:
                s0 = time.perf_counter(); d = predict(net, it); ms.append((time.perf_counter() - s0) * 1000 / it["x"].shape[2])
                evs += [{"t": round(t, 3), "line": it["line"], "pos": round(p, 3), "score": round(s, 3)} for t, p, s in d if s >= th]
            T = data[c["id"]][0]["x"].shape[2]; fps = data[c["id"]][0]["fps"]
            (out / f"{c['id']}.json").write_text(json.dumps({"method": {"name": "strip_net", "iters": a.iters, "held_out": cam, "th": round(float(th), 2)},
                "video_seconds": round(T / fps, 2), "infer_ms_avg": round(float(np.sum(ms)), 3), "events": sorted(evs, key=lambda e: e["t"])}, ensure_ascii=False))
            summary[c["id"]] = {"held_out": cam, "th": round(float(th), 2), "n": len(evs)}
        torch.save(net.state_dict(), out / f"net_wo_{cam}.pt")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
