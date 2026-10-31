"""기준선 시공간 띠 만들기(라벨러·prepare_clip과 같은 규칙: 선 위 점마다 3×3 평균). 결과는 캐시."""
import json
from pathlib import Path
import cv2, numpy as np


def sample_points(p1, p2):
    a, b = np.array(p1, float), np.array(p2, float)
    n = max(48, int(round(np.linalg.norm(b - a))))
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def build(video, lines_path, cache):
    cache = Path(cache)
    lines = json.loads(Path(lines_path).read_text(encoding="utf-8"))
    key = json.dumps([[L["name"], L["p1"], L["p2"]] for L in lines])
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if str(z["key"]) == key:
            return {L["name"]: z[L["name"]] for L in lines}, float(z["fps"]), lines
    cap = cv2.VideoCapture(str(video)); fps = cap.get(cv2.CAP_PROP_FPS)
    pts = {L["name"]: sample_points(L["p1"], L["p2"]) for L in lines}; cols = {k: [] for k in pts}
    while True:
        ok, f = cap.read()
        if not ok: break
        for k, ps in pts.items():
            cols[k].append(np.stack([f[max(int(y) - 1, 0):int(y) + 2, max(int(x) - 1, 0):int(x) + 2].reshape(-1, 3).mean(0) for x, y in ps]))
    strips = {k: np.stack(v, 1).astype(np.uint8) for k, v in cols.items()}  # (rows=선 위치, cols=프레임, 3)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, key=key, fps=fps, **strips)
    return strips, fps, lines
