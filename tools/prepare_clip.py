"""라벨러용 클립 준비.

영상과 카메라의 기준선 설정(counter/lines/<camera>.json)을 받아
라벨러가 읽는 파일을 만든다.

  tools/labeler/clips/<clip_id>/
    clip.mp4            웹 재생용 영상 (탐색이 빠르도록 키프레임 15장 간격)
    strip_<lane>.png    기준선별 시공간 띠 (가로 = 프레임, 세로 = 기준선 위 위치)
    cands.json          자동 라벨 [{lane, t, y}] — 밝기 덩어리 기반, 검출기와 무관
    meta.json           fps, 프레임 수, 기준선, 표시 정보
  tools/labeler/clips/index.json   클립 목록 (갱신)

  python3 tools/prepare_clip.py --video samples/x.mp4 --camera dongtan-jct \
      --clip-id dongtan-day-01 --title "경부선 동탄분기점 · 주간 120초 · 2026-10-27 13:10"
"""
import argparse, json, subprocess
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def sample_points(p1, p2):
    a, b = np.array(p1, float), np.array(p2, float)
    n = max(48, int(round(np.linalg.norm(b - a))))
    return [a + (b - a) * i / (n - 1) for i in range(n)]


def blob_candidates(strip, fps, thresh=40, min_area=60):
    img = strip.astype(np.float32)
    bg = np.median(img, axis=1, keepdims=True)
    d = np.abs(img - bg).max(axis=2)
    m = (cv2.GaussianBlur(d, (5, 5), 0) > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(m)
    out = [(round(float(cent[i][0]) / fps, 3), round(float(cent[i][1]) / img.shape[0], 3)) for i in range(1, n) if stats[i][4] >= min_area]
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--camera', required=True, help='counter/lines/<camera>.json')
    ap.add_argument('--clip-id', required=True)
    ap.add_argument('--title', required=True)
    ap.add_argument('--out', default=str(ROOT / 'tools/labeler/clips'))
    a = ap.parse_args()

    lanes = json.loads((ROOT / f'counter/lines/{a.camera}.json').read_text(encoding='utf-8'))
    out = Path(a.out) / a.clip_id
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pts = {L['name']: sample_points(L['p1'], L['p2']) for L in lanes}
    cols = {L['name']: [] for L in lanes}
    frames = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        for name, ps in pts.items():
            cols[name].append(np.stack([f[max(int(y) - 1, 0):int(y) + 2, max(int(x) - 1, 0):int(x) + 2].reshape(-1, 3).mean(0) for x, y in ps]))
        frames += 1
    cap.release()

    cands, meta_lanes = [], []
    for L in lanes:
        strip = np.stack(cols[L['name']], axis=1).astype(np.uint8)
        cv2.imwrite(str(out / f"strip_{L['name']}.png"), strip)
        cands += [{'lane': L['name'], 't': t, 'y': y} for t, y in blob_candidates(strip, fps)]
        meta_lanes.append({'name': L['name'], 'label': L.get('label', L['name']), 'note': L.get('note', ''),
                           'p1': L['p1'], 'p2': L['p2'], 'height': int(min(180, max(110, strip.shape[0] * 1.5)))})
    (out / 'cands.json').write_text(json.dumps(cands, separators=(',', ':')), encoding='utf-8')
    meta = {'id': a.clip_id, 'title': a.title, 'camera': a.camera, 'fps': round(fps, 4), 'frames': frames,
            'width': w, 'height': h, 'lanes': meta_lanes}
    (out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', a.video, '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '33',
                    '-g', '15', '-an', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'clip.mp4')], check=True)

    idx_p = Path(a.out) / 'index.json'
    idx = json.loads(idx_p.read_text(encoding='utf-8')) if idx_p.exists() else []
    idx = [c for c in idx if c['id'] != a.clip_id] + [{'id': a.clip_id, 'title': a.title}]
    idx_p.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'{a.clip_id}: {frames} frames, {len(cands)} auto labels')


if __name__ == '__main__':
    main()
