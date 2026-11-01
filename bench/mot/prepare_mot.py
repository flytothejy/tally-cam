"""MOT17 보행자 시퀀스를 tally-cam 벤치 형식으로 바꾼다 (두 번째 도메인 이식 시험).

- 영상: img1/*.jpg → samples/mot/<seq>.mp4 (원래 fps)
- 기준선: 사람이 고르지 않고 자동으로 여섯 개 — 가로선 h40/h60/h80(폭 5~95%), 세로선 v30/v50/v70(높이 30~95%)
- 정답: MOT 정답 궤적(보행자, consider=1)의 발 위치(상자 바닥 중앙)가 선을 처음 지나는 순간 = 사건
  (사람 하나가 선 하나를 여러 번 오가도 1번만 — 계수기의 '선마다 ID당 1번'과 같은 정의). 사람 라벨 불필요.
- 결과: eval/gt/mot/<seq>.csv(라벨러 CSV 형식), counter/lines/mot/<seq>.json, bench/dataset_mot.json

  python3 bench/mot/prepare_mot.py --root /home/claude/datasets/MOT17/train
"""
import argparse, configparser, csv, json, subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
STATIC = {"MOT17-02", "MOT17-04", "MOT17-09"}   # 고정 카메라(나머지는 움직이는 카메라)


def cross(a, b, p, q):
    """선분 ab와 pq의 교차 → (ab 위 비율 s, pq 위 비율 u) 또는 None"""
    d = (b[0] - a[0]) * (q[1] - p[1]) - (b[1] - a[1]) * (q[0] - p[0])
    if d == 0: return None
    s = ((p[0] - a[0]) * (q[1] - p[1]) - (p[1] - a[1]) * (q[0] - p[0])) / d
    u = ((p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])) / d
    return (s, u) if 0 < s <= 1 and 0 <= u <= 1 else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
    clips = []
    for d in sorted(Path(a.root).glob("MOT17-*-FRCNN")):
        seq = d.name.replace("-FRCNN", ""); ini = configparser.ConfigParser(); ini.read(d / "seqinfo.ini"); S = ini["Sequence"]
        fps, W, H, N = float(S["frameRate"]), int(S["imWidth"]), int(S["imHeight"]), int(S["seqLength"])
        vid = ROOT / f"samples/mot/{seq}.mp4"; vid.parent.mkdir(parents=True, exist_ok=True)
        if not vid.exists():
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", str(d / "img1/%06d.jpg"),
                            "-c:v", "libx264", "-crf", "16", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(vid)], check=True)
        lines = [{"name": f"h{k}", "label": f"가로선 {k}%", "note": "자동", "p1": [round(0.05 * W), round(k / 100 * H)], "p2": [round(0.95 * W), round(k / 100 * H)]} for k in (40, 60, 80)] + \
                [{"name": f"v{k}", "label": f"세로선 {k}%", "note": "자동", "p1": [round(k / 100 * W), round(0.3 * H)], "p2": [round(k / 100 * W), round(0.95 * H)]} for k in (30, 50, 70)]
        tr = defaultdict(list)
        for r in csv.reader(open(d / "gt/gt.txt")):
            f, i, x, y, w, h, keep, cls = int(r[0]), int(r[1]), *map(float, r[2:6]), int(r[6]), int(r[7])
            if keep == 1 and cls == 1: tr[i].append((f, x + w / 2, y + h))
        ev = []
        for i, pts in tr.items():
            pts.sort()
            for L in lines:
                for (f0, x0, y0), (f1, x1, y1) in zip(pts, pts[1:]):
                    if f1 - f0 > 5: continue
                    c = cross((x0, y0), (x1, y1), L["p1"], L["p2"])
                    if c:
                        s, u = c; ft = f0 + s * (f1 - f0)
                        ev.append((L["name"], (ft - 1) / fps, int(round(ft)) - 1, u, x0 + s * (x1 - x0), y0 + s * (y1 - y0), i)); break
        ev.sort(key=lambda e: e[1])
        gt = ROOT / f"eval/gt/mot/{seq}.csv"; gt.parent.mkdir(parents=True, exist_ok=True)
        with open(gt, "w") as fo:
            fo.write("clip,event_id,route,class,t_gate_sec,frame,gate_pos,x,y,source,reviewed\n")
            for k, (ln, t, f, u, x, y, i) in enumerate(ev):
                fo.write(f"{seq},e{k + 1:04d},{ln},person,{t:.3f},{f},{u:.3f},{x:.1f},{y:.1f},mot_gt,1\n")
            fo.write("# reviewed_ranges: []\n")
        gp = ROOT / f"counter/lines/mot/{seq}.json"; gp.parent.mkdir(parents=True, exist_ok=True); gp.write_text(json.dumps(lines, ensure_ascii=False, indent=1))
        clips.append({"id": seq, "camera": seq, "video": f"samples/mot/{seq}.mp4", "gates": f"counter/lines/mot/{seq}.json", "gt": f"eval/gt/mot/{seq}.csv",
                      "split": "mot", "light": "day", "density": "crowd", "view": "static" if seq in STATIC else "moving", "target": "person",
                      "note": f"MOT17 정답 궤적에서 자동 생성, {N}프레임 {fps:g}fps, 사람 {len(tr)}명, 통과 {len(ev)}건"})
        print(seq, clips[-1]["view"], clips[-1]["note"], flush=True)
    (ROOT / "bench/dataset_mot.json").write_text(json.dumps({"version": "0.1", "note": "MOT17 train(FRCNN 폴더 영상) — 비상업 라이선스(CC BY-NC-SA 3.0), 영상은 저장소에 넣지 않음",
                                                             "tolerance_sec": 1.0, "clips": clips}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
