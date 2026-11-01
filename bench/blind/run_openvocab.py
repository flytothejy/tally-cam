"""2단계(글자로 지정) 검출기로 세기 — 차량 dev 5클립 + MOT17 7개, 각 클립 뒤 절반만, N프레임마다 검출.
검출기 순서대로(gdino 전부 → owlv2 전부) 돌리고, 하나 끝날 때마다 채점 결과를 찍음(중간 결과 확보).
  python bench/blind/run_openvocab.py [--stride 3] [--backends gdino,owlv2] [--out-json /content/bl.json]
"""
import argparse, json, subprocess, sys
from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parent.parent.parent
ap = argparse.ArgumentParser(); ap.add_argument("--stride", type=int, default=3); ap.add_argument("--backends", default="gdino,owlv2")
ap.add_argument("--confs", default="0.2,0.3,0.4"); ap.add_argument("--out-json", default=""); a = ap.parse_args()
(ROOT / "bench/cache").mkdir(parents=True, exist_ok=True)
PROMPT = {"vehicle": "car. truck. bus.", "person": "person."}
SETS = [("bench/dataset.json", "dev", "vehicle", ["--agnostic", "--ext", "30", "--oneway"]),
        ("bench/dataset_mot.json", "mot", "person", ["--agnostic", "--anchor", "bottom"])]
py = sys.executable
for be in a.backends.split(","):
    for dsf, split, tgt, rule in SETS:
        if not (ROOT / dsf).exists(): print("없음:", dsf); continue
        tags = []
        for c in json.loads((ROOT / dsf).read_text())["clips"]:
            if c["split"] != split or not c["gt"]: continue
            cap = cv2.VideoCapture(str(ROOT / c["video"])); dur = cap.get(7) / cap.get(5)
            cache = ROOT / f"bench/cache/{c['id']}_{be}_s{a.stride}.npz"
            if not cache.exists():
                r = subprocess.run([py, str(ROOT / "bench/methods/detcache.py"), "--video", str(ROOT / c["video"]), "--backend", be, "--model", "x", "--prompt", PROMPT[tgt],
                                    "--conf", "0.1", "--from-seconds", str(dur / 2), "--stride", str(a.stride), "--out", str(cache)], capture_output=True, text=True)
                if r.returncode: print("검출 실패", c["id"], be, r.stderr[-800:], flush=True); continue
                print(c["id"], be, r.stdout.strip().splitlines()[-1], flush=True)
            for conf in a.confs.split(","):
                tag = ("BL-MOT-" if tgt == "person" else "BL-") + f"{be}-s{a.stride}-c{conf}"; tags.append(tag)
                subprocess.run([py, str(ROOT / "bench/methods/track_count.py"), "--cache", str(cache), "--lines", str(ROOT / c["gates"]),
                                "--out", str(ROOT / f"bench/preds/{tag}/{c['id']}.json"), "--conf", conf, "--stride", str(a.stride)] + rule, check=True, capture_output=True)
        cmd = [py, str(ROOT / "bench/analysis/score_half.py"), "--dataset", dsf, "--split", split] + sum([["--method", t] for t in dict.fromkeys(tags)], [])
        if a.out_json: cmd += ["--json", f"{a.out_json}.{be}.{tgt}.json"]
        print(subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout, flush=True)
