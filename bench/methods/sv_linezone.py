"""오픈소스 비교 대상: Roboflow supervision LineZone (기본 설정: 박스 네 모서리가 모두 넘어가야 셈).
검출은 같은 캐시, 추적은 Ultralytics ByteTrack 기본값. supervision 0.31부터 추적기가 빠져 있어 이렇게 조합.

  python3 bench/methods/sv_linezone.py --cache c.npz --lines counter/lines/x.json --out preds/sv-linezone/clip.json
"""
import argparse, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np, supervision as sv
from ultralytics.trackers.byte_tracker import BYTETracker
import sys; sys.path.insert(0, str(Path(__file__).parent))
from track_count import Dets

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True); ap.add_argument("--lines", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--conf", type=float, default=0.25, help="supervision 예제 기본에 맞춤")
ap.add_argument("--anchor", default="corners", help="corners(기본: 네 모서리) | center | bottom")
a = ap.parse_args()
z = np.load(a.cache, allow_pickle=True); det = z["det"]; F = int(z["frames"]); fps = float(z["fps"])
det = det[det[:, 5] >= a.conf]; starts = np.searchsorted(det[:, 0], np.arange(F + 1))
args = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25, track_buffer=30, match_thresh=0.8, fuse_score=True)
trk = BYTETracker(args)
lines = json.loads(Path(a.lines).read_text(encoding="utf-8"))
anch = {"center": [sv.Position.CENTER], "bottom": [sv.Position.BOTTOM_CENTER]}.get(a.anchor)
zones = {L["name"]: (sv.LineZone(start=sv.Point(*L["p1"]), end=sv.Point(*L["p2"]), triggering_anchors=anch) if anch else sv.LineZone(start=sv.Point(*L["p1"]), end=sv.Point(*L["p2"]))) for L in lines}
events = []
for f in range(F):
    d = det[starts[f]:starts[f + 1]]
    out = trk.update(Dets(d[:, 1:5].astype(np.float32), d[:, 5].astype(np.float32), d[:, 6].astype(np.float32)), None)
    if not len(out): continue
    dets = sv.Detections(xyxy=out[:, :4].astype(np.float32), confidence=out[:, 5].astype(np.float32), class_id=out[:, 6].astype(int), tracker_id=out[:, 4].astype(int))
    for name, zn in zones.items():
        cin, cout = zn.trigger(dets)
        for tid in np.r_[dets.tracker_id[cin], dets.tracker_id[cout]]:
            events.append({"t": round(f / fps, 3), "id": int(tid), "line": name})
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({"method": f"supervision LineZone (anchor={a.anchor}) + ByteTrack", "video_seconds": round(F / fps, 2),
                                   "infer_ms_avg": round(float(np.mean(z["det_ms"])), 1), "events": events}))
print(len(events), "events")
