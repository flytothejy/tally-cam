"""벤치마크용 CCTV 녹화.

API 목록은 세션(시간대)마다 한 번만 받고, 스트림은 동시 3개 이하로 받는다.
스트림 주소에는 서명이 들어 있으므로 어떤 출력에도 주소를 남기지 않는다.

  python3 scripts/record.py list            # 목록 1회 조회 → samples/campaign/<날짜>/<시각>/_list.json
  python3 scripts/record.py snap            # 카메라마다 한 프레임 + 모아 보기(sheet.jpg)
  python3 scripts/record.py rec --part 0/3  # 카메라를 3묶음으로 나눠 0번 묶음 녹화 (120초)
  python3 scripts/record.py retry           # 실패한 카메라 다시 녹화

세션 폴더는 가장 최근 list 결과를 쓴다(--session 으로 지정 가능).
"""
import argparse, json, re, subprocess, sys, time, urllib.parse, urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from snapshot import API, load_key  # noqa: E402

KST = timezone(timedelta(hours=9))
CAMP = ROOT / "samples" / "campaign"
CFG = json.loads((ROOT / "bench" / "cameras.json").read_text(encoding="utf-8"))
STAGGER, MAX_PAR = 15, 3  # 새 접속이 몰리면 ITS가 403으로 잠깐 막음 → 15초 간격


def clean(s):
    return re.sub(r"https?://\S+", "<url>", s or "")[-200:]


def latest_session():
    ss = sorted(CAMP.glob("*/*/_list.json"))
    if not ss:
        sys.exit("먼저 list를 실행하세요")
    return ss[-1].parent


def load_session(a):
    sess = Path(a.session) if a.session else latest_session()
    lst = {c["cctvname"]: c for c in json.loads((sess / "_list.json").read_text(encoding="utf-8"))}
    age = time.time() - (sess / "_list.json").stat().st_mtime
    if age > 110 * 60:
        sys.exit("목록이 110분보다 오래돼 스트림 주소가 만료됐을 수 있어요. list를 다시 실행하세요")
    return sess, lst


def manifest(sess):
    p = sess / "manifest.json"
    return p, (json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"session": sess.name, "clips": {}})


def cmd_list(a):
    b = CFG["bbox"]
    q = urllib.parse.urlencode({"apiKey": load_key(), "type": CFG["road_type"], "cctvType": 4, "minX": b["minX"], "maxX": b["maxX"],
                                "minY": b["minY"], "maxY": b["maxY"], "getType": "json"})
    data = json.load(urllib.request.urlopen(f"{API}?{q}", timeout=30))["response"]["data"]
    if isinstance(data, dict):
        data = [data]
    now = datetime.now(KST)
    sess = CAMP / now.strftime("%Y%m%d") / now.strftime("%H%M")
    sess.mkdir(parents=True, exist_ok=True)
    (sess / "_list.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    names = {c["cctvname"] for c in data}
    miss = [c["name"] for c in CFG["cameras"] if c["name"] not in names]
    print(f"session {sess.relative_to(ROOT)} · 목록 {len(data)}개 · 빠진 카메라 {miss or '없음'}")


def run_group(cams, lst, sess, seconds, snap=False):
    """cams를 STAGGER초 간격으로 시작, 동시 MAX_PAR개 이하."""
    procs, results = [], {}
    for c in cams:
        src = lst.get(c["name"])
        if not src:
            results[c["slug"]] = {"ok": False, "err": "목록에 없음"}
            continue
        while len([p for p in procs if p[1].poll() is None]) >= MAX_PAR:
            time.sleep(1)
        out = sess / (f"{c['slug']}.jpg" if snap else f"{c['slug']}.mp4")
        args = ["ffmpeg", "-v", "error", "-y", "-i", src["cctvurl"]] + (["-frames:v", "1", "-q:v", "3"] if snap else ["-t", str(seconds), "-c", "copy", "-an"]) + [str(out)]
        start = datetime.now(KST).isoformat(timespec="seconds")
        procs.append((c, subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True), out, start))
        time.sleep(STAGGER)
    for c, p, out, start in procs:
        try:
            _, err = p.communicate(timeout=seconds + 40)
        except subprocess.TimeoutExpired:
            p.kill(); err = "timeout"
        ok = p.returncode == 0 and out.exists() and out.stat().st_size > 10000
        results[c["slug"]] = {"ok": ok, "start_kst": start, "seconds": None if snap else seconds, "bytes": out.stat().st_size if out.exists() else 0,
                              "err": "" if ok else ("403" if "403" in (err or "") else clean(err))}
    return results


def cmd_snap(a):
    sess, lst = load_session(a)
    res = run_group(CFG["cameras"], lst, sess, 30, snap=True)
    import cv2, numpy as np
    tiles = []
    for c in CFG["cameras"]:
        im = cv2.imread(str(sess / f"{c['slug']}.jpg"))
        im = cv2.resize(im, (360, 240)) if im is not None else np.zeros((240, 360, 3), np.uint8)
        cv2.putText(im, c["slug"], (8, 26), 0, 0.7, (0, 255, 255), 2); tiles.append(im)
    while len(tiles) % 4:
        tiles.append(np.zeros((240, 360, 3), np.uint8))
    cv2.imwrite(str(sess / "sheet.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    print(json.dumps({k: (v["ok"], v["err"]) for k, v in res.items()}, ensure_ascii=False))


def cmd_rec(a, cams=None):
    sess, lst = load_session(a)
    if cams is None:
        k, n = map(int, a.part.split("/"))
        cams = [c for i, c in enumerate(CFG["cameras"]) if i % n == k]
    res = run_group(cams, lst, sess, a.seconds)
    mp, m = manifest(sess)
    m["clips"].update(res)
    mp.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: (v["ok"], v["bytes"] // 1000, v["err"]) for k, v in res.items()}, ensure_ascii=False))


def cmd_retry(a):
    sess, _ = load_session(a)
    _, m = manifest(sess)
    bad = [c for c in CFG["cameras"] if not m["clips"].get(c["slug"], {}).get("ok")]
    if not bad:
        print("다시 받을 카메라 없음"); return
    time.sleep(a.wait)
    cmd_rec(a, bad[:MAX_PAR])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "snap", "rec", "retry"])
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--wait", type=int, default=20, help="retry 전에 기다릴 초")
    ap.add_argument("--session", default="")
    a = ap.parse_args()
    {"list": cmd_list, "snap": cmd_snap, "rec": cmd_rec, "retry": cmd_retry}[a.cmd](a)
