"""경찰청 도시교통정보센터(UTIC) 개방데이터 CCTV — 시내 교차로 영상 받기.

인증: 키(.env의 UTIC_API_KEY) + 신청한 공인 IP 대역에서만 동작 → 집 PC(또는 같은 공유기의 엣지 장비)에서 실행.
준수: 신청 목적 외 사용 금지, 화면에 "경찰청 도시교통정보센터(UTIC) 제공" 표기,
      같은 주소 1초 5회 이상·오류 반복 접속 시 IP 차단 → 요청 사이 최소 1.5초, 목록은 하루 1회까지만 새로 받음.
키와 스트림 주소는 출력하지 않음(clean).

  python scripts/utic.py list [--refresh] [--center 서울교통정보센터] [--grep 사거리]
  python scripts/utic.py info L010010
  python scripts/utic.py rec L010010 --seconds 120 --out samples/utic/gangnam_terminal.mp4
"""
import argparse, json, re, subprocess, sys, time, urllib.parse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://www.utic.go.kr"
LIST_HTML = ROOT / "data/utic/list_raw.html"; LIST_JSON = ROOT / "data/utic/list.json"
_last = [0.0]


def key():
    for l in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if l.startswith("UTIC_API_KEY="): return l.split("=", 1)[1].strip()
    sys.exit("UTIC_API_KEY가 .env에 없음")


def clean(s):
    return re.sub(r"key=[^&\"' ]+", "key=<KEY>", str(s)).replace(key(), "<KEY>")


def get(url, referer=None, xhr=False):
    wait = 1.5 - (time.time() - _last[0])
    if wait > 0: time.sleep(wait)
    req = urllib.request.Request(url, headers={"User-Agent": "tally-cam research", **({"Referer": referer} if referer else {}),
                                               **({"X-Requested-With": "XMLHttpRequest"} if xhr else {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        _last[0] = time.time(); return r.read().decode("utf-8", "ignore")


def load_list(refresh=False):
    if refresh or not LIST_JSON.exists():   # 목록은 하루 1회 이하로만 새로 받기
        LIST_HTML.parent.mkdir(parents=True, exist_ok=True)
        s = get(f"{BASE}/guide/cctvOpenData.do?key={key()}"); LIST_HTML.write_text(s, encoding="utf-8")
        rows = re.findall(r"test\('([^']+)'\)\">([^<]*)</a></td>\s*<td[^>]*>\s*<a[^>]*>([^<]*)</a>", s)
        LIST_JSON.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return json.loads(LIST_JSON.read_text(encoding="utf-8"))


def info(cctvid):
    return json.loads(get(f"{BASE}/map/getCctvInfoById.do?cctvId={cctvid}", referer=f"{BASE}/guide/cctvOpenData.do", xhr=True))


def kind_of(i):
    p = i["CCTVID"][:3]
    return {"L01": "Seoul", "L02": "N", "L03": "O", "L04": "P", "L08": "d"}.get(p, i.get("KIND"))


def stream_url(i):
    """플레이어 페이지에서 HLS 주소를 찾음. 기관마다 형식이 달라 HLS가 없으면 None."""
    q = lambda v: urllib.parse.quote(urllib.parse.quote(str(v)))
    url = (f"{BASE}/jsp/map/openDataCctvStream.jsp?key={key()}&cctvid={i['CCTVID']}&cctvName={q(i.get('CCTVNAME', ''))}&kind={kind_of(i)}"
           f"&cctvip={i.get('CCTVIP', 'undefined')}&cctvch={i.get('CH', 'undefined')}&id={i.get('ID', 'undefined')}"
           f"&cctvpasswd={i.get('PASSWD', 'undefined')}&cctvport={i.get('PORT', 'undefined')}")
    s = get(url, referer=f"{BASE}/guide/cctvOpenData.do?key={key()}")
    cands = [u for u in re.findall(r"https?://[^\"' <>]+\.m3u8", s) if "undefined" not in u]
    return cands[0] if cands else None


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--refresh", action="store_true"); p.add_argument("--center"); p.add_argument("--grep")
    p = sub.add_parser("info"); p.add_argument("cctvid")
    p = sub.add_parser("snap", help="여러 카메라 화면 한 장씩(후보 고르기). ids 파일 한 줄에 ID 하나"); p.add_argument("ids"); p.add_argument("--outdir", default="samples/utic/snaps")
    p = sub.add_parser("rec"); p.add_argument("cctvid"); p.add_argument("--seconds", type=int, default=120); p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "list":
        rows = load_list(a.refresh)
        rows = [r for r in rows if (not a.center or r[2] == a.center) and (not a.grep or a.grep in r[1])]
        for r in rows: print("\t".join(r))
        print(f"{len(rows)}개", file=sys.stderr)
    elif a.cmd == "info":
        i = info(a.cctvid); print(json.dumps({k: v for k, v in i.items() if k not in ("PASSWD",)}, ensure_ascii=False))
        print("HLS:", "있음" if stream_url(i) else "없음(기관별 플레이어 형식)")
    elif a.cmd == "snap":
        out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True); meta = {}
        mp = out / "meta.json"
        if mp.exists(): meta = json.loads(mp.read_text(encoding="utf-8"))
        for cid in [l.strip() for l in Path(a.ids).read_text().splitlines() if l.strip()]:
            if (out / f"{cid}.jpg").exists(): continue
            try:
                i = info(cid); u = stream_url(i)
                if not u: meta[cid] = {"name": i.get("CCTVNAME"), "ok": False}; continue
                r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", u, "-frames:v", "1", str(out / f"{cid}.jpg")], capture_output=True, timeout=40)
                meta[cid] = {"name": i.get("CCTVNAME"), "center": i.get("CENTERNAME"), "x": i.get("XCOORD"), "y": i.get("YCOORD"), "ok": r.returncode == 0}
            except Exception as e:
                meta[cid] = {"ok": False, "err": clean(e)[:120]}
            mp.write_text(json.dumps(meta, ensure_ascii=False, indent=0), encoding="utf-8"); print(cid, meta[cid].get("name"), meta[cid]["ok"], flush=True)
    else:
        i = info(a.cctvid); u = stream_url(i)
        if not u: sys.exit("HLS 주소를 찾지 못함")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", u, "-t", str(a.seconds), "-c", "copy", a.out], capture_output=True, text=True)
        print(clean(r.stderr[-500:]) if r.returncode else f"저장: {a.out} ({i.get('CCTVNAME')})")


if __name__ == "__main__":
    main()
