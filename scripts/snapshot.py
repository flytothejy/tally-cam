"""ITS 국가교통정보센터 CCTV 스냅샷 수집.

API 1회 호출로 bbox 안의 CCTV 목록을 받아, config.json에 지정한 카메라만
ffmpeg로 한 프레임씩 저장한다. 스트림 URL에는 API 키가 들어 있으므로
어떤 출력에도 URL을 남기지 않는다.
"""
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KST = timezone(timedelta(hours=9))
API = "https://openapi.its.go.kr:9443/cctvInfo"


def load_key() -> str:
    key = os.environ.get("ITS_API_KEY", "").strip()
    if not key:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith("ITS_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        sys.exit("ITS_API_KEY가 없습니다 (환경변수 또는 .env)")
    return key


def fetch_cctv_list(key: str, cfg: dict) -> list:
    b = cfg["bbox"]
    q = urllib.parse.urlencode({
        "apiKey": key, "type": cfg["road_type"], "cctvType": 4,
        "minX": b["minX"], "maxX": b["maxX"], "minY": b["minY"], "maxY": b["maxY"],
        "getType": "json",
    })
    with urllib.request.urlopen(f"{API}?{q}", timeout=30) as r:
        return json.load(r)["response"]["data"]


def grab_frame(url: str, out: Path) -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", url, "-frames:v", "1", "-q:v", "3", str(out)],
            check=True, timeout=90, capture_output=True,
        )
        return out.exists() and out.stat().st_size > 0
    except Exception:
        return False


def main() -> None:
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    key = load_key()
    now = datetime.now(KST)
    stamp = now.strftime("%Y%m%d-%H%M")

    try:
        listing = {c["cctvname"]: c for c in fetch_cctv_list(key, cfg)}
    except Exception as e:
        sys.exit(f"API 호출 실패: {type(e).__name__}")

    index_path = DATA / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"cameras": {}}
    ok_count = 0

    for cam in cfg["cameras"]:
        src = listing.get(cam["name"])
        entry = index["cameras"].setdefault(cam["slug"], {"history": []})
        entry.update({"label": cam["label"], "name": cam["name"]})
        if not src:
            print(f"[skip] 목록에 없음: {cam['name']}")
            continue
        entry["coord"] = [float(src["coordy"]), float(src["coordx"])]
        cam_dir = DATA / cam["slug"]
        cam_dir.mkdir(parents=True, exist_ok=True)
        shot = cam_dir / f"{stamp}.jpg"
        if not grab_frame(src["cctvurl"], shot):
            print(f"[fail] 프레임 저장 실패: {cam['label']}")
            continue
        (cam_dir / "latest.jpg").write_bytes(shot.read_bytes())
        entry["updated"] = now.isoformat(timespec="minutes")
        entry["history"] = ([shot.name] + [h for h in entry["history"] if h != shot.name])
        for old in entry["history"][cfg["keep_history"]:]:
            (cam_dir / old).unlink(missing_ok=True)
        entry["history"] = entry["history"][: cfg["keep_history"]]
        ok_count += 1
        print(f"[ok] {cam['label']}")

    index["updated"] = now.isoformat(timespec="minutes")
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    if ok_count == 0:
        sys.exit("저장된 스냅샷이 없습니다")


if __name__ == "__main__":
    main()
