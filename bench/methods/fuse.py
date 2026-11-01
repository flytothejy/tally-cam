"""사건 단위 늦은 융합: 두 방법의 사건을 같은 방향 ±tol초로 1:1 매칭해 합친다.

  둘 다 본 사건 → 1개로 남김(시각은 A)
  A만 본 사건 → --keep-a 면 남김
  B만 본 사건 → 점수(score)가 --b-hi 이상이면 남김 (점수 없는 B는 --keep-b 로)
결과의 'agree' 필드: 2(둘 다), 1(한쪽만) → 불일치 사건 = 클라우드 확인 후보.

  python3 bench/methods/fuse.py --a A7-rfdetrS-c0.25 --b E1-stripnet --out bench/preds/F-union --keep-a --b-hi 0.3
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from score import match, direction


def fuse(A, B, tol, keep_a, b_hi, keep_b):
    out = []
    for d in sorted({direction(e["line"]) for e in A + B}):
        a = sorted([e for e in A if direction(e["line"]) == d], key=lambda e: e["t"]); b = sorted([e for e in B if direction(e["line"]) == d], key=lambda e: e["t"])
        m = match([e["t"] for e in a], [e["t"] for e in b], tol); ma = {i for i, _, _ in m}; mb = {j for _, j, _ in m}
        out += [dict(a[i], agree=2) for i, _, _ in m]
        if keep_a: out += [dict(e, agree=1, src="a") for i, e in enumerate(a) if i not in ma]
        out += [dict(e, agree=1, src="b") for j, e in enumerate(b) if j not in mb and (keep_b or e.get("score", 0) >= b_hi)]
    return sorted(out, key=lambda e: e["t"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tol", type=float, default=1.0); ap.add_argument("--keep-a", action="store_true"); ap.add_argument("--keep-b", action="store_true")
    ap.add_argument("--b-hi", type=float, default=2.0)
    a = ap.parse_args(); out = ROOT / "bench/preds" / a.out; out.mkdir(parents=True, exist_ok=True)
    for pa in (ROOT / "bench/preds" / a.a).glob("*.json"):
        pb = ROOT / "bench/preds" / a.b / pa.name
        if not pb.exists(): continue
        A, B = json.loads(pa.read_text()), json.loads(pb.read_text())
        ev = fuse(A["events"], B["events"], a.tol, a.keep_a, a.b_hi, a.keep_b)
        (out / pa.name).write_text(json.dumps({"method": vars(a), "video_seconds": A.get("video_seconds"), "events": ev,
                                               "disagree_rate": round(sum(e["agree"] == 1 for e in ev) / max(len(ev), 1), 3)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
