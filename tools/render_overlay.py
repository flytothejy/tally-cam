"""계수 결과를 방송용 영상으로 — 원본 위에 추적 상자·기준선·카운트 효과, 옆 패널에 숫자·차트가 쌓여 감.

  python3 bench/methods/track_count.py ... --out ev.json --dump trk.npz
  python3 tools/render_overlay.py --video samples/x.mp4 --events ev.json --trk trk.npz --gates gates.json \
      --t0 0 --t1 220 --title "서울 역삼역 테헤란로" --start "2026-11-03 09:06:00" --out out.mp4

화면 1920×1080: 위 띠(제목·시각) / 왼쪽 영상(원본을 키움) / 오른쪽 패널(방향별 숫자, 누적 그래프, 분당 막대, 차종 비율) / 아래 띠(출처).
"""
import argparse, json, math, subprocess, datetime as dt
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-{}.ttc"
WINFONT = {"Black": "malgunbd.ttf", "Bold": "malgunbd.ttf", "Medium": "malgun.ttf", "Regular": "malgun.ttf"}   # 윈도우: 맑은 고딕
def font(size, w="Bold"):
    for ww in (w, {"Black": "Bold", "Medium": "Regular"}.get(w, w), "Bold", "Regular"):   # 굵기 파일이 없으면 가까운 굵기로
        if Path(FONT.format(ww)).exists(): return ImageFont.truetype(FONT.format(ww), size, index=1)   # index 1 = KR
    for p in (Path("C:/Windows/Fonts") / WINFONT.get(w, "malgun.ttf"), Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")):
        if p.exists(): return ImageFont.truetype(str(p), size)
    return ImageFont.load_default(size)

KIND = {2: "승용", 3: "오토바이", 5: "버스", 7: "트럭", 0: "사람", 1: "자전거"}
BG, PANEL, CARD, GRID, TXT, SUB = (11, 14, 19), (17, 21, 28), (24, 29, 38), (44, 51, 64), (236, 240, 245), (138, 148, 163)
PALETTE = [(61, 214, 245), (255, 181, 71), (255, 95, 162), (140, 230, 120), (180, 140, 255)]
KCOL = {2: (210, 216, 226), 3: (140, 230, 120), 5: (120, 170, 255), 7: (255, 150, 90), 0: (255, 95, 162), 1: (255, 225, 110)}
ease = lambda x: 1 - (1 - min(max(x, 0), 1)) ** 3


def load_tracks(trk, f0, f1):
    """dump(f, id, x1, y1, x2, y2, conf, cls) → 프레임마다 상자(2프레임 간격 검출 사이는 선형 보간), 트랙별 다수결 종류."""
    T = trk[(trk[:, 0] >= f0 - 2) & (trk[:, 0] <= f1 + 2)]
    by, kind = {}, {}
    for tid in np.unique(T[:, 1]):
        t = T[T[:, 1] == tid]; t = t[np.argsort(t[:, 0])]
        v = {}
        for c, s in zip(t[:, 7].astype(int), t[:, 6]): v[c] = v.get(c, 0) + s
        kind[int(tid)] = max(v, key=v.get)
        for a, b in zip(t[:-1], t[1:]):
            if b[0] - a[0] > 6: continue
            for f in range(int(a[0]), int(b[0])):
                r = (f - a[0]) / (b[0] - a[0]); by.setdefault(f, []).append((int(tid), *(a[2:6] * (1 - r) + b[2:6] * r)))
        by.setdefault(int(t[-1, 0]), []).append((int(tid), *t[-1, 2:6]))
    return by, kind


def rounded(d, box, r, **kw): d.rounded_rectangle(box, r, **kw)


def main():
    ap = argparse.ArgumentParser()
    for k in ("video", "events", "trk", "gates", "out", "title"): ap.add_argument("--" + k, required=True)
    ap.add_argument("--t0", type=float, default=0.0); ap.add_argument("--t1", type=float, default=60.0)
    ap.add_argument("--start", default=None, help="영상 시작 시각(화면 시계용) 'YYYY-MM-DD HH:MM:SS'")
    ap.add_argument("--labels", default=None, help='선 이름 → 표시 이름 JSON (예: {"lane2":"강남역 방향"})')
    ap.add_argument("--source", default="영상: 경찰청 도시교통정보센터(UTIC) 제공")
    ap.add_argument("--note", default="")
    ap.add_argument("--crf", type=int, default=20)
    a = ap.parse_args()

    cap = cv2.VideoCapture(a.video); fps = cap.get(cv2.CAP_PROP_FPS); VW, VH = int(cap.get(3)), int(cap.get(4))
    f0, f1 = int(round(a.t0 * fps)), int(round(a.t1 * fps)); cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    gates = json.loads(Path(a.gates).read_text(encoding="utf-8"))
    names = json.loads(a.labels) if a.labels else {}
    for i, L in enumerate(gates):
        L["col"] = PALETTE[i % len(PALETTE)] if L.get("target") != "person" else PALETTE[2]
        L["show"] = names.get(L["name"], L.get("label", L["name"]))
    vi = 0
    for L in gates:   # 차 선은 파랑·주황 순서, 사람 선은 분홍
        if L.get("target") != "person": L["col"] = PALETTE[[0, 1, 3, 4][vi % 4]]; vi += 1
    G = {L["name"]: L for L in gates}
    ev = [e for e in json.loads(Path(a.events).read_text(encoding="utf-8"))["events"] if a.t0 <= e["t"] < a.t1]
    for e in ev: e["f"] = int(round(e["t"] * fps))
    z = np.load(a.trk); boxes, tkind = load_tracks(z["trk"], f0, f1)
    counted_at = {}   # 트랙 → (프레임, 선)
    for e in ev: counted_at.setdefault(e["id"], (e["f"], e["line"]))

    W, H, TOP, BOT = 1920, 1080, 64, 56
    S = min(1440 / VW, (H - TOP - BOT) / VH); DW, DH = int(VW * S), int(VH * S)
    OX, OY = 0 + (1440 - DW) // 2, TOP + (H - TOP - BOT - DH) // 2
    PX = 1440; PW = W - PX
    F = {k: font(*v) for k, v in {"title": (30,), "clock": (26, "Medium"), "small": (18, "Medium"), "tiny": (15, "Medium"), "big": (64, "Black"),
                                  "mid": (24,), "chip": (17,), "pop": (30, "Black"), "label": (20,)}.items()}
    start = dt.datetime.strptime(a.start, "%Y-%m-%d %H:%M:%S") if a.start else None

    # 고정 배경(패널 틀)
    base = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(base)
    d.rectangle((PX, 0, W, H), fill=PANEL)
    d.rectangle((0, 0, W, TOP), fill=(8, 10, 14)); d.rectangle((0, H - BOT, W, H), fill=(8, 10, 14))
    d.text((24, TOP // 2), "tally-cam", font=F["title"], fill=(255, 196, 64), anchor="lm")
    d.text((190, TOP // 2), a.title, font=F["title"], fill=TXT, anchor="lm")
    d.text((24, H - BOT // 2), a.source, font=F["small"], fill=SUB, anchor="lm")
    if a.note: d.text((W - 24, H - BOT // 2), a.note, font=F["small"], fill=SUB, anchor="rm")

    lines = [L for L in gates if L.get('target') != 'person'] + [L for L in gates if L.get('target') == 'person']; nl = len(lines)   # 차 먼저
    CARD_Y, CARD_H, CG = TOP + 14, 108, 10
    CH0 = CARD_Y + nl * (CARD_H + CG) + 50; CH1 = CH0 + 150
    BAR_Y = CH1 + 70; BAR_H = 110
    MIX_Y = BAR_Y + BAR_H + 38
    dur = a.t1 - a.t0
    for i, L in enumerate(lines):
        y = CARD_Y + i * (CARD_H + CG); rounded(d, (PX + 20, y, W - 20, y + CARD_H), 14, fill=CARD)
        d.rounded_rectangle((PX + 20, y, PX + 28, y + CARD_H), 4, fill=L["col"])
        d.text((PX + 44, y + 12), L["show"], font=F["mid"], fill=TXT)
        d.text((PX + 44, y + 44), "보행자" if L.get("target") == "person" else "차량", font=F["tiny"], fill=SUB)
    def frame_chart(y, h, title):
        d.text((PX + 24, y - 46), title, font=F["mid"], fill=TXT)
        for k in range(5): yy = y + h - h * k / 4; d.line((PX + 70, yy, W - 28, yy), fill=GRID, width=1)
    frame_chart(CH0, CH1 - CH0, "누적 통과")
    frame_chart(BAR_Y, BAR_H, "30초마다 통과한 차")
    d.text((PX + 24, MIX_Y - 4), "차종 비율", font=F["mid"], fill=TXT)

    # 분당이 아니라 30초 묶음(짧은 클립에서도 막대가 여러 개 쌓이게)
    BIN = 30.0; nb = int(math.ceil(dur / BIN))
    vlines = [L for L in lines if L.get("target") != "person"]
    tot_final = max(1, max(sum(1 for e in ev if e["line"] == L["name"]) for L in lines))
    ymax_cum = int(math.ceil(tot_final / 40.0) * 40)
    bins_final = max([sum(1 for e in ev if e["line"] == L["name"] and int((e["t"] - a.t0) // BIN) == b) for L in vlines for b in range(nb)] + [1])
    ymax_bin = int(math.ceil(bins_final / 20.0) * 20)
    for k in range(5):
        d.text((PX + 62, CH0 + (CH1 - CH0) * (1 - k / 4)), str(ymax_cum * k // 4), font=F["tiny"], fill=SUB, anchor="rm")
        d.text((PX + 62, BAR_Y + BAR_H * (1 - k / 4)), str(ymax_bin * k // 4), font=F["tiny"], fill=SUB, anchor="rm")
    cx0, cx1 = PX + 76, W - 32
    for b in range(nb + 1):
        if b % 2 == 0 or nb <= 8:
            xx = cx0 + (cx1 - cx0) * b / nb; d.text((xx, BAR_Y + BAR_H + 8), f"{int(b * BIN // 60)}:{int(b * BIN % 60):02d}", font=F["tiny"], fill=SUB, anchor="mt")

    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
                             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(a.crf),
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", a.out], stdin=subprocess.PIPE)
    sx = lambda x: OX + x * S; sy = lambda y: OY + y * S
    shown = {L["name"]: 0.0 for L in lines}   # 숫자가 굴러 올라가는 표시값
    for fi in range(f0, f1):
        ok, fr = cap.read()
        if not ok: break
        t = fi / fps
        img = base.copy()
        vid = cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), (DW, DH), interpolation=cv2.INTER_CUBIC)
        img.paste(Image.fromarray(vid), (OX, OY))
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); o = ImageDraw.Draw(ov)
        past = [e for e in ev if e["f"] <= fi]
        recent = [e for e in past if fi - e["f"] < fps * 1.2]

        # 기준선: 반투명 띠 + 최근 통과 때 밝게
        for L in lines:
            p1, p2 = (sx(L["p1"][0]), sy(L["p1"][1])), (sx(L["p2"][0]), sy(L["p2"][1]))
            hot = max([1 - (fi - e["f"]) / (fps * 0.5) for e in recent if e["line"] == L["name"]] + [0])
            c = L["col"]; o.line((p1, p2), fill=c + (int(70 + 150 * hot),), width=int(10 + 8 * hot)); o.line((p1, p2), fill=(255, 255, 255, int(120 + 135 * hot)), width=2)

        # 추적 상자: 아직 안 센 건 흰 모서리, 센 건 선 색으로 채움 + 종류 칩
        for tid, x1, y1, x2, y2 in boxes.get(fi, []):
            k = tkind.get(tid, 2)
            if k not in KIND or (y2 - y1) < 12: continue   # 멀리 있는 아주 작은 상자는 생략(화면이 어지러움)
            X1, Y1, X2, Y2 = sx(x1), sy(y1), sx(x2), sy(y2)
            ca = counted_at.get(tid)
            if ca and ca[0] <= fi:
                c = G[ca[1]]["col"]; age = (fi - ca[0]) / fps; pulse = max(0, 1 - age / 0.6)
                fade = 1.0 if age < 3 else max(0.35, 1 - (age - 3) / 2)
                o.rounded_rectangle((X1, Y1, X2, Y2), 6, outline=c + (int(255 * fade),), width=3 if age < 3 else 2, fill=c + (int((40 + 90 * pulse) * fade),))
                if age < 2.5:
                    tw = o.textlength(KIND[k], font=F["tiny"]) + 12
                    o.rounded_rectangle((X1, Y1 - 22, X1 + tw, Y1 - 2), 5, fill=c + (230,)); o.text((X1 + 6, Y1 - 12), KIND[k], font=F["tiny"], fill=(10, 12, 16, 255), anchor="lm")
            else:
                al = 150 if k != 0 else 90; L_ = max(6, min(X2 - X1, Y2 - Y1) * 0.28); w = 2
                for (px, py, dx, dy) in ((X1, Y1, 1, 1), (X2, Y1, -1, 1), (X1, Y2, 1, -1), (X2, Y2, -1, -1)):
                    o.line((px, py, px + dx * L_, py), fill=(255, 255, 255, al), width=w); o.line((px, py, px, py + dy * L_), fill=(255, 255, 255, al), width=w)

        # 기준선 이름표(상자 위에): 선의 왼쪽 끝 위
        for L in lines:
            p1, p2 = (sx(L["p1"][0]), sy(L["p1"][1])), (sx(L["p2"][0]), sy(L["p2"][1])); q = min(p1, p2); c = L["col"]
            tw = o.textlength(L["show"], font=F["chip"]) + 18
            o.rounded_rectangle((q[0], q[1] - 34, q[0] + tw, q[1] - 8), 8, fill=(10, 12, 16, 200)); o.text((q[0] + 9, q[1] - 21), L["show"], font=F["chip"], fill=c + (255,), anchor="lm")

        # 통과 순간: 퍼지는 고리 + 떠오르는 +1
        for e in recent:
            age = (fi - e["f"]) / fps; c = G[e["line"]]["col"]; X, Y = sx(e["pt"][0]), sy(e["pt"][1])
            r = 10 + 46 * ease(age / 0.7); al = int(255 * max(0, 1 - age / 0.7))
            o.ellipse((X - r, Y - r, X + r, Y + r), outline=c + (al,), width=4)
            if age < 1.2:
                up = 70 * ease(age / 1.2); al2 = int(255 * (1 - max(0, age - 0.6) / 0.6)); txt = f"+1 {KIND.get(e['kind'], '')}"
                tw = o.textlength(txt, font=F["label"]) + 20
                o.rounded_rectangle((X - tw / 2, Y - 46 - up, X + tw / 2, Y - 14 - up), 10, fill=c + (int(al2 * 0.92),))
                o.text((X, Y - 30 - up), txt, font=F["label"], fill=(10, 12, 16, al2), anchor="mm")
        img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"); d = ImageDraw.Draw(img)

        # 시계
        clock = (start + dt.timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S") if start else f"{t:6.1f}s"
        d.ellipse((1440 - 250, TOP // 2 - 7, 1440 - 236, TOP // 2 + 7), fill=(255, 70, 70) if int(t * 2) % 2 == 0 else (120, 30, 30))
        d.text((1440 - 226, TOP // 2), "REPLAY", font=F["small"], fill=(255, 120, 120), anchor="lm")
        d.text((W - 24, TOP // 2), clock, font=F["clock"], fill=TXT, anchor="rm")

        # 카드: 숫자 굴러 올라가기 + 방금 센 카드 번쩍
        for i, L in enumerate(lines):
            y = CARD_Y + i * (CARD_H + CG); mine = [e for e in past if e["line"] == L["name"]]; n = len(mine)
            shown[L["name"]] += (n - shown[L["name"]]) * 0.35
            if abs(n - shown[L["name"]]) < 0.02: shown[L["name"]] = n
            flash = max([1 - (fi - e["f"]) / (fps * 0.4) for e in mine if fi - e["f"] < fps * 0.4] + [0])
            if flash > 0:
                d.rounded_rectangle((PX + 20, y, W - 20, y + CARD_H), 14, outline=L["col"], width=int(2 + 3 * flash))
            d.text((W - 40, y + 38), f"{int(round(shown[L['name']])):,}", font=F["big"], fill=TXT, anchor="rm")
            if L.get("target") != "person":   # 종류 칩
                kc = {}
                for e in mine: kc[e["kind"]] = kc.get(e["kind"], 0) + 1
                xx = PX + 44
                for k in (2, 5, 7, 3):
                    if not kc.get(k): continue
                    s = f"{KIND[k]} {kc[k]}"; tw = d.textlength(s, font=F["tiny"]) + 16
                    d.rounded_rectangle((xx, y + 74, xx + tw, y + 98), 7, fill=(36, 42, 54)); d.ellipse((xx + 6, y + 83, xx + 12, y + 89), fill=KCOL[k])
                    d.text((xx + 15, y + 86), s, font=F["tiny"], fill=TXT, anchor="lm"); xx += tw + 6
            else:
                d.text((PX + 44, y + 86), f"양방향 합계 · 최근 1분 {sum(1 for e in mine if t - e['t'] < 60)}명", font=F["tiny"], fill=SUB, anchor="lm")

        # 누적 그래프: 지금까지 그려짐 + 끝점 빛남
        ch0, ch1 = CH0, CH1
        for L in lines:
            ts = [e["t"] - a.t0 for e in past if e["line"] == L["name"]]
            pts = [(cx0, ch1)]
            for k, tt in enumerate(ts): x = cx0 + (cx1 - cx0) * tt / dur; pts += [(x, ch1 - (ch1 - ch0) * k / ymax_cum), (x, ch1 - (ch1 - ch0) * (k + 1) / ymax_cum)]
            xn = cx0 + (cx1 - cx0) * (t - a.t0) / dur; yn = ch1 - (ch1 - ch0) * len(ts) / ymax_cum; pts.append((xn, yn))
            d.line(pts, fill=L["col"], width=3); d.ellipse((xn - 5, yn - 5, xn + 5, yn + 5), fill=L["col"])
        d.line((cx0 + (cx1 - cx0) * (t - a.t0) / dur, ch0, cx0 + (cx1 - cx0) * (t - a.t0) / dur, ch1), fill=(90, 100, 118), width=1)

        # 30초 막대: 지금 묶음은 차오르는 중
        bw = (cx1 - cx0) / nb; nv = max(1, len(vlines)); gap = 6
        for b in range(nb):
            lo = a.t0 + b * BIN
            if lo > t: break
            for j, L in enumerate(vlines):
                n = sum(1 for e in past if e["line"] == L["name"] and lo <= e["t"] < lo + BIN)
                x = cx0 + b * bw + gap + j * (bw - 2 * gap) / nv; w_ = (bw - 2 * gap) / nv - 3
                hgt = BAR_H * n / ymax_bin
                if hgt > 0: d.rounded_rectangle((x, BAR_Y + BAR_H - hgt, x + w_, BAR_Y + BAR_H), 4, fill=L["col"] if lo + BIN <= t else tuple(int(c * 0.7) for c in L["col"]))

        # 차종 비율 막대
        kc = {}
        for e in past:
            if G[e["line"]].get("target") != "person": kc[e["kind"]] = kc.get(e["kind"], 0) + 1
        tot = sum(kc.values()); bx0, bx1, by = PX + 24, W - 24, MIX_Y + 42
        d.rounded_rectangle((bx0, by, bx1, by + 22), 8, fill=(36, 42, 54))
        if tot:
            x = bx0
            for k in (2, 5, 7, 3):
                if not kc.get(k): continue
                w_ = (bx1 - bx0) * kc[k] / tot; d.rectangle((x, by, x + w_, by + 22), fill=KCOL[k]); x += w_
            xx = bx0
            for k in (2, 5, 7, 3):
                if not kc.get(k): continue
                s = f"{KIND[k]} {kc[k] * 100 / tot:.0f}%"; d.ellipse((xx, by + 36, xx + 10, by + 46), fill=KCOL[k]); d.text((xx + 15, by + 41), s, font=F["tiny"], fill=TXT, anchor="lm")
                xx += d.textlength(s, font=F["tiny"]) + 34
        proc.stdin.write(np.asarray(img).tobytes())
        if (fi - f0) % 300 == 0: print(f"{t:.0f}s", flush=True)
    proc.stdin.close(); proc.wait()
    print("저장:", a.out)


if __name__ == "__main__":
    main()
