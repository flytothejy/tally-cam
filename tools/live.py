"""라이브 계수 방송 — CCTV 스트림을 받아 실시간으로 검출·추적·계수하고, 방송 화면(상자·기준선·+1 효과·숫자·차트)을 내보냄.

집 PC(UTIC 신청 IP)에서 실행:
  python tools/live.py --utic L010426 --gates counter/lines/utic/L010426_v2.json --title "서울 역삼역 앞 테헤란로" \
      --labels '{"lane2":"↓ 강남역 방향","lane3":"↑ 선릉역 방향","cw":"횡단보도"}' --out preview
  --out preview          화면 창으로만 보기(밖으로 내보내지 않음, q로 끝)
  --out live.mp4         파일로 저장
  --out rtmp             유튜브 라이브: 환경변수 YT_STREAM_KEY 필요(키는 출력하지 않음)
녹화 파일로 시험:  --file samples/utic/am0906_L010426.mp4  (실시간 속도로 흘려 보냄)

구조: 받는 스레드(최신 프레임만 유지) / 검출·추적·계수 스레드(검출기가 낼 수 있는 속도로) / 그리기 루프(--fps로 일정하게).
검출이 느리면(CPU) 상자가 조금 늦게 따라오고 추적이 끊기기 쉬움 → GPU면 small, CPU면 --model nano 권장.
"""
import argparse, json, math, os, subprocess, sys, threading, time, datetime as dt
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "bench/methods")); sys.path.insert(0, str(ROOT / "scripts"))
from render_overlay import font, KIND, BG, PANEL, CARD, GRID, TXT, SUB, PALETTE, KCOL, ease   # noqa: E402
from track_count import Dets, side, cross_pt, BYTETracker   # noqa: E402  같은 계수 규칙

TGT = {"vehicle": ([2, 3, 5, 7], "center"), "person": ([0], "bottom"), "any": ([0, 2, 3, 5, 7], "center")}
COCO = {1: 0, 2: 1, 3: 2, 4: 3, 6: 5, 8: 7}   # rfdetr(COCO 91번호) → 80번호: 사람·자전거·승용·오토바이·버스·트럭


class Source(threading.Thread):
    """최신 프레임 하나만 들고 있음(밀리면 버림). 파일은 실시간 속도로."""
    def __init__(self, url, realtime):
        super().__init__(daemon=True); self.cap = cv2.VideoCapture(url); self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.realtime, self.frame, self.n, self.ok, self.t0 = realtime, None, 0, True, time.time()
        self.W, self.H = int(self.cap.get(3)), int(self.cap.get(4))
    def run(self):
        while self.ok:
            ok, f = self.cap.read()
            if not ok:
                if not self.realtime: time.sleep(0.5); continue   # 스트림이 잠깐 끊기면 기다림
                self.ok = False; break
            self.n += 1; self.frame = (self.n, f)
            if self.realtime:
                lag = self.t0 + self.n / self.fps - time.time()
                if lag > 0: time.sleep(lag)


class Counter(threading.Thread):
    def __init__(self, src, gates, model, conf):
        super().__init__(daemon=True); self.src, self.gates, self.conf = src, gates, conf
        import rfdetr
        self.m = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}[model]()
        try: self.m.optimize_for_inference()
        except Exception: pass
        self.trk = BYTETracker(SimpleNamespace(tracker_type="bytetrack", track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25, track_buffer=30,
                                               match_thresh=0.8, fuse_score=True, gmc_method="none", proximity_thresh=0.5, appearance_thresh=0.8, with_reid=False, model="auto"))
        for L in gates:
            c, an = TGT[L.get("target", "vehicle")]; L.setdefault("classes", c); L.setdefault("anchor", an); L["oneway"] = L.get("dir", "one") == "one"
            p1, p2 = np.array(L["p1"], float), np.array(L["p2"], float); u = (p2 - p1) / np.linalg.norm(p2 - p1)
            L["q1"], L["q2"], L["sg"] = tuple(p1 - u * 30), tuple(p2 + u * 30), []
        self.boxes, self.events, self.last, self.votes, self.counted = [], [], {}, {}, set()
        self.det_ms, self.lock, self.ok = 0.0, threading.Lock(), True
    def detect(self, f):
        d = self.m.predict(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)), threshold=self.conf)
        keep = [i for i, c in enumerate(d.class_id) if int(c) in COCO]
        return d.xyxy[keep], d.confidence[keep], np.array([COCO[int(c)] for c in d.class_id[keep]])
    def run(self):
        import torch, torchvision
        seen = 0
        while self.ok and (self.src.ok or self.src.frame and self.src.frame[0] > seen):
            fr = self.src.frame
            if not fr or fr[0] == seen: time.sleep(0.005); continue
            seen, f = fr; t0 = time.time(); t = seen / self.src.fps
            xy, sc, cl = self.detect(f)
            if len(xy):   # 오토바이·자전거 위 사람 지우기 + 종류 무관 겹침 정리
                tw = np.where(np.isin(cl, (1, 3)))[0]; pi = np.where(cl == 0)[0]
                if len(tw) and len(pi):
                    fx, fy = (xy[pi, 0] + xy[pi, 2]) / 2, xy[pi, 3]; b = xy[tw]; mx = (b[:, 2] - b[:, 0]) * 0.1
                    ins = ((fx[:, None] >= b[None, :, 0] - mx) & (fx[:, None] <= b[None, :, 2] + mx) & (fy[:, None] >= b[None, :, 1]) & (fy[:, None] <= b[None, :, 3] + (b[None, :, 3] - b[None, :, 1]) * 0.15)).any(1)
                    k = np.setdiff1d(np.arange(len(xy)), pi[ins]); xy, sc, cl = xy[k], sc[k], cl[k]
                k = np.sort(torchvision.ops.nms(torch.tensor(xy, dtype=torch.float32), torch.tensor(sc, dtype=torch.float32), 0.6).numpy()); xy, sc, cl = xy[k], sc[k], cl[k]
            out = self.trk.update(Dets(xy.astype(np.float32), sc.astype(np.float32), cl.astype(np.float32)), None)
            boxes, new = [], []
            for x1, y1, x2, y2, tid, s, c, _ in (out if len(out) else []):
                tid = int(tid); v = self.votes.setdefault(tid, {}); v[int(c)] = v.get(int(c), 0) + float(s); kind = max(v, key=v.get)
                cen, bot = ((x1 + x2) / 2, (y1 + y2) / 2), ((x1 + x2) / 2, y2); prev = self.last.get(tid); self.last[tid] = (cen, bot)
                boxes.append((tid, x1, y1, x2, y2, kind))
                if not prev: continue
                for L in self.gates:
                    if (L["name"], tid) in self.counted or int(c) not in L["classes"]: continue
                    a, b = (prev[1], bot) if L["anchor"] == "bottom" else (prev[0], cen)
                    cp = cross_pt(a, b, L["q1"], L["q2"])
                    if cp is None: continue
                    sg = side(L["q1"], L["q2"], b); L["sg"].append(sg)
                    if L["oneway"] and len(L["sg"]) >= 5 and sg != (1 if sum(L["sg"]) >= 0 else -1): continue   # 처음 5대로 주 방향을 정하고 반대 방향은 버림
                    self.counted.add((L["name"], tid)); new.append({"t": t, "wall": time.time(), "id": tid, "line": L["name"], "pt": cp, "kind": kind})
            with self.lock:
                self.boxes = boxes; self.events += new
            self.det_ms = self.det_ms * 0.9 + (time.time() - t0) * 100


class Screen:
    """방송 화면(1280×720 기본). render_overlay.py와 같은 모양, 다만 길이를 모르니 차트는 지난 10분을 굴려 가며."""
    def __init__(self, W, H, vw, vh, gates, title, source, note):
        k = W / 1920; self.k, self.W, self.H = k, W, H; z = lambda v: int(round(v * k))
        self.TOP, self.BOT, self.PX = z(64), z(56), z(1440)
        s = min(self.PX / vw, (H - self.TOP - self.BOT) / vh); self.s = s; self.DW, self.DH = int(vw * s), int(vh * s)
        self.OX, self.OY = (self.PX - self.DW) // 2, self.TOP + (H - self.TOP - self.BOT - self.DH) // 2
        self.F = {n: font(max(9, z(sz)), w) for n, (sz, w) in {"title": (30, "Bold"), "clock": (26, "Medium"), "small": (18, "Medium"), "tiny": (15, "Medium"),
                                                               "big": (64, "Black"), "mid": (24, "Bold"), "chip": (17, "Bold"), "label": (20, "Bold")}.items()}
        self.lines = [L for L in gates if L.get("target") != "person"] + [L for L in gates if L.get("target") == "person"]
        vi = 0
        for L in self.lines:
            if L.get("target") == "person": L["col"] = PALETTE[2]
            else: L["col"] = PALETTE[[0, 1, 3, 4][vi % 4]]; vi += 1
        self.G = {L["name"]: L for L in self.lines}; self.z = z; self.shown = {L["name"]: 0.0 for L in self.lines}
        b = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(b); F = self.F
        d.rectangle((self.PX, 0, W, H), fill=PANEL); d.rectangle((0, 0, W, self.TOP), fill=(8, 10, 14)); d.rectangle((0, H - self.BOT, W, H), fill=(8, 10, 14))
        d.text((z(24), self.TOP // 2), "tally-cam", font=F["title"], fill=(255, 196, 64), anchor="lm"); d.text((z(190), self.TOP // 2), title, font=F["title"], fill=TXT, anchor="lm")
        d.text((z(24), H - self.BOT // 2), source, font=F["small"], fill=SUB, anchor="lm")
        if note: d.text((W - z(24), H - self.BOT // 2), note, font=F["small"], fill=SUB, anchor="rm")
        self.CY, self.CH, self.CG = self.TOP + z(14), z(108), z(10); n = len(self.lines)
        self.C0 = self.CY + n * (self.CH + self.CG) + z(50); self.C1 = self.C0 + z(150); self.B0 = self.C1 + z(70); self.BH = z(110); self.MIX = self.B0 + self.BH + z(38)
        for i, L in enumerate(self.lines):
            y = self.CY + i * (self.CH + self.CG); d.rounded_rectangle((self.PX + z(20), y, W - z(20), y + self.CH), z(14), fill=CARD)
            d.rounded_rectangle((self.PX + z(20), y, self.PX + z(28), y + self.CH), z(4), fill=L["col"])
            d.text((self.PX + z(44), y + z(12)), L["show"], font=F["mid"], fill=TXT); d.text((self.PX + z(44), y + z(44)), "보행자" if L.get("target") == "person" else "차량", font=F["tiny"], fill=SUB)
        for y0, h, tt in ((self.C0, self.C1 - self.C0, "누적 통과"), (self.B0, self.BH, "1분마다 통과한 차 (최근 10분)")):
            d.text((self.PX + z(24), y0 - z(46)), tt, font=F["mid"], fill=TXT)
            for q in range(5): yy = y0 + h - h * q / 4; d.line((self.PX + z(70), yy, W - z(28), yy), fill=GRID, width=1)
        d.text((self.PX + z(24), self.MIX - z(4)), "차종 비율", font=F["mid"], fill=TXT)
        self.base = b; self.cx0, self.cx1 = self.PX + z(76), W - z(32)

    def draw(self, f, now, boxes, events, t_start, fps_info):
        z, F, W, H = self.z, self.F, self.W, self.H
        img = self.base.copy(); img.paste(Image.fromarray(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (self.DW, self.DH))), (self.OX, self.OY))
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); o = ImageDraw.Draw(ov)
        sx, sy = (lambda x: self.OX + x * self.s), (lambda y: self.OY + y * self.s)
        recent = [e for e in events if now - e["wall"] < 1.2]; cnt = {}
        for e in events: cnt.setdefault(e["id"], (e["wall"], e["line"]))
        for L in self.lines:
            p1, p2 = (sx(L["p1"][0]), sy(L["p1"][1])), (sx(L["p2"][0]), sy(L["p2"][1])); hot = max([1 - (now - e["wall"]) / 0.5 for e in recent if e["line"] == L["name"]] + [0])
            o.line((p1, p2), fill=L["col"] + (int(70 + 150 * hot),), width=z(10 + 8 * hot)); o.line((p1, p2), fill=(255, 255, 255, int(120 + 135 * hot)), width=max(1, z(2)))
        for tid, x1, y1, x2, y2, k in boxes:
            if k not in KIND or (y2 - y1) < 12: continue
            X1, Y1, X2, Y2 = sx(x1), sy(y1), sx(x2), sy(y2); ca = cnt.get(tid)
            if ca:
                c = self.G[ca[1]]["col"]; age = now - ca[0]; pulse = max(0, 1 - age / 0.6); fade = 1.0 if age < 3 else max(0.35, 1 - (age - 3) / 2)
                o.rounded_rectangle((X1, Y1, X2, Y2), z(6), outline=c + (int(255 * fade),), width=max(1, z(3)), fill=c + (int((40 + 90 * pulse) * fade),))
                if age < 2.5:
                    tw = o.textlength(KIND[k], font=F["tiny"]) + z(12); o.rounded_rectangle((X1, Y1 - z(22), X1 + tw, Y1 - z(2)), z(5), fill=c + (230,)); o.text((X1 + z(6), Y1 - z(12)), KIND[k], font=F["tiny"], fill=(10, 12, 16, 255), anchor="lm")
            else:
                Ln = max(4, min(X2 - X1, Y2 - Y1) * 0.28); al = 150 if k else 90
                for px, py, dx, dy in ((X1, Y1, 1, 1), (X2, Y1, -1, 1), (X1, Y2, 1, -1), (X2, Y2, -1, -1)):
                    o.line((px, py, px + dx * Ln, py), fill=(255, 255, 255, al), width=max(1, z(2))); o.line((px, py, px, py + dy * Ln), fill=(255, 255, 255, al), width=max(1, z(2)))
        for L in self.lines:
            q = min((sx(L["p1"][0]), sy(L["p1"][1])), (sx(L["p2"][0]), sy(L["p2"][1]))); tw = o.textlength(L["show"], font=F["chip"]) + z(18)
            o.rounded_rectangle((q[0], q[1] - z(34), q[0] + tw, q[1] - z(8)), z(8), fill=(10, 12, 16, 200)); o.text((q[0] + z(9), q[1] - z(21)), L["show"], font=F["chip"], fill=L["col"] + (255,), anchor="lm")
        for e in recent:
            age = now - e["wall"]; c = self.G[e["line"]]["col"]; X, Y = sx(e["pt"][0]), sy(e["pt"][1]); r = z(10) + z(46) * ease(age / 0.7); al = int(255 * max(0, 1 - age / 0.7))
            o.ellipse((X - r, Y - r, X + r, Y + r), outline=c + (al,), width=max(1, z(4)))
            up = z(70) * ease(age / 1.2); al2 = int(255 * (1 - max(0, age - 0.6) / 0.6)); txt = f"+1 {KIND.get(e['kind'], '')}"; tw = o.textlength(txt, font=F["label"]) + z(20)
            o.rounded_rectangle((X - tw / 2, Y - z(46) - up, X + tw / 2, Y - z(14) - up), z(10), fill=c + (int(al2 * 0.92),)); o.text((X, Y - z(30) - up), txt, font=F["label"], fill=(10, 12, 16, al2), anchor="mm")
        img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB"); d = ImageDraw.Draw(img)
        d.ellipse((self.PX - z(250), self.TOP // 2 - z(7), self.PX - z(236), self.TOP // 2 + z(7)), fill=(255, 60, 60) if int(now * 2) % 2 == 0 else (120, 30, 30))
        d.text((self.PX - z(226), self.TOP // 2), "LIVE", font=F["small"], fill=(255, 110, 110), anchor="lm")
        d.text((W - z(24), self.TOP // 2), dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), font=F["clock"], fill=TXT, anchor="rm")
        d.text((self.PX - z(24), self.TOP // 2), fps_info, font=F["tiny"], fill=SUB, anchor="rm")
        for i, L in enumerate(self.lines):
            y = self.CY + i * (self.CH + self.CG); mine = [e for e in events if e["line"] == L["name"]]; n = len(mine)
            self.shown[L["name"]] += (n - self.shown[L["name"]]) * 0.35
            if abs(n - self.shown[L["name"]]) < 0.02: self.shown[L["name"]] = n
            fl = max([1 - (now - e["wall"]) / 0.4 for e in mine if now - e["wall"] < 0.4] + [0])
            if fl > 0: d.rounded_rectangle((self.PX + z(20), y, W - z(20), y + self.CH), z(14), outline=L["col"], width=max(1, z(2 + 3 * fl)))
            d.text((W - z(40), y + z(38)), f"{int(round(self.shown[L['name']])):,}", font=F["big"], fill=TXT, anchor="rm")
            if L.get("target") != "person":
                kc = {}
                for e in mine: kc[e["kind"]] = kc.get(e["kind"], 0) + 1
                xx = self.PX + z(44)
                for kk in (2, 5, 7, 3):
                    if not kc.get(kk): continue
                    s_ = f"{KIND[kk]} {kc[kk]}"; tw = d.textlength(s_, font=F["tiny"]) + z(16)
                    d.rounded_rectangle((xx, y + z(74), xx + tw, y + z(98)), z(7), fill=(36, 42, 54)); d.ellipse((xx + z(6), y + z(83), xx + z(12), y + z(89)), fill=KCOL[kk])
                    d.text((xx + z(15), y + z(86)), s_, font=F["tiny"], fill=TXT, anchor="lm"); xx += tw + z(6)
            else: d.text((self.PX + z(44), y + z(86)), f"양방향 합계 · 최근 1분 {sum(1 for e in mine if now - e['wall'] < 60)}명", font=F["tiny"], fill=SUB, anchor="lm")
        el = max(now - t_start, 1.0); span = max(300.0, el); cx0, cx1 = self.cx0, self.cx1
        top = max([1] + [sum(1 for e in events if e["line"] == L["name"]) for L in self.lines]); ym = max(20, int(math.ceil(top / 20.0) * 20))
        for q in range(5): d.text((self.PX + z(62), self.C0 + (self.C1 - self.C0) * (1 - q / 4)), str(ym * q // 4), font=F["tiny"], fill=SUB, anchor="rm")
        for L in self.lines:
            ts = [e["wall"] - t_start for e in events if e["line"] == L["name"]]; pts = [(cx0, self.C1)]
            for j, tt in enumerate(ts): x = cx0 + (cx1 - cx0) * tt / span; pts += [(x, self.C1 - (self.C1 - self.C0) * j / ym), (x, self.C1 - (self.C1 - self.C0) * (j + 1) / ym)]
            xn = cx0 + (cx1 - cx0) * el / span; yn = self.C1 - (self.C1 - self.C0) * len(ts) / ym; pts.append((xn, yn)); d.line(pts, fill=L["col"], width=max(1, z(3))); d.ellipse((xn - z(5), yn - z(5), xn + z(5), yn + z(5)), fill=L["col"])
        vl = [L for L in self.lines if L.get("target") != "person"]; nb = 10; bw = (cx1 - cx0) / nb
        cur = int(el // 60); bins = list(range(max(0, cur - nb + 1), cur + 1))
        vals = [[sum(1 for e in events if e["line"] == L["name"] and int((e["wall"] - t_start) // 60) == b) for L in vl] for b in bins]
        bm = max(20, int(math.ceil(max([max(v) for v in vals] + [1]) / 20.0) * 20))
        for q in range(5): d.text((self.PX + z(62), self.B0 + self.BH * (1 - q / 4)), str(bm * q // 4), font=F["tiny"], fill=SUB, anchor="rm")
        for j, (b, v) in enumerate(zip(bins, vals)):
            for i, L in enumerate(vl):
                x = cx0 + j * bw + z(6) + i * (bw - z(12)) / max(1, len(vl)); w_ = (bw - z(12)) / max(1, len(vl)) - z(3); hh = self.BH * v[i] / bm
                if hh > 0: d.rounded_rectangle((x, self.B0 + self.BH - hh, x + w_, self.B0 + self.BH), z(4), fill=L["col"] if b < cur else tuple(int(c * 0.7) for c in L["col"]))
            d.text((cx0 + j * bw + bw / 2, self.B0 + self.BH + z(8)), f"{b}분", font=F["tiny"], fill=SUB, anchor="mt")
        kc = {}
        for e in events:
            if self.G[e["line"]].get("target") != "person": kc[e["kind"]] = kc.get(e["kind"], 0) + 1
        tot = sum(kc.values()); bx0, bx1, by = self.PX + z(24), W - z(24), self.MIX + z(42); d.rounded_rectangle((bx0, by, bx1, by + z(22)), z(8), fill=(36, 42, 54))
        if tot:
            x = bx0; xx = bx0
            for kk in (2, 5, 7, 3):
                if not kc.get(kk): continue
                w_ = (bx1 - bx0) * kc[kk] / tot; d.rectangle((x, by, x + w_, by + z(22)), fill=KCOL[kk]); x += w_
                s_ = f"{KIND[kk]} {kc[kk] * 100 / tot:.0f}%"; d.ellipse((xx, by + z(36), xx + z(10), by + z(46)), fill=KCOL[kk]); d.text((xx + z(15), by + z(41)), s_, font=F["tiny"], fill=TXT, anchor="lm"); xx += d.textlength(s_, font=F["tiny"]) + z(34)
        return img


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument("--utic"); g.add_argument("--file"); g.add_argument("--url")
    ap.add_argument("--gates", required=True); ap.add_argument("--title", required=True); ap.add_argument("--labels")
    ap.add_argument("--out", default="preview"); ap.add_argument("--size", default="1280x720"); ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--model", default="small"); ap.add_argument("--conf", type=float, default=0.25); ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--source", default="영상: 경찰청 도시교통정보센터(UTIC) 제공"); ap.add_argument("--note", default="자동 계수 · 실시간")
    a = ap.parse_args()
    if a.utic:
        import utic; url = utic.stream_url(utic.info(a.utic))
        if not url: sys.exit("HLS 주소를 찾지 못함")
    else: url = a.file or a.url
    src = Source(url, realtime=bool(a.file)); src.start()
    while src.frame is None and src.ok: time.sleep(0.05)
    gates = json.loads(Path(a.gates).read_text(encoding="utf-8")); names = json.loads(a.labels) if a.labels else {}
    for L in gates: L["show"] = names.get(L["name"], L.get("label", L["name"]))
    cnt = Counter(src, gates, a.model, a.conf); cnt.start()
    W, H = map(int, a.size.split("x")); scr = Screen(W, H, src.W, src.H, gates, a.title, a.source, a.note)
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-g", str(int(a.fps * 2))]
    proc = None
    if a.out == "rtmp":
        key = os.environ.get("YT_STREAM_KEY") or sys.exit("YT_STREAM_KEY 환경변수가 필요해요")
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-",
                                 "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-shortest", *enc, "-b:v", "3500k", "-maxrate", "3500k", "-bufsize", "7000k",
                                 "-c:a", "aac", "-b:a", "64k", "-f", "flv", f"rtmp://a.rtmp.youtube.com/live2/{key}"], stdin=subprocess.PIPE)
    elif a.out != "preview":
        proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-", *enc, "-crf", "23", a.out], stdin=subprocess.PIPE)
    t_start = time.time(); nxt = t_start; frames = 0
    try:
        while src.ok or (src.frame and cnt.is_alive()):
            now = time.time()
            if a.seconds and now - t_start > a.seconds: break
            if now < nxt: time.sleep(nxt - now); continue
            nxt += 1 / a.fps
            with cnt.lock: boxes, events = list(cnt.boxes), list(cnt.events)
            img = scr.draw(src.frame[1], time.time(), boxes, events, t_start, f"검출 {cnt.det_ms:.0f}ms")
            frames += 1
            if proc: proc.stdin.write(np.asarray(img).tobytes())
            else:
                cv2.imshow("tally-cam live", cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord("q"): break
            if not src.ok and a.file: break
    finally:
        src.ok = cnt.ok = False
        if proc: proc.stdin.close(); proc.wait()
        el = time.time() - t_start
        print(f"{el:.0f}초 · 화면 {frames / el:.1f}fps · 검출 {cnt.det_ms:.0f}ms · 계수 {len(cnt.events)}건")


if __name__ == "__main__":
    main()
