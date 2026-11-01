"""RF-DETR small 미세조정 (클라우드 GPU용) — 의사 상자(bench/pseudo) 데이터, 카메라 하나를 빼고 학습해 그 카메라로 검증.

  python3 bench/train/train_rfdetr.py --data data/pseudo_coco --hold songpa --epochs 15 --out runs
  # 끝나면 runs/wo_songpa/checkpoint_best_total.pth 로 빠진 카메라 클립 검출 → 세기 → 채점까지 자동(--eval)

--hold none 이면 모든 dev 클립으로 학습(최종 모델용; 이때 검증 점수는 의미 없음).
"""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def merge(clip_dirs, dst):
    """여러 클립 COCO를 한 split 폴더로 합침(이미지는 심볼릭 링크)."""
    (dst / "images").mkdir(parents=True, exist_ok=True); imgs, anns = [], []
    for d in clip_dirs:
        c = json.loads((d / "_annotations.coco.json").read_text()); remap = {}
        for im in c["images"]:
            nid = len(imgs) + 1; remap[im["id"]] = nid; src = (d / im["file_name"]).resolve()
            link = dst / "images" / src.name
            if not link.exists(): os.symlink(src, link)
            imgs.append(dict(im, id=nid, file_name=f"images/{src.name}"))
        for a in c["annotations"]:
            anns.append(dict(a, id=len(anns) + 1, image_id=remap[a["image_id"]]))
    (dst / "_annotations.coco.json").write_text(json.dumps({"images": imgs, "annotations": anns,
        "categories": [{"id": 1, "name": "vehicle", "supercategory": "vehicle"}]}))
    return len(imgs), len(anns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/pseudo_coco"); ap.add_argument("--hold", required=True, help="빼는 카메라 이름(dataset.json camera) 또는 none")
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--out", default="runs"); ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    ds = json.loads((ROOT / "bench/dataset.json").read_text())
    dev = [c for c in ds["clips"] if c["split"] == "dev" and (Path(a.data) / c["id"]).exists()]
    hold = [c for c in dev if c["camera"] == a.hold]; trn = [c for c in dev if c["camera"] != a.hold]
    fold = Path(a.out) / f"wo_{a.hold}"; data_dir = fold / "data"
    if data_dir.exists(): shutil.rmtree(data_dir)
    n = merge([Path(a.data) / c["id"] for c in trn], data_dir / "train")
    v = merge([Path(a.data) / c["id"] for c in (hold or trn[:1])], data_dir / "valid")
    merge([Path(a.data) / c["id"] for c in (hold or trn[:1])], data_dir / "test")
    print(f"학습 {n[0]}장/{n[1]}상자 ({[c['id'] for c in trn]}), 검증 {v[0]}장 ({[c['id'] for c in hold]})", flush=True)

    from rfdetr import RFDETRSmall
    model = RFDETRSmall()
    model.train(dataset_dir=str(data_dir), epochs=a.epochs, batch_size=a.bs, grad_accum_steps=a.accum, lr=a.lr, output_dir=str(fold))
    w = fold / "checkpoint_best_total.pth"
    print("가중치:", w, w.exists(), flush=True)

    if a.eval and hold:
        m = f"FT-rfdetrS-wo_{a.hold}"
        for c in hold:
            cache = ROOT / f"bench/cache/{c['id']}_ftS_wo_{a.hold}.npz"
            subprocess.run([sys.executable, str(ROOT / "bench/methods/detcache.py"), "--video", str(ROOT / c["video"]), "--backend", "rfdetr", "--model", "small",
                            "--weights", str(w), "--out", str(cache)], check=True)
            subprocess.run([sys.executable, str(ROOT / "bench/methods/track_count.py"), "--cache", str(cache), "--lines", str(ROOT / c["gates"]),
                            "--out", str(ROOT / f"bench/preds/{m}/{c['id']}.json"), "--agnostic", "--ext", "30", "--conf", "0.25", "--oneway"], check=True)
        subprocess.run([sys.executable, str(ROOT / "bench/score.py"), "--dataset", str(ROOT / "bench/dataset.json"), "--method", m, "--split", "dev"])


if __name__ == "__main__":
    main()
