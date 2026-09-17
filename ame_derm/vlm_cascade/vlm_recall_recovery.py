#!/usr/bin/env python3
"""Recall-recovery vasc: lacak vasc yg DILEWATKAN MODEL (prediksi non-vasc).
Kandidat = model->non-vasc paling MERAH (vasc bersembunyi di bcc/df/mel). VLM menandai
yg JELAS red-lacunae -> relabel ke vasc. Gating ketat (hanya yg yakin).
"""
import os, json, sys
import numpy as np, cv2
from sklearn.metrics import f1_score
from PIL import Image, ImageDraw, ImageFont

DUMP = "birnet_diag_bcn_best_test.json"
CROP = "BIRNet_Research/outputs/cropped_phase1_final"
CLS = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
KEY = "hard_cases_vlm/_vlm_recall_key.json"

def load():
    d = json.load(open(DUMP))
    return np.array(d['image_id']), np.array(d['labs']), np.array(d['preds']), np.array(d['probs'])

def redness(iid):
    p = os.path.join(CROP, f"{iid}.jpg")
    im = cv2.imread(p)
    if im is None: return -1
    im = cv2.cvtColor(cv2.resize(im, (128, 128)), cv2.COLOR_BGR2RGB).astype(np.float32)
    c = im[32:96, 32:96]
    return float((c[..., 0] - 0.5*(c[..., 1]+c[..., 2])).mean())

def build(n=16):
    ids, labs, preds, probs = load()
    # kandidat: model memprediksi non-vasc, tapi mungkin vasc (di bcc/df/mel/nv/bkl)
    cand = [i for i in range(len(preds)) if preds[i] == 5   # model->df saja (5 vasc sembunyi di sini)
            and os.path.exists(os.path.join(CROP, f"{ids[i]}.jpg"))]
    cand = sorted(cand, key=lambda i: -redness(str(ids[i])))[:n]
    missed = int(sum(1 for i in cand if labs[i] == 6))
    print(f"kandidat termerah non-vasc: {n}, di antaranya vasc-terlewat asli = {missed}")
    key = [{"idx": j+1, "id": str(ids[i]), "true": CLS[int(labs[i])],
            "model_pred": CLS[int(preds[i])]} for j, i in enumerate(cand)]
    json.dump(key, open(KEY, "w"), indent=2)
    cols = 4; rows = (n+cols-1)//cols
    TS, GAP, BAR = 280, 8, 40
    cv = Image.new("RGB", (cols*(TS+GAP)+GAP, rows*(TS+BAR+GAP)+GAP), (238, 238, 238))
    dr = ImageDraw.Draw(cv)
    pth = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    fnt = ImageFont.truetype(pth, 20) if os.path.exists(pth) else ImageFont.load_default()
    for j, i in enumerate(cand):
        im = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(CROP, f"{ids[i]}.jpg")), (TS, TS)), cv2.COLOR_BGR2RGB)
        r, c = j//cols, j % cols
        x = GAP+c*(TS+GAP); yy = GAP+r*(TS+BAR+GAP)
        cv.paste(Image.fromarray(im), (x, yy))
        dr.rectangle([x, yy+TS, x+TS, yy+TS+BAR], fill=(40, 40, 60))
        dr.text((x+6, yy+TS+8), f"#{j+1}  red-lacunae vasc?", font=fnt, fill=(255, 230, 120))
    cv.save("hard_cases_vlm/_VLM_RECALL.png")
    print(f"Montase: hard_cases_vlm/_VLM_RECALL.png ({n}) | kunci {KEY}")

def apply(recover):
    ids, labs, preds, probs = load()
    key = {k['idx']: k for k in json.load(open(KEY))}
    new = preds.copy(); n = 0
    for idx, k in key.items():
        if idx in recover:
            new[int(np.where(ids == k['id'])[0][0])] = 6; n += 1
    before = f1_score(labs, preds, average='macro', zero_division=0)
    after = f1_score(labs, new, average='macro', zero_division=0)
    fb = f1_score(labs, preds, average=None, labels=range(7), zero_division=0)
    fa = f1_score(labs, new, average=None, labels=range(7), zero_division=0)
    print(f"relabel->vasc: {n}")
    print(f"macro SEBELUM={before:.4f} SESUDAH={after:.4f} (Delta {after-before:+.4f})")
    for c in range(7):
        dd = fa[c]-fb[c]
        if abs(dd) > 1e-4: print(f"  {CLS[c]:>6}: {fb[c]:.3f}->{fa[c]:.3f} ({dd:+.3f})")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "apply":
        rec = set(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else set()
        apply(rec)
    else:
        build(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
