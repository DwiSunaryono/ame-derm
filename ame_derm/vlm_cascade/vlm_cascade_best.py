#!/usr/bin/env python3
"""VLM-cascade pada model TERBAIK (dump birnet_diag_bcn_best_test.json, ada image_id).
Gating KETAT: default = pertahankan prediksi model. VLM cuma menolak yg JELAS
bukan-vasc (no lacunae). Yg ditolak -> reassign 2nd-choice HANYA bila prob-2 kuat.

build: montase semua model->vasc + kunci.
apply "<reject_idx>": idx yg VLM YAKIN bukan-vasc -> reassign; selebihnya biarkan model.
"""
import os, json, sys
import numpy as np, cv2
from sklearn.metrics import f1_score
from PIL import Image, ImageDraw, ImageFont

DUMP = os.environ.get("BCN_DUMP", "birnet_diag_bcn_best_test.json")
CROP = "BIRNet_Research/outputs/cropped_phase1_final"
CLS = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
KEY = "hard_cases_vlm/_vlm_cascade_best_key.json"

def load():
    d = json.load(open(DUMP))
    return (np.array(d['image_id'] if 'image_id' in d else d['ids']),
            np.array(d['labs']), np.array(d['preds']), np.array(d['probs']))

def build():
    ids, labs, preds, probs = load()
    mv = np.where(preds == 6)[0]
    print(f"model->vasc: {len(mv)} (asli={int((labs[mv]==6).sum())}, palsu={int((labs[mv]!=6).sum())})")
    sel = [i for i in mv if os.path.exists(os.path.join(CROP, f"{ids[i]}.jpg"))]
    key = []
    for n, i in enumerate(sel):
        p = probs[i].copy(); p[6] = -1
        key.append({"idx": n+1, "id": str(ids[i]), "true": CLS[int(labs[i])],
                    "second": CLS[int(np.argmax(p))], "second_lbl": int(np.argmax(p)),
                    "second_prob": float(np.max(p))})
    json.dump(key, open(KEY, "w"), indent=2)
    cols = 4; rows = (len(sel)+cols-1)//cols
    TS, GAP, BAR = 280, 8, 40
    cv = Image.new("RGB", (cols*(TS+GAP)+GAP, rows*(TS+BAR+GAP)+GAP), (238, 238, 238))
    dr = ImageDraw.Draw(cv)
    pth = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    fnt = ImageFont.truetype(pth, 20) if os.path.exists(pth) else ImageFont.load_default()
    for n, i in enumerate(sel):
        im = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(CROP, f"{ids[i]}.jpg")), (TS, TS)), cv2.COLOR_BGR2RGB)
        r, c = n//cols, n % cols
        x = GAP+c*(TS+GAP); yy = GAP+r*(TS+BAR+GAP)
        cv.paste(Image.fromarray(im), (x, yy))
        dr.rectangle([x, yy+TS, x+TS, yy+TS+BAR], fill=(40, 40, 60))
        dr.text((x+6, yy+TS+8), f"#{n+1}  ada lacunae merah?", font=fnt, fill=(255, 230, 120))
    cv.save("hard_cases_vlm/_VLM_CASCADE_BEST.png")
    print(f"Montase: hard_cases_vlm/_VLM_CASCADE_BEST.png ({len(sel)}) | kunci {KEY}")

def apply(reject, min_prob2=0.0):
    ids, labs, preds, probs = load()
    key = {k['idx']: k for k in json.load(open(KEY))}
    new = preds.copy(); n_re = 0
    for idx, k in key.items():
        if idx in reject and k['second_prob'] >= min_prob2:   # VLM yakin bukan-vasc dan pilihan-2 kuat
            new[_row(ids, k['id'])] = k['second_lbl']; n_re += 1
    before = f1_score(labs, preds, average='macro', zero_division=0)
    after = f1_score(labs, new, average='macro', zero_division=0)
    fb = f1_score(labs, preds, average=None, labels=range(7), zero_division=0)
    fa = f1_score(labs, new, average=None, labels=range(7), zero_division=0)
    print(f"reassign {n_re}/{len(key)} (VLM yakin bukan-vasc, prob2>={min_prob2})")
    print(f"macro SEBELUM={before:.4f} SESUDAH={after:.4f}  (Delta {after-before:+.4f})")
    for c in range(7):
        dd = fa[c]-fb[c]
        if abs(dd) > 1e-4: print(f"  {CLS[c]:>6}: {fb[c]:.3f}->{fa[c]:.3f} ({dd:+.3f})")

def _row(ids, iid):
    return int(np.where(ids == iid)[0][0])

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "apply":
        rej = set(int(x) for x in sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[2] else set()
        mp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
        apply(rej, mp)
    else:
        build()
