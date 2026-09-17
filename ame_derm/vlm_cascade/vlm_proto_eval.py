#!/usr/bin/env python3
"""Prototype-retrieval eval: build kunci-dari-TRAIN + montase test model->vasc utk 1 fold.
build <seed> <dump>  -> _KEY_<seed>.png, _TEST_<seed>.png, _vlm_proto<seed>_key.json
apply <seed> <dump> <reject_idx> -> reassign + macro
"""
import sys, json, os
import numpy as np, cv2
from ame_derm.external_validation import retrain_bcn as R
from sklearn.metrics import f1_score
from PIL import Image, ImageDraw, ImageFont
CROP = "BIRNet_Research/outputs/cropped_phase1_final"
CLS = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']

def ex(i): return os.path.exists(os.path.join(CROP, f"{i}.jpg"))
def redn(i):
    im = cv2.imread(os.path.join(CROP, f"{i}.jpg"));
    if im is None: return -99
    im = cv2.cvtColor(cv2.resize(im, (128, 128)), cv2.COLOR_BGR2RGB).astype(float); c = im[32:96, 32:96]
    return float((c[..., 0]-0.5*(c[..., 1]+c[..., 2])).mean())
def dark(i):
    im = cv2.imread(os.path.join(CROP, f"{i}.jpg"));
    if im is None: return 999
    return float(cv2.cvtColor(cv2.resize(im, (128, 128)), cv2.COLOR_BGR2GRAY)[32:96, 32:96].mean())
def sheet(items, path, sz=280):
    cols = 4; rows = (len(items)+cols-1)//cols; BAR = 44; GAP = 8
    cv = Image.new("RGB", (cols*(sz+GAP)+GAP, rows*(sz+BAR+GAP)+GAP), (245, 245, 245)); dr = ImageDraw.Draw(cv)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 19)
    for n, (lab, i) in enumerate(items):
        im = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(CROP, f"{i}.jpg")), (sz, sz)), cv2.COLOR_BGR2RGB)
        r, c = n//cols, n % cols; x = GAP+c*(sz+GAP); y = GAP+r*(sz+BAR+GAP)
        cv.paste(Image.fromarray(im), (x, y)); dr.rectangle([x, y+sz, x+sz, y+sz+BAR], fill=(40, 40, 60))
        dr.text((x+6, y+sz+10), lab, font=f, fill=(255, 230, 120))
    cv.save(path)

def load(seed, dump):
    bcn = R.build_bcn_df("wp>crop")
    train_df, _, test_df = R.lesion_grouped_split(bcn, seed)
    d = json.load(open(dump))
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), np.array(d['labs']), np.array(d['preds']), np.array(d['probs']), np.array(d['image_id'])

def build(seed, dump):
    train_df, test_df, labs, preds, probs, ids = load(seed, dump)
    def pool(l): return [i for i in train_df[train_df.label == l].image_id.astype(str) if ex(i)]
    vt = pool(6)
    keys = [('VASC-merah', i) for i in sorted(vt, key=lambda i:-redn(i))[:3]] + \
           [('VASC-biru', i) for i in sorted(vt, key=dark)[:3]] + \
           [('BCC-hitam', i) for i in sorted(pool(2), key=dark)[:2]] + \
           [('MEL-hitam', i) for i in sorted(pool(0), key=dark)[:2]] + [('NV-gelap', i) for i in sorted(pool(1), key=dark)[:1]]
    sheet(keys, f"hard_cases_vlm/_KEY_{seed}.png")
    mv = [j for j in range(len(preds)) if preds[j] == 6 and ex(str(ids[j]))]
    tk = []
    for n, j in enumerate(mv):
        pp = probs[j].copy(); pp[6] = -1
        tk.append({"idx": n+1, "id": str(ids[j]), "true": CLS[int(labs[j])], "second_lbl": int(np.argmax(pp))})
    json.dump(tk, open(f"hard_cases_vlm/_vlm_proto{seed}_key.json", "w"), indent=2)
    sheet([(f"#{n+1}", str(ids[j])) for n, j in enumerate(mv)], f"hard_cases_vlm/_TEST_{seed}.png")
    print(f"seed {seed}: model->vasc={len(mv)} (asli={int((labs[[j for j in mv]]==6).sum())},palsu={int((labs[[j for j in mv]]!=6).sum())})")
    print(f" -> _KEY_{seed}.png , _TEST_{seed}.png")

def apply(seed, dump, rej):
    _, _, labs, preds, probs, ids = load(seed, dump)
    key = {k['idx']: k for k in json.load(open(f"hard_cases_vlm/_vlm_proto{seed}_key.json"))}
    tv_rej = [i for i in rej if key[i]['true'] == 'vasc']; fr = [i for i in rej if key[i]['true'] != 'vasc']
    tvk = [i for i in key if key[i]['true'] == 'vasc' and i not in rej]
    new = preds.copy(); n = 0
    for idx, k in key.items():
        if idx in rej:
            new[int(np.where(ids == k['id'])[0][0])] = k['second_lbl']; n += 1
    b = f1_score(labs, preds, average='macro', zero_division=0); a = f1_score(labs, new, average='macro', zero_division=0)
    print(f"seed {seed}: REJECT {len(rej)} (false-vasc={len(fr)}, SALAH-buang-vasc={len(tv_rej)} {tv_rej}, vasc-dijaga={len(tvk)})")
    print(f"  macro {b:.4f} -> {a:.4f}  (Delta {a-b:+.4f})  vasc-F1 {f1_score(labs,preds,average=None,labels=[6])[0]:.3f}->{f1_score(labs,new,average=None,labels=[6])[0]:.3f}")

if __name__ == "__main__":
    cmd, seed, dump = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    if cmd == "build": build(seed, dump)
    else: apply(seed, dump, set(int(x) for x in sys.argv[4].split(",")) if len(sys.argv) > 4 and sys.argv[4] else set())
