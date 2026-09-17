#!/usr/bin/env python3
"""PoC VLM-verifier di BCN: verifikasi prediksi MODEL->vasc (presisi cuma 0.18).

Gating benar (koreksi user): VLM HANYA menyentuh prediksi model utk kelas lemah
(vasc), BUKAN nv. Tugas VLM: dari 39 prediksi-vasc model, tolak 32 palsu (bukan
lacunae) & pertahankan 7 asli. Kalau VLM bisa memisahkan -> presisi vasc melonjak.

Align: diag preds/labs dibuat pd split lesion-grouped seed-42 -> rebuild split,
verifikasi labs cocok, ambil image_id yg model-prediksi-vasc. Montase BLIND.
"""
import os, json, random
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
import birnet_retrain_bcn as R

CROP = "BIRNet_Research/outputs/cropped_phase1_final"
CLS = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
random.seed(5)

# rebuild split IDENTIK diag (seed 42, img_mode wp>crop)
bcn = R.build_bcn_df("wp>crop")
_, _, test_df = R.lesion_grouped_split(bcn, 42)
diag = json.load(open("birnet_diag_bcn_gen_test.json"))
labs = np.array(diag['labs']); preds = np.array(diag['preds'])

assert len(test_df) == len(labs), f"panjang beda {len(test_df)} vs {len(labs)}"
align = (test_df['label'].values == labs).mean()
print(f"Alignment labs vs split: {align:.3f} (harus ~1.0)")
if align < 0.99:
    print("!! alignment gagal — urutan diag != split. Stop."); raise SystemExit

test_df = test_df.reset_index(drop=True)
ids = test_df['image_id'].astype(str).values

# kasus model->vasc (pred==6)
mv = np.where(preds == 6)[0]
true_v = [i for i in mv if labs[i] == 6]
false_v = [i for i in mv if labs[i] != 6]
print(f"model->vasc: {len(mv)} (asli={len(true_v)}, palsu={len(false_v)})")

random.shuffle(false_v)
sel_idx = list(true_v) + false_v[:13]        # semua asli + 13 palsu
sel = [(i, ids[i], int(labs[i])) for i in sel_idx if os.path.exists(os.path.join(CROP, f"{ids[i]}.jpg"))]
random.shuffle(sel)

cols = 4; rows = (len(sel) + cols - 1) // cols
TS, GAP, BAR = 300, 10, 44
canvas = Image.new("RGB", (cols*(TS+GAP)+GAP, rows*(TS+BAR+GAP)+GAP), (240, 240, 240))
draw = ImageDraw.Draw(canvas)
pth = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
fnt = ImageFont.truetype(pth, 22) if os.path.exists(pth) else ImageFont.load_default()

key = []
for n, (i, iid, tc) in enumerate(sel):
    key.append({"idx": n+1, "id": iid, "true": CLS[tc], "model_pred": "vasc", "is_vasc": bool(tc == 6)})
    im = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(CROP, f"{iid}.jpg")), (TS, TS)), cv2.COLOR_BGR2RGB)
    r, c = n//cols, n % cols
    x = GAP+c*(TS+GAP); yy = GAP+r*(TS+BAR+GAP)
    canvas.paste(Image.fromarray(im), (x, yy))
    draw.rectangle([x, yy+TS, x+TS, yy+TS+BAR], fill=(40, 40, 60))
    draw.text((x+8, yy+TS+8), f"#{n+1}  vasc asli? YA/TIDAK", font=fnt, fill=(255, 230, 120))

canvas.save("hard_cases_vlm/_VLM_BCN_VASC.png")
json.dump(key, open("hard_cases_vlm/_vlm_bcn_vasc_key.json", "w"), indent=2)
print(f"Montase: hard_cases_vlm/_VLM_BCN_VASC.png (n={len(sel)}: semua model->vasc; aku tolak yg bukan lacunae)")
