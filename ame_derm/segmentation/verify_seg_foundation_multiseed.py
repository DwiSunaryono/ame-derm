"""Summarizes the 3-seed FOUNDATION segmentation training (UnetPlusPlus,
efficientnet-b4, paper-matched loss, NO BGFF, fully deterministic via
bf.set_deterministic) -- the first checkpoint in the systematic bottom-up
rebuild. Confirms whether the new deterministic setup itself produces
stable, low-variance results (i.e. the fix actually worked), before this
checkpoint becomes the reference mask source for all later classification
experiments (Phase 2 onward).

Usage:
    python birnet_verify_seg_foundation_multiseed.py
"""
import re
import numpy as np

SEEDS = [42, 43, 44]
LOG_TEMPLATE = "run_logs/53_seg_foundation_seed{seed}.log"


def parse_test_dice(log_path):
    try:
        text = open(log_path).read()
    except FileNotFoundError:
        return None
    m = re.search(r"RESULT_SUMMARY.*?test_dice=([\d.]+)", text)
    return float(m.group(1)) if m else None


def main():
    vals = []
    print(f"{'seed':<6}{'test_dice':<12}")
    for seed in SEEDS:
        d = parse_test_dice(LOG_TEMPLATE.format(seed=seed))
        if d is None:
            print(f"{seed:<6}{'MISSING':<12}")
            continue
        vals.append(d)
        print(f"{seed:<6}{d:<12.4f}")

    if not vals:
        print("\nERROR: tidak ada hasil yang lengkap.")
        return

    vals = np.array(vals)
    print(f"\nFoundation (efficientnet-b4, deterministic): mean={vals.mean():.4f} std={vals.std():.4f}")
    print(f"Rentang: {vals.min():.4f} - {vals.max():.4f} (selisih={vals.max()-vals.min():.4f})")
    print(f"\nPembanding lama (unetpp_final_best.pth, non-deterministic, 1 run saja): 0.8904")

    if vals.std() < 0.01:
        verdict = "STABIL -- determinisme berhasil menekan noise antar-seed ke level rendah"
    else:
        verdict = "MASIH ADA VARIANSI CUKUP BESAR -- perlu investigasi lanjut sumber noise lain"
    print(f"\nVerdict: {verdict}")
    print(f"RESULT_SUMMARY model=seg_foundation_multiseed_verify mean={vals.mean():.4f} std={vals.std():.4f}")


if __name__ == "__main__":
    main()
