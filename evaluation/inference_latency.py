#!/usr/bin/env python3
"""
AME-Derm inference-latency benchmark (for the paper's Computational Complexity section).

Measures WALL-CLOCK forward-pass latency of the actual backbones on THIS machine's GPU,
composed exactly as the pipeline runs at inference:
  - Phase-2 classification per image = (generalist ViT-L/14 + 1 routed expert ViT-B/14) x 6-view TTA
  - Phase-1 segmentation per image   = 3-model ensemble x 4-view flip TTA   (optional; needs smp)
Latency depends on architecture + input size + hardware + precision, NOT on trained weights,
so models are built with pretrained=False (no download needed); timing is valid.

Run:
  python birnet_bench_inference.py --precision fp16 --repeats 50 --n_test 11982
  python birnet_bench_inference.py --precision fp32          # if eval uses fp32

Paste the printed "PASTE-READY" block back.
"""
import argparse, time, statistics, platform
import torch

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def time_module(model, x, repeats, warmup, amp_dtype):
    """Return (mean_ms, std_ms) per forward call of `model` on input `x`."""
    model.eval()
    use_amp = amp_dtype is not None
    with torch.no_grad():
        for _ in range(warmup):
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                model(x)
        sync()
        ts = []
        for _ in range(repeats):
            sync(); t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                model(x)
            sync(); ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(ts), (statistics.pstdev(ts) if len(ts) > 1 else 0.0)

def build_cls(name, device, img):
    import timm
    # DINOv2(+reg) timm models default to img_size=518; force the paper's 224x224
    # (pos-embed is interpolated to the 224/14 = 16x16 patch grid).
    try:
        m = timm.create_model(name, pretrained=False, num_classes=7, img_size=img)
    except TypeError:
        m = timm.create_model(name, pretrained=False, num_classes=7, dynamic_img_size=True)
    return m.to(device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--tta_cls", type=int, default=6, help="classification TTA views")
    ap.add_argument("--tta_seg", type=int, default=4, help="segmentation TTA views")
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--n_test", type=int, default=11982, help="images to extrapolate full-set time (BCN20000=11982, HAM10000=10015)")
    ap.add_argument("--gen", default="vit_large_patch14_reg4_dinov2.lvd142m")
    ap.add_argument("--exp", default="vit_base_patch14_reg4_dinov2.lvd142m")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[args.precision]
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor()

    print(f"# device={dev}  gpu={gpu}  torch={torch.__version__}  precision={args.precision}")
    print(f"# img={args.img}  tta_cls={args.tta_cls}  tta_seg={args.tta_seg}  repeats={args.repeats}")

    # ---- Phase-2 classification: generalist + one expert, batch = TTA views ----
    xb_cls = torch.randn(args.tta_cls, 3, args.img, args.img, device=dev)
    results = {}
    for role, name in [("generalist (ViT-L/14)", args.gen), ("expert (ViT-B/14)", args.exp)]:
        try:
            m = build_cls(name, dev, args.img)
            mean, std = time_module(m, xb_cls, args.repeats, args.warmup, amp_dtype)
            results[role] = (mean, std)
            print(f"# {role:24s} {name:38s}  {mean:7.2f} +/- {std:4.2f} ms  (per image, {args.tta_cls}-view TTA batch)")
            del m; torch.cuda.empty_cache() if dev == "cuda" else None
        except Exception as e:
            print(f"# {role}: FAILED ({e})")
            results[role] = None

    cls_total = None
    if results.get("generalist (ViT-L/14)") and results.get("expert (ViT-B/14)"):
        cls_total = results["generalist (ViT-L/14)"][0] + results["expert (ViT-B/14)"][0]

    # ---- Phase-1 segmentation ensemble (optional; needs segmentation_models_pytorch) ----
    seg_total = None
    try:
        import segmentation_models_pytorch as smp
        xb_seg = torch.randn(args.tta_seg, 3, args.img, args.img, device=dev)
        seg_specs = [
            ("SegFormer/MiT-B3", lambda: smp.Segformer(encoder_name="mit_b3", encoder_weights=None, classes=1)),
            ("UPerNet/ConvNeXt", lambda: smp.UPerNet(encoder_name="tu-convnext_tiny", encoder_weights=None, classes=1)),
            ("UNet++/ResNet50",  lambda: smp.UnetPlusPlus(encoder_name="resnet50", encoder_weights=None, classes=1)),
        ]
        seg_total = 0.0
        for sname, ctor in seg_specs:
            try:
                m = ctor().to(dev)
                mean, std = time_module(m, xb_seg, args.repeats, args.warmup, amp_dtype)
                seg_total += mean
                print(f"# seg {sname:20s}  {mean:7.2f} +/- {std:4.2f} ms  (per image, {args.tta_seg}-view TTA batch)")
                del m; torch.cuda.empty_cache() if dev == "cuda" else None
            except Exception as e:
                print(f"# seg {sname}: FAILED ({e})")
                seg_total = None; break
    except Exception as e:
        print(f"# segmentation timing skipped (segmentation_models_pytorch not usable: {e})")

    # ---- Summary (paste-ready) ----
    print("\n===== PASTE-READY =====")
    print(f"GPU: {gpu} | precision: {args.precision} | input {args.img}x{args.img}")
    if cls_total is not None:
        ips = 1000.0 / cls_total
        print(f"Phase-2 classification (generalist + 1 expert, 6-view TTA): {cls_total:.2f} ms/image  ({ips:.1f} img/s)")
        print(f"  -> full classification of {args.n_test} images: {cls_total*args.n_test/1000:.1f} s ({cls_total*args.n_test/60000:.2f} min)")
    if seg_total is not None:
        print(f"Phase-1 segmentation ensemble (3 models, 4-view TTA): {seg_total:.2f} ms/image")
    if cls_total is not None and seg_total is not None:
        e2e = cls_total + seg_total
        print(f"End-to-end (seg + classification): {e2e:.2f} ms/image ({1000.0/e2e:.1f} img/s)")
        print(f"  -> full pipeline on {args.n_test} images: {e2e*args.n_test/60000:.2f} min")
    print("VLM cascade: queried only on predicted rare-class images (small fraction); report count if desired.")
    print("=======================")

if __name__ == "__main__":
    main()
