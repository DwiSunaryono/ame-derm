import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, f1_score

import birnet_final as bf
from train_specialist_expert import MultimodalClassifier, EXPERT_MAPPING, set_active_meta_fields
from train_classifier_phase2_final import CROP_DIR
from train_moe_smart_router import SmartRouter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = "BIRNet_Research/outputs/checkpoints/phase2_multimodal"

def get_expert_name(loc):
    for expert, locs in EXPERT_MAPPING.items():
        if loc in locs:
            return expert
    return "generalist"

def tta_views(img_t):
    return [
        img_t,
        torch.flip(img_t, dims=[3]),
        torch.flip(img_t, dims=[2]),
        torch.rot90(img_t, k=1, dims=[2, 3]),
        torch.rot90(img_t, k=2, dims=[2, 3]),
        torch.rot90(img_t, k=3, dims=[2, 3]),
    ]

def extract_probs(models, loader, cfg):
    probs_tta = {k: [] for k in models.keys()}
    true_labels = []
    all_metas = []
    
    with torch.no_grad():
        for imgs, metas, labels in tqdm(loader, leave=False):
            imgs, metas = imgs.to(DEVICE), metas.to(DEVICE)
            true_labels.extend(labels.numpy())
            all_metas.extend(metas.cpu().numpy())
            
            views = tta_views(imgs)
            for m_name, model in models.items():
                if model is not None:
                    p_sum = torch.zeros((imgs.shape[0], cfg.num_classes), device=DEVICE)
                    for i, view in enumerate(views):
                        p_sum += torch.softmax(model(view, metas), dim=1)
                    probs_tta[m_name].extend((p_sum / len(views)).cpu().numpy())
                    
    return probs_tta, np.array(true_labels), np.array(all_metas)

def train_router_ensemble(X_prob_gen, X_prob_exp, X_metas, Y_labels, n_routers=5, epochs=100):
    routers = []
    X_prob_gen = torch.tensor(X_prob_gen).float()
    X_prob_exp = torch.tensor(X_prob_exp).float()
    X_metas = torch.tensor(X_metas).float()
    Y_labels = torch.tensor(Y_labels).long()
    
    valid = Y_labels >= 0
    X_prob_gen, X_prob_exp, X_metas, Y = X_prob_gen[valid], X_prob_exp[valid], X_metas[valid], Y_labels[valid]
    
    dataset = torch.utils.data.TensorDataset(X_prob_gen, X_prob_exp, X_metas, Y)
    
    for i in range(n_routers):
        router = SmartRouter().to(DEVICE)
        optimizer = torch.optim.Adam(router.parameters(), lr=1e-3, weight_decay=1e-4)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        router.train()
        for _ in range(epochs):
            for p_gen, p_exp, meta, label in loader:
                p_gen, p_exp, meta, label = p_gen.to(DEVICE), p_exp.to(DEVICE), meta.to(DEVICE), label.to(DEVICE)
                w_exp = router(p_gen, p_exp, meta)
                p_merged = w_exp * p_exp + (1 - w_exp) * p_gen
                loss = nn.NLLLoss()(torch.log(p_merged + 1e-8), label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        router.eval()
        routers.append(router)
    return routers

def eval_logic(test_probs_gen, test_probs_exp, test_metas, test_labels, routers=None, weight_expert=0.8):
    final_preds = []
    with torch.no_grad():
        for i in range(len(test_labels)):
            gen_prob = test_probs_gen[i]
            exp_prob = test_probs_exp[i]
            
            if routers is None:
                merged = (weight_expert * exp_prob) + ((1.0 - weight_expert) * gen_prob)
            else:
                p_gen_t = torch.tensor(gen_prob).unsqueeze(0).float().to(DEVICE)
                p_exp_t = torch.tensor(exp_prob).unsqueeze(0).float().to(DEVICE)
                meta_t = torch.tensor(test_metas[i]).unsqueeze(0).float().to(DEVICE)
                
                w_exp_sum = 0
                for r in routers:
                    w_exp_sum += r(p_gen_t, p_exp_t, meta_t).item()
                w_exp_avg = w_exp_sum / len(routers)
                
                merged = (w_exp_avg * exp_prob) + ((1.0 - w_exp_avg) * gen_prob)
                
            final_preds.append(np.argmax(merged))
            
    valid = test_labels >= 0
    return f1_score(test_labels[valid], np.array(final_preds)[valid], average="macro")

def build_exp_arrays(probs_dict, df, head_ver="head_v1"):
    exp_arr = []
    for i, row in df.iterrows():
        exp_type = get_expert_name(row["localization"])
        if exp_type == "head":
            exp_arr.append(probs_dict[head_ver][i])
        elif exp_type in probs_dict:
            exp_arr.append(probs_dict[exp_type][i])
        else:
            exp_arr.append(probs_dict["gen_pure"][i])
    return np.array(exp_arr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="vit_base_patch14_reg4_dinov2.lvd142m")
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    set_active_meta_fields("localization")
    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)
    
    has_crop = ham_df["image_id"].apply(lambda iid: os.path.exists(os.path.join(CROP_DIR, f"{iid}.jpg")))
    ham_df = ham_df[has_crop].reset_index(drop=True)
    
    from birnet_representative_val_split import stratified_ham_kfold_fully_representative
    from train_specialist_expert import detect_metadata_fields
    bucketed, _ = detect_metadata_fields(ham_df, n_bins_numeric=9)
    overlap = [c for c in bucketed.columns if c in ham_df.columns]
    ham_meta = pd.concat([ham_df.drop(columns=overlap), bucketed], axis=1)
    
    for _, fold_i, _, val_idx, tst_idx in stratified_ham_kfold_fully_representative(
            ham_meta, cfg, ["dataset"], n_splits=5, n_repeats=1):
        if fold_i == args.fold:
            val_df = ham_df.iloc[val_idx].reset_index(drop=True)
            test_df = ham_df.iloc[tst_idx].reset_index(drop=True)
            break

    ckpts = {
        "gen_pure": f"BIRNet_Research/outputs/checkpoints/phase2_multimodal_ablation/cls_{args.backbone}_multimodal_fold{args.fold}_best.pth",
        "torso": f"BIRNet_Research/outputs/checkpoints/phase2_multimodal/cls_{args.backbone}_expert_torso_fold{args.fold}_best.pth",
        "limb": f"BIRNet_Research/outputs/checkpoints/phase2_multimodal/cls_{args.backbone}_expert_limb_fold{args.fold}_best.pth",
        "head_v1": f"BIRNet_Research/outputs/checkpoints/phase2_multimodal/cls_{args.backbone}_expert_head_fold{args.fold}_best.pth",
        "head_v2": f"BIRNet_Research/outputs/checkpoints/phase2_multimodal/cls_{args.backbone}_expert_head_fold{args.fold}_active_noise_best.pth",
    }
    
    models = {}
    for key, path in ckpts.items():
        if os.path.exists(path):
            print(f"Loading {key}...")
            m = MultimodalClassifier(args.backbone, num_classes=cfg.num_classes, meta_dim=15, img_size=cfg.img_size).to(DEVICE)
            m.load_state_dict(torch.load(path, map_location=DEVICE)["model_state"], strict=True)
            m.eval()
            models[key] = m
        else:
            print(f"FAILED TO FIND {key} at {path}")
            return

    from train_specialist_expert import make_loader
    _, val_aug = bf.build_augmentations(cfg.img_size, seed=42)
    val_loader = make_loader(val_df, cfg, val_aug, shuffle=False)
    test_loader = make_loader(test_df, cfg, val_aug, shuffle=False)

    print("\\nExtracting VAL probs...")
    val_tta, val_labels, val_metas = extract_probs(models, val_loader, cfg)
    print("Extracting TEST probs...")
    test_tta, test_labels, test_metas = extract_probs(models, test_loader, cfg)

    # Base arrays
    val_gen_pure = np.array(val_tta["gen_pure"])
    test_gen_pure = np.array(test_tta["gen_pure"])

    # Combos
    combos = [
        ("Kombinasi 1: Murni 100% (Torso v1, Limb v1, Head v1)", "head_v1"),
        ("Kombinasi 2: Murni + Sintetis Kepala (Torso v1, Limb v1, Head v2)", "head_v2")
    ]

    for name, head_key in combos:
        print(f"\\n{'='*60}\\n{name}\\n{'='*60}")
        val_exp = build_exp_arrays(val_tta, val_df, head_ver=head_key)
        test_exp = build_exp_arrays(test_tta, test_df, head_ver=head_key)
        
        # 1. Static 80:20
        f1_static = eval_logic(test_gen_pure, test_exp, test_metas, test_labels, routers=None)
        
        # 2. Smart Router (Ensemble of 10 routers to eliminate variance)
        print("Training Ensemble 10 Smart Routers...")
        routers = train_router_ensemble(val_gen_pure, val_exp, val_metas, val_labels, n_routers=10)
        f1_router = eval_logic(test_gen_pure, test_exp, test_metas, test_labels, routers=routers)
        
        print(f"➔ F1 Statis 80:20       : {f1_static:.4f}")
        print(f"➔ F1 Smart Router (Ens) : {f1_router:.4f}")

if __name__ == "__main__":
    main()
