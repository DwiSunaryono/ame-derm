import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from ame_derm.common import birnet_final as bf
from ame_derm.experts.train_specialist_expert import MultimodalClassifier, SkinLesionDatasetMultimodal, EXPERT_MAPPING, set_active_meta_fields, make_loader
from ame_derm.data_splits.representative_val_split import stratified_ham_kfold_fully_representative
from ame_derm.experts.train_classifier_phase2_final import CROP_DIR
from ame_derm.data_splits.diagnose_metadata_imbalance import detect_metadata_fields

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class SmartRouter(nn.Module):
    def __init__(self, num_classes=7, meta_dim=15):
        super().__init__()
        in_features = num_classes * 2 + meta_dim
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, prob_gen, prob_exp, metas):
        x = torch.cat([prob_gen, prob_exp, metas], dim=1)
        return self.net(x)

def get_expert_name(loc):
    for expert, locs in EXPERT_MAPPING.items():
        if loc in locs:
            return expert
    return "generalist"

def tta_views(img_t: torch.Tensor):
    return [
        img_t,
        torch.flip(img_t, dims=[3]),
        torch.flip(img_t, dims=[2]),
        torch.rot90(img_t, k=1, dims=[2, 3]),
        torch.rot90(img_t, k=2, dims=[2, 3]),
        torch.rot90(img_t, k=3, dims=[2, 3]),
    ]

def extract_probs(model, imgs, metas):
    views = tta_views(imgs)
    p_sum = torch.zeros((imgs.shape[0], 7), device=DEVICE)
    for view in views:
        p_sum += torch.softmax(model(view, metas), dim=1)
    return p_sum / len(views)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="vit_base_patch14_reg4_dinov2.lvd142m")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()

    set_active_meta_fields("localization")
    cfg = bf.Config()
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)
    has_crop = ham_df["image_id"].apply(lambda iid: os.path.exists(os.path.join(CROP_DIR, f"{iid}.jpg")))
    ham_df = ham_df[has_crop].reset_index(drop=True)

    bucketed, _ = detect_metadata_fields(ham_df, n_bins_numeric=9)
    overlap = [c for c in bucketed.columns if c in ham_df.columns]
    ham_meta = pd.concat([ham_df.drop(columns=overlap), bucketed], axis=1)

    for repeat_i, fold_i, trn_idx, val_idx, tst_idx in stratified_ham_kfold_fully_representative(
            ham_meta, cfg, ["dataset"], n_splits=5, n_repeats=1):
        if fold_i == args.fold:
            val_idx_target = val_idx
            break
            
    val_df = ham_df.iloc[val_idx_target].reset_index(drop=True)
    _, val_aug = bf.build_augmentations(cfg.img_size, seed=42)
    val_loader = make_loader(val_df, cfg, val_aug, shuffle=False)

    ckpts = {
        "generalist": f"BIRNet_Research/outputs/checkpoints/phase3_hybrid_multimodal/cls_{args.backbone}_multimodal_fold{args.fold}_best.pth",
        "torso": f"BIRNet_Research/outputs/checkpoints/phase2_multimodal/cls_{args.backbone}_expert_torso_fold{args.fold}_best.pth",
        "limb": f"BIRNet_Research/outputs/checkpoints/phase2_multimodal/cls_{args.backbone}_expert_limb_fold{args.fold}_best.pth",
        "head": f"BIRNet_Research/outputs/checkpoints/phase2_multimodal/cls_{args.backbone}_expert_head_fold{args.fold}_best.pth"
    }

    models = {}
    for expert, ckpt_path in ckpts.items():
        if os.path.exists(ckpt_path):
            print(f"Loading {expert.upper()} model...")
            model = MultimodalClassifier(args.backbone, num_classes=cfg.num_classes, meta_dim=15, img_size=cfg.img_size).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            model.load_state_dict(ckpt["model_state"], strict=True)
            model.eval()
            models[expert] = model

    print("Extracting representations from Validation Set...")
    X_prob_gen = []
    X_prob_exp = []
    X_metas = []
    Y_labels = []

    with torch.no_grad():
        for i, (imgs, metas, labels) in enumerate(tqdm(val_loader)):
            imgs = imgs.to(DEVICE)
            metas = metas.to(DEVICE)
            
            # Predict with Generalist
            prob_gen = extract_probs(models["generalist"], imgs, metas)
            
            # Predict with Experts
            prob_exp = torch.zeros_like(prob_gen)
            # Find which expert applies to each image in batch
            batch_start_idx = i * val_loader.batch_size
            for j in range(imgs.shape[0]):
                loc = val_df.iloc[batch_start_idx + j]["localization"]
                expert_type = get_expert_name(loc)
                if expert_type in models and models[expert_type] is not None:
                    # Single image prediction
                    img_single = imgs[j:j+1]
                    meta_single = metas[j:j+1]
                    prob_exp[j:j+1] = extract_probs(models[expert_type], img_single, meta_single)
                else:
                    prob_exp[j:j+1] = prob_gen[j:j+1]
                    
            X_prob_gen.append(prob_gen.cpu())
            X_prob_exp.append(prob_exp.cpu())
            X_metas.append(metas.cpu())
            Y_labels.append(labels)

    X_prob_gen = torch.cat(X_prob_gen)
    X_prob_exp = torch.cat(X_prob_exp)
    X_metas = torch.cat(X_metas)
    Y_labels = torch.cat(Y_labels)

    # Filter out unknown labels
    valid = Y_labels >= 0
    X_prob_gen, X_prob_exp, X_metas, Y_labels = X_prob_gen[valid], X_prob_exp[valid], X_metas[valid], Y_labels[valid]

    print(f"Training Smart Router on {len(Y_labels)} validation samples...")
    router = SmartRouter().to(DEVICE)
    optimizer = torch.optim.Adam(router.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # We want to minimize the Cross Entropy of the merged probabilities
    dataset = torch.utils.data.TensorDataset(X_prob_gen, X_prob_exp, X_metas, Y_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    best_loss = float('inf')
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        router.train()
        total_loss = 0
        preds = []
        labs = []
        for p_gen, p_exp, meta, label in loader:
            p_gen, p_exp, meta, label = p_gen.to(DEVICE), p_exp.to(DEVICE), meta.to(DEVICE), label.to(DEVICE)
            
            w_exp = router(p_gen, p_exp, meta) # (B, 1)
            p_merged = w_exp * p_exp + (1 - w_exp) * p_gen
            
            # Avoid log(0)
            loss = nn.NLLLoss()(torch.log(p_merged + 1e-8), label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds.extend(p_merged.argmax(dim=1).cpu().numpy())
            labs.extend(label.cpu().numpy())
            
        acc = (np.array(preds) == np.array(labs)).mean()
        if loss.item() < best_loss or acc > best_acc:
            best_loss = loss.item()
            best_acc = max(best_acc, acc)
            torch.save(router.state_dict(), f"router_best_fold{args.fold}.pth")
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(loader):.4f} - Acc: {acc:.4f}")

    print(f"Router training completed. Best Acc: {best_acc:.4f}. Saved to router_best_fold{args.fold}.pth")

if __name__ == "__main__":
    main()
