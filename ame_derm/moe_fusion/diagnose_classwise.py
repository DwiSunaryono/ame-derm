import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from ame_derm.common import birnet_final as bf
from ame_derm.experts.train_phase3_hybrid_multimodal import (
    MultimodalClassifier, 
    make_loader, 
    set_active_meta_fields
)
from ame_derm.data_splits.diagnose_metadata_imbalance import detect_metadata_fields
from ame_derm.data_splits.representative_val_split import stratified_ham_kfold_fully_representative

def main():
    cfg = bf.Config()
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    paths = bf.DatasetPaths(cfg)
    ham_df = bf.build_ham10000_dataframe(paths)
    from ame_derm.experts.train_classifier_phase2_final import CROP_DIR
    has_crop = ham_df["image_id"].apply(lambda iid: os.path.exists(os.path.join(CROP_DIR, f"{iid}.jpg")))
    ham_df = ham_df[has_crop].reset_index(drop=True)

    bucketed, field_names = detect_metadata_fields(ham_df, n_bins_numeric=9)
    overlap = [c for c in bucketed.columns if c in ham_df.columns]
    ham_meta = pd.concat([ham_df.drop(columns=overlap), bucketed], axis=1)

    fold_i, train_idx, val_idx, test_idx = next(stratified_ham_kfold_fully_representative(ham_meta, cfg, ["dataset"], n_splits=5, n_repeats=1))[1:]
    
    test_df = ham_df.iloc[test_idx].reset_index(drop=True)
    _, val_aug = bf.build_augmentations(cfg.img_size, seed=42)
    
    set_active_meta_fields("localization")
    test_loader = make_loader(test_df, cfg, val_aug, shuffle=False)
    
    dino_ckpt = "BIRNet_Research/outputs/checkpoints/phase2_multimodal_ablation/cls_vit_base_patch14_reg4_dinov2.lvd142m_multimodal_fold0_best.pth"
    dino_model = MultimodalClassifier("vit_base_patch14_reg4_dinov2.lvd142m", num_classes=7, meta_dim=16, img_size=cfg.img_size)
    ckpt2 = torch.load(dino_ckpt, map_location=DEVICE, weights_only=False)
    dino_model.load_state_dict(ckpt2["model_state"])
    dino_model.to(DEVICE)
    dino_model.eval()
    
    dino_probs = []
    true_labels = []
    
    with torch.no_grad():
        for imgs, metas, labels in test_loader:
            imgs = imgs.to(DEVICE)
            metas = metas.to(DEVICE)
            
            if metas.shape[1] == 15:
                metas_padded = torch.zeros((metas.size(0), 16), dtype=metas.dtype, device=metas.device)
                metas_padded[:, :15] = metas
                logits_dino = dino_model(imgs, metas_padded)
            else:
                logits_dino = dino_model(imgs, metas)
                
            probs = torch.softmax(logits_dino, dim=1).cpu().numpy()
            
            valid = labels >= 0
            dino_probs.extend(probs[valid.numpy()])
            true_labels.extend(labels[valid].numpy())
            
    dino_preds = np.argmax(np.array(dino_probs), axis=1)
    target_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    print(classification_report(true_labels, dino_preds, target_names=target_names, digits=4))

if __name__ == "__main__":
    main()
