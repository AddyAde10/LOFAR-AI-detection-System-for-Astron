#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multilabel YOLO-ish classifier for LOFAR dynspecs

- Single real H5 (typically *filtered* labelled_dataset_7classes_filtered.h5)
- Optional GAN H5s appended **only to TRAIN split**
- Class-balanced sampler
- Focal loss (default)
- Early stopping
- Per-class thresholds learned on VAL, applied on TEST
- Multilabel confusion matrix printed and saved
"""

import os, sys, json, random, argparse, warnings
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import h5py
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    multilabel_confusion_matrix
)

warnings.filterwarnings("ignore")

# -------------------- CLI --------------------
def get_args():
    p = argparse.ArgumentParser()
    # REAL data (filtered)
    p.add_argument("--h5", type=str, required=True,
                   help="Real labelled H5 (e.g. labelled_dataset_7classes_filtered.h5)")
    # Optional GAN H5s (train-only)
    p.add_argument("--h5_gan", type=str, nargs="+", default=None,
                   help="Optional GAN H5 files; their samples are appended ONLY to the TRAIN split.")
    # Persistent split file
    p.add_argument("--split_path", type=str, default=None,
                   help="Optional npz with tr_idx/va_idx/te_idx. If exists -> reuse, else create.")

    p.add_argument("--outdir", type=str, default="runs/yolo_multilabel")
    p.add_argument("--labels", type=int, nargs="+", default=[1,2,3,4,5,6,7])
    p.add_argument("--filter_mode", type=str, default="keep_any", choices=["keep_any","strict","drop"])
    p.add_argument("--img", type=int, nargs=2, default=[224,224], help="H W")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=7.14e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.15)
    p.add_argument("--threshold", type=float, default=0.5, help="scalar fallback threshold")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=8, help="LR scheduler patience")
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--save_name", type=str, default="yolo_multilabel_subset.pth")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--quiet", action="store_true")

    # Sampler, loss, early-stopping, thresholds
    p.add_argument("--class_sampler", type=str, default="balanced", choices=["none","balanced"])
    p.add_argument("--loss", type=str, default="focal", choices=["bce","focal"])
    p.add_argument("--focal_gamma", type=float, default=2.0)
    p.add_argument("--focal_alpha", type=float, default=0.25)   # scalar alpha
    p.add_argument("--es_patience", type=int, default=12)
    p.add_argument("--es_min_delta", type=float, default=0.0)
    p.add_argument("--threshold_mode", type=str, default="perclass", choices=["perclass","scalar"])
    return p.parse_args()

# -------------------- Repro --------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# -------------------- Label utils --------------------
def parse_label_string(x) -> List[int]:
    if isinstance(x, (int, np.integer)):
        s = str(int(x))
    elif isinstance(x, (bytes, np.bytes_)):
        s = x.decode("utf-8")
    else:
        s = str(x)
    labs = []
    for ch in s.strip():
        if ch.isdigit():
            k = int(ch)
            if 1 <= k <= 9:
                labs.append(k)
    return sorted(set(labs))

def to_subset_multihot(labels_1idx: List[int], selected: List[int],
                       num_classes: int, class_to_index: dict) -> np.ndarray:
    v = np.zeros(num_classes, dtype=np.float32)
    for k in labels_1idx:
        j = class_to_index.get(k, None)
        if j is not None:
            v[j] = 1.0
    return v

def filter_keep(label_list: List[int], selected: List[int], mode: str) -> bool:
    s = set(selected); L = set(label_list)
    if mode == "keep_any": return len(L & s) > 0
    if mode == "strict":   return L.issubset(s)
    if mode == "drop":     return len(L) == 1 and list(L)[0] in s
    raise ValueError(f"Unknown FILTER_MODE: {mode}")

# -------------------- Splits --------------------
def multilabel_stratified_splits(X, Y, test_size=0.15, val_size=0.15, seed=42):
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tv_idx, te_idx = next(msss1.split(np.zeros(len(X)), Y))
        Y_tv = Y[tv_idx]
        msss2 = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=val_size/(1.0-test_size), random_state=seed
        )
        tr_rel, va_rel = next(msss2.split(np.zeros(len(tv_idx)), Y_tv))
        tr_idx = tv_idx[tr_rel]; va_idx = tv_idx[va_rel]
        return tr_idx, va_idx, te_idx
    except Exception:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(X)); rng.shuffle(idx)
        n_te = int(round(test_size * len(X)))
        te_idx = idx[:n_te]; tv_idx = idx[n_te:]
        n_va = int(round(val_size * len(tv_idx)))
        va_idx = tv_idx[:n_va]; tr_idx = tv_idx[n_va:]
        print("[WARN] 'iterstrat' not available; used random split.")
        return tr_idx, va_idx, te_idx

# -------------------- Dataset --------------------
class DynspecMultiLabelDataset(Dataset):
    def __init__(self, imgs, labels_multi_hot, target_size=(224,224)):
        self.imgs = imgs.astype(np.float32)
        self.labels = labels_multi_hot.astype(np.float32)
        self.target_size = target_size
    def __len__(self): return len(self.imgs)
    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = np.where(np.isfinite(img), img, 0.0)
        mn = np.nanmin(img); mx = np.nanmax(img); denom = mx - mn
        if not np.isfinite(denom) or denom <= 0:
            img = np.clip(img, 0.0, 1.0)
        else:
            img = np.clip((img - mn) / max(denom, 1e-8), 0.0, 1.0)
        img = torch.from_numpy(img.transpose(2,0,1))  # (1,H,W)
        if img.shape[-2:] != self.target_size:
            img = F.interpolate(img.unsqueeze(0), size=self.target_size,
                                mode='bilinear', align_corners=False).squeeze(0)
        img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0).contiguous()
        label = torch.from_numpy(self.labels[idx])
        return img, label

# -------------------- Model --------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class C2F(nn.Module):
    def __init__(self, ch, expansion=0.5, n=1):
        super().__init__()
        hid = int(ch*expansion)
        self.cv1 = ConvBlock(ch, hid, 1)
        self.cv2 = ConvBlock(ch, hid, 1)
        self.blocks = nn.Sequential(*[ConvBlock(hid, hid, 3, p=1) for _ in range(n)])
        self.cv3 = ConvBlock(hid*2, ch, 1)
    def forward(self, x):
        y1 = self.cv1(x); y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, self.blocks(y2)], dim=1))

class SPPF(nn.Module):
    def __init__(self, in_ch, out_ch, k=5):
        super().__init__()
        hid = in_ch // 2
        self.cv1 = ConvBlock(in_ch, hid, 1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k//2)
        self.cv2 = ConvBlock(hid*4, out_ch, 1)
    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x); y2 = self.pool(y1); y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))

class YOLOClassifier(nn.Module):
    def __init__(self, num_classes=7, input_hw=(224,224)):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1,   32, 3, s=1, p=1),
            ConvBlock(32,  64, 3, s=2, p=1),
            C2F(64,  0.5, n=1),
            ConvBlock(64, 128, 3, s=2, p=1),
            C2F(128, 0.5, n=3),
            ConvBlock(128, 256, 3, s=2, p=1),
            C2F(256, 0.5, n=3),
            ConvBlock(256, 512, 3, s=2, p=1),
            C2F(512, 0.5, n=1),
        )
        self.neck = SPPF(512, 512, k=5)
        with torch.no_grad():
            dummy = torch.zeros(1,1,*input_hw)
            feat  = self.neck(self.backbone(dummy))
            flat_dim = feat.numel()
        self.flatten = nn.Flatten()
        self.fc1  = nn.Linear(flat_dim, 128); self.drop1=nn.Dropout(0.5)
        self.fc2  = nn.Linear(128, 128);     self.drop2=nn.Dropout(0.5)
        self.fc3  = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.backbone(x); x = self.neck(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x)); x = self.drop1(x)
        x = F.relu(self.fc2(x)); x = self.drop2(x)
        return self.fc3(x)

# -------------------- Losses --------------------
class FocalLoss(nn.Module):
    """
    Multilabel focal loss with logits.
    alpha: scalar or None
    gamma: focusing parameter
    reduction: 'mean' over batch/classes
    """
    def __init__(self, alpha: Optional[float]=0.25, gamma: float=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)                           # (B,C)
        probs = torch.sigmoid(logits)
        pt = probs*targets + (1.0 - probs)*(1.0 - targets)        # p_t
        focal = (1.0 - pt).clamp(min=1e-6).pow(self.gamma) * bce  # (B,C)
        if self.alpha is not None:
            focal = self.alpha*targets*focal + (1-self.alpha)*(1-targets)*focal
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal

def make_pos_weight(y_bin_train: np.ndarray, eps=1e-6) -> torch.Tensor:
    N = y_bin_train.shape[0]; P = y_bin_train.sum(axis=0)
    pw = (N - P) / np.clip(P, eps, None)
    pw = np.clip(pw, 1.0, 100.0)
    return torch.tensor(pw, dtype=torch.float32)

# -------------------- Metrics helpers --------------------
def _prepare_threshold(threshold, num_classes: int, device, dtype):
    import numpy as np
    if isinstance(threshold, torch.Tensor):
        t = threshold.to(device=device, dtype=dtype)
        if t.numel() == 1:
            return t.view(1, 1).expand(1, num_classes)
        assert t.numel() == num_classes, f"threshold has {t.numel()} values, expected {num_classes}"
        return t.view(1, -1)
    elif isinstance(threshold, (list, tuple, np.ndarray)):
        t = torch.as_tensor(threshold, device=device, dtype=dtype)
        if t.numel() == 1:
            return t.view(1, 1).expand(1, num_classes)
        assert t.numel() == num_classes, f"threshold has {t.numel()} values, expected {num_classes}"
        return t.view(1, -1)
    else:
        return torch.tensor(float(threshold), device=device, dtype=dtype).view(1, 1).expand(1, num_classes)

@torch.no_grad()
def f1_scores_from_logits(logits, targets, threshold, eps=1e-8):
    probs = torch.sigmoid(logits)
    C = probs.shape[1]
    thr = _prepare_threshold(threshold, C, probs.device, probs.dtype)

    preds = (probs >= thr).float()
    TP = (preds * targets).sum(dim=0)
    FP = (preds * (1 - targets)).sum(dim=0)
    FN = ((1 - preds) * targets).sum(dim=0)

    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1_per_c  = 2 * precision * recall / (precision + recall + eps)

    macro_f1 = f1_per_c.mean().item()
    TPm, FPm, FNm = TP.sum(), FP.sum(), FN.sum()
    micro_prec = TPm / (TPm + FPm + eps)
    micro_rec  = TPm / (TPm + FNm + eps)
    micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec + eps)).item()
    subset_acc = (preds.eq(targets).all(dim=1).float().mean().item())
    return micro_f1, macro_f1, subset_acc

@torch.no_grad()
def collect_targets_probs(model, loader, device):
    model.eval()
    all_t, all_l = [], []
    for imgs, targets in loader:
        imgs = torch.nan_to_num(imgs, nan=0.0, posinf=1.0, neginf=0.0).to(device)
        targets = targets.to(device)
        logits = model(imgs)
        all_t.append(targets.cpu())
        all_l.append(logits.cpu())
    y_true = torch.cat(all_t, 0).numpy()
    y_prob = 1.0/(1.0 + np.exp(-torch.cat(all_l,0).numpy()))
    return y_true, y_prob

def compute_per_class_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    C = y_true.shape[1]
    ths = np.zeros(C, dtype=np.float32)
    from sklearn.metrics import precision_recall_curve
    for c in range(C):
        y_t = y_true[:, c]
        y_p = y_prob[:, c]
        if len(np.unique(y_t)) < 2:
            ths[c] = 0.5
            continue
        prec, rec, thr = precision_recall_curve(y_t, y_p)
        f1 = 2*prec*rec/(prec+rec+1e-12)
        if thr.size == 0:
            ths[c] = 0.5
            continue
        best_idx = np.nanargmax(f1[1:])
        ths[c] = float(np.clip(thr[best_idx], 0.05, 0.95))
    return ths

# -------------------- Train/Eval loops --------------------
def train_one_epoch(model, loader, criterion, optimizer, device, threshold, max_grad_norm=5.0):
    model.train()
    loss_sum, n_items = 0.0, 0
    f1_micro_sum = f1_macro_sum = subset_acc_sum = 0.0
    n_batches = 0
    for imgs, targets in loader:
        imgs = torch.nan_to_num(imgs, nan=0.0, posinf=1.0, neginf=0.0).to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = torch.nan_to_num(model(imgs), nan=0.0, posinf=1e3, neginf=-1e3)
        loss = criterion(logits, targets)
        if not torch.isfinite(loss): continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        B = imgs.size(0)
        loss_sum += float(loss) * B
        n_items  += B
        mi, ma, sa = f1_scores_from_logits(logits.detach(), targets, threshold)
        f1_micro_sum += mi; f1_macro_sum += ma; subset_acc_sum += sa
        n_batches += 1
    return (loss_sum / max(n_items,1),
            f1_micro_sum/max(n_batches,1),
            f1_macro_sum/max(n_batches,1),
            subset_acc_sum/max(n_batches,1))

@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold):
    model.eval()
    loss_sum, n_items = 0.0, 0
    f1_micro_sum = f1_macro_sum = subset_acc_sum = 0.0
    n_batches = 0
    for imgs, targets in loader:
        imgs = torch.nan_to_num(imgs, nan=0.0, posinf=1.0, neginf=0.0).to(device)
        targets = targets.to(device)
        logits = model(imgs)
        loss = criterion(logits, targets)
        B = imgs.size(0)
        loss_sum += float(loss) * B
        n_items  += B
        mi, ma, sa = f1_scores_from_logits(logits, targets, threshold)
        f1_micro_sum += mi; f1_macro_sum += ma; subset_acc_sum += sa
        n_batches += 1
    return (loss_sum/max(n_items,1),
            f1_micro_sum/max(n_batches,1),
            f1_macro_sum/max(n_batches,1),
            subset_acc_sum/max(n_batches,1))

class EarlyStopper:
    def __init__(self, patience=12, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad = 0
    def step(self, val_loss):
        if self.best is None or (self.best - val_loss) > self.min_delta:
            self.best = val_loss
            self.bad = 0
            return False
        self.bad += 1
        return self.bad > self.patience

# -------------------- Main --------------------
def main():
    args = get_args()
    set_seed(args.seed)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(args.outdir, ts)
    os.makedirs(outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.quiet:
        print("Device:", device); print("Saving to:", outdir)

    with open(os.path.join(outdir, "RUN_INFO.txt"), "w") as f:
        f.write(f"started: {ts}\n")
        for k in sorted(vars(args).keys()):
            f.write(f"{k}: {getattr(args,k)}\n")
        for k in ["SLURM_JOB_ID","SLURM_JOB_NAME","SLURM_NODELIST","CUDA_VISIBLE_DEVICES"]:
            if k in os.environ: f.write(f"{k}: {os.environ[k]}\n")

    # ---------- Load REAL H5 (filtered) ----------
    with h5py.File(args.h5, "r") as f:
        X = f["data"][:]
        raw_labels = f["labels"][:]
    if X.ndim == 3: X = X[..., None]
    elif X.ndim == 4 and X.shape[-1] == 1: pass
    else: raise ValueError(f"Unexpected X shape {X.shape}")

    selected = list(args.labels)
    class_to_index = {c: i for i, c in enumerate(selected)}
    class_names = [f"Type {c}" if c != 6 else "Nothing (6)" for c in selected]
    num_classes = len(selected)

    label_lists = [parse_label_string(v) for v in raw_labels]
    keep_mask   = np.array([filter_keep(L, selected, args.filter_mode)
                            for L in label_lists], dtype=bool)
    X = X[keep_mask]
    label_lists = [label_lists[i] for i in np.where(keep_mask)[0]]
    Y = np.stack([to_subset_multihot(L, selected, num_classes, class_to_index)
                  for L in label_lists], axis=0)

    if not args.quiet:
        print(f"Loaded REAL: X {X.shape} | Y {Y.shape} (subset={selected}, mode={args.filter_mode})")

    # ---------- Splits on REAL data only ----------
    if args.split_path is not None and os.path.isfile(args.split_path):
        if not args.quiet:
            print(f"Loading existing splits from {args.split_path}")
        data = np.load(args.split_path)
        tr_idx, va_idx, te_idx = data["tr_idx"], data["va_idx"], data["te_idx"]
    else:
        tr_idx, va_idx, te_idx = multilabel_stratified_splits(
            X, Y, args.test_split, args.val_split, args.seed
        )
        if args.split_path is not None:
            os.makedirs(os.path.dirname(args.split_path), exist_ok=True)
            np.savez(args.split_path, tr_idx=tr_idx, va_idx=va_idx, te_idx=te_idx)
            if not args.quiet:
                print(f"Saved new splits to {args.split_path}")

    X_train, Y_train = X[tr_idx], Y[tr_idx]
    X_val,   Y_val   = X[va_idx], Y[va_idx]
    X_test,  Y_test  = X[te_idx], Y[te_idx]

    if not args.quiet:
        print(f"Initial REAL-only splits -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
        print("REAL train positives per class (1..C):", Y_train.sum(axis=0).astype(int))

    # ---------- Optional: append GAN data to TRAIN ONLY ----------
    if args.h5_gan:
        X_gan_all, Y_gan_all = [], []
        for gpath in args.h5_gan:
            if not os.path.isfile(gpath):
                if not args.quiet:
                    print(f"[WARN] GAN file not found, skipping: {gpath}")
                continue
            if not args.quiet:
                print(f"Loading GAN H5: {gpath}")
            with h5py.File(gpath, "r") as fg:
                Xg = fg["data"][:]
                raw_g = fg["labels"][:]
            if Xg.ndim == 3:
                Xg = Xg[..., None]
            elif Xg.ndim == 4 and Xg.shape[-1] == 1:
                pass
            else:
                raise ValueError(f"Unexpected GAN X shape {Xg.shape} in {gpath}")
            label_lists_g = [parse_label_string(v) for v in raw_g]
            keep_mask_g = np.array(
                [filter_keep(L, selected, args.filter_mode) for L in label_lists_g],
                dtype=bool
            )
            if keep_mask_g.sum() == 0:
                if not args.quiet:
                    print(f"[WARN] No GAN samples kept after filtering from {gpath}")
                continue
            Xg = Xg[keep_mask_g]
            label_lists_g = [label_lists_g[i] for i in np.where(keep_mask_g)[0]]
            Yg = np.stack(
                [to_subset_multihot(L, selected, num_classes, class_to_index) for L in label_lists_g],
                axis=0
            )
            X_gan_all.append(Xg)
            Y_gan_all.append(Yg)

        if X_gan_all:
            X_gan_all = np.concatenate(X_gan_all, axis=0)
            Y_gan_all = np.concatenate(Y_gan_all, axis=0)
            if not args.quiet:
                print(f"GAN samples added to TRAIN: {len(X_gan_all)}")
                print("GAN positives per class (1..C):", Y_gan_all.sum(axis=0).astype(int))
            X_train = np.concatenate([X_train, X_gan_all], axis=0)
            Y_train = np.concatenate([Y_train, Y_gan_all], axis=0)
            if not args.quiet:
                print(f"Final TRAIN size (REAL + GAN): {len(X_train)}")
                print("Total train positives per class (1..C):", Y_train.sum(axis=0).astype(int))
        else:
            if not args.quiet:
                print("[INFO] No usable GAN samples loaded; training on REAL only.")

    # ---------- DataLoaders ----------
    img_hw = tuple(args.img)
    train_ds = DynspecMultiLabelDataset(X_train, Y_train, img_hw)
    val_ds   = DynspecMultiLabelDataset(X_val,   Y_val,   img_hw)
    test_ds  = DynspecMultiLabelDataset(X_test,  Y_test,  img_hw)

    pin_mem = torch.cuda.is_available()

    # Class-balanced sampler (after GAN augmentation)
    if args.class_sampler == "balanced":
        class_pos = np.clip(Y_train.sum(axis=0), 1, None)
        inv = 1.0 / class_pos
        samp_w = (Y_train * inv[None, :]).sum(axis=1)
        samp_w = np.asarray(samp_w, dtype=np.float64)
        sampler = WeightedRandomSampler(weights=samp_w,
                                        num_samples=len(train_ds),
                                        replacement=True)
        shuffle = False
    else:
        sampler = None; shuffle = True

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=shuffle, sampler=sampler,
                              num_workers=args.num_workers,
                              pin_memory=pin_mem, persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=pin_mem, persistent_workers=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=args.num_workers,
                              pin_memory=pin_mem, persistent_workers=False)

    # ---------- Model/Loss/Opt ----------
    model = YOLOClassifier(num_classes=num_classes, input_hw=img_hw).to(device)

    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha,
                              gamma=args.focal_gamma, reduction="mean")
    else:
        pos_weight = make_pos_weight(Y_train).to(device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                   betas=(args.beta1, args.beta2),
                                   weight_decay=args.wd)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=args.patience, min_lr=args.min_lr
    )
    earlystop  = EarlyStopper(patience=args.es_patience,
                              min_delta=args.es_min_delta)

    # ---------- Train ----------
    history = []
    best_metric, best_state = -np.inf, None
    best_epoch = -1
    train_threshold = args.threshold  # scalar during training

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_f1mic, tr_f1mac, tr_sacc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            threshold=train_threshold
        )
        va_loss, va_f1mic, va_f1mac, va_sacc = evaluate(
            model, val_loader, criterion, device,
            threshold=train_threshold
        )
        scheduler.step(va_loss)

        row = dict(epoch=epoch,
                   train_loss=tr_loss, train_microF1=tr_f1mic,
                   train_macroF1=tr_f1mac, train_subsetAcc=tr_sacc,
                   val_loss=va_loss,   val_microF1=va_f1mic,
                   val_macroF1=va_f1mac,   val_subsetAcc=va_sacc,
                   lr=float(optimizer.param_groups[0]["lr"]))
        history.append(row)
        if not args.quiet:
            print(f"Epoch {epoch:03d} | "
                  f"train_loss {tr_loss:.4f} microF1 {tr_f1mic:.4f} macroF1 {tr_f1mac:.4f} subset {tr_sacc:.4f} | "
                  f"val_loss {va_loss:.4f} microF1 {va_f1mic:.4f} macroF1 {va_f1mac:.4f} subset {va_sacc:.4f} | "
                  f"lr {row['lr']:.2e}")

        if va_f1mac > best_metric:
            best_metric = va_f1mac
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = epoch

        if earlystop.step(va_loss):
            if not args.quiet:
                print(f"[EarlyStopping] No val_loss improvement for {earlystop.patience} epochs — stopping.")
            break

    # Save history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(outdir, "metrics.csv"), index=False)

    # Save best & final
    if best_state is not None:
        model.load_state_dict(best_state)
        best_path = os.path.join(outdir, f"best_{args.save_name}")
        torch.save(model.state_dict(), best_path)
        if not args.quiet:
            print(f"Saved BEST model to {best_path} (val_macroF1={best_metric:.4f}, epoch={best_epoch})")
    final_path = os.path.join(outdir, f"final_{args.save_name}")
    torch.save(model.state_dict(), final_path)

    # ---------- Per-class thresholds from VAL ----------
    yv_true, yv_prob = collect_targets_probs(model, val_loader, device)
    if args.threshold_mode == "perclass":
        ths = compute_per_class_thresholds(yv_true, yv_prob)
        with open(os.path.join(outdir, "per_class_thresholds.json"), "w") as f:
            json.dump({str(i): float(t) for i, t in enumerate(ths)}, f, indent=2)
        thr_eval = ths
        if not args.quiet:
            print("Per-class thresholds:", np.round(ths, 3))
    else:
        thr_eval = args.threshold

    # ---------- Test ----------
    te_loss, te_f1mic, te_f1mac, te_sacc = evaluate(
        model, test_loader, criterion, device, threshold=thr_eval
    )
    test_report = {
        "loss": round(te_loss, 6),
        "microF1": round(te_f1mic, 6),
        "macroF1": round(te_f1mac, 6),
        "subset_acc": round(te_sacc, 6),
        "class_order_1idx": selected,
        "class_names": [str(n) for n in class_names],
        "best_val_macroF1": round(best_metric, 6),
        "best_epoch": int(best_epoch)
    }

    # --- Per-class metrics on TEST ---
    yt_true, yt_prob = collect_targets_probs(model, test_loader, device)
    if isinstance(thr_eval, (list, np.ndarray)):
        thr_arr = np.array(thr_eval, dtype=np.float32)[None, :]
        yt_pred = (yt_prob >= thr_arr).astype(int)
    else:
        yt_pred = (yt_prob >= args.threshold).astype(int)

    prec, rec, f1, support = precision_recall_fscore_support(
        yt_true, yt_pred, average=None, zero_division=0
    )

    roc_aucs, pr_aucs = [], []
    for c in range(yt_true.shape[1]):
        try: roc_aucs.append(roc_auc_score(yt_true[:, c], yt_prob[:, c]))
        except ValueError: roc_aucs.append(np.nan)
        try: pr_aucs.append(average_precision_score(yt_true[:, c], yt_prob[:, c]))
        except ValueError: pr_aucs.append(np.nan)

    per_class_rows = []
    for i, cname in enumerate(class_names):
        per_class_rows.append({
            "class_index": i,
            "class_name": cname,
            "label_1idx": selected[i],
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "roc_auc": None if np.isnan(roc_aucs[i]) else float(roc_aucs[i]),
            "pr_auc":  None if np.isnan(pr_aucs[i]) else float(pr_aucs[i]),
            "threshold_used": float(thr_eval[i]) if isinstance(thr_eval, (list, np.ndarray)) else float(args.threshold)
        })
    pd.DataFrame(per_class_rows).to_csv(os.path.join(outdir, "per_class_metrics.csv"), index=False)
    with open(os.path.join(outdir, "per_class_metrics.json"), "w") as f:
        json.dump(per_class_rows, f, indent=2)

    # --- Multilabel confusion matrix (per-class 2x2) on TEST ---
    mcm = multilabel_confusion_matrix(yt_true, yt_pred)
    cm_rows = []
    for i, cname in enumerate(class_names):
        tn, fp, fn, tp = int(mcm[i,0,0]), int(mcm[i,0,1]), int(mcm[i,1,0]), int(mcm[i,1,1])
        cm_rows.append({"class_index": i, "class_name": cname,
                        "TN": tn, "FP": fp, "FN": fn, "TP": tp})
    pd.DataFrame(cm_rows).to_csv(os.path.join(outdir, "confusion_matrices.csv"), index=False)
    with open(os.path.join(outdir, "confusion_matrices.json"), "w") as f:
        json.dump(cm_rows, f, indent=2)

    if not args.quiet:
        print("\n=== TEST: Per-class metrics (thresholds) ===")
        colw = 14
        header = f"{'Class':<{colw}} {'P':>6} {'R':>6} {'F1':>6} {'Sup':>6} {'ROC-AUC':>8} {'PR-AUC':>8} {'Thr':>6}"
        print(header); print("-"*len(header))
        for r in per_class_rows:
            print(f"{r['class_name']:<{colw}} "
                  f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} "
                  f"{r['support']:>6d} "
                  f"{(r['roc_auc'] if r['roc_auc'] is not None else float('nan')):>8.3f} "
                  f"{(r['pr_auc']  if r['pr_auc']  is not None else float('nan')):>8.3f} "
                  f"{r['threshold_used']:>6.2f}")

        print("\n=== TEST: Multilabel confusion matrix (per-class TN FP / FN TP) ===")
        for r in cm_rows:
            print(f"{r['class_name']:<{colw}} TN={r['TN']:4d} FP={r['FP']:4d} | FN={r['FN']:4d} TP={r['TP']:4d}")

    with open(os.path.join(outdir, "test_report.json"), "w") as f:
        json.dump(test_report, f, indent=2)

    # ---------- Plots ----------
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            # Loss
            plt.figure(figsize=(6,4))
            plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train")
            plt.plot(hist_df["epoch"], hist_df["val_loss"],   label="val")
            plt.xlabel("Epoch"); plt.ylabel("Loss")
            plt.title("Loss ({} loss)".format(args.loss.upper()))
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=150); plt.close()
            # F1
            plt.figure(figsize=(6,4))
            plt.plot(hist_df["epoch"], hist_df["train_macroF1"], label="train macro-F1")
            plt.plot(hist_df["epoch"], hist_df["val_macroF1"],   label="val macro-F1")
            plt.xlabel("Epoch"); plt.ylabel("F1"); plt.title("Macro-F1")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(outdir, "f1_curve.png"), dpi=150); plt.close()
        except Exception as e:
            print("[WARN] Plotting failed:", e)

if __name__ == "__main__":
    main()
