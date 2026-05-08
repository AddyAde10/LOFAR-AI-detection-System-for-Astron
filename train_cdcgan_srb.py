#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cDCGAN for LOFAR dynspec (multi/composite labels supported)
- Target labels can include composites, e.g.: --target_labels "2,4,14,35,34,135"
- If H5 labels are multi-hot, we map active indices -> canonical composite int (sorted digits, e.g., {1,3,5} -> 135)
- Aspect-preserving resize to (256,160)
- Dynspec-aware DiffAug on D (+ optional strong affine jitter)
- Hinge + R1 + feature matching  (or)  WGAN-GP + feature matching  (switch via --loss)
- Class-balanced sampler (weighted) or optional per-batch balancer
- Saves EXACTLY --num_gen total samples per checkpoint into OUTDIR/gan_samples/ every --save_every epochs
- Calculates FID score every --save_every epochs
"""

import os, math, json, random, argparse, warnings, copy
from pathlib import Path
warnings.filterwarnings("ignore")

import h5py
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler

# <-- NEW: Imports for augmentation and FID -->
import scipy.ndimage
from scipy.linalg import sqrtm
import torchvision.models

# -------------------- Utils --------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def safe(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    finite_vals = arr[finite]
    med = float(np.median(finite_vals)); lo = float(np.min(finite_vals)); hi = float(np.max(finite_vals))
    return np.nan_to_num(arr, nan=med, posinf=hi, neginf=lo).astype(np.float32)

def robust_minmax(x: np.ndarray, lo=1, hi=99) -> np.ndarray:
    x = safe(x)
    xf = x[np.isfinite(x)]
    if xf.size == 0: return np.zeros_like(x, dtype=np.float32)
    a, b = np.percentile(xf, lo), np.percentile(xf, hi)
    if not np.isfinite(a): a = float(np.min(xf))
    if not np.isfinite(b): b = float(np.max(xf))
    if b <= a: return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, a, b)
    x = (x - a) / (b - a + 1e-8)
    return (x * 2.0 - 1.0).astype(np.float32)

def resize_hw(img: np.ndarray, H=256, W=160) -> np.ndarray:
    t = torch.from_numpy(img[None, None, ...].astype(np.float32))
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t[0,0].numpy()

# -------------------- Dynspec-aware augmentations --------------------
def dynspec_diffaug(x, translate_t=6, translate_f=4, cut_rows=0.05, cut_cols=0.05, contrast=0.15, noise_std=0.02):
    B, C, H, W = x.shape
    device = x.device

    if translate_t > 0 or translate_f > 0:
        sh = torch.randint(-translate_f, translate_f+1, (B,), device=device)  # freq shift
        sw = torch.randint(-translate_t, translate_t+1, (B,), device=device)  # time shift
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device),
                                        torch.arange(W, device=device), indexing='ij')
        grid_y = (grid_y.unsqueeze(0) - sh.view(-1,1,1)).clamp(0, H-1)
        grid_x = (grid_x.unsqueeze(0) - sw.view(-1,1,1)).clamp(0, W-1)
        x = x.gather(2, grid_y.unsqueeze(1).expand(-1,1,-1,-1)).gather(3, grid_x.unsqueeze(1).expand(-1,1,-1,-1))

    if cut_rows > 0:
        num_rows = max(1, int(H * cut_rows))
        for b in range(B):
            r0 = torch.randint(0, H - num_rows + 1, (1,), device=device).item()
            x[b, :, r0:r0+num_rows, :].zero_()

    if cut_cols > 0:
        num_cols = max(1, int(W * cut_cols))
        for b in range(B):
            c0 = torch.randint(0, W - num_cols + 1, (1,), device=device).item()
            x[b, :, :, c0:c0+num_cols].zero_()

    if contrast > 0:
        m = x.mean(dim=(2,3), keepdim=True)
        alpha = 1.0 + (torch.rand(B,1,1,1, device=device)*2-1)*contrast
        x = (x - m) * alpha + m
        x = x.clamp(-1, 1)

    if noise_std > 0:
        x = (x + torch.randn_like(x) * noise_std).clamp(-1, 1)

    return x

def affine_timefreq(x, scale_t=0.05, scale_f=0.05, shear=0.03):
    """Differentiable scale + gentle shear (slope jitter) in time/frequency."""
    B, C, H, W = x.shape
    device = x.device
    st = 1.0 + (torch.rand(B, device=device)*2-1)*scale_t
    sf = 1.0 + (torch.rand(B, device=device)*2-1)*scale_f
    sh = (torch.rand(B, device=device)*2-1)*shear  # shear in time
    theta = torch.zeros(B,2,3, device=device)
    theta[:,0,0] = sf;  theta[:,0,1] = sh;  theta[:,0,2] = 0.0
    theta[:,1,0] = 0.0; theta[:,1,1] = st;  theta[:,1,2] = 0.0
    grid = F.affine_grid(theta, size=x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

def dynspec_diffaug_strong(x):
    # base augs that already backprop fine
    x = dynspec_diffaug(x, translate_t=6, translate_f=4, cut_rows=0.05, cut_cols=0.05, contrast=0.15, noise_std=0.02)

    # slope-ish jitter via slight anisotropic scaling (interpolate has backward everywhere)
    B, C, H, W = x.shape
    # random scales around 1.0 (±6%)
    scale_f = 1.0 + (torch.rand(B, device=x.device)*2-1)*0.06
    scale_t = 1.0 + (torch.rand(B, device=x.device)*2-1)*0.06

    outs = []
    for b in range(B):
        h2 = max(1, int(round(H * float(scale_f[b]))))
        w2 = max(1, int(round(W * float(scale_t[b]))))
        xb = F.interpolate(x[b:b+1], size=(h2, w2), mode="bilinear", align_corners=False)

        # center-crop or pad back to HxW
        dh = max(0, h2 - H); dw = max(0, w2 - W)
        if dh > 0 or dw > 0:
            top = dh//2; left = dw//2
            xb = xb[:, :, top:top+min(H,h2), left:left+min(W,w2)]
        ph = H - xb.size(2); pw = W - xb.size(3)
        if ph > 0 or pw > 0:
            xb = F.pad(xb, (pw//2, pw - pw//2, ph//2, ph - ph//2), mode="replicate")

        outs.append(xb)
    return torch.cat(outs, dim=0)


# -------------------- Composite label helpers --------------------
def vec_to_composite_int(vec: np.ndarray) -> int:
    idx = np.where(vec > 0.5)[0].tolist()
    if len(idx) == 0: return 0
    s = ''.join(str(i) for i in sorted(idx))
    try: return int(s)
    except ValueError: return 0

def normalize_target_labels(target_str: str):
    labs = sorted({int(s.strip()) for s in target_str.split(',') if s.strip()!=''})
    return labs

# -------------------- Dataset --------------------
class H5CompositeDataset(Dataset):
    # <-- NEW: Added unfiltered_h5_path and changed min_per_class to aug_target_count -->
    def __init__(self, h5_path, target_labels, unfiltered_h5_path=None, H=256, W=160, aug_target_count=500, pre_aug=True, seed=123):
        set_seed(seed)
        self.H, self.W = H, W
        self.lab_list = sorted(list(target_labels))
        lab2idx = {lab:i for i,lab in enumerate(self.lab_list)}
        idx2lab = {i:lab for i,lab in enumerate(self.lab_list)}  # <-- local map available early

        # --- Load Filtered Data ---
        with h5py.File(h5_path, "r") as f:
            data = f["data"][:]      # (N,H0,W0)
            labels = f["labels"][:]    # (N,) ints OR (N,C) multi-hot
        if labels.ndim == 1:
            comp = labels.astype(int)
        else:
            comp = np.array([vec_to_composite_int(v) for v in labels], dtype=int)
        
        keep_mask = np.isin(comp, self.lab_list)
        X_raw_list = [data[keep_mask]]
        Y_comp_list = [comp[keep_mask]]
        print(f"[Dataset] Found {len(Y_comp_list[0])} matching samples in filtered file: {h5_path}")

        # <-- NEW: Load Unfiltered Type 2 Data -->
        if unfiltered_h5_path is not None and 2 in self.lab_list:
            print(f"[Dataset] Loading unfiltered Type 2 bursts from: {unfiltered_h5_path}")
            try:
                with h5py.File(unfiltered_h5_path, "r") as f_u:
                    data_u = f_u["data"][:]
                    labels_u = f_u["labels"][:]

                if labels_u.ndim == 1:
                    comp_u = labels_u.astype(int)
                else:
                    comp_u = np.array([vec_to_composite_int(v) for v in labels_u], dtype=int)
                
                # We ONLY want pure Type 2 from the unfiltered set
                u_mask = (comp_u == 2)
                
                if u_mask.any():
                    X_raw_list.append(data_u[u_mask])
                    Y_comp_list.append(comp_u[u_mask])
                    print(f"[Dataset] Added {u_mask.sum()} unfiltered Type 2 samples.")
                else:
                    print("[Dataset] No Type 2 samples found in unfiltered file.")
            except Exception as e:
                print(f"[Dataset] WARNING: Could not load unfiltered file {unfiltered_h5_path}. Error: {e}")

        X_raw = np.concatenate(X_raw_list, axis=0)
        Y_comp = np.concatenate(Y_comp_list, axis=0)
        print(f"[Dataset] Total raw samples before preprocessing: {len(X_raw)}")
        # --- End New Data Loading ---

        # Preprocess
        Xp, Yp = [], []
        for i in range(len(X_raw)):
            im = robust_minmax(resize_hw(X_raw[i], H=H, W=W), lo=1, hi=99)
            Xp.append(im); Yp.append(lab2idx[int(Y_comp[i])])

        if len(Xp) == 0:
            raise RuntimeError("No samples matched target_labels. Check --target_labels vs your H5 labels.")
        Xp = np.stack(Xp, axis=0).astype(np.float32)
        Yp = np.array(Yp, dtype=np.int64)

        # Offline expansion
        if pre_aug:
            X_list, Y_list = [], []
            counts = np.bincount(Yp, minlength=len(self.lab_list))
            for cls_idx in range(len(self.lab_list)):
                Xc = Xp[Yp == cls_idx]
                # <-- NEW: Use aug_target_count -->
                need = max(0, aug_target_count - Xc.shape[0])
                
                # keep originals
                if Xc.shape[0] > 0:
                    X_list.append(Xc)
                    Y_list.append(np.full(Xc.shape[0], cls_idx, dtype=np.int64))
                else:
                    print(f"[Dataset] WARNING: No samples for class {idx2lab[cls_idx]} (local idx {cls_idx}). Cannot augment.")
                    continue

                # synth copies
                if need > 0 and Xc.shape[0] > 0:
                    for _ in range(need):
                        src = Xc[np.random.randint(0, Xc.shape[0])].copy()
                        is_cls2 = (idx2lab[cls_idx] == 2)

                        # contrast
                        if np.random.rand() < (0.9 if is_cls2 else 0.7):
                            m = src.mean()
                            alpha = 1.0 + (np.random.rand()*2 - 1) * (0.25 if is_cls2 else 0.15)
                            src = ((src - m) * alpha + m).clip(-1, 1)

                        # noise (slightly lower for type-2)
                        if np.random.rand() < (0.5 if is_cls2 else 0.7):
                            sigma = 0.01 if is_cls2 else 0.02
                            src = (src + np.random.randn(*src.shape) * sigma).clip(-1, 1)

                        # small rolls
                        if np.random.rand() < 0.7:  # time
                            src = np.roll(src, shift=np.random.randint(-4, 5), axis=1)
                        if np.random.rand() < 0.7:  # freq
                            src = np.roll(src, shift=np.random.randint(-3, 4), axis=0)

                        # <-- NEW: Blurring -->
                        if np.random.rand() < 0.3:
                            sigma = np.random.uniform(0.5, 1.2)
                            src = scipy.ndimage.gaussian_filter(src, sigma=sigma)

                        # <-- NEW: Random Crop/Zoom -->
                        if np.random.rand() < 0.3:
                            zoom_f = np.random.uniform(1.0, 1.15) # Zoom factor for freq
                            zoom_t = np.random.uniform(1.0, 1.15) # Zoom factor for time
                            h, w = src.shape
                            src_z = scipy.ndimage.zoom(src, (zoom_f, zoom_t))
                            h_z, w_z = src_z.shape
                            top = max(0, (h_z - h) // 2)
                            left = max(0, (w_z - w) // 2)
                            src = src_z[top:top+h, left:left+w]
                            # Pad if zoom was < 1.0 (though we set > 1.0)
                            if src.shape[0] < H or src.shape[1] < W:
                                ph = max(0, H - src.shape[0]); pw = max(0, W - src.shape[1])
                                src = np.pad(src, ((ph//2, ph - ph//2), (pw//2, pw - pw//2)), 'constant', constant_values=-1.0)

                        # <-- NEW: Random Cutout (Feature Removal) -->
                        if np.random.rand() < 0.3:
                            num_cutouts = np.random.randint(1, 4)
                            for _ in range(num_cutouts):
                                h, w = src.shape
                                cut_h = np.random.randint(h//10, h//5)
                                cut_w = np.random.randint(w//10, w//5)
                                r0 = np.random.randint(0, max(1, h - cut_h))
                                c0 = np.random.randint(0, max(1, w - cut_w))
                                src[r0:r0+cut_h, c0:c0+cut_w] = -1.0 # Fill with min value

                        X_list.append(src[None, ...])
                        Y_list.append(np.array([cls_idx], dtype=np.int64))

            Xp = np.concatenate(X_list, axis=0)
            Yp = np.concatenate(Y_list, axis=0)

        self.X = Xp
        self.y = Yp
        self.idx2lab = idx2lab  # <-- now assign to self

        # diagnostics
        counts = np.bincount(self.y, minlength=len(self.lab_list))
        print(f"[Dataset] Total {len(self.X)} after expansion. Per-class:")
        for i, c in enumerate(counts):
            print(f"  Label {self.idx2lab[i]} -> {c}")


    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i]).unsqueeze(0).float()
        y = int(self.y[i]); return x, y

# -------------------- Optional per-batch balancer --------------------
class BalancedBatchSampler(Sampler):
    def __init__(self, labels: np.ndarray, num_classes: int, batch_size: int):
        self.labels = np.asarray(labels)
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.per = max(1, batch_size // num_classes)
        self.idxs_by_c = [np.where(self.labels == c)[0] for c in range(num_classes)]
        self.num_batches = int(math.ceil(len(self.labels) / float(batch_size)))

    def __len__(self): return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            idxs = []
            for c in range(self.num_classes):
                pool = self.idxs_by_c[c]
                if len(pool) == 0: continue
                pick = np.random.choice(pool, size=self.per, replace=len(pool) < self.per)
                idxs.append(pick)
            idxs = np.concatenate(idxs) if len(idxs) else np.array([], dtype=np.int64)
            if len(idxs) < self.batch_size:
                extra = np.random.choice(len(self.labels), size=self.batch_size - len(idxs), replace=True)
                idxs = np.concatenate([idxs, extra])
            np.random.shuffle(idxs)
            yield idxs.tolist()

# -------------------- Models --------------------
def sn(m): return nn.utils.spectral_norm(m)

class MinibatchStdDev(nn.Module):
    def __init__(self, group_size=16, eps=1e-8):
        super().__init__()
        self.group_size = group_size
        self.eps = eps

    def forward(self, x):
        # x: [N, C, H, W]  ->  concat an extra channel [N, 1, H, W]
        N, C, H, W = x.shape
        g = min(self.group_size, N)
        # ensure g divides N
        while N % g != 0 and g > 1:
            g -= 1
        m = N // g

        y = x.view(g, m, C, H, W)
        y = y - y.mean(dim=0, keepdim=True)          # [g, m, C, H, W]
        y = torch.sqrt((y ** 2).mean(dim=0) + self.eps)     # [m, C, H, W]
        y = y.mean(dim=(1, 2, 3), keepdim=True)        # [m, 1, 1, 1]

        # broadcast to [g, m, 1, H, W] then flatten to [N, 1, H, W]
        y = y.unsqueeze(0).expand(g, -1, -1, H, W).contiguous().view(N, 1, H, W)
        return torch.cat([x, y], dim=1)


class Gen(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=1, num_classes=3, cond_dim=64):
        super().__init__()
        self.embed = nn.Embedding(num_classes, cond_dim)
        in_latent = nz + cond_dim
        self.fc = nn.Linear(in_latent, ngf*8*16*10)
        self.blocks = nn.Sequential(
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, (4,4), (2,2), (1,1), bias=False), nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, (4,4), (2,2), (1,1), bias=False), nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   (4,4), (2,2), (1,1), bias=False), nn.BatchNorm2d(ngf),   nn.ReLU(True),
            nn.ConvTranspose2d(ngf,   nc,    (4,4), (2,2), (1,1), bias=False),
            nn.Tanh()
        )
    def forward(self, z, y):
        e = self.embed(y); h = torch.cat([z, e], dim=1)
        h = self.fc(h).view(h.size(0), -1, 16, 10)
        return self.blocks(h)

class Disc(nn.Module):
    def __init__(self, ndf=64, nc=1, num_classes=3, emb_dim=128, use_strong_aug=False):
        super().__init__()
        self.use_strong_aug = use_strong_aug
        self.net = nn.Sequential(
            sn(nn.Conv2d(nc,     ndf,   4, 2, 1, bias=True)),  nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(ndf,    ndf*2, 4, 2, 1, bias=True)),  nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(ndf*2,  ndf*4, 4, 2, 1, bias=True)),  nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(ndf*4,  ndf*8, 4, 2, 1, bias=True)),  nn.LeakyReLU(0.2, True),
            MinibatchStdDev(),  # <- added
        )
        self.emb = nn.Embedding(num_classes, emb_dim)
        self.lin = sn(nn.Linear((ndf*8 + 1)*16*10, emb_dim))
        self.out = sn(nn.Linear(emb_dim, 1))

    def forward(self, x, y, augment=True):
        x = x.to(torch.float32)
        if augment:
            x = dynspec_diffaug_strong(x) if self.use_strong_aug else dynspec_diffaug(x)
        fmaps = []
        h = x
        for layer in self.net:
            h = layer(h)
            if isinstance(layer, nn.LeakyReLU):
                fmaps.append(h)
        h = h.view(h.size(0), -1)
        feat = torch.tanh(self.lin(h))
        logits = self.out(feat)
        proj = (self.emb(y) * feat).sum(dim=1, keepdim=True)
        return logits + proj, feat, fmaps

# -------------------- Losses --------------------
def d_hinge_real(logits):  return torch.relu(1.0 - logits).mean()
def d_hinge_fake(logits):  return torch.relu(1.0 + logits).mean()
def g_hinge_fake(logits):  return -logits.mean()

# DELETE the old r1_penalty() that re-calls netD

def r1_from_real_logits(real, real_logits):
    # real: [B,1,H,W] with requires_grad=True
    grad = torch.autograd.grad(
        outputs=real_logits.sum(),
        inputs=real,
        create_graph=True,  # needed so D gets the R1 gradient
        retain_graph=True,  # keep graph for the rest of the D loss terms
        only_inputs=True
    )[0]
    gp = grad.view(grad.size(0), -1).pow(2).sum(dim=1).mean()
    return gp


def feature_matching_loss(fake_fmaps, real_fmaps):
    loss = 0.0
    for fr, rr in zip(fake_fmaps, real_fmaps):
        loss = loss + F.l1_loss(fr.mean(dim=0), rr.detach().mean(dim=0))
    return loss

# WGAN-GP helpers
def d_wgan(real_logits, fake_logits):
    return (-real_logits.mean() + fake_logits.mean())
def g_wgan(fake_logits):
    return -fake_logits.mean()
def gp_wgan(disc, gen, real, y, nz, lam=10.0):
    b = real.size(0)
    with torch.no_grad():
        z = torch.randn(b, nz, device=real.device)
        fake = gen(z, y)
    eps = torch.rand(b, 1, 1, 1, device=real.device)
    inter = eps*real + (1-eps)*fake
    inter.requires_grad_(True)
    out, _, _ = disc(inter, y, augment=True)
    grad = torch.autograd.grad(out.sum(), inter, create_graph=True)[0]
    return ((grad.view(b, -1).norm(2, dim=1) - 1.0) ** 2).mean() * lam


# -------------------- FID Calculation (NEW) --------------------

class InceptionV3Features(nn.Module):
    """Wrapper for InceptionV3 to extract features."""
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.inception_v3(weights='Inception_V3_Weights.DEFAULT', aux_logits=True)
        # Hook the 'Mixed_7c' layer (output is 2048-dim feature vector)
        self.model.Mixed_7c.register_forward_hook(self.hook)
        self.features = None

    def hook(self, module, input, output):
        self.features = F.adaptive_avg_pool2d(output, (1, 1)).view(output.size(0), -1)

    def forward(self, x):
        # x expected to be [N, 3, 299, 299] and normalized
        self.model(x)
        return self.features

def preprocess_for_inception(img_tensor):
    """Prepares a batch of 1-channel [-1, 1] images for InceptionV3."""
    # 1. Upscale to 299x299
    # (Input is [N, 1, H, W])
    img_tensor = F.interpolate(img_tensor, size=(299, 299), mode='bilinear', align_corners=False)
    # 2. Rescale from [-1, 1] to [0, 1]
    img_tensor = (img_tensor + 1.0) / 2.0
    # 3. Repeat channel to make [N, 3, 299, 299]
    img_tensor = img_tensor.repeat(1, 3, 1, 1)
    # 4. Normalize with InceptionV3 stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=img_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img_tensor.device).view(1, 3, 1, 1)
    return (img_tensor - mean) / std

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of FID."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Means must have the same shape"
    assert sigma1.shape == sigma2.shape, "Covariances must have the same shape"

    # Compute sum of squared difference in means
    ssdiff = np.sum((mu1 - mu2)**2.0)

    # Compute sqrt of product of covariances
    covmean = sqrtm(sigma1.dot(sigma2))

    # Check for imaginary numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # FID formula
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

@torch.no_grad()
def precompute_real_features(dataset, model, device, batch_size=32):
    """Pre-calculates InceptionV3 features for the entire real dataset."""
    model.eval()
    all_features = []
    # Use a simple dataloader for the dataset's .X field
    num_batches = int(math.ceil(len(dataset.X) / batch_size))
    
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(dataset.X))
        
        # Get raw numpy data, convert to tensor
        batch_np = dataset.X[start:end]
        batch_torch = torch.from_numpy(batch_np).unsqueeze(1).float().to(device)
        
        # Preprocess and get features
        batch_preprocessed = preprocess_for_inception(batch_torch)
        features = model(batch_preprocessed)
        all_features.append(features.cpu().numpy())
        
    all_features = np.concatenate(all_features, axis=0)
    mu = np.mean(all_features, axis=0)
    sigma = np.cov(all_features, rowvar=False)
    return mu, sigma

@torch.no_grad()
def calculate_fake_features_and_fid(netG_eval, inception_model, num_classes, nz, num_fid_samples, real_mu, real_sigma, device):
    """Generates fake images and computes their features and FID."""
    netG_eval.eval()
    all_features = []
    num_per_class = max(1, num_fid_samples // num_classes)
    num_to_gen = num_per_class * num_classes
    
    for i in range(num_classes):
        z = torch.randn(num_per_class, nz, device=device)
        y = torch.full((num_per_class,), i, dtype=torch.long, device=device)
        
        fakes = netG_eval(z, y)
        fakes_preprocessed = preprocess_for_inception(fakes)
        features = inception_model(fakes_preprocessed)
        all_features.append(features.cpu().numpy())
    
    all_features = np.concatenate(all_features, axis=0)
    mu = np.mean(all_features, axis=0)
    sigma = np.cov(all_features, rowvar=False)
    
    fid_score = calculate_fid(real_mu, real_sigma, mu, sigma)
    return fid_score

def generate_synthetic_h5(args, netG, ema, label_map, num_classes, device):
    """
    Generate a synthetic H5 with keys:
      ['data', 'freq_range', 'indices', 'labels', 'time_range', 'timestamps']

    - data: (N, H_orig, W_orig), float32 in [0, 1]
    - labels: (N,), int, using original label ids from label_map
    - indices: (N,), int64, 0..N-1
    - freq_range/time_range/timestamps: copied/sampled from the original H5

    Only called AFTER training is finished.
    """
    if args.num_h5 <= 0:
        return

    import math

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Decide output filename
    if args.h5_outname is not None:
        h5_out_path = outdir / args.h5_outname
    else:
        # e.g. synthetic_labels_2_4_14_N1000.h5
        lab_str = "_".join(str(l) for l in label_map)
        h5_out_path = outdir / f"synthetic_labels_{lab_str}_N{args.num_h5}.h5"

    print(f"[H5] Generating {args.num_h5} synthetic samples -> {h5_out_path}")

    # Build an EMA-eval generator (if EMA is used), otherwise use netG as-is
    if ema is not None:
        netG_eval = Gen(args.nz, args.ngf, 1, num_classes=num_classes).to(device)
        netG_eval.load_state_dict(ema.shadow, strict=False)
    else:
        netG_eval = netG
    netG_eval.eval()

    # Read original H5 shape + metadata
    with h5py.File(args.h5, "r") as f_in:
        if "data" not in f_in:
            raise RuntimeError(f"[H5] Input file {args.h5} has no 'data' dataset.")
        data_shape = f_in["data"].shape  # (N_base, H_orig, W_orig)
        if len(data_shape) != 3:
            raise RuntimeError(f"[H5] Expected 'data' to be 3D, got shape {data_shape}")
        base_N, H_orig, W_orig = data_shape
        print(f"[H5] Base data shape: N={base_N}, H={H_orig}, W={W_orig}")

        # Load per-sample/global metadata if present
        base_meta_per_sample = {}
        base_meta_global = {}
        for key in ["freq_range", "time_range", "timestamps"]:
            if key in f_in:
                arr = f_in[key][:]
                arr = np.asarray(arr)
                # Treat as per-sample if first dim == base_N
                if arr.ndim >= 1 and arr.shape[0] == base_N:
                    base_meta_per_sample[key] = arr
                    print(f"[H5] Found per-sample meta '{key}' with shape {arr.shape}")
                else:
                    base_meta_global[key] = arr
                    print(f"[H5] Found global meta '{key}' with shape {arr.shape}")
            else:
                print(f"[H5] WARNING: key '{key}' not found in {args.h5}; will be omitted from synthetic file.")

    N_total = int(args.num_h5)
    if N_total <= 0:
        print("[H5] num_h5 <= 0, nothing to do.")
        return

    # Allocate output H5 and datasets
    with h5py.File(h5_out_path, "w") as f_out:
        # Main datasets
        d_data = f_out.create_dataset(
            "data",
            shape=(N_total, H_orig, W_orig),
            dtype="float32",
            compression="gzip",
            chunks=(min(64, N_total), H_orig, W_orig),
            shuffle=True,
        )
        d_labels = f_out.create_dataset(
            "labels",
            shape=(N_total,),
            dtype="int64",
            compression="gzip",
            shuffle=True,
        )
        d_indices = f_out.create_dataset(
            "indices",
            data=np.arange(N_total, dtype="int64"),
            dtype="int64",
            compression="gzip",
            shuffle=True,
        )

        # Metadata datasets
        meta_out = {}
        # per-sample: shape (N_total, ...) and we will fill by sampling
        for key, base in base_meta_per_sample.items():
            extra_shape = base.shape[1:]  # e.g. (2,) or (k,...)
            meta_out[key] = f_out.create_dataset(
                key,
                shape=(N_total,) + extra_shape,
                dtype=base.dtype,
                compression="gzip",
                shuffle=True,
            )
        # global: broadcast to all samples
        for key, base in base_meta_global.items():
            base = np.asarray(base)
            extra_shape = base.shape  # broadcast entire thing
            d = f_out.create_dataset(
                key,
                shape=(N_total,) + extra_shape,
                dtype=base.dtype,
                compression="gzip",
                shuffle=True,
            )
            # broadcast write
            d[...] = np.broadcast_to(base, (N_total,) + extra_shape)
            meta_out[key] = d

        # store label_map as attribute for later reference
        f_out.attrs["label_map"] = np.array(label_map, dtype=np.int64)

        # Decide how many per class (roughly equal)
        counts = [0] * num_classes
        if num_classes > 0:
            per_class = max(1, N_total // num_classes)
            leftover = N_total - per_class * num_classes
            for c in range(num_classes):
                counts[c] = per_class
            for c in range(leftover):
                counts[c] += 1
        else:
            # no classes? shouldn't happen
            counts = [N_total]

        print(f"[H5] Per-class synthetic counts (local class ids): {counts}")

        # Generation loop
        ptr = 0
        batch_size = max(1, args.batch)

        for c_local, n_c in enumerate(counts):
            if n_c <= 0:
                continue
            label_global = int(label_map[c_local])

            remaining = n_c
            while remaining > 0:
                cur = min(batch_size, remaining)
                remaining -= cur

                # Latent + labels
                z = torch.randn(cur, args.nz, device=device)
                y_local = torch.full((cur,), c_local, dtype=torch.long, device=device)

                with torch.no_grad():
                    fake = netG_eval(z, y_local)  # [cur,1,256,160] in [-1,1]
                    fake = fake.to(torch.float32)
                    # Resize back to original H/W and map to [0,1]
                    fake_up = F.interpolate(
                        fake, size=(H_orig, W_orig),
                        mode="bilinear", align_corners=False
                    )
                    fake_up = (fake_up + 1.0) / 2.0  # [-1,1] -> [0,1]
                    fake_up = fake_up.clamp(0.0, 1.0)

                fake_np = fake_up.squeeze(1).cpu().numpy().astype("float32")

                # Write into H5
                j0, j1 = ptr, ptr + cur
                d_data[j0:j1, :, :] = fake_np
                d_labels[j0:j1] = label_global

                # Metadata: per-sample -> random rows from base
                # we re-open base file here only for shapes? No, we already loaded base arrays.
                for key, base in base_meta_per_sample.items():
                    base_N = base.shape[0]
                    idxs = np.random.randint(0, base_N, size=cur)
                    meta_out[key][j0:j1, ...] = base[idxs]

                ptr = j1

        print(f"[H5] Done writing {N_total} samples to {h5_out_path}")


# -------------------- Viz --------------------
def save_grid(samples, labels, out_png, idx2lab):
    x = samples.detach().cpu().numpy()[:,0]
    x = (x + 1.0) / 2.0
    N = x.shape[0]; cols = min(10, N); rows = math.ceil(N / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(1.6*cols, 1.6*rows), constrained_layout=True)
    if rows == 1: axes = np.array([axes]); axes = axes.reshape(rows, cols)
    for i in range(rows*cols):
        ax = axes[i//cols, i%cols]
        if i < N:
            ax.imshow(x[i], aspect="auto", origin="lower", cmap="inferno")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"L{idx2lab[int(labels[i].item())]}", fontsize=8)
        else:
            ax.axis("off")
    fig.suptitle(Path(out_png).name, fontsize=10)
    fig.savefig(out_png, dpi=140); plt.close(fig)

# -------------------- EMA --------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                self.shadow[k] = v.detach().clone()
            else:
                # keep non-float buffers/params as a plain copy (no averaging)
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                self.shadow[k] = v.detach().clone()
                continue
            if torch.is_floating_point(v):
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                # overwrite non-float tensors directly
                self.shadow[k].copy_(v.detach())


# -------------------- Train --------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True, help="Path to the primary (filtered) H5 dataset.")
    # <-- NEW: Args for unfiltered data, aug count, and FID -->
    p.add_argument("--unfiltered_h5_path", default=None, help="Path to unfiltered H5 dataset to source extra Type 2 bursts.")
    p.add_argument("--aug_target_count", type=int, default=500, help="Target number of samples per class after offline augmentation.")
    p.add_argument("--fid_num_samples", type=int, default=1000, help="Number of fake samples to use for FID calculation.")
    p.add_argument("--outdir", required=True)
    p.add_argument("--target_labels", type=str, default="2,4,14,35,34,135")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgH", type=int, default=256)
    p.add_argument("--imgW", type=int, default=160)
    p.add_argument("--nz", type=int, default=128)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lrG", type=float, default=None, help="Override G LR (TTUR). If None, uses --lr")
    p.add_argument("--lrD", type=float, default=None, help="Override D LR (TTUR). If None, uses --lr")
    p.add_argument("--betas", type=float, nargs=2, default=(0.0, 0.9))
    p.add_argument("--loss", type=str, default="hinge", choices=["hinge","wgan-gp"])
    p.add_argument("--gp_lambda", type=float, default=10.0)
    p.add_argument("--r1_gamma", type=float, default=10.0)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--strong_aug", action="store_true", help="use stronger affine jitter")
    p.add_argument("--balanced_batch", action="store_true", help="per-batch class balancing")
    p.add_argument("--save_every", type=int, default=50, help="save samples and calculate FID every N epochs")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--num_gen", type=int, default=10, help="TOTAL samples to generate per checkpoint for viz")
    p.add_argument("--g_steps", type=int, default=2)
    # ---- Synthetic H5 output ----
    p.add_argument("--num_h5", type=int, default=0,
                   help="Number of synthetic samples to generate into a final H5 file (0 = disable).")
    
    p.add_argument("--h5_outname", type=str, default=None,
                   help="Output filename of the synthetic H5 file (inside outdir).")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir = Path(args.outdir)
    (outdir / "gan_samples").mkdir(parents=True, exist_ok=True)
    with open(outdir/"config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    target_labels = normalize_target_labels(args.target_labels)
    ds = H5CompositeDataset(args.h5, target_labels=target_labels,
                            # <-- NEW: Pass new args -->
                            unfiltered_h5_path=args.unfiltered_h5_path,
                            H=args.imgH, W=args.imgW,
                            aug_target_count=args.aug_target_count, 
                            pre_aug=True, seed=args.seed)
    num_classes = len(target_labels)

    counts = np.bincount(ds.y, minlength=num_classes)
    if np.any(counts == 0):
        print("WARNING: Some classes have 0 samples after augmentation. This may cause errors.")
        
    class_weights = 1.0 / (counts + 1e-6)
    sample_weights = class_weights[ds.y]

    if args.balanced_batch:
        batch_sampler = BalancedBatchSampler(ds.y, num_classes, args.batch)
        loader = DataLoader(ds, batch_sampler=batch_sampler,
                            num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    else:
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                        num_samples=len(ds), replacement=True)
        loader = DataLoader(ds, batch_size=args.batch, sampler=sampler,
                            num_workers=args.num_workers, drop_last=True, pin_memory=torch.cuda.is_available())

    netG = Gen(args.nz, args.ngf, 1, num_classes=num_classes).to(device)
    netD = Disc(args.ndf, 1, num_classes=num_classes, use_strong_aug=args.strong_aug).to(device)

    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if getattr(m, "bias", None) is not None: nn.init.zeros_(m.bias.data)
        elif isinstance(m, (nn.BatchNorm2d,)):
            nn.init.normal_(m.weight.data, 1.0, 0.02); nn.init.zeros_(m.bias.data)
    netG.apply(weights_init); netD.apply(weights_init)

    # TTUR
    lrG = args.lr if args.lrG is None else args.lrG
    lrD = args.lr if args.lrD is None else args.lrD
    optG = torch.optim.Adam(netG.parameters(), lr=lrG, betas=tuple(args.betas))
    optD = torch.optim.Adam(netD.parameters(), lr=lrD, betas=tuple(args.betas))
    print(f"TTUR: lrG={lrG} lrD={lrD} (betas={args.betas})")

    ema = EMA(netG, decay=args.ema_decay) if args.use_ema else None

    # Fixed noise for monitoring (even split over classes)
    per_class = max(1, args.num_gen // num_classes) if num_classes > 0 else 0
    leftover = args.num_gen - per_class*num_classes
    cls_counts = [per_class]*num_classes
    for i in range(leftover): cls_counts[i] += 1
    fixed_y = []
    for c in range(num_classes): fixed_y += [c]*cls_counts[c]
    fixed_y = torch.tensor(fixed_y, device=device, dtype=torch.long)
    fixed_z = torch.randn(fixed_y.size(0), args.nz, device=device)

    label_map = [ds.idx2lab[i] for i in range(num_classes)]
    print(f"Label map: {label_map}. Class counts(after expand)={counts.tolist()}")
    
    # <-- NEW: Pre-compute real features for FID -->
    print("Initializing InceptionV3 model for FID...")
    inception_model = InceptionV3Features().to(device).eval()
    print("Pre-computing real features for FID... (this may take a minute)")
    real_mu, real_sigma = precompute_real_features(ds, inception_model, device, args.batch)
    print("...Real features pre-computed.")

    # ----- Track losses -----
    G_losses, D_losses, FID_scores = [], [], []

    torch.backends.cudnn.benchmark = True
    save_epochs = set(range(args.save_every, args.epochs+1, args.save_every))
    log_path = outdir / "train_log.txt"
    with open(log_path, "w") as lf:
        # <-- NEW: Added fid to log -->
        lf.write("epoch,lossD,lossG,lossD_real,lossD_fake,gp,fm,fid\n")

    for epoch in range(1, args.epochs + 1):
        epoch_G_loss, epoch_D_loss = 0.0, 0.0
        num_batches = 0

        for real, y in loader:
            real = real.to(device, non_blocking=True, dtype=torch.float32)
            y = y.to(device, non_blocking=True)
            b = real.size(0)
            num_batches += 1

            # ----- D -----
            optD.zero_grad(set_to_none=True)
            
            # fake path
            z = torch.randn(b, args.nz, device=device)
            y_fake = torch.randint(0, num_classes, (b,), device=device)
            with torch.no_grad():
                fake = netG(z, y_fake)
            fake_logits, _, _ = netD(fake, y_fake, augment=True)
            lossD_fake = d_hinge_fake(fake_logits)
            
            # real path (make real require grad BEFORE forward)
            real.requires_grad_(True)
            real_logits, real_feat, real_fmaps = netD(real, y, augment=True)
            real_fmaps_ref = [t.detach() for t in real_fmaps]
            lossD_real = d_hinge_real(real_logits)
            
            # R1 on the SAME forward
            gp = (args.r1_gamma * 0.5) * r1_from_real_logits(real, real_logits)
            
            # total D loss / backward
            lossD = lossD_real + lossD_fake + gp
            lossD.backward()                  # no retain_graph needed now
            nn.utils.clip_grad_norm_(netD.parameters(), 1.0)
            optD.step()
            
            # we won’t need real’s graph anymore
            real = real.detach()

        
            # ----- G (possibly multiple steps) -----
            for _ in range(args.g_steps):
                optG.zero_grad(set_to_none=True)
                z = torch.randn(b, args.nz, device=device)
                y_fake = torch.randint(0, num_classes, (b,), device=device)
                fake = netG(z, y_fake)
                fake_logits, fake_feat, fake_fmaps = netD(fake, y_fake, augment=True)
                adv = g_hinge_fake(fake_logits)  # or g_wgan
                fm  = feature_matching_loss(fake_fmaps, real_fmaps_ref).clamp_min(0.0)
                lossG = adv + 10.0 * fm
                lossG.backward()
                nn.utils.clip_grad_norm_(netG.parameters(), 1.0)
                optG.step()
                if ema: ema.update(netG)
                epoch_G_loss += lossG.item()
            epoch_D_loss += lossD.item()


        # ----- epoch summary -----
        epoch_G_loss /= max(1, num_batches * args.g_steps)
        epoch_D_loss /= max(1, num_batches)
        G_losses.append(epoch_G_loss)
        D_losses.append(epoch_D_loss)
        
        # Log line (component terms are from the last batch)
        # <-- NEW: Print/log logic separated from save logic -->
        print(f"[{epoch:03d}/{args.epochs}]  D:{epoch_D_loss:.4f} | G:{epoch_G_loss:.4f}", end="")
        
        # ----- sample saves & FID calc -----
        if epoch in save_epochs or epoch == args.epochs:
            netG_eval = netG
            if ema:
                netG_eval = Gen(args.nz, args.ngf, 1, num_classes=num_classes).to(device)
                netG_eval.load_state_dict(ema.shadow, strict=False)
            netG_eval.eval()
            
            # Save viz grid
            with torch.no_grad():
                fakes = netG_eval(fixed_z, fixed_y)
            grid_path = outdir / "gan_samples" / f"samples_epoch_{epoch:03d}.png"
            npy_path  = outdir / "gan_samples" / f"samples_epoch_{epoch:03d}.npy"
            save_grid(fakes, fixed_y, str(grid_path), ds.idx2lab)
            np.save(npy_path, fakes.detach().cpu().numpy())
            
            # <-- NEW: Calculate and log FID -->
            print("  Calculating FID...", end="")
            fid_score = calculate_fake_features_and_fid(netG_eval, inception_model, num_classes, args.nz, args.fid_num_samples, real_mu, real_sigma, device)
            FID_scores.append(fid_score)
            print(f" FID: {fid_score:.3f}")
            
            if netG_eval is not netG:
                del netG_eval
            netG.train()
            
            # Write to log file
            with open(log_path, "a") as lf:
                lf.write(f"{epoch},{epoch_D_loss:.6f},{epoch_G_loss:.6f},"
                         f"{lossD_real.item():.6f},{lossD_fake.item():.6f},{gp.item():.6f},{fm.item():.6f},"
                         f"{fid_score:.6f}\n")
        
        else:
            # Not a save epoch, just write losses (FID is 0 or nan)
            print("") # newline
            with open(log_path, "a") as lf:
                lf.write(f"{epoch},{epoch_D_loss:.6f},{epoch_G_loss:.6f},"
                         f"{lossD_real.item():.6f},{lossD_fake.item():.6f},{gp.item():.6f},{fm.item():.6f},"
                         f"nan\n")


    # ----- Save weights -----
    torch.save({"G": netG.state_dict(), "D": netD.state_dict(),
                "G_ema": ema.shadow if ema else None,
                "args": vars(args), "label_map": label_map},
               outdir / "final_ckpt.pt")

    # ----- Optional: generate synthetic H5 dataset with final generator -----
    generate_synthetic_h5(args, netG, ema, label_map, num_classes, device)


    # ----- Plot PNG for losses -----
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    # Plot G and D losses on left y-axis
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('G/D Loss', color=color)
    ax1.plot(D_losses, label="Discriminator", color="red", alpha=0.8)
    ax1.plot(G_losses, label="Generator", color="blue", alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3, ls="--")

    # Create a second y-axis for FID scores
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('FID Score', color=color)
    
    # Get epochs where FID was calculated
    fid_epochs = sorted(list(save_epochs))
    if args.epochs not in fid_epochs:
        fid_epochs.append(args.epochs)
    
    # Ensure FID_scores matches fid_epochs length
    valid_fid_scores = [f for f in FID_scores if not np.isnan(f)]
    
    ax2.plot(fid_epochs, valid_fid_scores, label="FID Score", color="green", marker='o', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.suptitle("GAN Training Metrics")
    fig.tight_layout()
    plt.savefig(outdir / "loss_and_fid_curves.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()