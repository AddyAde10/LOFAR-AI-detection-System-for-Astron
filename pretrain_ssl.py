#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-supervised pretraining (SimCLR) for LOFAR dynspecs (MULTI-SOURCE)

Uses:
  - Labeled twins (filtered + unfiltered) for TRAIN split only (train_ids from splits_npz)
  - Unlabeled filtered dynspecs
  - Synthetic dynspecs (all as UNLABELED)

Guarantees:
  - Val/test labeled images (val_ids + test_ids) are NEVER used in SSL
    (neither filtered nor unfiltered views).
  - Only the 70% TRAIN split of the labeled set is used as twins.

Dynspec-aware augmentations:
  - Jitter (gamma, contrast, brightness, small noise)
  - Mild warp
  - SpecAug-style time/frequency drop
  - Optional Sobel edge as second channel (2-ch input)

Example usage (Spider, from sbatch):
  python pretrain_ssl.py \
    --h5_labeled_filtered labeled_filtered.h5 \
    --h5_labeled_unfiltered labeled_unfiltered.h5 \
    --h5_unlabeled_filtered unlabelled_filtered.h5 \
    --splits_npz splits_labelled7_unfiltered_seed42.npz \
    --synth_h5 synthetic_type2_1500.h5 synthetic_type4,14_1500.h5 synthetic_type15,35_1500.h5 \
    --outdir runs/ssl_pretrain \
    --epochs 50 --batch 256 --img 224 224 --sobel --num_workers 8
"""

import os
import math
import argparse
import random
import time

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# ----------------- utils -----------------
def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True


def robust_minmax(x):
    x = np.asarray(x, np.float32)
    m = np.isfinite(x)
    if not m.any():
        return np.zeros_like(x, dtype=np.float32)
    lo = np.nanpercentile(x[m], 1)
    hi = np.nanpercentile(x[m], 99)
    rng = hi - lo
    if not np.isfinite(rng) or rng <= 1e-6:
        y = np.clip(x, 0, 1)
    else:
        y = (x - lo) / rng
        y = np.clip(y, 0, 1)
    y[~np.isfinite(y)] = 0.0
    return y


def _is_invalid(arr):
    if not np.isfinite(arr).any():
        return True
    if np.nanstd(arr) < 1e-12:
        return True
    return False


# ----------------- Augmentations (dynspec-safe) -----------------
def spec_augment(img, tmask=0.06, fmask=0.06):
    c, h, w = img.shape
    # time mask
    tw = max(1, int(w * tmask * np.random.uniform(0.5, 1.0)))
    t0 = np.random.randint(0, max(1, w - tw + 1))
    img[:, :, t0:t0 + tw] *= 0.0
    # freq mask
    fh = max(1, int(h * fmask * np.random.uniform(0.5, 1.0)))
    f0 = np.random.randint(0, max(1, h - fh + 1))
    img[:, f0:f0 + fh, :] *= 0.0
    return img


def jitter(img):
    # gamma
    g = 2.0 ** np.random.uniform(-0.25, 0.25)
    img = img ** g
    # contrast/brightness
    c = np.random.uniform(0.9, 1.1)
    b = np.random.uniform(-0.02, 0.02)
    img = np.clip(img * c + b, 0., 1.)
    # small Gaussian noise
    img += np.random.normal(0, 0.01, size=img.shape).astype(np.float32)
    return np.clip(img, 0., 1.)


def warp(img):
    c, h, w = img.shape
    nh = int(round(h * np.random.uniform(0.98, 1.02)))
    nw = int(round(w * np.random.uniform(0.98, 1.02)))
    img_t = torch.from_numpy(img)[None]
    img_t = F.interpolate(img_t, size=(nh, nw), mode='bilinear', align_corners=False)
    img_t = F.interpolate(img_t, size=(h, w), mode='bilinear', align_corners=False)[0]
    return img_t.numpy()


# ----------------- Multi-source SSL dataset -----------------
class MultiSSLDataset(Dataset):
    """
    Combines:
      - Labeled twins (filtered + unfiltered) for train_ids (kind='twin')
      - Unlabeled filtered (kind='ufil')
      - Synthetic dynspecs (kind='syn')

    Returns two augmented views (x1, x2) for each item.
    All of these are treated as UNLABELED from the SSL perspective.
    """

    def __init__(
        self,
        h5_labeled_filtered,
        h5_labeled_unfiltered,
        train_ids,
        h5_unlabeled_filtered=None,
        synth_h5_list=None,
        size=(224, 224),
        sobel=False,
        max_samples=None,
        resample_tries=5,
    ):
        super().__init__()
        self.h5_lfil_path = h5_labeled_filtered
        self.h5_lunf_path = h5_labeled_unfiltered
        self.h5_ufil_path = h5_unlabeled_filtered
        self.synth_paths = synth_h5_list or []
        self.size = size
        self.sobel = sobel
        self.resample_tries = resample_tries

        # Lazily opened per worker
        self._h5_lfil = None
        self._h5_lunf = None
        self._h5_ufil = None
        self._h5_synth = [None] * len(self.synth_paths)

        # Labeled sizes
        with h5py.File(self.h5_lfil_path, 'r', swmr=True, libver='latest') as f:
            n_lfil = f['data'].shape[0]
        with h5py.File(self.h5_lunf_path, 'r', swmr=True, libver='latest') as f:
            n_lunf = f['data'].shape[0]
        if n_lfil != n_lunf:
            raise ValueError(f"Labeled filtered ({n_lfil}) and unfiltered ({n_lunf}) lengths differ.")

        self.train_ids = np.asarray(train_ids, dtype=np.int64)
        assert self.train_ids.ndim == 1

        # Unlabeled filtered size
        self.n_ufil = 0
        if self.h5_ufil_path is not None:
            with h5py.File(self.h5_ufil_path, 'r', swmr=True, libver='latest') as f:
                self.n_ufil = f['data'].shape[0]

        # Synthetic sizes
        self.synth_sizes = []
        for p in self.synth_paths:
            with h5py.File(p, 'r', swmr=True, libver='latest') as f:
                self.synth_sizes.append(f['data'].shape[0])

        # Build index table: list of (kind, info)
        entries = []
        # Twins from labeled TRAIN only
        for idx in self.train_ids:
            entries.append(('twin', int(idx)))
        # Unlabeled filtered
        for j in range(self.n_ufil):
            entries.append(('ufil', int(j)))
        # Synthetic
        for k, n_k in enumerate(self.synth_sizes):
            for j in range(n_k):
                entries.append(('syn', (k, int(j))))

        # Optional downsampling of SSL pool
        if max_samples is not None and max_samples < len(entries):
            rng = np.random.default_rng(42)
            chosen = rng.choice(len(entries), size=max_samples, replace=False)
            entries = [entries[i] for i in chosen]

        self.entries = entries
        self._len = len(entries)

    def __len__(self):
        return self._len

    # ---- lazy open ----
    def _ensure_open_labeled(self):
        if self._h5_lfil is None:
            self._h5_lfil = h5py.File(self.h5_lfil_path, 'r', swmr=True, libver='latest')
        if self._h5_lunf is None:
            self._h5_lunf = h5py.File(self.h5_lunf_path, 'r', swmr=True, libver='latest')

    def _ensure_open_ufil(self):
        if self.h5_ufil_path is None:
            return
        if self._h5_ufil is None:
            self._h5_ufil = h5py.File(self.h5_ufil_path, 'r', swmr=True, libver='latest')

    def _ensure_open_synth(self, k):
        if self._h5_synth[k] is None:
            self._h5_synth[k] = h5py.File(self.synth_paths[k], 'r', swmr=True, libver='latest')

    # ---- core I/O ----
    def _prep_arr(self, arr):
        """Robust minmax + resize to target size, returns np (1,H,W)."""
        x = robust_minmax(arr)[None, ...]  # (1,H,W)
        xt = torch.from_numpy(x)
        if tuple(xt.shape[-2:]) != self.size:
            xt = F.interpolate(xt[None], size=self.size,
                               mode='bilinear', align_corners=False)[0]
        x = xt.numpy()
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        return x

    def _sobelify(self, x1, x2):
        import cv2

        def sobel_edge(img1c):
            gx = cv2.Sobel(img1c, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(img1c, cv2.CV_32F, 0, 1, ksize=3)
            e = np.clip(np.sqrt(gx * gx + gy * gy), 0, 1)
            return e

        e1 = sobel_edge(x1[0])
        e2 = sobel_edge(x2[0])
        x1 = np.stack([x1[0], e1], 0)
        x2 = np.stack([x2[0], e2], 0)
        return x1, x2

    def _augment_pair(self, base1, base2=None):
        """
        If base2 is None, use base1 for both branches.
        base1/base2 are np arrays (1,H,W) or (C,H,W).
        """
        if base2 is None:
            base2 = base1
        x1 = spec_augment(warp(jitter(base1.copy())), 0.06, 0.06)
        x2 = spec_augment(warp(jitter(base2.copy())), 0.06, 0.06)
        if self.sobel:
            x1, x2 = self._sobelify(x1, x2)
        return torch.from_numpy(x1), torch.from_numpy(x2)

    def __getitem__(self, idx):
        tries = 0
        while True:
            kind, info = self.entries[idx]

            if kind == 'twin':
                self._ensure_open_labeled()
                i = info
                arr_fil = self._h5_lfil['data'][i]
                arr_unf = self._h5_lunf['data'][i]
                if _is_invalid(arr_fil) and _is_invalid(arr_unf):
                    if tries >= self.resample_tries:
                        arr_fil = np.zeros_like(self._h5_lfil['data'][0], dtype=np.float32)
                        arr_unf = arr_fil
                        break
                    idx = np.random.randint(0, self._len)
                    tries += 1
                    continue
                if _is_invalid(arr_fil):
                    arr_fil = arr_unf
                if _is_invalid(arr_unf):
                    arr_unf = arr_fil
                x_fil = self._prep_arr(arr_fil)
                x_unf = self._prep_arr(arr_unf)
                x1, x2 = self._augment_pair(x_fil, x_unf)
                return x1, x2

            elif kind == 'ufil':
                self._ensure_open_ufil()
                j = info
                arr = self._h5_ufil['data'][j]
                if _is_invalid(arr):
                    if tries >= self.resample_tries:
                        arr = np.zeros_like(self._h5_ufil['data'][0], dtype=np.float32)
                        break
                    idx = np.random.randint(0, self._len)
                    tries += 1
                    continue
                x = self._prep_arr(arr)
                x1, x2 = self._augment_pair(x)
                return x1, x2

            elif kind == 'syn':
                k, j = info
                self._ensure_open_synth(k)
                arr = self._h5_synth[k]['data'][j]
                if _is_invalid(arr):
                    if tries >= self.resample_tries:
                        arr = np.zeros_like(self._h5_synth[k]['data'][0], dtype=np.float32)
                        break
                    idx = np.random.randint(0, self._len)
                    tries += 1
                    continue
                x = self._prep_arr(arr)
                x1, x2 = self._augment_pair(x)
                return x1, x2

        # Fallback if we really can't get valid data
        z = np.zeros((2 if self.sobel else 1, *self.size), dtype=np.float32)
        return torch.from_numpy(z), torch.from_numpy(z)


# ----------------- Encoder (= backbone twin) -----------------
class ConvBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BlurPool(nn.Module):
    def __init__(self, ch, stride=2):
        super().__init__()
        k = torch.tensor([1., 2., 1.])
        filt = (k[:, None] * k[None, :])
        filt /= filt.sum()
        self.register_buffer('f', filt[None, None, ...].repeat(ch, 1, 1, 1))
        self.stride = stride
        self.ch = ch

    def forward(self, x):
        return F.conv2d(x, self.f, stride=self.stride, padding=1, groups=self.ch)


class C2F(nn.Module):
    def __init__(self, ch, n=1):
        super().__init__()
        self.cv1 = ConvBlock(ch, ch // 2, 1, 1, 0)
        self.cv2 = ConvBlock(ch, ch // 2, 1, 1, 0)
        self.blocks = nn.Sequential(*[ConvBlock(ch // 2, ch // 2, 3, 1, 1) for _ in range(n)])
        self.cv3 = ConvBlock(ch, ch, 1, 1, 0)

    def forward(self, x):
        a = self.cv1(x)
        b = self.blocks(self.cv2(x))
        return self.cv3(torch.cat([a, b], 1))


class SPPF(nn.Module):
    def __init__(self, c, k=5):
        super().__init__()
        h = c // 2
        self.cv1 = ConvBlock(c, h, 1, 1, 0)
        self.pool = nn.MaxPool2d(k, 1, k // 2)
        self.cv2 = ConvBlock(h * 4, c, 1, 1, 0)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], 1))


class Encoder(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.stem = ConvBlock(in_ch, 32, 3, 1, 1)
        self.down1 = nn.Sequential(BlurPool(32), ConvBlock(32, 64, 3, 1, 1), C2F(64, 1))
        self.down2 = nn.Sequential(BlurPool(64), ConvBlock(64, 128, 3, 1, 1), C2F(128, 2))
        self.mid = nn.Sequential(ConvBlock(128, 256, 3, 1, 2), C2F(256, 2))
        self.tail = nn.Sequential(
            BlurPool(256),
            ConvBlock(256, 512, 3, 1, 1),
            C2F(512, 1),
            nn.Dropout2d(0.1),
            SPPF(512),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.mid(x)
        x = self.tail(x)
        return x


# ----------------- SimCLR head & loss -----------------
class ProjHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
        )

    def forward(self, f):
        z = self.gap(f).flatten(1)
        return F.normalize(self.mlp(z), dim=1)


def ntxent(z1, z2, tau=0.2):
    z = torch.cat([z1, z2], 0)  # (2B, D)
    sim = z @ z.t()
    mask = torch.eye(sim.size(0), dtype=torch.bool, device=sim.device)
    sim = sim[~mask].view(sim.size(0), -1)
    pos = (z1 @ z2.t()).diag()
    pos = torch.cat([pos, pos], 0)
    logits = torch.cat([pos[:, None], sim], 1) / tau
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=sim.device)
    return F.cross_entropy(logits, labels)


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5_labeled_filtered', required=True)
    ap.add_argument('--h5_labeled_unfiltered', required=True)
    ap.add_argument('--h5_unlabeled_filtered', required=True)
    ap.add_argument('--splits_npz', required=True)
    ap.add_argument('--synth_h5', nargs='*', default=[],
                    help="Optional synthetic H5s used as UNLABELED")
    ap.add_argument('--outdir', default='runs/ssl_pretrain')
    ap.add_argument('--img', type=int, nargs=2, default=[224, 224])
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--sobel', action='store_true')
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--prefetch', type=int, default=2)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_samples', type=int, default=None,
                    help="Optional cap on total SSL samples per epoch")
    args = ap.parse_args()

    # HDF5 on shared FS (Spider): disable locking
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- load splits and use ONLY train_ids for twins ----
    splits = np.load(args.splits_npz)
    if not all(k in splits for k in ["train_ids", "val_ids", "test_ids"]):
        raise KeyError(
            f"[SSL] Expected keys train_ids/val_ids/test_ids in {args.splits_npz}, "
            f"got {list(splits.keys())}"
        )
    train_ids = splits['train_ids'].astype(np.int64)
    val_ids = splits['val_ids'].astype(np.int64)
    test_ids = splits['test_ids'].astype(np.int64)

    print(f"[SSL] Using {len(train_ids)} labeled TRAIN indices as twins for SSL.")
    print(f"[SSL] (Val+Test) labeled indices excluded from SSL: {len(val_ids) + len(test_ids)} total.")
    print(f"[SSL] Synth H5s: {args.synth_h5}")

    # ---- build dataset & dataloader ----
    ds = MultiSSLDataset(
        h5_labeled_filtered=args.h5_labeled_filtered,
        h5_labeled_unfiltered=args.h5_labeled_unfiltered,
        train_ids=train_ids,
        h5_unlabeled_filtered=args.h5_unlabeled_filtered,
        synth_h5_list=args.synth_h5,
        size=tuple(args.img),
        sobel=args.sobel,
        max_samples=args.max_samples,
    )
    print(f"[SSL] Total SSL items (twins + unlabeled + synth): {len(ds)}")
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=args.prefetch,
    )

    in_ch = 2 if args.sobel else 1
    enc = Encoder(in_ch=in_ch).to(dev)

    with torch.no_grad():
        dummy = torch.zeros(2, in_ch, args.img[0], args.img[1], device=dev)
        f = enc(dummy)
        C = f.shape[1]
    head = ProjHead(C).to(dev)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    total = args.epochs * len(loader)
    warm = max(10, int(0.05 * total))

    def lr_sched(step):
        if step < warm:
            return step / max(1, warm)
        t = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * t))

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    hist = []
    step = 0

    for ep in range(1, args.epochs + 1):
        enc.train()
        head.train()
        run_loss = 0.0
        t0 = time.time()

        for it, (x1, x2) in enumerate(loader):
            x1 = x1.to(dev, non_blocking=True)
            x2 = x2.to(dev, non_blocking=True)

            for pg in opt.param_groups:
                pg['lr'] = args.lr * lr_sched(step)
            step += 1

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=True):
                z1 = head(enc(x1))
                z2 = head(enc(x2))
                loss = ntxent(z1, z2, tau=0.2)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            scaler.step(opt)
            scaler.update()

            run_loss += float(loss.detach())

            if (it + 1) % 50 == 0:
                print(f"[ep {ep:03d}] it {it+1}/{len(loader)} | loss {run_loss/(it+1):.4f}")

        epoch_loss = run_loss / len(loader)
        dt_min = (time.time() - t0) / 60.0
        print(f"Epoch {ep:03d} | loss {epoch_loss:.4f} | {dt_min:.1f} min")

        hist.append({
            "epoch": ep,
            "loss": float(epoch_loss),
            "lr": float(opt.param_groups[0]["lr"]),
            "minutes": round(dt_min, 3),
        })

    ckpt = os.path.join(args.outdir, 'encoder_ssl.pth')
    torch.save(enc.state_dict(), ckpt)

    # Save SSL training history
    try:
        df = pd.DataFrame(hist)
        df.to_csv(os.path.join(args.outdir, "ssl_metrics.csv"), index=False)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6, 4))
            plt.plot(df["epoch"], df["loss"], label="SSL train loss")
            plt.xlabel("Epoch")
            plt.ylabel("NT-Xent loss")
            plt.title("SSL Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, "ssl_loss_curve.png"), dpi=150)
            plt.close()
        except Exception as e:
            print("[WARN] SSL plotting failed:", e)
    except Exception as e:
        print("[WARN] saving ssl_metrics.csv failed:", e)

    print(f"[saved] {ckpt}")


if __name__ == '__main__':
    main()
