#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROBUST Multilabel fine-tuning on top of SSL encoder (SimCLR), with strict splits.

Data usage:
  - Labeled UNFILTERED H5  (ground-truth labels, split anchor)
  - Labeled FILTERED H5    (cleaned view used for TRAIN only)
  - Optional GAN H5s       (e.g. synthetic_type2_1500.h5) used as LABELED TRAIN only
  - Splits NPZ             (train_ids, val_ids, test_ids) on UNFILTERED indices

Guarantees:
  - SSL pretraining used only TRAIN indices from labeled twins.
  - Fine-tuning uses only TRAIN indices + GAN H5.
  - Val/test are ONLY real UNFILTERED data, never used in any training stage.
"""

import os, math, json, argparse, random
from datetime import datetime
import numpy as np, pandas as pd, h5py

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score
)
warnings = __import__("warnings"); warnings.filterwarnings("ignore")

# ---------------- utils ----------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = True

def robust_minmax(x):
    x = np.asarray(x, np.float32)
    m = np.isfinite(x)
    if not m.any(): return np.zeros_like(x, dtype=np.float32)
    lo = np.nanpercentile(x[m], 1); hi = np.nanpercentile(x[m], 99)
    rng = hi - lo
    if not np.isfinite(rng) or rng <= 1e-6:
        y = np.clip(x, 0, 1)
    else:
        y = np.clip((x - lo) / max(rng, 1e-6), 0, 1)
    y[~np.isfinite(y)] = 0.0
    return y

# ---------------- dataset ----------------
class DynspecDataset(Dataset):
    def __init__(self, X, Y, size=(224,224), sobel=False, augment=False, resample_tries=3):
        self.X = X
        self.Y = Y.astype(np.float32)
        self.size = size
        self.sobel = sobel
        self.augment = augment
        self.resample_tries = resample_tries

    def _is_invalid(self, x):
        if not np.isfinite(x).any(): return True
        if np.nanstd(x) < 1e-12: return True
        return False

    def _prep(self, x):
        x = robust_minmax(x)[None, ...]       # (1,H,W)
        xt = torch.from_numpy(x)
        if tuple(xt.shape[-2:]) != self.size:
            xt = F.interpolate(xt[None], size=self.size,
                               mode='bilinear', align_corners=False)[0]
        return xt.numpy()

    def __getitem__(self, i):
        x = self.X[i]; tries = 0
        while self._is_invalid(x) and tries < self.resample_tries:
            i = np.random.randint(0, len(self.X)); x = self.X[i]; tries += 1
        if self._is_invalid(x):
            x = np.zeros_like(self.X[0], dtype=np.float32)

        x = self._prep(x)
        if self.sobel:
            import cv2
            base = x[0]
            gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
            e = np.clip(np.sqrt(gx*gx+gy*gy), 0, 1)
            x = np.stack([x[0], e], 0)

        if self.augment:
            x = self._augment(x)

        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        y = self.Y[i]
        return torch.from_numpy(x), torch.from_numpy(y)

    def _augment(self, x):
        c,h,w = x.shape
        # Spec-augment style masks
        if np.random.rand() < 0.5:
            tw = max(1, int(w * 0.06 * np.random.uniform(0.3, 1.0)))
            t0 = np.random.randint(0, max(1, w - tw + 1))
            x[:, :, t0:t0+tw] *= 0.0
        if np.random.rand() < 0.5:
            fh = max(1, int(h * 0.06 * np.random.uniform(0.3, 1.0)))
            f0 = np.random.randint(0, max(1, h - fh + 1))
            x[:, f0:f0+fh, :] *= 0.0
        # Mild warp
        if np.random.rand() < 0.5:
            nh = int(round(h * np.random.uniform(0.98, 1.02)))
            nw = int(round(w * np.random.uniform(0.98, 1.02)))
            xt = torch.from_numpy(x)[None]
            xt = F.interpolate(xt, size=(nh, nw), mode='bilinear', align_corners=False)
            xt = F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=False)[0].numpy()
            x = xt
        # Intensity jitter
        g  = 2.0 ** np.random.uniform(-0.15, 0.15); x = x ** g
        c1 = np.random.uniform(0.95, 1.05); b = np.random.uniform(-0.01, 0.01)
        x  = np.clip(x * c1 + b, 0, 1)
        # Small noise
        x += np.random.normal(0, 0.008, size=x.shape).astype(np.float32)
        x  = np.clip(x, 0, 1)
        return x

    def __len__(self): return len(self.X)

# ---------------- label parsing ----------------
def parse_label_string(x):
    s = str(int(x)) if isinstance(x,(int,np.integer)) else (x.decode() if isinstance(x,(bytes,np.bytes_)) else str(x))
    labs = []
    for ch in s.strip():
        if ch.isdigit():
            k = int(ch)
            if 1 <= k <= 9:
                labs.append(k)
    return sorted(set(labs))

# ---------------- model (same encoder as SSL) ----------------
class ConvBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__()
        self.conv=nn.Conv2d(c1,c2,k,s,p,bias=False)
        self.bn=nn.BatchNorm2d(c2)
        self.act=nn.SiLU()
    def forward(self,x): return self.act(self.bn(self.conv(x)))

class BlurPool(nn.Module):
    def __init__(self, ch, stride=2):
        super().__init__()
        k=torch.tensor([1.,2.,1.]); f=(k[:,None]*k[None,:]); f/=f.sum()
        self.register_buffer('f', f[None,None,...].repeat(ch,1,1,1))
        self.stride=stride; self.ch=ch
    def forward(self,x):
        return F.conv2d(x, self.f, stride=self.stride, padding=1, groups=self.ch)

class C2F(nn.Module):
    def __init__(self,ch,n=1):
        super().__init__()
        self.cv1=ConvBlock(ch,ch//2,1,1,0)
        self.cv2=ConvBlock(ch,ch//2,1,1,0)
        self.blocks=nn.Sequential(*[ConvBlock(ch//2,ch//2,3,1,1) for _ in range(n)])
        self.cv3=ConvBlock(ch,ch,1,1,0)
    def forward(self,x):
        a=self.cv1(x); b=self.blocks(self.cv2(x))
        return self.cv3(torch.cat([a,b],1))

class SPPF(nn.Module):
    def __init__(self,c,k=5):
        super().__init__()
        h=c//2
        self.cv1=ConvBlock(c,h,1,1,0)
        self.pool=nn.MaxPool2d(k,1,k//2)
        self.cv2=ConvBlock(h*4,c,1,1,0)
    def forward(self,x):
        x=self.cv1(x); y1=self.pool(x); y2=self.pool(y1); y3=self.pool(y2)
        return self.cv2(torch.cat([x,y1,y2,y3],1))

class Encoder(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.stem = ConvBlock(in_ch,32,3,1,1)
        self.down1 = nn.Sequential(BlurPool(32),  ConvBlock(32,64,3,1,1),  C2F(64,1))
        self.down2 = nn.Sequential(BlurPool(64),  ConvBlock(64,128,3,1,1), C2F(128,2))
        self.mid   = nn.Sequential(               ConvBlock(128,256,3,1,2), C2F(256,2))
        self.tail  = nn.Sequential(BlurPool(256), ConvBlock(256,512,3,1,1),
                                   C2F(512,1), nn.Dropout2d(0.1), SPPF(512))
    def forward(self,x):
        x = self.stem(x); x = self.down1(x); x = self.down2(x); x = self.mid(x); x = self.tail(x)
        return x  # (B,512,h,w)

class SSLClassifierHead(nn.Module):
    def __init__(self, in_ch=1, num_classes=7):
        super().__init__()
        self.encoder = Encoder(in_ch=in_ch)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        f = self.encoder(x)              # (B,512,h,w)
        z = self.gap(f).flatten(1)       # (B,512)
        return self.fc(z)                # logits

# ---------------- losses (stable) ----------------
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8):
        super().__init__(); self.gp=gamma_pos; self.gn=gamma_neg; self.clip=clip; self.eps=eps
    def forward(self, logits, targets):
        logits = logits.float(); targets = targets.float()
        x = torch.sigmoid(logits)
        if self.clip and self.clip>0:
            x = (x - self.clip).clamp(min=0, max=1-1e-7) / (1 - self.clip)
        x = x.clamp(min=self.eps, max=1.0 - self.eps)
        xs_pos = x; xs_neg = 1.0 - x
        los_pos = targets * torch.log(xs_pos)
        los_neg = (1.0 - targets) * torch.log(xs_neg)
        if self.gp>0 or self.gn>0:
            pt_pos = xs_pos * targets; pt_neg = xs_neg * (1.0 - targets)
            los_pos *= (1.0 - pt_pos).pow(self.gp)
            los_neg *= (pt_neg).pow(self.gn)
        return - (los_pos + los_neg).mean()

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, eps=1e-8):
        super().__init__(); self.g=gamma; self.a=alpha; self.eps=eps
    def forward(self, logits, targets):
        logits = logits.float(); targets = targets.float()
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p*targets + (1-p)*(1-targets)
        a_t = self.a*targets + (1-self.a)*(1-targets)
        return (a_t*(1-p_t).pow(self.g)*ce).mean()

def build_loss(name, y_train):
    name = name.lower()
    if name=='asl':   return AsymmetricLoss()
    if name=='focal': return FocalLoss()
    if name=='bce':
        pos_w = (y_train.shape[0]-y_train.sum(0))/np.clip(y_train.sum(0),1,1e9)
        pos_w = np.clip(pos_w, 1.0, 100.0)
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32))
    raise ValueError('loss?')

# ---------------- metrics ----------------
@torch.no_grad()
def f1_from_logits(logits, targets, thr=0.5):
    p = torch.sigmoid(logits.float()); pred=(p>=thr).float()
    TP=(pred*targets).sum(0); FP=(pred*(1-targets)).sum(0); FN=((1-pred)*targets).sum(0)
    prec = TP/(TP+FP+1e-8); rec=TP/(TP+FN+1e-8); f1=2*prec*rec/(prec+rec+1e-8)
    micro = (2*TP.sum())/(2*TP.sum()+FP.sum()+FN.sum()+1e-8)
    subset = (pred.eq(targets).all(1).float().mean())
    return micro.item(), f1.mean().item(), subset.item()

# ---------------- EMA ----------------
class EMA:
    def __init__(self, model, decay=0.999, device=None):
        self.decay=float(decay); self.shadow={}; self.device=device
        for k,p in model.named_parameters():
            if p.requires_grad and p.data.dtype.is_floating_point:
                self.shadow[k]=p.data.detach().clone().to(device=device, dtype=torch.float32)
        for k,b in model.named_buffers():
            if b.dtype.is_floating_point and ('running_mean' in k or 'running_var' in k):
                self.shadow[f'@buf@{k}']=b.detach().clone().to(device=device, dtype=torch.float32)
    @torch.no_grad()
    def update(self, model):
        for k,p in model.named_parameters():
            if k in self.shadow and p.data.dtype.is_floating_point:
                s=self.shadow[k]; p32=p.data.detach().to(dtype=torch.float32, device=s.device)
                s.mul_(self.decay).add_(p32, alpha=1.0-self.decay)
        for k,b in model.named_buffers():
            key=f'@buf@{k}'
            if key in self.shadow and b.dtype.is_floating_point:
                s=self.shadow[key]; b32=b.detach().to(dtype=torch.float32, device=s.device)
                s.mul_(self.decay).add_(b32, alpha=1.0-self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        for k,p in model.named_parameters():
            if k in self.shadow and p.data.dtype.is_floating_point:
                p.copy_(self.shadow[k].to(dtype=p.dtype, device=p.device))
        for k,b in model.named_buffers():
            key=f'@buf@{k}'
            if key in self.shadow and b.dtype.is_floating_point:
                b.copy_(self.shadow[key].to(dtype=b.dtype, device=b.device))

# ---------------- helpers ----------------
def freeze_module(m, requires_grad: bool):
    for p in m.parameters(): p.requires_grad = requires_grad

def set_bn_eval(m):
    if isinstance(m, nn.BatchNorm2d):
        m.eval()
        for p in m.parameters(): p.requires_grad=False

def cosine_with_warmup(step, total, warm):
    if step < warm: return step/max(1,warm)
    t=(step-warm)/max(1,total-warm); return 0.5*(1+math.cos(math.pi*t))

def per_class_thresholds(y_true, y_prob):
    C=y_true.shape[1]; th=np.zeros(C, dtype=np.float32)
    y_prob = np.clip(y_prob, 1e-6, 1-1e-6)
    for c in range(C):
        if not np.isfinite(y_prob[:,c]).any():
            th[c]=0.5; continue
        best=0.5; bestf=-1
        for t in np.linspace(0.1,0.9,17):
            y=(y_prob[:,c]>=t).astype(int)
            _,_,f,_=precision_recall_fscore_support(y_true[:,c], y, average='binary', zero_division=0)
            if f>bestf: bestf=f; best=t
        th[c]=best
    return th

# ---------------- main ----------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--h5_labeled_filtered',   required=True)
    ap.add_argument('--h5_labeled_unfiltered', required=True)
    ap.add_argument('--splits_npz',            required=True)
    ap.add_argument('--gan_h5', nargs='*', default=[],
                    help="Synthetic H5s used as LABELED TRAIN (e.g. type2 only).")
    ap.add_argument('--outdir', default='runs/yolo_multilabel_ssl')
    ap.add_argument('--labels', type=int, nargs='+', default=[1,2,3,4,5,6,7])
    ap.add_argument('--filter_mode', type=str, default='keep_any', choices=['keep_any','strict','drop'])
    ap.add_argument('--img', type=int, nargs=2, default=[224,224])
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--loss', type=str, default='bce', choices=['bce','asl','focal'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--ssl_ckpt', type=str, default='')
    ap.add_argument('--head_warmup', type=int, default=5)
    ap.add_argument('--unfreeze_stages_at', type=int, nargs='*', default=[10,20])
    ap.add_argument('--lrd', type=float, default=0.1)
    ap.add_argument('--use_sampler', action='store_true')
    ap.add_argument('--sobel', action='store_true')
    ap.add_argument('--no_plots', action='store_true')
    ap.add_argument('--quiet', action='store_true')
    args=ap.parse_args(); set_seed(args.seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ts=datetime.now().strftime('%Y%m%d_%H%M%S'); outdir=os.path.join(args.outdir, ts); os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir,'RUN_INFO.txt'),'w') as f:
        for k,v in vars(args).items(): f.write(f'{k}: {v}\n')

    # ----- load labeled H5s -----
    with h5py.File(args.h5_labeled_unfiltered,'r') as fu:
        X_unf = fu['data'][:]
        raw_unf = fu['labels'][:]
    with h5py.File(args.h5_labeled_filtered,'r') as ff:
        X_fil = ff['data'][:]
        raw_fil = ff['labels'][:]

    if X_unf.ndim==3: pass
    elif X_unf.ndim==4 and X_unf.shape[-1]==1: X_unf=X_unf[...,0]
    else: raise ValueError(f'bad X_unf {X_unf.shape}')
    if X_fil.ndim==3: pass
    elif X_fil.ndim==4 and X_fil.shape[-1]==1: X_fil=X_fil[...,0]
    else: raise ValueError(f'bad X_fil {X_fil.shape}')
    if X_unf.shape != X_fil.shape:
        raise ValueError(f"Unfiltered and filtered labeled shapes differ: {X_unf.shape} vs {X_fil.shape}")

    N = X_unf.shape[0]
    print(f"[DATA] Labeled unfiltered: {X_unf.shape}, filtered: {X_fil.shape}")

    selected=list(args.labels)
    cls_to_idx={c:i for i,c in enumerate(selected)}; C=len(selected)

    def to_multi(L):
        v=np.zeros(C, np.float32)
        for k in L:
            j=cls_to_idx.get(k,None)
            if j is not None: v[j]=1.0
        return v

    labels_list=[parse_label_string(v) for v in raw_unf]  # use UNFILTERED labels as truth

    def keep(L,mode):
        s=set(selected); Ls=set(L)
        if mode=='keep_any': return len(Ls & s) > 0
        if mode=='strict':   return Ls.issubset(s)
        if mode=='drop':     return (len(Ls)==1 and list(Ls)[0] in s)
        raise ValueError(mode)

    keep_mask = np.array([keep(L,args.filter_mode) for L in labels_list], bool)
    Y_full = np.stack([to_multi(L) for L in labels_list], 0)

    # ----- splits from NPZ (on UNFILTERED indices) -----
    splits = np.load(args.splits_npz)
    tr_ids_raw = splits['train_ids']
    va_ids_raw = splits['val_ids']
    te_ids_raw = splits['test_ids']

    # Apply keep_mask to each split
    tr_idx = np.array([i for i in tr_ids_raw if keep_mask[i]], dtype=int)
    va_idx = np.array([i for i in va_ids_raw if keep_mask[i]], dtype=int)
    te_idx = np.array([i for i in te_ids_raw if keep_mask[i]], dtype=int)

    Xtr_fil, Xva_unf, Xte_unf = X_fil[tr_idx], X_unf[va_idx], X_unf[te_idx]
    Ytr,     Yva,     Yte     = Y_full[tr_idx], Y_full[va_idx], Y_full[te_idx]

    print(f"[DATA] Train (filtered real): {Xtr_fil.shape[0]} | Val (unf): {Xva_unf.shape[0]} | Test (unf): {Xte_unf.shape[0]}")
    print("[DATA] Train positives per class:", Ytr.sum(axis=0).astype(int))

    # ----- GAN H5s as labeled TRAIN (all classes you provide) -----
    if args.gan_h5:
        X_g_list = []
        Y_g_list = []
        for p in args.gan_h5:
            with h5py.File(p,'r') as fg:
                Xg = fg['data'][:]
                lab_g = fg['labels'][:]
            if Xg.ndim==3: pass
            elif Xg.ndim==4 and Xg.shape[-1]==1: Xg=Xg[...,0]
            else: raise ValueError(f'bad GAN X {Xg.shape} in {p}')
            labs_g = [parse_label_string(v) for v in lab_g]
            Yg = np.stack([to_multi(L) for L in labs_g], 0)
            km_g = np.array([keep(L,args.filter_mode) for L in labs_g], bool)
            X_g_list.append(Xg[km_g])
            Y_g_list.append(Yg[km_g])
            print(f"[GAN] {p}: kept {km_g.sum()} / {len(km_g)} synthetic samples")
        if X_g_list:
            X_g = np.concatenate(X_g_list,0)
            Y_g = np.concatenate(Y_g_list,0)
            Xtr_fil = np.concatenate([Xtr_fil, X_g], 0)
            Ytr     = np.concatenate([Ytr,     Y_g], 0)
            print(f"[DATA] After GAN: Train (filtered+GAN): {Xtr_fil.shape[0]}")
            print("[DATA] Train positives per class (real+GAN):", Ytr.sum(axis=0).astype(int))

    # ----- build datasets -----
    in_ch = 2 if args.sobel else 1
    ds_tr = DynspecDataset(Xtr_fil, Ytr,     size=tuple(args.img), sobel=args.sobel, augment=True)
    ds_va = DynspecDataset(Xva_unf, Yva,     size=tuple(args.img), sobel=args.sobel, augment=False)
    ds_te = DynspecDataset(Xte_unf, Yte,     size=tuple(args.img), sobel=args.sobel, augment=False)

    if args.use_sampler:
        pos = Ytr.sum(0) + 1e-6
        w = (1.0/pos) @ Ytr.T
        w = torch.tensor(np.asarray(w, np.float32))
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        tr_loader=DataLoader(ds_tr,batch_size=args.batch,sampler=sampler,
                             num_workers=args.num_workers,pin_memory=True)
    else:
        tr_loader=DataLoader(ds_tr,batch_size=args.batch,shuffle=True,
                             num_workers=args.num_workers,pin_memory=True)
    va_loader=DataLoader(ds_va,batch_size=args.batch,shuffle=False,
                         num_workers=args.num_workers,pin_memory=True)
    te_loader=DataLoader(ds_te,batch_size=args.batch,shuffle=False,
                         num_workers=args.num_workers,pin_memory=True)

    # ----- model -----
    model = SSLClassifierHead(in_ch=in_ch, num_classes=C).to(dev)

    # load SSL encoder
    if args.ssl_ckpt and os.path.isfile(args.ssl_ckpt):
        sd = torch.load(args.ssl_ckpt, map_location='cpu')
        model.encoder.load_state_dict(sd, strict=True)
        print('[SSL] loaded encoder ckpt (strict=True).')

    # Head warmup: freeze encoder params and BN stats
    freeze_module(model.encoder, False)
    model.encoder.apply(set_bn_eval)
    freeze_module(model.encoder, False)
    for p in model.encoder.parameters(): p.requires_grad = False  # encoder frozen at start

    base_lr = args.lr

    head_params = [p for p in model.fc.parameters() if p.requires_grad]
    enc_params  = [p for p in model.encoder.parameters() if p.requires_grad]

    opt = torch.optim.AdamW([{'params': head_params, 'lr': base_lr}],
                            lr=base_lr, weight_decay=args.wd)

    for pg in opt.param_groups:
        pg.setdefault('lr_scale', 1.0)
        pg['base_lr_snapshot'] = args.lr
        pg['lr'] = args.lr * pg['lr_scale']

    crit = build_loss(args.loss, Ytr)
    if isinstance(crit, nn.BCEWithLogitsLoss):
        crit.pos_weight=crit.pos_weight.to(dev)
    crit=crit.to(dev)

    ema = EMA(model, decay=0.999, device=dev)

    total=args.epochs*len(tr_loader); warm=max(50, int(0.1*total))
    scaler=torch.cuda.amp.GradScaler(enabled=True)

    history=[]; best_macro=-1; best_state=None; best_ep=-1
    no_improve=0; patience=15   # <-- tweak this if you want earlier/later early stopping
    step=0

    def maybe_unfreeze(ep):
        nonlocal opt, head_params, enc_params

        changed = False
        # At end of head-warmup, start by unfreezing the tail
        if ep == args.head_warmup:
            for p in model.encoder.tail.parameters(): p.requires_grad = True
            changed = True

        # Progressively unfreeze deeper blocks
        for k, tgt_ep in enumerate(sorted(args.unfreeze_stages_at)):
            if ep == tgt_ep:
                if k == 0:
                    for p in model.encoder.mid.parameters(): p.requires_grad = True
                elif k == 1:
                    for p in model.encoder.down2.parameters(): p.requires_grad = True
                    for p in model.encoder.down1.parameters(): p.requires_grad = True
                    for p in model.encoder.stem.parameters():  p.requires_grad = True
                changed = True

        if changed:
            head_params = [p for p in model.fc.parameters() if p.requires_grad]
            enc_params  = [p for p in model.encoder.parameters() if p.requires_grad]
            params = []
            if head_params:
                params.append({'params': head_params, 'lr': base_lr})
            if enc_params:
                params.append({'params': enc_params,  'lr': base_lr * args.lrd})
            opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=args.wd)

    for ep in range(1, args.epochs+1):
        maybe_unfreeze(ep)

        model.train()
        run={'tr_loss':0,'tr_mi':0,'tr_ma':0,'tr_sa':0}; nb=0
        for xb,yb in tr_loader:
            xb=xb.to(dev,non_blocking=True); yb=yb.to(dev,non_blocking=True)
            xb = torch.nan_to_num(xb, nan=0.0, posinf=1.0, neginf=0.0)

            sched = cosine_with_warmup(step, total, warm)
            if len(opt.param_groups) == 1:
                opt.param_groups[0]['lr'] = base_lr * sched
            else:
                opt.param_groups[0]['lr'] = base_lr * sched
                opt.param_groups[1]['lr'] = base_lr * args.lrd * sched

            step+=1
            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=True):
                logits=model(xb)

            loss = crit(logits.float(), yb.float())
            if not torch.isfinite(loss):
                if not args.quiet:
                    print("[WARN] non-finite loss detected; skipping batch")
                opt.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            mi,ma,sa = f1_from_logits(logits.detach(), yb, thr=0.5)
            B=xb.size(0); run['tr_loss']+=float(loss)*B; run['tr_mi']+=mi; run['tr_ma']+=ma; run['tr_sa']+=sa; nb+=1
        for k in run: run[k]/= (len(ds_tr) if k=='tr_loss' else max(nb,1))

        # ----- validate with EMA weights -----
        model.eval()
        orig_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        ema.apply_to(model)

        va={'loss':0,'mi':0,'ma':0,'sa':0}; nb=0
        with torch.no_grad():
            for xb,yb in va_loader:
                xb=xb.to(dev); yb=yb.to(dev)
                with torch.cuda.amp.autocast(enabled=True):
                    logits=model(xb)
                loss=crit(logits.float(),yb.float())
                if not torch.isfinite(loss): continue
                mi,ma,sa=f1_from_logits(logits, yb, thr=0.5)
                B=xb.size(0); va['loss']+=float(loss)*B; va['mi']+=mi; va['ma']+=ma; va['sa']+=sa; nb+=1
        va['loss']/=max(len(ds_va),1); va['mi']/=max(nb,1); va['ma']/=max(nb,1); va['sa']/=max(nb,1)

        try:
            xb_, yb_ = next(iter(va_loader))
            xb_ = xb_.to(dev)
            with torch.cuda.amp.autocast(enabled=True):
                p_ = torch.sigmoid(model(xb_)).detach().float().cpu().numpy()
            if not args.quiet:
                print(f"[dbg] ep{ep:03d} val prob stats: mean={p_.mean():.4f} min={p_.min():.2e} max={p_.max():.2e}")
        except StopIteration:
            pass

        model.load_state_dict(orig_state, strict=True)
        history.append({'epoch':ep,'train_loss':run['tr_loss'],
                        'train_microF1':run['tr_mi'],'train_macroF1':run['tr_ma'],
                        'train_subsetAcc':run['tr_sa'],'val_loss':va['loss'],
                        'val_microF1':va['mi'],'val_macroF1':va['ma'],'val_subsetAcc':va['sa'],
                        'lr_head': opt.param_groups[0]['lr'],
                        'lr_enc': (opt.param_groups[1]['lr'] if len(opt.param_groups)>1 else 0.0)})
        if not args.quiet:
            print(f"Ep {ep:03d} | tr_loss {run['tr_loss']:.4f} trMa {run['tr_ma']:.3f} | va_loss {va['loss']:.4f} vaMa{va['ma']:.3f}")

        if va['ma'] > best_macro + 1e-6:
            best_macro=va['ma']; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_ep=ep; no_improve=0
        else:
            no_improve+=1
            if no_improve>=patience:
                print(f'[EarlyStop] no improve {patience} epochs. best at {best_ep} (val macro-F1={best_macro:.4f}).')
                break
        

    hist_df=pd.DataFrame(history) 
    hist_df.to_csv(os.path.join(outdir,'metrics.csv'), index=False)
    if not args.no_plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,4))
            plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train")
            plt.plot(hist_df["epoch"], hist_df["val_loss"],   label="val")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss (BCE/ASL)")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=150)
            plt.close()

            plt.figure(figsize=(6,4))
            plt.plot(hist_df["epoch"], hist_df["train_macroF1"], label="train macro-F1")
            plt.plot(hist_df["epoch"], hist_df["val_macroF1"],   label="val macro-F1")
            plt.xlabel("Epoch"); plt.ylabel("F1"); plt.title("Macro-F1")
            plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(outdir, "f1_curve.png"), dpi=150)
            plt.close()
        except Exception as e:
            print("[WARN] Plotting failed:", e)

    model.load_state_dict(best_state, strict=True)
    best_path=os.path.join(outdir,'best_yolo_multilabel_ssl.pth'); torch.save(model.state_dict(), best_path)

    # --- evaluate with per-class thresholds ---
    @torch.no_grad()
    def collect(loader):
        model.eval(); ts=[]; ps=[]
        for xb,yb in loader:
            xb=xb.to(dev); yb=yb.to(dev)
            with torch.cuda.amp.autocast(enabled=True):
                logits=model(xb)
            pr = torch.sigmoid(logits.float()).float().cpu().numpy()
            ts.append(yb.cpu().numpy()); ps.append(pr)
        Y = np.concatenate(ts,0); P = np.concatenate(ps,0)
        P = np.clip(P, 1e-6, 1.0-1e-6)
        return Y, P

    yv, pv = collect(va_loader)
    thr = per_class_thresholds(yv, pv)
    np.save(os.path.join(outdir,'thresholds.npy'), thr)
    with open(os.path.join(outdir,'thresholds.json'),'w') as f: json.dump([float(t) for t in thr.tolist()], f, indent=2)

    yt, pt = collect(te_loader)
    preds = (pt >= thr[None,:]).astype(int)

    # aggregate test accuracy metrics
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        yt, preds, average='micro', zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        yt, preds, average='macro', zero_division=0
    )
    subset_acc = float((preds == yt).all(axis=1).mean())

    p,r,f,s = precision_recall_fscore_support(yt, preds, average=None, zero_division=0)
    roc_aucs=[]; pr_aucs=[]
    for c in range(C):
        try: roc_aucs.append(roc_auc_score(yt[:,c], pt[:,c]))
        except: roc_aucs.append(np.nan)
        try: pr_aucs.append(average_precision_score(yt[:,c], pt[:,c]))
        except: pr_aucs.append(np.nan)
    class_names=[f"Type {c}" if c!=6 else "Nothing (6)" for c in selected]
    rows=[]
    for i,cn in enumerate(class_names):
        rows.append(dict(class_index=i, class_name=cn, label_1idx=selected[i],
                         precision=float(p[i]), recall=float(r[i]), f1=float(f[i]),
                         support=int(s[i]),
                         roc_auc=None if np.isnan(roc_aucs[i]) else float(roc_aucs[i]),
                         pr_auc=None if np.isnan(pr_aucs[i]) else float(pr_aucs[i])))
    pd.DataFrame(rows).to_csv(os.path.join(outdir,'per_class_metrics.csv'), index=False)

    agg = {
        "best_val_macroF1": round(float(best_macro),4),
        "best_epoch": int(best_ep),
        "thresholds": [round(float(t),3) for t in thr.tolist()],
        "test_micro_precision": float(micro_p),
        "test_micro_recall": float(micro_r),
        "test_micro_f1": float(micro_f1),
        "test_macro_precision": float(macro_p),
        "test_macro_recall": float(macro_r),
        "test_macro_f1": float(macro_f1),
        "test_subset_accuracy": float(subset_acc),
        "test_num_samples": int(yt.shape[0])
    }
    with open(os.path.join(outdir,'test_report.json'),'w') as f: json.dump(agg, f, indent=2)
    print('[Saved]', best_path, 'and thresholds.json')

    # ---- SAMPLE VISUALIZATION (test examples with true vs predicted) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        samples_dir = os.path.join(outdir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        # choose up to 12 random test samples
        num_samples = min(12, len(ds_te))
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(len(ds_te), size=num_samples, replace=False)

        model.eval()
        def labs_from_bin(bin_vec):
            return [str(selected[i]) for i, v in enumerate(bin_vec) if v > 0.5]

        for k, idx in enumerate(sample_indices):
            x_t, y_t = ds_te[idx]        # tensors
            x_np = x_t.numpy()
            y_np = y_t.numpy()

            x_in = x_t.unsqueeze(0).to(dev)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                logits = model(x_in)
                prob = torch.sigmoid(logits.float())[0].cpu().numpy()
            pred_bin = (prob >= thr).astype(int)

            img = x_np[0]   # channel 0 for visualization
            true_labs = labs_from_bin(y_np)
            pred_labs = labs_from_bin(pred_bin)

            plt.figure(figsize=(4.5, 3.5))
            plt.imshow(img, aspect='auto', origin='lower', cmap='inferno')
            plt.colorbar(fraction=0.046, pad=0.04)
            title = f"idx={idx} | True: {','.join(true_labs) or 'None'} | Pred: {','.join(pred_labs) or 'None'}"
            plt.title(title, fontsize=8)
            plt.tight_layout()
            out_path = os.path.join(samples_dir, f"test_idx{idx:04d}.png")
            plt.savefig(out_path, dpi=150)
            plt.close()

        print(f"[VIS] Saved {num_samples} test sample PNGs to {samples_dir}")
    except Exception as e:
        print("[WARN] Visualization failed:", e)

if __name__=='__main__': main()
