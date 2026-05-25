#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orientation + Background Mitigation (Ir) for UNLABELLED H5

- Applies orientation fix first: rotate 90° CCW then flip top→bottom
- Runs paper §4.2 background mitigation with soft low-band cross-fade
- Copies all non-data keys unchanged; writes new `data` with Ir
- No label logic anywhere (safe for unlabelled datasets)

Output filename defaults to: <stem>_oriented_ir.h5 (in --outdir)
"""

import os, sys, time, argparse, h5py, numpy as np
from typing import Optional, Tuple

DATA_KEY = "data"
FREQ_RANGE_KEY = "freq_range"    # optional
TS_KEY = "timestamps"            # optional

# -------------------- Orientation --------------------
def fix_orientation(arr: np.ndarray) -> np.ndarray:
    """
    Rotate 90° anticlockwise, then flip top-to-bottom.
    Works for (H,W) and (N,H,W).
    """
    if arr.ndim == 2:
        x = np.rot90(arr, k=1, axes=(0, 1))
        x = np.flip(x, axis=0)
        return x
    elif arr.ndim == 3:
        x = np.rot90(arr, k=1, axes=(1, 2))
        x = np.flip(x, axis=1)
        return x
    raise ValueError(f"Unsupported array rank for fix_orientation: {arr.shape}")

# -------------------- Background mitigation --------------------
def rolling_mean_along_time(image: np.ndarray, win: int) -> np.ndarray:
    H, W = image.shape
    if win < 1: return image.copy()
    if win % 2 == 0: win += 1
    padL = win // 2
    padR = padL + 1  # keep width exact
    padded = np.pad(image, ((0,0),(padL,padR)), mode='edge')
    csum = np.cumsum(padded, axis=1, dtype=np.float64)
    out = (csum[:, win:] - csum[:, :-win]) / win
    if out.shape[1] != W:
        raise RuntimeError(f"rolling_mean width {out.shape[1]} != {W}")
    return out.astype(np.float32)

def long_stats_per_row(Ic: np.ndarray, robust: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    if robust:
        med = np.median(Ic, axis=1)
        mad = np.median(np.abs(Ic - med[:, None]), axis=1)
        return med.astype(np.float32), (1.4826 * mad).astype(np.float32)
    return Ic.mean(axis=1).astype(np.float32), Ic.std(axis=1, ddof=0).astype(np.float32)

def build_freq_axis_from_range(freq_rng: np.ndarray, H: int) -> np.ndarray:
    fmin, fmax = float(freq_rng[0]), float(freq_rng[1])
    return np.linspace(fmin, fmax, H, dtype=np.float32)

def soft_lowfreq_weight(freqs: np.ndarray, f_cut: float, df: float) -> np.ndarray:
    w = np.ones_like(freqs, dtype=np.float32)
    lo, hi = f_cut - df, f_cut + df
    w[freqs >= hi] = 0.0
    sel = (freqs > lo) & (freqs < hi)
    ramp = (freqs[sel] - lo) / (2*df) * np.pi
    w[sel] = 0.5 * (1 + np.cos(ramp))  # 1→0
    return w

def mitigate_background_42(
    Ic: np.ndarray,
    freq_range_for_sample: Optional[np.ndarray],
    k: float,
    T_short_s: float,
    use_robust: bool,
    f_cut_MHz: Optional[float],
    low_band_fraction: Optional[float],
    ramp_df_MHz: float,
    z_protect: float,
    sub_fraction_cap: float,
    bias_match_highband: bool = True,
) -> np.ndarray:
    H, W = Ic.shape
    sec_per_col = 900.0 / float(W)
    win_cols = int(round(T_short_s / sec_per_col))
    if win_cols < 1: win_cols = 1
    if win_cols % 2 == 0: win_cols += 1

    lb = rolling_mean_along_time(Ic, win=win_cols)
    lr, rr = long_stats_per_row(Ic, robust=use_robust)
    lr_full = np.broadcast_to(lr[:, None], Ic.shape)
    rr_full = np.broadcast_to(rr[:, None], Ic.shape)

    if freq_range_for_sample is not None:
        f_axis = build_freq_axis_from_range(freq_range_for_sample, H)
    else:
        f_axis = np.linspace(0.0, 1.0, H, dtype=np.float32)

    if f_cut_MHz is not None:
        w_rows = soft_lowfreq_weight(f_axis, f_cut=float(f_cut_MHz), df=float(ramp_df_MHz))
    elif low_band_fraction is not None:
        frac = float(np.clip(low_band_fraction, 0.0, 1.0))
        cutoff = np.quantile(f_axis, frac)
        w_rows = soft_lowfreq_weight(f_axis, f_cut=float(cutoff), df=float(ramp_df_MHz))
    else:
        w_rows = np.ones(H, dtype=np.float32)
    w2 = w_rows[:, None].astype(np.float32)

    threshold = lr_full + k * rr_full
    use_lb_base = (lb <= threshold)
    burst_mask = (Ic >= (lr_full + z_protect * rr_full))
    use_lb = np.where(burst_mask, False, use_lb_base)

    sub_amount = np.where(use_lb, lb, lr_full).astype(np.float32)
    if sub_fraction_cap is not None and sub_fraction_cap > 0:
        cap = sub_fraction_cap * np.maximum(Ic, 1e-6)
        sub_amount = np.minimum(sub_amount, cap).astype(np.float32)

    Ir = Ic - (w2 * sub_amount)

    if bias_match_highband:
        thr = np.quantile(w_rows, 0.9)  # top ~10% lowest weights
        high_mask = (w_rows <= thr)
        if np.any(high_mask):
            mu_ref = float(np.median(Ir[high_mask, :].mean(axis=1)))
            mu_row = Ir.mean(axis=1)
            delta = (mu_ref - mu_row).astype(np.float32)
            Ir += (w_rows[:, None] * delta[:, None]).astype(np.float32)

    return np.clip(Ir, 0.0, None).astype(np.float32)

# -------------------- H5 utils --------------------
def copy_attrs(src_obj, dst_obj):
    for k, v in src_obj.attrs.items():
        dst_obj.attrs[k] = v

def copy_tree_except_data(src_grp, dst_grp, data_key=DATA_KEY):
    """Copy everything except dataset named `data_key` at each level."""
    copy_attrs(src_grp, dst_grp)
    for name, item in src_grp.items():
        if isinstance(item, h5py.Dataset):
            if name == data_key and item.parent == src_grp:
                continue
            dset = dst_grp.create_dataset(
                name,
                data=item[...],
                dtype=item.dtype,
                compression=item.compression if item.compression else None,
                shuffle=getattr(item, "shuffle", None),
                fletcher32=getattr(item, "fletcher32", None),
                chunks=item.chunks
            )
            copy_attrs(item, dset)
        elif isinstance(item, h5py.Group):
            new_grp = dst_grp.create_group(name)
            copy_tree_except_data(item, new_grp, data_key=data_key)

def make_progress(total, bar_len=30):
    start = time.time()
    def _update(i):
        frac = (i + 1) / total
        filled = int(bar_len * frac)
        bar = "█" * filled + "-" * (bar_len - filled)
        elapsed = time.time() - start
        itps = (i + 1) / max(elapsed, 1e-6)
        eta = (total - (i + 1)) / max(itps, 1e-6)
        sys.stdout.write(f"\r[{bar}] {i+1}/{total}  {itps:5.1f} it/s  ETA {eta:5.1f}s")
        sys.stdout.flush()
        if i + 1 == total:
            sys.stdout.write("\n")
    return _update

# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Compute Ir for unlabelled H5 with orientation fix.")
    ap.add_argument("--h5", required=True, help="Path to unlabelled .h5 with dataset 'data'")
    ap.add_argument("--outdir", default=".", help="Directory for output file")
    ap.add_argument("--suffix", default="_oriented_ir", help="Suffix for output filename stem")
    ap.add_argument("--k", type=float, default=1.2)
    ap.add_argument("--T_short_s", type=float, default=45.0)
    ap.add_argument("--robust", action="store_true", default=True, help="Use median/MAD long stats")
    ap.add_argument("--no-robust", dest="robust", action="store_false")
    ap.add_argument("--f_cut_MHz", type=float, default=40.0)
    ap.add_argument("--low_band_fraction", type=float, default=None)
    ap.add_argument("--ramp_df_MHz", type=float, default=3.0)
    ap.add_argument("--z_protect", type=float, default=2.0)
    ap.add_argument("--sub_fraction_cap", type=float, default=0.6)
    return ap.parse_args()

def main():
    args = parse_args()
    src = os.path.abspath(args.h5)
    os.makedirs(args.outdir, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(src))
    dst = os.path.join(args.outdir, f"{stem}{args.suffix}{ext}")

    print(f"Source: {src}")
    print(f"Output: {dst}")

    with h5py.File(src, "r") as fr:
        if DATA_KEY not in fr:
            raise KeyError(f"'{DATA_KEY}' not found in {src}")
        N, H, W = fr[DATA_KEY].shape
        print(f"Loaded data shape: (N,H,W) = {N,H,W}")

        # Load entire data to RAM (HPC node OK) and apply orientation
        data = fr[DATA_KEY][...].astype(np.float32)
        print("Applying orientation fix to full dataset in RAM…")
        data_oriented = fix_orientation(data)  # (N,H,W)
        del data

        freq_ranges_ds = fr.get(FREQ_RANGE_KEY, None)

        # Create destination and copy all non-data keys first
        with h5py.File(dst, "w") as fw:
            copy_tree_except_data(fr, fw, data_key=DATA_KEY)
            dset_out = fw.create_dataset(
                DATA_KEY,
                shape=data_oriented.shape,
                dtype=np.float32,
                compression="gzip",
                chunks=(1, data_oriented.shape[1], data_oriented.shape[2])
            )
            # Copy attrs and annotate
            copy_attrs(fr[DATA_KEY], dset_out)
            dset_out.attrs["orientation_fixed"] = np.string_("rot90_ccw_then_flip_tb")
            dset_out.attrs["filtered_stage"] = np.string_("background_only")
            dset_out.attrs["paper_section"] = np.string_("4.2 background mitigation (soft low-band cross-fade)")

            # Process each frame
            update = make_progress(N)
            for i in range(N):
                Ic = data_oriented[i]
                frng = None
                if freq_ranges_ds is not None:
                    frng = np.asarray(freq_ranges_ds[i], dtype=np.float32)

                # Guard invalid frames
                if not np.isfinite(Ic).all() or Ic.max() <= Ic.min():
                    dset_out[i] = np.clip(Ic, 0.0, None).astype(np.float32)
                    update(i); continue

                Ir = mitigate_background_42(
                    Ic, frng,
                    k=args.k,
                    T_short_s=args.T_short_s,
                    use_robust=args.robust,
                    f_cut_MHz=(None if args.f_cut_MHz is None else float(args.f_cut_MHz)),
                    low_band_fraction=args.low_band_fraction,
                    ramp_df_MHz=args.ramp_df_MHz,
                    z_protect=args.z_protect,
                    sub_fraction_cap=args.sub_fraction_cap,
                    bias_match_highband=True
                )
                dset_out[i] = Ir
                update(i)

    print("Done.")

if __name__ == "__main__":
    main()
