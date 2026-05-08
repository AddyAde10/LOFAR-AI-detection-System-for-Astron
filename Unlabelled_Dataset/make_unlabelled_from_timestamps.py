#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_unlabelled_from_timestamps.py  (FAST)
- Builds unlabelled H5 by excluding labelled timestamps.
- SPEED: Writes per-sample datasets in contiguous RUNS (avoid fancy indexing).
- Uses LZF (fast) by default and row-blocked chunks for 'data'.
- Shows tqdm progress bars.

Usage:
  python -u make_unlabelled_from_timestamps.py \
    --full dset_2025.h5 \
    --labelled labelled_dataset_7classes.h5 \
    --compression lzf
"""

import argparse, os
import h5py, numpy as np
from tqdm import tqdm

def is_per_sample_key(f, key, n0):
    try:
        d = f[key]
        return (len(d.shape) >= 1) and (d.shape[0] == n0)
    except Exception:
        return False

def ensure_bytes(arr):
    if arr.dtype.kind == 'S': return arr
    if arr.dtype.kind in ('U','O'):
        return np.array([x.encode('utf-8') if isinstance(x, str) else x for x in arr], dtype='S')
    return arr.astype('S')

def build_keep_mask(ts_full_b, labelled_set):
    # vectorized-ish membership w/ Python set
    return np.fromiter((t not in labelled_set for t in ts_full_b.tolist()), dtype=bool, count=len(ts_full_b))

def mask_to_runs(mask):
    """Return list of (start,end) half-open contiguous runs where mask is True."""
    N = len(mask)
    runs = []
    s = None
    for i in range(N):
        if mask[i] and s is None:
            s = i
        if (not mask[i] and s is not None) or (i == N-1 and s is not None):
            e = i if not mask[i] else i+1
            runs.append((s, e))
            s = None
    return runs

def create_output_dset_like(src, fout, name, n_keep, compression, chunks=None):
    shape = (n_keep,) + src.shape[1:]
    ds = fout.create_dataset(name, shape=shape, dtype=src.dtype,
                             compression=compression, chunks=chunks)
    # copy attrs
    for k, v in src.attrs.items():
        ds.attrs[k] = v
    return ds

def copy_per_sample_runs(src, dst, runs, desc):
    """Copy using contiguous runs; show a single tqdm over kept rows."""
    total = sum(b - a for (a, b) in runs)
    p = 0
    with tqdm(total=total, ncols=80, desc=desc) as t:
        w = 0
        for (a, b) in runs:
            n = b - a
            dst[w:w+n] = src[a:b]     # contiguous slice = fast
            w += n
            p += n
            t.update(n)

def copy_1d_string_per_sample(src, fout, name, keep_mask, runs, compression):
    # 1D strings: safer to bulk copy per run then assign
    n_keep = int(keep_mask.sum())
    # Create string dtype similar to src
    if src.dtype.kind == 'S':
        maxlen = int(src.dtype.itemsize)
        dt = f"S{maxlen}"
    else:
        dt = h5py.string_dtype(encoding='utf-8')
    out = fout.create_dataset(name, shape=(n_keep,), dtype=dt, compression=compression, chunks=True)
    w = 0
    with tqdm(total=n_keep, ncols=80, desc=f"Copying {name}") as t:
        for (a, b) in runs:
            block = src[a:b]
            out[w:w+(b-a)] = block
            w += (b - a)
            t.update(b - a)
    for k, v in src.attrs.items():
        out.attrs[k] = v

def main(args):
    # Resolve output path next to labelled H5
    labelled_dir = os.path.dirname(os.path.abspath(args.labelled))
    out_path = os.path.join(labelled_dir, "unlabelled_dataset.h5")
    print(f"[INFO] Output will be saved at: {out_path}")

    # Load labelled timestamps
    print(f"[INFO] Loading labelled timestamps from {args.labelled}")
    with h5py.File(args.labelled, "r") as flab:
        lab_ts = flab["timestamps"][...]
        lab_ts_b = ensure_bytes(lab_ts)
        labelled_set = set(lab_ts_b.tolist())
    print(f"[INFO] labelled unique: {len(labelled_set)}")

    # Open full
    print(f"[INFO] Opening full dataset: {args.full}")
    with h5py.File(args.full, "r") as ffull:
        full_ts = ffull["timestamps"][...]
        full_ts_b = ensure_bytes(full_ts)
        N = len(full_ts_b)
        print(f"  Total samples in full dataset: {N}")

        keep_mask = build_keep_mask(full_ts_b, labelled_set)
        n_keep = int(keep_mask.sum())
        print(f"  Keeping {n_keep} samples (excluding {N - n_keep} labelled)")
        if n_keep == 0:
            raise RuntimeError("No unlabelled samples remain.")
        runs = mask_to_runs(keep_mask)
        print(f"[INFO] Contiguous runs: {len(runs)}")

        # Write output fast
        with h5py.File(out_path, "w") as fout:
            # file attrs
            for k, v in ffull.attrs.items():
                fout.attrs[k] = v

            # decide keys
            keys = list(ffull.keys())
            per_sample, global_keys = [], []
            for k in keys:
                try:
                    if is_per_sample_key(ffull, k, N):
                        per_sample.append(k)
                    else:
                        global_keys.append(k)
                except Exception:
                    global_keys.append(k)

            # exclude labels from per-sample
            per_sample = [k for k in per_sample if k != "labels"]

            # DATA first with optimal chunks: (rows, H, W)
            if "data" in per_sample:
                src = ffull["data"]
                H, W = src.shape[1], src.shape[2]
                # Good throughput choice: row blocks 16 (tune if needed)
                chunks = (min(16, n_keep), H, W)
                compression = None if args.compression.lower() == "none" else args.compression
                dst = create_output_dset_like(src, fout, "data", n_keep, compression=compression, chunks=chunks)
                copy_per_sample_runs(src, dst, runs, desc="Copying data")
                per_sample.remove("data")

            # Other per-sample datasets
            compression = None if args.compression.lower() == "none" else args.compression
            for name in per_sample:
                src = ffull[name]
                # string 1D?
                if (src.dtype.kind in ('O', 'S')) and src.ndim == 1:
                    copy_1d_string_per_sample(src, fout, name, keep_mask, runs, compression=compression)
                else:
                    # make dst with similar chunks: for 1D use (min(8192,n_keep),), else row-block
                    if src.ndim == 1:
                        chunks = (min(8192, n_keep),)
                    else:
                        rowblk = 1024 if src.ndim == 1 else min(1024, n_keep)
                        chunks = (rowblk,) + src.shape[1:]
                    dst = create_output_dset_like(src, fout, name, n_keep, compression=compression, chunks=chunks)
                    copy_per_sample_runs(src, dst, runs, desc=f"Copying {name}")

            # Copy globals verbatim
            for name in global_keys:
                if name == "labels":
                    continue
                obj = ffull[name]
                if isinstance(obj, h5py.Dataset):
                    out = fout.create_dataset(name, data=obj[...], dtype=obj.dtype, compression=compression, chunks=True)
                    for k, v in obj.attrs.items():
                        out.attrs[k] = v
                else:
                    g = fout.create_group(name)
                    for k, v in obj.attrs.items():
                        g.attrs[k] = v
                    for subk, subobj in obj.items():
                        if isinstance(subobj, h5py.Dataset):
                            d = g.create_dataset(subk, data=subobj[...], dtype=subobj.dtype, compression=compression, chunks=True)
                            for ak, av in subobj.attrs.items():
                                d.attrs[ak] = av
                        else:
                            g.create_group(subk)

            # sanity
            assert "timestamps" in fout, "timestamps missing"
            assert len(fout["timestamps"]) == n_keep, "timestamps length mismatch"

    print(f"[DONE] Unlabelled dataset saved to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full", required=True)
    p.add_argument("--labelled", required=True)
    p.add_argument("--compression", default="lzf", help="lzf | gzip | none")
    args = p.parse_args()
    main(args)
