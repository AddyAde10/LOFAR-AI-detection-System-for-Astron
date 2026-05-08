#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orientation-only for UNLABELLED H5 (unfiltered)
- Rotate 90° CCW, then flip top->bottom
- No background mitigation
- Copies all non-data keys; rewrites only 'data'
- Output: <stem>_oriented_unf.h5 (in --outdir)
"""

import os, sys, time, argparse, h5py, numpy as np

DATA_KEY = "data"

def fix_orientation_2d(img: np.ndarray) -> np.ndarray:
    x = np.rot90(img, k=1, axes=(0, 1))  # rotate CCW
    x = np.flip(x, axis=0)               # flip top->bottom
    return x

def copy_attrs(src_obj, dst_obj):
    for k, v in src_obj.attrs.items():
        dst_obj.attrs[k] = v

def copy_tree_except_data(src_grp, dst_grp, data_key=DATA_KEY):
    """Recursively copy everything except the dataset named `data_key` at each level."""
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
            g = dst_grp.create_group(name)
            copy_tree_except_data(item, g, data_key=data_key)

def make_progress(total, bar_len=30):
    start = time.time()
    def _u(i):
        frac = (i + 1) / total
        filled = int(bar_len * frac)
        bar = "█" * filled + "-" * (bar_len - filled)
        elapsed = time.time() - start
        itps = (i + 1) / max(elapsed, 1e-6)
        eta = (total - (i + 1)) / max(itps, 1e-6)
        sys.stdout.write(f"\r[{bar}] {i+1}/{total}  {itps:5.1f} it/s  ETA {eta:6.1f}s")
        sys.stdout.flush()
        if i + 1 == total:
            sys.stdout.write("\n")
    return _u

def parse_args():
    ap = argparse.ArgumentParser(description="Orientation-only (unfiltered) for unlabelled H5.")
    ap.add_argument("--h5", required=True, help="Path to unlabelled .h5 with dataset 'data'")
    ap.add_argument("--outdir", default=".", help="Output directory")
    ap.add_argument("--suffix", default="_oriented_unf", help="Output suffix for filename stem")
    ap.add_argument("--compression", default="lzf", choices=["gzip","lzf","none"], help="HDF5 compression")
    return ap.parse_args()

def main():
    args = parse_args()
    src = os.path.abspath(args.h5)
    os.makedirs(args.outdir, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(src))
    dst = os.path.join(args.outdir, f"{stem}{args.suffix}{ext}")

    print(f"Source: {src}")
    print(f"Output: {dst}")

    with h5py.File(src, "r") as fr, h5py.File(dst, "w") as fw:
        if DATA_KEY not in fr:
            raise KeyError(f"'{DATA_KEY}' not found in {src}")

        # Copy all metadata/side datasets first
        copy_tree_except_data(fr, fw, data_key=DATA_KEY)

        src_data = fr[DATA_KEY]
        N, H, W = src_data.shape
        print(f"Data shape: (N,H,W) = ({N}, {H}, {W})")

        # Determine oriented shape from a small dummy frame (no need to load big data)
        dummy = np.zeros((H, W), dtype=np.float32)
        H2, W2 = fix_orientation_2d(dummy).shape  # typically (W, H)

        # Prepare output dataset with oriented shape
        comp = None if args.compression == "none" else args.compression
        dset_out = fw.create_dataset(
            DATA_KEY,
            shape=(N, H2, W2),
            dtype=np.float32,
            compression=comp,
            chunks=(1, H2, W2)
        )
        copy_attrs(src_data, dset_out)
        dset_out.attrs["orientation_fixed"] = np.string_("rot90_ccw_then_flip_tb")
        dset_out.attrs["filtered_stage"] = np.string_("orientation_only_unfiltered")

        # Stream through frames and write oriented result (always orient, even if "invalid")
        update = make_progress(N)
        for i in range(N):
            Ic = np.asarray(src_data[i], dtype=np.float32)  # (H,W)
            oriented = fix_orientation_2d(Ic)               # (H2,W2)
            # If you want to guard weird values, clip after orientation:
            oriented = np.clip(oriented, 0.0, None, dtype=np.float32)
            dset_out[i] = oriented
            update(i)

    print("Done.")

if __name__ == "__main__":
    main()
