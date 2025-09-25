import os
import h5py
import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import uniform_filter

# ------------------------------
# Helpers
# ------------------------------

def robust_uint8(x, lo=1, hi=99):
    """Percentile-based rescale of a float array to uint8 [0..255]."""
    x = np.asarray(x, dtype=np.float32)
    # Handle NaNs/Infs
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=np.nanmin(x[np.isfinite(x)]) if np.isfinite(x).any() else 0.0)

    p_lo, p_hi = np.percentile(x, [lo, hi])
    if p_hi <= p_lo:
        # fallback: simple min/max
        x_min, x_max = float(np.min(x)), float(np.max(x))
    else:
        x_min, x_max = p_lo, p_hi

    if x_max == x_min:
        y = np.zeros_like(x, dtype=np.uint8)
    else:
        y = (np.clip(x, x_min, x_max) - x_min) / (x_max - x_min)
        y = (255.0 * y).round().astype(np.uint8)
    return y

def apply_lightroom_like_steps(pil_img, brightness=0.6, contrast=1.35, sharpness=1.0, saturation=None):
    """
    Apply 'Lightroom-ish' edits to a PIL Image.
    - brightness < 1.0 darkens, > 1.0 brightens
    - contrast > 1.0 increases contrast
    - sharpness > 1.0 increases sharpness
    - saturation can be ignored for grayscale; if provided, convert to RGB first
    """
    img = pil_img

    # If saturation requested on grayscale, convert to RGB first
    if saturation is not None and img.mode != "RGB":
        img = img.convert("RGB")

    # Brightness
    img = ImageEnhance.Brightness(img).enhance(brightness)
    # Contrast
    img = ImageEnhance.Contrast(img).enhance(contrast)
    # Sharpness
    img = ImageEnhance.Sharpness(img).enhance(sharpness)

    # Optional saturation (only meaningful in RGB)
    if saturation is not None:
        img = ImageEnhance.Color(img).enhance(saturation)

    return img

def process_frame_float2d(frame2d,
                          rotate_deg=270,
                          flip_top_bottom=True,
                          local_mean_size=5,
                          percentile_low=1, percentile_high=99,
                          brightness=0.6, contrast=1.35, sharpness=1.0, saturation=None):
    """
    Take a 2D float array (freq x time or time x freq), perform local-mean normalization,
    robustly rescale to uint8, apply orientation, then 'Lightroom-like' steps.
    Returns a PIL Image.
    """
    arr = np.asarray(frame2d, dtype=np.float32)

    # Local-mean normalize on floats (same idea you had before)
    if local_mean_size and local_mean_size > 1:
        slm = uniform_filter(arr, size=local_mean_size, mode="nearest")
        sbck = np.mean(arr[np.isfinite(arr)]) if np.isfinite(arr).any() else 0.0
        arr = arr - slm + sbck

    # Robust rescale to 8-bit
    u8 = robust_uint8(arr, lo=percentile_low, hi=percentile_high)

    # Make grayscale image
    img = Image.fromarray(u8, mode="L")

    # Orientation fixes (match your PNG flow)
    if rotate_deg:
        img = img.rotate(rotate_deg, expand=True)
    if flip_top_bottom:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Lightroom-like steps
    img = apply_lightroom_like_steps(
        img,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        saturation=saturation,  # keep None for grayscale
    )
    return img

# ------------------------------
# H5 processing
# ------------------------------

def process_h5_file(h5_path, key="data", out_dir="h5_filtered_out",
                    first_n=None,
                    # tuning knobs; match PNG look by adjusting these
                    rotate_deg=270, flip_top_bottom=True,
                    local_mean_size=5, percentile_low=1, percentile_high=99,
                    brightness=0.6, contrast=1.35, sharpness=1.0, saturation=None):
    """
    Reads an HDF5 dataset (shape: N, H, W) or (N, H, W, C) and writes PNGs.
    """
    os.makedirs(out_dir, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        ds = f[key]
        n = ds.shape[0]
        if first_n is not None:
            n = min(n, int(first_n))

        for i in range(n):
            frame = ds[i]
            # Accept both 2D or 3D; if 3D, pick a single channel or reduce
            if frame.ndim == 3:
                # If it’s (H, W, 3) but all channels are same, just take one
                frame = frame[..., 0]
            elif frame.ndim != 2:
                # Skip incompatible shapes
                continue

            img = process_frame_float2d(
                frame,
                rotate_deg=rotate_deg,
                flip_top_bottom=flip_top_bottom,
                local_mean_size=local_mean_size,
                percentile_low=percentile_low,
                percentile_high=percentile_high,
                brightness=brightness,
                contrast=contrast,
                sharpness=sharpness,
                saturation=saturation,
            )

            img.save(os.path.join(out_dir, f"{i:06d}.png"))

# ------------------------------
# PNG passthrough (optional)
# ------------------------------

def process_png_file(png_path, out_path,
                     rotate_deg=270, flip_top_bottom=True,
                     local_mean_size=5, percentile_low=1, percentile_high=99,
                     brightness=0.6, contrast=1.35, sharpness=1.0, saturation=None):
    """
    Pass a PNG through the same steps for consistency.
    """
    img = Image.open(png_path)
    # Ensure grayscale pipeline: convert to float2d then reuse same logic
    arr = np.asarray(img.convert("L"), dtype=np.float32)  # 0..255 to float
    return_img = process_frame_float2d(
        arr,
        rotate_deg=rotate_deg,
        flip_top_bottom=flip_top_bottom,
        local_mean_size=local_mean_size,
        percentile_low=percentile_low,
        percentile_high=percentile_high,
        brightness=brightness,
        contrast=contrast,
        sharpness=sharpness,
        saturation=saturation,
    )
    return_img.save(out_path)

# ------------------------------
# Example usage
# ------------------------------
process_h5_file("/Volumes/External SSD 512gb/dset_with_labels.h5", key="data", out_dir="h5_filtered_out", first_n=10)
process_png_file("fits files/outputw copy.png", "output.png")
