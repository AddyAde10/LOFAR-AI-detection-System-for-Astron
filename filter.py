
# This does not work when i am trting to apply it on all 6641 now labelled images because 
# this filter algorithm works for png and not the h5 files. The h5 files might need a different way to handle
# their complexity. 

import h5py
from PIL import Image, ImageEnhance
import numpy as np
from scipy.ndimage import uniform_filter

def apply_lightroom_filter(image_array):
    img = Image.fromarray(image_array)

    # Apply orientation changes first
    # 1. Rotate the image 90 degrees to the left (counter-clockwise)
    img = img.rotate(270, expand=True) # Use expand=True to ensure the whole image is visible after rotation

    # 2. Flip the image on the vertical axis
    img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Convert to numpy array for channel normalization and intensity enhancement
    img_array = np.array(img).astype(np.float32)

    # --- Channel Normalization Filter ---
    # S = Ŝ - SLM + Sbck
    # Ŝ is input spectrum (img_array)
    # SLM is local mean of the spectrum
    # Sbck is the mean of the whole spectrum adds with the global background

    # Calculate local mean (SLM) using a uniform filter (neighborhood averaging)
    filter_size_norm = 5 # Example size, needs fine-tuning for normalization
    slm = uniform_filter(img_array, size=filter_size_norm)

    # Calculate global mean (Sbck) of the whole spectrum
    sbck = np.mean(img_array)

    # Apply the normalization equation
    normalized_array = img_array - slm + sbck

    # Ensure non-negativity of S
    normalized_array = np.clip(normalized_array, 0, 255) # Clip to 0-255 range for image data

    # # --- Image Intensity Enhancement Filter (Neighborhood Averaging) ---
    # # This process can increase the significant edge in the solar radio spectrum
    # # using neighborhood averaging.
    # filter_size_enhance = 3 # Example size for enhancement, needs fine-tuning
    # enhanced_array = uniform_filter(normalized_array, size=filter_size_enhance)
    # enhanced_array = np.clip(enhanced_array, 0, 255) # Ensure values are within valid range

    # img = Image.fromarray(enhanced_array.astype(np.uint8))

    # Apply Exposure, Contrast, Saturation
    brightness_factor = 0.6
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)

    contrast_factor = 2.0
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)

    saturation_factor = 0.69
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)

    # Convert to grayscale (luminance) to prepare for green mapping
    gray_img = img.convert("L")
    gray_array = np.array(gray_img)

    # Create an empty RGB array for the output, initialized to black
    output_img_array = np.zeros((gray_array.shape[0], gray_array.shape[1], 3), dtype=np.uint8)

    # Apply a non-linear mapping to enhance the black background and bright green.
    normalized_gray = gray_array / 255.0
    exponent = 2.0
    adjusted_intensity = np.power(normalized_gray, exponent) * 255
    adjusted_intensity = np.clip(adjusted_intensity, 0, 255).astype(np.uint8)

    # Map the adjusted intensity to the green channel
    output_img_array[:, :, 1] = adjusted_intensity

    return Image.fromarray(output_img_array)

def process_h5_file(input_h5_path, output_h5_path):
    with h5py.File(input_h5_path, "r") as f_in:
        if "data" not in f_in:
            raise KeyError("H5 file must contain a dataset named 'data'")

        original_images = f_in["data"]

        with h5py.File(output_h5_path, "w") as f_out:
            if original_images.shape[0] > 0:
                temp_img = Image.fromarray(original_images[0])
                temp_img = temp_img.rotate(270, expand=True)
                temp_img = temp_img.transpose(Image.FLIP_TOP_BOTTOM)
                new_height, new_width, new_channels = np.array(temp_img).shape
            else:
                new_height, new_width, new_channels = 0, 0, 0

            processed_images_dataset = f_out.create_dataset(
                "data",
                shape=(original_images.shape[0], new_height, new_width, new_channels),
                dtype=original_images.dtype,
                chunks=True
            )

            for i in range(original_images.shape[0]):
                print(f"Processing image {i+1}/{original_images.shape[0]}")
                image_array = original_images[i]
                processed_image = apply_lightroom_filter(image_array)
                processed_images_dataset[i] = np.array(processed_image)

if __name__ == "__main__":
    num_images = 4
    image_height = 800 
    image_width = 500
    
    image_paths = [
        'fits files/output.png',
        'fits files/outputw copy 2.png',
        'fits files/outputw copy.png',
        'fits files/outputw.png'
    ]
    
    image_arrays = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = img.resize((image_width, image_height))
        image_arrays.append(np.array(img))

    dummy_data = np.array(image_arrays)

    with h5py.File("input_images.h5", "w") as f:
        f.create_dataset("data", data=dummy_data)

    print("Dummy input_images.h5 created with example PNGs.")

    process_h5_file("input_images.h5", "output_filtered_images.h5")
    print("Processing complete. Filtered images saved to output_filtered_images.h5")

    with h5py.File("output_filtered_images.h5", "r") as f_out:
        if "data" in f_out:
            for i in range(f_out["data"].shape[0]):
                filtered_image = f_out["data"][i]
                Image.fromarray(filtered_image).save(f"filtered_image_{i+1}.png")
                print(f"Saved filtered image {i+1} to filtered_image_{i+1}.png")


