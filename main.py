# main.py
import os
import glob
import time
import subprocess
import torch
import warnings
from ultralytics import FastSAM, YOLO
from PIL import Image  # Pillow for image loading, though ultralytics also handles it

# --- Configuration ---
IMAGE_DIR = "images"  # Directory containing your test images
SUPPORTED_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]  # Includes BMP
FASTSAM_MODEL_NAME = 'FastSAM-x.pt'  # Using the extra large model
MIN_CUDA_COMPUTE_CAPABILITY = 3.7  # Minimum compute capability supported by modern PyTorch


# --- Helper Functions ---

def get_gpu_info():
    """
    Retrieves and prints NVIDIA GPU information using nvidia-smi.
    Returns True if an NVIDIA GPU is detected, False otherwise.
    """
    print("-" * 50)
    print("GPU Information:")
    print("-" * 50)
    try:
        # Execute nvidia-smi command
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print(result.stdout)
        if "NVIDIA-SMI" in result.stdout:
            print("NVIDIA GPU detected via nvidia-smi.\n")
            return True
        else:
            print("nvidia-smi output doesn't contain expected header. Assuming no NVIDIA GPU.\n")
            return False
    except FileNotFoundError:
        print("nvidia-smi command not found. NVIDIA drivers might not be installed or not in PATH.")
        print("Assuming no NVIDIA GPU available for testing.\n")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error executing nvidia-smi: {e}")
        print(e.stderr)
        print("Assuming no NVIDIA GPU available for testing.\n")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while fetching GPU info: {e}")
        print("Assuming no NVIDIA GPU available for testing.\n")
        return False


def check_gpu_compatibility(min_capability=MIN_CUDA_COMPUTE_CAPABILITY):
    """
    Checks if PyTorch can use CUDA and if the detected GPU meets minimum compute capability.
    Returns True if a compatible GPU is usable by PyTorch, False otherwise.
    """
    if not torch.cuda.is_available():
        print("PyTorch CUDA available: False. GPU is not usable by PyTorch.\n")
        return False

    try:
        device_index = 0  # Assuming use of the first detected GPU
        device_name = torch.cuda.get_device_name(device_index)
        capability_str = torch.cuda.get_device_capability(device_index)
        capability = float(f"{capability_str[0]}.{capability_str[1]}")
        total_memory_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)

        print(f"PyTorch CUDA available: True")
        print(f"Detected GPU [Device {device_index}]: {device_name}")
        print(f"Compute Capability: {capability}")
        print(f"Total VRAM: {total_memory_gb:.2f} GB")

        if capability < min_capability:
            warnings.warn(
                f"Detected GPU ({device_name}) with compute capability {capability} "
                f"is below the minimum required ({min_capability}) for this PyTorch build. "
                f"GPU acceleration will likely fail or be unsupported.",
                UserWarning
            )
            print("GPU is too old for this PyTorch version. Treating as unavailable for benchmark.\n")
            return False
        else:
            print("GPU meets minimum compute capability requirements.\n")
            return True

    except Exception as e:
        print(f"Error checking GPU compatibility: {e}")
        print("Assuming GPU is not compatible.\n")
        return False


def load_model(model_name, device):
    """
    Loads the FastSAM model onto the specified device.
    Reports peak VRAM usage after loading if on CUDA device.
    """
    print(f"Loading model '{model_name}' onto {device}...")
    peak_vram_after_load_gb = 0.0

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats(0)  # Reset stats for device 0
        torch.cuda.empty_cache()  # Clear cache before loading

    try:
        model = YOLO(model_name)
        model.to(device)
        print(f"Model '{model_name}' loaded successfully on {device}.")

        # Perform a dummy inference for potential JIT compilation or setup
        # Use a reasonable input size for FastSAM
        dummy_input = torch.zeros(1, 3, 640, 640).to(device)
        _ = model.predict(dummy_input, verbose=False)
        if device == 'cuda':
            torch.cuda.synchronize(0)  # Wait for operations to finish
            peak_vram_after_load_bytes = torch.cuda.max_memory_allocated(0)
            peak_vram_after_load_gb = peak_vram_after_load_bytes / (1024 ** 3)
            print(f"Peak VRAM after model load & warm-up on {device}: {peak_vram_after_load_gb:.3f} GB")
        print(f"Model warm-up on {device} complete.")

        return model, peak_vram_after_load_gb  # Return model and peak VRAM

    except Exception as e:
        print(f"Error loading or warming up model {model_name} on {device}: {e}")
        return None, peak_vram_after_load_gb


def segment_image(model, image_path, device):
    """
    Performs segmentation on a single image and returns the time taken.
    """
    try:
        start_time = time.perf_counter()
        results = model.predict(source=image_path, device=device, verbose=False)
        # Ensure CUDA operations finish for accurate timing if on GPU
        if device == 'cuda':
            torch.cuda.synchronize(0)
        end_time = time.perf_counter()

        return end_time - start_time
    except Exception as e:
        print(f"Error segmenting image {image_path} on {device}: {e}")
        return None


def run_benchmark(model_name, image_paths, device_name):
    """
    Runs the segmentation benchmark for all images on the specified device.
    Returns timings, total time, and peak VRAM usage during the run (if CUDA).
    """
    peak_vram_during_run_gb = 0.0
    if not image_paths:
        print(f"No images found to benchmark on {device_name}.")
        return [], 0.0, peak_vram_during_run_gb

    # Reset peak memory stats before starting the benchmark run for this device
    if device_name == 'cuda':
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.empty_cache()

    model, peak_vram_after_load_gb = load_model(model_name, device_name)
    if model is None:
        return [], 0.0, peak_vram_after_load_gb  # Return VRAM after failed load attempt

    # Update peak VRAM based on loading/warmup phase
    peak_vram_during_run_gb = peak_vram_after_load_gb

    print(f"\n--- Starting Benchmark Processing on {device_name.upper()} ---")

    timings = []
    total_time_for_device = 0.0

    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i + 1}/{len(image_paths)}: {os.path.basename(img_path)} on {device_name}...")
        duration = segment_image(model, img_path, device_name)
        if duration is not None:
            timings.append(duration)
            total_time_for_device += duration
            print(f"Time taken: {duration:.4f} seconds")
            # Update peak VRAM after each image processing if on CUDA
            if device_name == 'cuda':
                current_peak_bytes = torch.cuda.max_memory_allocated(0)
                current_peak_gb = current_peak_bytes / (1024 ** 3)
                peak_vram_during_run_gb = max(peak_vram_during_run_gb, current_peak_gb)
                # Optional: Print per-image peak if needed
                # print(f"  Peak VRAM after this image: {current_peak_gb:.3f} GB")
        else:
            print(f"Skipping image {os.path.basename(img_path)} due to error.")

    if timings:
        avg_time = sum(timings) / len(timings)
        print(f"Average time per image on {device_name.upper()}: {avg_time:.4f} seconds")
    else:
        print(f"No images were successfully processed on {device_name.upper()}.")

    print(f"Total time for all images on {device_name.upper()}: {total_time_for_device:.4f} seconds")
    # Report final peak VRAM for the entire run on this device
    if device_name == 'cuda':
        final_peak_bytes = torch.cuda.max_memory_allocated(0)
        peak_vram_during_run_gb = final_peak_bytes / (1024 ** 3)  # Get final peak accurately
        print(f"Peak VRAM during entire {device_name.upper()} run: {peak_vram_during_run_gb:.3f} GB")

    print(f"--- Benchmark on {device_name.upper()} Finished ---")
    return timings, total_time_for_device, peak_vram_during_run_gb


# --- Main Execution ---
if __name__ == "__main__":
    # Check physical GPU presence first
    has_nvidia_gpu = get_gpu_info()

    # Check if PyTorch can use a *compatible* CUDA GPU
    gpu_usable_by_pytorch = check_gpu_compatibility()

    # Collect image paths
    image_paths = []
    for ext in SUPPORTED_EXTENSIONS:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))

    if not image_paths:
        print(f"No images found in the '{IMAGE_DIR}' directory with extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
        print("Please add some images to the 'images' folder and try again.")
        exit()

    print(f"Found {len(image_paths)} images for testing in '{IMAGE_DIR}'.\n")

    # --- CPU Benchmark ---
    # VRAM usage is not applicable/tracked easily for CPU like this
    cpu_timings, cpu_total_time, _ = run_benchmark(FASTSAM_MODEL_NAME, image_paths, "cpu")

    # --- GPU Benchmark ---
    gpu_timings = []
    gpu_total_time = 0.0
    gpu_peak_vram = 0.0
    if gpu_usable_by_pytorch:
        gpu_timings, gpu_total_time, gpu_peak_vram = run_benchmark(FASTSAM_MODEL_NAME, image_paths, "cuda")
    else:
        print("\nSkipping GPU benchmark as a compatible CUDA GPU is not available to PyTorch.")

    # --- Results Summary ---
    print("\n" + "=" * 50)
    print("Benchmark Results Summary")
    print("=" * 50)

    if cpu_timings:
        avg_cpu_time = sum(cpu_timings) / len(cpu_timings)
        print(f"CPU Average Time per Image: {avg_cpu_time:.4f} seconds")
        print(f"CPU Total Time for {len(cpu_timings)} images: {cpu_total_time:.4f} seconds")
    else:
        print("CPU benchmark did not produce any results.")

    if gpu_timings:
        avg_gpu_time = sum(gpu_timings) / len(gpu_timings)
        print(f"GPU Average Time per Image: {avg_gpu_time:.4f} seconds")
        print(f"GPU Total Time for {len(gpu_timings)} images: {gpu_total_time:.4f} seconds")
        print(f"GPU Peak VRAM Usage during run: {gpu_peak_vram:.3f} GB")  # Display peak VRAM
        if cpu_timings and avg_gpu_time > 0:  # Ensure no division by zero
            speedup = avg_cpu_time / avg_gpu_time
            print(f"\nGPU Speedup over CPU (based on average time): {speedup:.2f}x")
    elif gpu_usable_by_pytorch:
        print("GPU benchmark attempted but did not produce any results.")
    else:
        print("GPU benchmark was skipped due to incompatibility or unavailability.")

    print("=" * 50)
    print("Benchmark Complete.")
