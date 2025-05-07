# main.py
import os
import glob
import time
import subprocess
import torch
from ultralytics import FastSAM, YOLO
from PIL import Image  # Pillow for image loading, though ultralytics also handles it

# --- Configuration ---
IMAGE_DIR = "images"  # Directory containing your test images
SUPPORTED_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]  # Supported image file extensions
FASTSAM_MODEL_NAME = 'FastSAM-s.pt'  # Or 'FastSAM-x.pt' for a larger model


# --- Helper Functions ---

def get_gpu_info():
    """
    Retrieves and prints NVIDIA GPU information using nvidia-smi.
    Returns True if a GPU is detected, False otherwise.
    """
    print("-" * 50)
    print("GPU Information:")
    print("-" * 50)
    try:
        # Execute nvidia-smi command
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print(result.stdout)
        if "CUDA Version" in result.stdout:
            print("NVIDIA GPU with CUDA support detected.\n")
            return True
        else:
            print("NVIDIA GPU detected, but CUDA version string not found in nvidia-smi output.\n")
            return False  # Or True, depending on how strict you want to be
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


def load_model(model_name, device):
    """
    Loads the FastSAM model onto the specified device.
    Ultralytics' YOLO class can load FastSAM models directly.
    """
    print(f"Loading model '{model_name}' onto {device}...")
    try:
        model = YOLO(model_name)  # Or FastSAM(model_name) if using FastSAM class directly
        model.to(device)  # Ensure model is on the correct device
        print(f"Model '{model_name}' loaded successfully on {device}.")
        return model
    except Exception as e:
        print(f"Error loading model {model_name} on {device}: {e}")
        return None


def segment_image(model, image_path, device):
    """
    Performs segmentation on a single image and returns the time taken.
    """
    try:
        # Ultralytics models can take image paths directly
        # For more control or if using Pillow explicitly:
        # img = Image.open(image_path).convert("RGB")

        start_time = time.perf_counter()
        # The predict method handles moving data to the model's device implicitly
        # if the input is a path, PIL image, or numpy array.
        # Forcing device here is more for the model loading itself.
        results = model.predict(source=image_path, device=device, verbose=False)
        end_time = time.perf_counter()

        # You can process results here if needed, e.g., count masks, save images
        # For speed testing, the predict call itself is the main work.
        # num_masks = len(results[0].masks) if results[0].masks is not None else 0
        # print(f"Found {num_masks} masks in {os.path.basename(image_path)}")

        return end_time - start_time
    except Exception as e:
        print(f"Error segmenting image {image_path} on {device}: {e}")
        return None


def run_benchmark(model_name, image_paths, device_name):
    """
    Runs the segmentation benchmark for all images on the specified device.
    """
    if not image_paths:
        print(f"No images found to benchmark on {device_name}.")
        return [], 0.0

    model = load_model(model_name, device_name)
    if model is None:
        return [], 0.0

    print(f"\n--- Starting Benchmark on {device_name.upper()} ---")

    # Optional: Warm-up run (can help stabilize timings, especially for GPU)
    # print(f"Performing a warm-up run on {device_name}...")
    # segment_image(model, image_paths[0], device_name)
    # print("Warm-up complete.")

    timings = []
    total_time_for_device = 0.0

    for i, img_path in enumerate(image_paths):
        print(f"Processing image {i + 1}/{len(image_paths)}: {os.path.basename(img_path)} on {device_name}...")
        duration = segment_image(model, img_path, device_name)
        if duration is not None:
            timings.append(duration)
            total_time_for_device += duration
            print(f"Time taken: {duration:.4f} seconds")
        else:
            print(f"Skipping image {os.path.basename(img_path)} due to error.")

    if timings:
        avg_time = sum(timings) / len(timings)
        print(f"Average time per image on {device_name.upper()}: {avg_time:.4f} seconds")
    else:
        print(f"No images were successfully processed on {device_name.upper()}.")

    print(f"Total time for all images on {device_name.upper()}: {total_time_for_device:.4f} seconds")
    print(f"--- Benchmark on {device_name.upper()} Finished ---")
    return timings, total_time_for_device


# --- Main Execution ---
if __name__ == "__main__":
    gpu_available_for_torch = torch.cuda.is_available()
    has_nvidia_gpu = get_gpu_info()  # Prints nvidia-smi output

    if not gpu_available_for_torch and has_nvidia_gpu:
        print("\nWARNING: nvidia-smi detects a GPU, but PyTorch cannot find a CUDA-enabled device.")
        print("Please ensure your PyTorch installation is compatible with your CUDA drivers.")
        print(
            "You can check PyTorch's website for correct installation commands: https://pytorch.org/get-started/locally/")
    elif gpu_available_for_torch:
        print(f"PyTorch CUDA available: True (Using: {torch.cuda.get_device_name(0)})\n")
    else:
        print("PyTorch CUDA available: False. GPU testing will be skipped if no compatible GPU is found.\n")

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
    cpu_timings, cpu_total_time = run_benchmark(FASTSAM_MODEL_NAME, image_paths, "cpu")

    # --- GPU Benchmark ---
    gpu_timings = []
    gpu_total_time = 0.0
    if gpu_available_for_torch:  # Only run GPU benchmark if PyTorch sees CUDA
        gpu_timings, gpu_total_time = run_benchmark(FASTSAM_MODEL_NAME, image_paths, "cuda")
    else:
        print("\nSkipping GPU benchmark as CUDA is not available to PyTorch.")

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
        if cpu_timings:
            speedup = avg_cpu_time / avg_gpu_time
            print(f"\nGPU Speedup over CPU (based on average time): {speedup:.2f}x")
    elif gpu_available_for_torch:  # If we tried to run GPU but got no results
        print("GPU benchmark did not produce any results despite CUDA being available.")
    else:
        print("GPU benchmark was skipped.")

    print("=" * 50)
    print("Benchmark Complete.")
