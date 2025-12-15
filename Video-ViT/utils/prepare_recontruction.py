import os
import sys
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import subprocess
import random
import gc
import torch.multiprocessing as mp  # Added for multiprocessing
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================

# SETTINGS TO FIX OOM
VAE_CHUNK_SIZE = 8     

target_w = 640
target_h = 360
# Defined purely for print/logic usage if needed, though mostly handled by target_w/h logic now
MAX_RESOLUTION = 640 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_ROOT = os.path.join(PROJECT_ROOT, "dataset", "original", "300_clip_640x360_panda70m")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "dataset", "CustomGen", "reconstructed_8_frame_VAE")

PYRAMID_FLOW_REPO_PATH = os.path.join(SCRIPT_DIR, "Pyramid-Flow")
VAE_CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "pyramid-flow-sd3", "causal_video_vae")

# Set allocator config
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ==============================================================================
# 1. VAE WRAPPER (Optimized)
# ==============================================================================
class PyramidFlowVAEWrapper(nn.Module):
    def __init__(self, device='cuda', use_fp16=True):
        super().__init__()
        self.device = device
        self.use_fp16 = use_fp16

        if PYRAMID_FLOW_REPO_PATH not in sys.path:
            sys.path.insert(0, PYRAMID_FLOW_REPO_PATH)

        try:
            from video_vae import CausalVideoVAE
        except ImportError as e:
            print(f"[CRITICAL] Import failed: {e}")
            sys.exit(1)

        print(f"[{device}] Loading VAE weights...")
        self.vae = CausalVideoVAE.from_pretrained(VAE_CHECKPOINT_PATH)
        # compile can sometimes cause issues in multiprocessing spawn if not handled carefully, 
        # but usually fine. If it hangs, comment out the compile line.
        self.vae = torch.compile(self.vae, mode="max-autotune")

        if use_fp16:
            self.vae = self.vae.to(dtype=torch.bfloat16)

        self.vae.to(device)
        self.vae.eval()
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        x_in = (x * 2.0) - 1.0
        if self.use_fp16: 
            x_in = x_in.to(dtype=torch.bfloat16, device=self.device)
        else:
            x_in = x_in.to(self.device)
        
        chunk_size = VAE_CHUNK_SIZE 
        decoded_chunks = []
        
        gc.collect()
        torch.cuda.empty_cache()
        
        for i in range(0, T, chunk_size):
            end_idx = min(i + chunk_size, T)
            x_chunk = x_in[:, :, i:end_idx, :, :]
            actual_frames = x_chunk.shape[2]
            
            # Padding Logic
            pad_t = 0
            if actual_frames < chunk_size and actual_frames > 1:
                pad_t = chunk_size - actual_frames
                last_frame = x_chunk[:, :, -1:, :, :]
                padding = last_frame.repeat(1, 1, pad_t, 1, 1)
                x_chunk = torch.cat([x_chunk, padding], dim=2)

            with torch.no_grad():
                posterior = self.vae.encode(x_chunk)
                if hasattr(posterior, 'latent_dist'):
                    latents = posterior.latent_dist.sample()
                elif hasattr(posterior, 'sample'):
                    latents = posterior.sample()
                else:
                    latents = posterior
                
                decoded_chunk = self.vae.decode(latents)
                if hasattr(decoded_chunk, 'sample'):
                    decoded_chunk = decoded_chunk.sample
            
            # Remove padding
            if pad_t > 0:
                decoded_chunk = decoded_chunk[:, :, :actual_frames, :, :]

            # --- OPTIMIZATION START ---
            decoded_chunk = (decoded_chunk + 1.0) / 2.0
            decoded_chunk = torch.clamp(decoded_chunk, 0.0, 1.0)
            
            # Move to CPU immediately
            decoded_chunks.append(decoded_chunk.to('cpu').float()) 
            # --- OPTIMIZATION END ---

            del x_chunk, posterior, latents, decoded_chunk
            torch.cuda.empty_cache()
            
        x_out = torch.cat(decoded_chunks, dim=2)
        
        # Interpolate if needed
        _, _, T_out, H_out, W_out = x_out.shape
        if T_out != T or H_out != H or W_out != W:
            x_out = F.interpolate(
                x_out, size=(T, H, W), mode='trilinear', align_corners=False
            )
        
        return x_out 

# ==============================================================================
# 2. COMPRESSION
# ==============================================================================
def strict_compression_simulation(video_tensor, temp_path="temp_rec.mp4"):
    B, T, C, H, W = video_tensor.shape
    
    video_np = video_tensor.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
    video_np = (video_np * 255).astype(np.uint8)

    # Make temp filename unique per process/thread to avoid collision
    unique_suffix = f"_{random.randint(0, 999999)}"
    intermediate_path = f"raw_temp{unique_suffix}.avi"
    unique_temp_path = temp_path.replace(".mp4", f"{unique_suffix}.mp4")

    out = cv2.VideoWriter(intermediate_path, cv2.VideoWriter_fourcc(*'FFV1'), 30.0, (W, H))
    if not out.isOpened():
         out = cv2.VideoWriter(intermediate_path, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W, H))
         
    for frame in video_np:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

    crf_value = random.randint(16, 30)
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-v', 'error',
        '-i', intermediate_path,
        '-vf', 'format=yuv420p',       
        '-c:v', 'libx264', 
        '-profile:v', 'main',           
        '-level', '3.1',                
        '-crf', str(crf_value), 
        unique_temp_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True)
    except subprocess.CalledProcessError:
        pass

    if not os.path.exists(unique_temp_path):
        if os.path.exists(intermediate_path): os.remove(intermediate_path)
        return video_tensor 
        
    cap = cv2.VideoCapture(unique_temp_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[:2] != (H, W):
            frame = cv2.resize(frame, (W, H))
        frames.append(frame)
    cap.release()

    if os.path.exists(intermediate_path): os.remove(intermediate_path)
    if os.path.exists(unique_temp_path): os.remove(unique_temp_path)

    if len(frames) == 0: return video_tensor 
    while len(frames) < T: frames.append(frames[-1])
    frames = frames[:T]
    
    processed_np = np.stack(frames)
    tensor = torch.from_numpy(processed_np).float() / 255.0
    return tensor.permute(0, 3, 1, 2).unsqueeze(0)

# ==============================================================================
# 3. WORKER FUNCTION
# ==============================================================================
def worker_process(rank, video_folders):
    """
    rank: The GPU ID (0 or 1)
    video_folders: List of folder paths to process on this GPU
    """
    device = f'cuda:{rank}'
    print(f"--- Worker {rank} started on {device} with {len(video_folders)} tasks ---")
    
    try:
        vae = PyramidFlowVAEWrapper(device=device, use_fp16=True)
    except Exception as e:
        print(f"[Worker {rank}] Failed to initialize VAE: {e}")
        return

    # Use 'position' to stack progress bars cleanly in terminal
    pbar = tqdm(video_folders, desc=f"GPU {rank}", position=rank, leave=True)
    
    for video_folder in pbar:
        folder_name = os.path.basename(video_folder)
        
        # Paths
        if "real" in folder_name:
            recon_name = folder_name.replace("real", "")
        else:
            recon_name = f"{folder_name}"
        output_dir = os.path.join(OUTPUT_ROOT, recon_name)

        frame_paths = sorted(glob.glob(os.path.join(video_folder, "*.png")))
        if not frame_paths: continue

        # Resume Logic
        num_input_frames = len(frame_paths)
        if os.path.exists(output_dir):
            existing_frames = glob.glob(os.path.join(output_dir, "*.png"))
            if len(existing_frames) >= num_input_frames - 1:
                continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize variables to None
        video_tensor = None
        recon = None
        compressed = None
        video_np = None
        frames = []

        try:
            # 1. Load Frames
            for p in frame_paths:
                img = Image.open(p).convert("RGB")
                w, h = img.size
                
                if w != target_w or h != target_h:
                    img = img.resize((target_w, target_h), Image.BILINEAR)

                frames.append(np.array(img)) 
            
            if not frames: continue
            
            video_np = np.stack(frames)
            # Create tensor and move to specific GPU device
            video_tensor = torch.from_numpy(video_np).float() / 255.0
            video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0).to(device)
            
            # 2. Inference
            recon = vae(video_tensor) 
            recon = recon.permute(0, 2, 1, 3, 4)
            
            # 3. Compression Simulation
            # Note: Compression happens on CPU, but we clear GPU cache first
            torch.cuda.empty_cache() 
            compressed = strict_compression_simulation(recon)
            
            # 4. Save
            compressed_np = (compressed.squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype(np.uint8)
            
            for i, frame in enumerate(compressed_np):
                if i < len(frame_paths):
                    fname = os.path.basename(frame_paths[i])
                    Image.fromarray(frame).save(os.path.join(output_dir, fname))
                    
        except RuntimeError as e:
            if "out of memory" in str(e):
                pbar.write(f"\n[GPU {rank} FAIL] OOM on {folder_name}.")
            else:
                pbar.write(f"\n[GPU {rank} Error] {folder_name}: {e}")
        except Exception as e:
            pbar.write(f"\n[GPU {rank} Error] {folder_name}: {e}")

        # --- MEMORY CLEANUP ---
        finally:
            if video_tensor is not None: del video_tensor
            if recon is not None: del recon
            if compressed is not None: del compressed
            if video_np is not None: del video_np
            del frames
            gc.collect()
            torch.cuda.empty_cache()

# ==============================================================================
# 4. MAIN LAUNCHER
# ==============================================================================
if __name__ == "__main__":
    # Required for PyTorch multiprocessing with CUDA
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    print(f"--- CONFIG ---")
    print(f"Chunk Size:      {VAE_CHUNK_SIZE}")
    print(f"Input Data:      {INPUT_ROOT}")
    print(f"----------------")

    if not os.path.exists(INPUT_ROOT):
        print(f"Error: Input directory not found: {INPUT_ROOT}")
        sys.exit(1)

    all_video_folders = sorted(glob.glob(os.path.join(INPUT_ROOT, "*")))
    total_videos = len(all_video_folders)
    print(f"Found {total_videos} videos total.")
    
    # Split the dataset into 2 chunks
    # We use slicing [::2] and [1::2] to distribute load more evenly 
    # (in case videos are sorted by length, this prevents one GPU getting all short ones)
    subset_0 = all_video_folders[0::2]
    subset_1 = all_video_folders[1::2]
    
    print(f"GPU 0 will process {len(subset_0)} videos")
    print(f"GPU 1 will process {len(subset_1)} videos")
    
    # Create processes
    p0 = mp.Process(target=worker_process, args=(0, subset_0))
    p1 = mp.Process(target=worker_process, args=(1, subset_1))
    
    p0.start()
    p1.start()
    
    p0.join()
    p1.join()
    
    print("All processes finished.")