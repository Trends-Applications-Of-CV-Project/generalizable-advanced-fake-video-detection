import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, default_collate
from torchvision.transforms import v2 as Tv2
from torchvision.transforms.v2 import functional as F_v2

from utils.swm import HaarWavelet2D, apply_waverep

class Video_dataset(Dataset):
    def __init__(self, opt, root):
        super().__init__()

        # --- Configurations ---
        if opt.model == 'vit_panda':
            self.size = 224
            # CLIP stats
            self.norm_mean = [0.48145466, 0.4578275, 0.40821073]
            self.norm_std = [0.26862954, 0.26130258, 0.27577711]
            self.scale = True

        self.samples = []
        
        # --- Augmentation Pipeline ---
        # Separate "Geometric" and "Pixel" transforms to insert SWM in the correct spotù.
        
        if opt.split == 'train':
            self.augment = True
            # These apply after the SWM
            self.geometric_transform = Tv2.Compose([
                Tv2.RandomResizedCrop(
                    self.size, 
                    scale=(0.85, 1.0), 
                    ratio=(0.9, 1.1),
                    antialias=True
                ),
                Tv2.RandomHorizontalFlip(p=0.5)
            ])
            self.final_norm = Tv2.Compose([
                Tv2.Normalize(mean=self.norm_mean, std=self.norm_std)
            ])
        else:
            self.augment = False
            self.geometric_transform = Tv2.Resize((self.size, self.size), antialias=True)
            self.final_norm = Tv2.Compose([
                Tv2.Normalize(mean=self.norm_mean, std=self.norm_std)
            ])

        self.enable_swm = opt.enable_swm
        self.wavelet_processor = HaarWavelet2D() 
        self.waverep_prob = getattr(opt, 'swm_percentage', 0.1)
        
        # --- Lookup paths ---
        self.real_root_lookup = os.path.join(opt.data_root, 'clips_original', 'dataset')
        self.recon_root_lookup = os.path.join(opt.data_root, 'SWM_Gen', 'dataset_gen')

        self.dilation = opt.dilation
        self.inverse = opt.invert_labels
        self.n_frames = opt.n_frames
        self.num_batches = opt.num_batches

        # --- Data Loading ---
        split_dir = opt.split_path if opt.split_path else 'split'
        if opt.split is not None:
            split_file = os.path.join(split_dir, f'{opt.split}.txt')
            with open(split_file) as f:
                videos = [line.strip() for line in f.readlines()]
        else:
            videos = []

        self.videos = []
        
        for subdir in os.listdir(root):
            if opt.split is not None and subdir not in videos:
                continue
            
            limit = 105 if 'clips_original' in root else 1000
            
            # Check if we are also dealing with reconstructed videos
            is_recon = 'SWM_Gen' in root
            
            target = (1.0 - self.inverse) if ('clips_original' in root or 'original_size' in root) else (1.0 * self.inverse)
            
            # Store: [Paths, Target, Is_Recon_Flag]
            self.samples.append([self._get_image_paths(f'{root}/{subdir}')[:limit], target, is_recon])
            self.videos.append(f'{root}/{subdir}')

            # Class Balancing for Training
            if opt.split == 'train' and 'clips_original' in root:
                self.samples.append([self._get_image_paths(f'{root}/{subdir}')[:limit], target, False])
                self.videos.append(f'{root}/{subdir}')

    # --- Helper Methods ---
    def _get_image_paths(self, directory):
        """Helper to find images with various extensions"""
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        images = []
        for ext in extensions:
            images.extend(glob.glob(os.path.join(directory, ext)))
        return sorted(images)

    def __len__(self):
        return len(self.samples)

    def _get_aligned_swm_pair(self, video_id):
        # Extract folder name (ID) from full path
        video_name = os.path.basename(video_id)
        
        if video_name.endswith('_reconn'):
            real_name = video_name.replace('_reconn', '_real')
        else:
            real_name = video_name
            
        real_dir = os.path.join(self.real_root_lookup, real_name)
        recon_dir = os.path.join(self.recon_root_lookup, video_name)

        if os.path.exists(real_dir) and os.path.exists(recon_dir):
            real_paths = self._get_image_paths(real_dir)
            recon_paths = self._get_image_paths(recon_dir)
            
            min_len = min(len(real_paths), len(recon_paths))
            if min_len < self.n_frames:
                print("[WARNING] Sequence length is less than n_frames. Skipping.")
                return None, None 

            return real_paths[:min_len], recon_paths[:min_len]
        else:
            print(f"[ERROR] Failed to find paths:\n  Real: {real_dir} (Exists: {os.path.exists(real_dir)})\n  Recon: {recon_dir} (Exists: {os.path.exists(recon_dir)})")
            raise Exception(f"No file present")

    def _pad_to_divisor(self, t, divisor=8):
        # Padding is required for Wavelet Transforms which perform downsampling.
        # We ensure dimensions are divisible by 8 to avoid shape mismatch during reconstruction.
        h, w = t.shape[-2:]
        pad_h = (divisor - h % divisor) % divisor
        pad_w = (divisor - w % divisor) % divisor
        if pad_h > 0 or pad_w > 0:
            return torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode='reflect')
        return t

    def load_clip_tensor(self, paths, indices):
        """Loads a sequence of images directly into a [T, C, H, W] float32 tensor."""
        tensors = []
        for idx in indices:
            safe_idx = min(idx, len(paths) - 1)
            img = Image.open(paths[safe_idx]).convert("RGB")
            img_tensor = F_v2.to_dtype(F_v2.to_image(img), torch.float32, scale=True)
            tensors.append(img_tensor)
        return torch.stack(tensors)
    
    def __getitem__(self, index):
        paths, target, is_recon = self.samples[index]
        video_path_str = self.videos[index]

        if not paths:
            raise ValueError(f"No image files found in {video_path_str}")

        in_tens_arr_batch = []

        for i in range(self.num_batches):
            # Temporal Sampling
            if self.augment:
                start_frame = random.randint(0, max(0, len(paths) - (self.n_frames - 1) * self.dilation - 1))
            else:
                viable = list(range(0, max(1, len(paths) - (self.n_frames - 1) * self.dilation - 1)))
                offset = len(viable) // self.num_batches
                start_frame = viable[(i * offset) % len(viable)]
        
            frame_indices = [min(start_frame + k * self.dilation, len(paths)-1) for k in range(self.n_frames)]

            # Load main clip as float32 [T, C, H, W]
            current_clip = self.load_clip_tensor(paths, frame_indices)

            # SWM Injection
            # Applied before geometric transforms to preserve frequency bands
            if self.augment and is_recon and self.wavelet_processor is not None:
                if random.random() < self.waverep_prob:
                    try:
                        real_p, recon_p = self._get_aligned_swm_pair(video_path_str)
                        
                        if real_p and recon_p and frame_indices[-1] < len(recon_p):
                            swm_real = self.load_clip_tensor(real_p, frame_indices)
                            swm_recon = self.load_clip_tensor(recon_p, frame_indices)

                            # Resize Real to match Recon if needed (Bilinear)
                            if swm_real.shape[-2:] != swm_recon.shape[-2:]:
                                swm_real = F_v2.resize(
                                    swm_real, 
                                    size=swm_recon.shape[-2:], 
                                    interpolation=Tv2.InterpolationMode.BILINEAR, 
                                    antialias=True
                                )

                            # Pad for Wavelet
                            real_in = self._pad_to_divisor(swm_real)
                            fake_in = self._pad_to_divisor(swm_recon)
                            
                            # Probabilistic mode selection
                            rnd = random.random()
                            mode = 'full' if rnd < 0.6 else ('horizontal' if rnd < 0.8 else 'base')
                            
                            # Apply Wavelet Replacement
                            # Output remains Float32 [T, C, H, W]
                            aug = apply_waverep(real_in, fake_in, self.wavelet_processor, mode=mode)

                            # Crop back to original dims
                            h_orig, w_orig = swm_recon.shape[-2], swm_recon.shape[-1]
                            current_clip = aug[..., :h_orig, :w_orig]

                    except Exception as e:
                        print(f"[WARNING] SWM Error in {video_path_str}: {e}")
                        # Fallback to original current_clip
                        
                    
            # Geometric Augmentations
            # Apply same crop/flip to all frames in the clip
            # [T, C, H, W] is treated as a batch by Tv2
            current_clip = self.geometric_transform(current_clip)

            # Pixel Augmentations (Blur/JPEG)
            if self.augment and random.random() < 0.3:
                if random.random() < 0.5:
                    sigma = random.uniform(0.1, 2.0)
                    current_clip = F_v2.gaussian_blur(current_clip, kernel_size=15, sigma=sigma)
                else:
                    # JPEG compression simulation
                    clip_u8 = (current_clip * 255).clamp(0, 255).to(torch.uint8)
                    quality = random.randint(65, 95)
                    clip_u8 = F_v2.jpeg(clip_u8, quality)
                    current_clip = clip_u8.to(torch.float32) / 255.0

            if self.augment and random.random() < 0.2:
                 # Video-specific blur
                 current_clip = F_v2.gaussian_blur(current_clip, kernel_size=(3, 3), sigma=random.uniform(0.1, 1.0))

            # Final Normalization
            current_clip = self.final_norm(current_clip)

            final_tensor = current_clip.permute(1, 0, 2, 3)

            if self.num_batches == 1:
                return final_tensor, target, video_path_str
            else:
                in_tens_arr_batch.append((final_tensor, target, video_path_str))

        return in_tens_arr_batch

def create_dataloader(opt, subdir='.', is_train=True):
    if subdir == "train":
        techniques = ["pyramidflow_whole", "SWM_Gen"]
        datasets = ["dataset_gen"]
        opt.split = 'train'
        opt.batch_size = max(1, opt.batch_size // opt.num_batches)

    elif subdir == "val":
        techniques = ["pyramidflow_whole", "SWM_Gen"]
        datasets = ["dataset_gen"]
        opt.split = 'val'
        opt.batch_size = max(1, opt.batch_size // opt.num_batches)
    
    elif subdir == "test":
        techniques = ["Cogvideo", "Hunyuan", "LUMA_AI", "RunawayML", "SORA", "Veo"]
        datasets = ["dataset"]
        opt.split = 'test'
        opt.batch_size = max(1, opt.batch_size // opt.num_batches)

    dset_lst = []
    for technique in techniques:
        print(technique)
        for dataset in datasets:
            root = os.path.join(opt.data_root, technique, dataset)
            dset = Video_dataset(opt, root)
            dset_lst.append(dset)

    print('clip_original')
    for dataset in datasets:
        root = os.path.join(opt.data_root, 'clips_original', 'dataset')
        dset = Video_dataset(opt, root)
        dset_lst.append(dset)

    dataset = torch.utils.data.ConcatDataset(dset_lst)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        num_workers=int(opt.num_threads),
        shuffle = True if is_train else False,
        collate_fn=(lambda x: default_collate([p for v in x for p in v])) if opt.num_batches > 1 else None,
    )
    return data_loader