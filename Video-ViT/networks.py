import sys
sys.path.append('ViT_panda')
from ViT_panda.get_features import get_extractor as get_extractor_panda

import torch.nn as nn
import torch
import torch.nn.functional as F

def compute_restrav_features(x):
    """
    Computes Perceptual Straightening features (RestraV).
    
    This function analyzes the temporal trajectory of the feature representation.
    DeepFake videos often exhibit unstable, jittery latent trajectories compared to real videos.
    
    Metrics:
    1. Straightness: Ratio of Euclidean end-to-end distance to path length. Low values indicate meandering.
    2. Stepwise Distance: Statistics of frame-to-frame feature displacement (speed stability).
    3. Curvature/Angles: Statistics of directional changes between consecutive steps (angular stability).
    
    x: [B, T, D]
    Returns: [B, 5]
    """
    B, T, D = x.shape
    # FORCE FLOAT32 for precision
    x = x.float() 
    if T < 2:
        return torch.zeros(B, 5, device=x.device, dtype=x.dtype)

    v = x[:, 1:] - x[:, :-1] # [B, T-1, D]
    
    # Check if frames are identical
    v_abs_mean = v.abs().mean()
    if v_abs_mean < 1e-5:
        print(f"[WARNING] Frame differences are near zero! Mean Diff: {v_abs_mean.item()}")
    
    # Robust Norm:
    # We add a small epsilon (1e-6) to avoid division by zero 
    # when the feature vector is close to zero
    dist = torch.sqrt(torch.sum(v ** 2, dim=2) + 1e-6) # [B, T-1]
    
    # Path length
    path_len = dist.sum(dim=1) # [B]
    
    # End-to-End distance
    diff_e2e = x[:, -1] - x[:, 0]
    e2e = torch.sqrt(torch.sum(diff_e2e ** 2, dim=1) + 1e-6) # [B]
    
    # Straightness Index
    straightness = e2e / (path_len + 1e-6)
    
    # Stepwise statistics
    mean_step = dist.mean(dim=1)
    
    var_step = ((dist - mean_step.unsqueeze(1)) ** 2).mean(dim=1)
    std_step = torch.sqrt(var_step + 1e-6)
    
    # Curvature (Angles)
    if T < 3:
        mean_ang = torch.zeros(B, device=x.device, dtype=x.dtype)
        std_ang = torch.zeros(B, device=x.device, dtype=x.dtype)
    else:
        v1 = v[:, :-1]
        v2 = v[:, 1:]

        # Cosine similarity
        v1_norm = torch.sqrt(torch.sum(v1 ** 2, dim=2) + 1e-6)
        v2_norm = torch.sqrt(torch.sum(v2 ** 2, dim=2) + 1e-6)
        dot_product = torch.sum(v1 * v2, dim=2)
        
        cos_sim = dot_product / (v1_norm * v2_norm + 1e-6)
        
        # Cosine Distance:
        dists = 1.0 - cos_sim
        mean_ang = dists.mean(dim=1)
        
        # Robust Std
        var_ang = ((dists - mean_ang.unsqueeze(1)) ** 2).mean(dim=1)
        std_ang = torch.sqrt(var_ang + 1e-6)
        
    out = torch.stack([straightness, mean_step, std_step, mean_ang, std_ang], dim=1) # [B, 5]
    
    return out

class ScoresLayer(nn.Module):
    """
    Prototype-based Classification Layer.
    
    Instead of a standard hyperplane (linear layer), this layer learns "centers" (prototypes)
    and computes a distance-based similarity score. Often used in anomaly detection 
    or open-set recognition tasks to better model the manifold of "Real" vs "Fake" data.
    """
    def __init__(self, input_dim, num_centers):
        super().__init__()
        self.input_dim = input_dim
        self.num_centers = num_centers
        self.centers = nn.Parameter(torch.zeros(num_centers, input_dim), requires_grad=True)
        self.logsigmas = nn.Parameter(torch.zeros(num_centers), requires_grad=True)

    def forward(self, x):
        batch_size = x.size(0)
        out = x.view(batch_size, self.input_dim, 1, 1) # [batch, C, 1, 1]

        centers = self.centers[None, :, :, None, None]  # [1, K, C, 1, 1]
        diff = out.unsqueeze(1) - centers  # [batch, K, C, 1, 1]

        sum_diff = torch.sum(diff, dim=2)  # [batch, K, 1, 1]
        sign = torch.sign(sum_diff)

        squared_diff = torch.sum(diff ** 2, dim=2)  # [batch, K, 1, 1]

        logsigmas = nn.functional.relu(self.logsigmas)
        denominator = 2 * torch.exp(2 * logsigmas)
        part1 = (sign * squared_diff) / denominator.view(1, -1, 1, 1)

        part2 = self.input_dim * logsigmas
        part2 = part2.view(1, -1, 1, 1)

        scores = part1 + part2
        output = scores.sum(dim=(1, 2, 3)).view(-1, 1)  # [batch, 1]

        return output
    
def get_network(settings):
    name = settings.model

    if name == 'vit_panda':
        class VideoClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                # 32 -> 8: Dimensionality reduction for efficiency
                self.fc1 = nn.Linear(32, 8 ) # 32 x 4096 > 8 x 4096
                self.dropout = nn.Dropout(p=settings.dropout)
                self.fc2 = nn.Linear(4096, 512) # 8 x 4096 > 8 x 512
                
                # Restrav modification
                self.enable_restrav = getattr(settings, 'enable_restrav', False)
                fc3_in = 4096
                if self.enable_restrav:
                     fc3_in += 5 # 5 features from restrav
                     self.restrav_bn = nn.BatchNorm1d(5)
                     
                self.fc3 = nn.Linear(fc3_in, 1) # 8 x 512 > 1 (Original) or 4096+5 > 1
                if settings.prototype:
                    self.proto = ScoresLayer(input_dim=self.fc3.out_features, num_centers=1)

            def forward(self, x, frame_feats=None):
                if self.enable_restrav:
                    if frame_feats is None:
                        raise ValueError("enable_restrav is True, but frame_feats were not provided to forward()")
                    
                    restrav_feats = compute_restrav_features(frame_feats)
                    # Cast back to model dtype (fp16/bf16) for BN and concatenation
                    restrav_feats = restrav_feats.to(dtype=x.dtype) 
                    restrav_feats = self.restrav_bn(restrav_feats)
                    
                x = self.fc1(x.permute(0,2,1)).permute(0,2,1)
                x = self.dropout(x)
                x = self.fc2(x).flatten(1)
                x = self.dropout(x)
                
                if self.enable_restrav:
                    x = torch.cat((x, restrav_feats), dim=1)

                x = self.fc3(x)
                if settings.prototype:
                    x = self.proto(x)
                return x

        model = VideoClassifier()

        for param in model.parameters():
            param.requires_grad = True

        extractor = get_extractor_panda(settings)
    
    else:
        raise NotImplementedError('model not recognized')

    return model, extractor


