import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.getcwd())
from networks import get_network
from parser import get_parser

class UnifiedModel(nn.Module):
    """
    Wrapper to align with train.py structure.
    This allows the 'accelerate' library to manage the model as a single entity.
    """
    def __init__(self, extractor_model, classifier):
        super().__init__()
        self.extractor_model = extractor_model
        self.classifier = classifier
        
    def forward(self, x):
        outputs = self.extractor_model.inference(x)
        
        frame_feats = None
        if isinstance(outputs, tuple):
             features = outputs[0]
             if len(outputs) > 1:
                 frame_feats = outputs[1]
                 # Match dtype
                 frame_feats = frame_feats.to(self.classifier.fc1.weight.dtype)
        else:
             features = outputs
             
        features = features.to(self.classifier.fc1.weight.dtype)
        
        # Classifier Forward
        output = self.classifier(features, frame_feats=frame_feats)
        return output

class EnsembleModel(nn.Module):
    """
    Late-Fusion Ensemble Model.
    
    Loads two pre-trained backbones (frozen) and learns a lightweight fusion head (MLP)
    on top of their concatenated intermediate features. 
    Uses forward hooks to extract features from specific layers without modifying the backbone code.
    """
    def __init__(self, model1_dir, model2_dir, settings, device='cuda'):
        super().__init__()
        
        print(f"[INFO] Loading Model 1 from {model1_dir}...")
        self.model1 = self._load_single_model(model1_dir, settings, device, use_lora=getattr(settings, 'use_lora', False))
        
        print(f"[INFO] Loading Model 2 from {model2_dir}...")
        self.model2 = self._load_single_model(model2_dir, settings, device, use_lora=getattr(settings, 'use_lora', False))
        
        # Freeze both models
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False
            
        # Hook storage
        self.feat1 = None
        self.feat2 = None
        
        # Register Hooks to intercept features before final layer
        # which is already rich in semantic/fake-detection information.
        self.model1.classifier.fc2.register_forward_hook(self._hook1)
        self.model2.classifier.fc2.register_forward_hook(self._hook2)
        
        # Fusion MLP
        # Input: 4096 * 2 = 8192
        self.fusion = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1)
        )
        
    def _load_single_model(self, model_dir, base_settings, device, use_lora=False):
        import copy
        from peft import PeftModel
        settings = copy.deepcopy(base_settings)
        
        # Heuristic: Adjust settings based on folder name
        dirname = os.path.basename(model_dir.rstrip('/'))
        
        if 'restrav' in dirname:
            settings.enable_restrav = True
            print(f"  [INFO] [Auto-Config] {dirname}: Enabled Restrav")
        else:
            settings.enable_restrav = False
            
        if 'swm' in dirname:
            settings.enable_swm = True
            print(f"  [INFO] [Auto-Config] {dirname}: Enabled SWM")
        else:
            settings.enable_swm = False

        # Create base model structure
        classifier, extractor = get_network(settings)
        
        # Load Classifier Weights
        weight_path = os.path.join(model_dir, 'models', 'best.pt')
        lora_path = os.path.join(model_dir, 'models', 'best_lora')
                
        if os.path.exists(weight_path):
            print(f"  [INFO] Loading weights from {weight_path}")
            state_dict = torch.load(weight_path, map_location='cpu')
            classifier.load_state_dict(state_dict)
        else:
            print(f"  [WARNING] No weights found in {model_dir}/models/. Using random init.")

        # Load LoRA if requested
        if use_lora:
            if os.path.exists(lora_path):
                 print(f"  Loading LoRA adapters from {lora_path}")
                 if hasattr(extractor.model, 'visual_encoder'):
                     extractor.model.visual_encoder = PeftModel.from_pretrained(extractor.model.visual_encoder, lora_path)
                     extractor.model.visual_encoder.to(device)
                 else:
                     print(f"  [WARNING] use_lora=True but extractor.model has no visual_encoder.")
            else:
                 print(f"  [WARNING] use_lora=True but path {lora_path} does not exist. Skipping LoRA.")

        model = UnifiedModel(extractor.model, classifier)
        model.to(device)
        model.eval()
        return model

    def _hook1(self, module, input, output):
        # Flatten(1) converts [B, C, H, W] or [B, D] to [B, Features] 
        # to ensure compatibility with the MLP input
        self.feat1 = output.flatten(1) 

    def _hook2(self, module, input, output):
        self.feat2 = output.flatten(1)

    def forward(self, x):
        # Reset features
        self.feat1 = None
        self.feat2 = None
        
        # Run forward passes, to activate hooks
        with torch.no_grad():
            _ = self.model1(x)
            _ = self.model2(x)
            
        if self.feat1 is None or self.feat2 is None:
            raise RuntimeError("Hooks did not capture features. Check model architecture.")
            
        # Concatenate
        combined = torch.cat([self.feat1, self.feat2], dim=1)
        
        # Fusion
        logits = self.fusion(combined)
        return logits
