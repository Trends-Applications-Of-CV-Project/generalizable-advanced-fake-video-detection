import os
import glob
import torch
import shutil
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from accelerate import Accelerator

from networks import get_network
from parser import get_parser
from dataset import create_dataloader

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.benchmark = False
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        # Not used on our case
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs are logits
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) 
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class UnifiedModel(nn.Module):
    """
    Adapter class to Unify the VideoLLAMA extraction with a standard Classifier.
    
    This wrapper allows the 'accelerate' library to see a single model entity, 
    managing device placement and mixed-precision casting for both the 
    large Feature Extractor (VideoLLAMA) and the lightweight Classifier.
    """
    def __init__(self, extractor_model, classifier, device):
        super().__init__()
        self.extractor_model = extractor_model
        self.classifier = classifier
        self.device = device
        
    def forward(self, x):
        outputs = self.extractor_model.inference(x)
        
        frame_feats = None
        if isinstance(outputs, tuple):
             features = outputs[0]
             if len(outputs) > 1:
                 frame_feats = outputs[1]
                 frame_feats = frame_feats.to(self.classifier.fc1.weight.dtype)
        else:
             features = outputs
             
        features = features.to(self.classifier.fc1.weight.dtype)
        
        output = self.classifier(features, frame_feats=frame_feats)
        return output

def check_accuracy(loader, model, settings, accelerator):
    model.eval()
    
    label_array = torch.empty(0, dtype=torch.int64, device=accelerator.device)
    pred_array = torch.empty(0, dtype=torch.int64, device=accelerator.device)
    
    with torch.no_grad():
        disable_tqdm = not accelerator.is_local_main_process
        with tqdm(loader, unit='batch', mininterval=0.5, disable=disable_tqdm) as tbatch:
            tbatch.set_description(f'Validation')
            for (data, label, _) in tbatch:
                # data is already on device
                scores = model(data).squeeze(1)
                pred = torch.round(torch.sigmoid(scores)).int()
                
                label_array = torch.cat((label_array, label))
                pred_array = torch.cat((pred_array, pred))
    
    if accelerator.is_main_process:
        pass # [DEBUG] Validation Loop Done.
    
    all_label_array = accelerator.gather(label_array)
    all_pred_array = accelerator.gather(pred_array)
    
    if accelerator.is_main_process:
        pass # [DEBUG] Gather Done. Computing metrics...
    
    zerosamples = torch.count_nonzero(all_label_array==0) * (2 if settings.invert_labels else 1)
    onesamples = torch.count_nonzero(all_label_array==1) * (1 if settings.invert_labels else 2) 
    totalsamples = zerosamples + onesamples

    zerocorrect = torch.count_nonzero(all_pred_array[all_label_array==0]==0) * (2 if settings.invert_labels else 1) 
    onecorrect = torch.count_nonzero(all_pred_array[all_label_array==1]==1) * (1 if settings.invert_labels else 2) 
    totalcorrect = zerocorrect + onecorrect

    if totalsamples == 0:
        totalaccuracy = 0.0
    else:
        totalaccuracy = float(totalcorrect/totalsamples)
        
    if zerosamples == 0:
        zeroaccuracy = 0.0
    else:
        zeroaccuracy = float(zerocorrect/zerosamples)
        
    if onesamples == 0:
        oneaccuracy = 0.0
    else:
        oneaccuracy = float(onecorrect/onesamples)

    # Logging
    if accelerator.is_main_process:
        print(f'[INFO] Accuracy: {totalaccuracy*100:.2f}% (Real: {zeroaccuracy*100:.2f}%, Fake: {oneaccuracy*100:.2f}%)')
        
    return totalaccuracy, oneaccuracy


def set_model_mode(model, settings):
    """
    Sets the model mode (train/eval) component-wise to avoid
    overwriting frozen batch normalization statistics in the backbone.
    """
    # Set everything to eval
    model.eval()
    
    # Train the classifier
    if hasattr(model, 'module'):
        # If wrapped by DDP/FSDP
        unified_ptr = model.module
    else:
        unified_ptr = model
        
    unified_ptr.classifier.train()
    
    # Handle Visual Encoder / Feature Extractor
    vid_model = unified_ptr.extractor_model
    
    # If using LoRA
    if settings.lora_visual_encoder:
        # Set .train() for LoRA, but freeze BN stats in backbone.
        if hasattr(vid_model, 'visual_encoder'):
            vid_model.visual_encoder.train() 
            
            if not settings.unfreeze_visual_encoder:
                for module in vid_model.visual_encoder.modules():
                    if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm3d):
                        module.eval()
                        
    # Handle Unfrozen Components
    if settings.unfreeze_qformer:
        if hasattr(vid_model, 'Qformer'):
             vid_model.Qformer.train()
        if hasattr(vid_model, 'query_tokens'):
             pass
        if hasattr(vid_model, 'video_Qformer'):
             vid_model.video_Qformer.train()
        if hasattr(vid_model, 'llama_proj'):
             vid_model.llama_proj.train()
             
    if settings.unfreeze_visual_encoder:
        # If full unfreeze, full train including BN
        if hasattr(vid_model, 'visual_encoder'):
            vid_model.visual_encoder.train()
        if hasattr(vid_model, 'ln_vision'):
            vid_model.ln_vision.train()
            
    if settings.unfreeze_patch_embed:
        # If only patch embed is unfrozen, put it in train, keep rest frozen
        if hasattr(vid_model.visual_encoder, 'patch_embed'):
            vid_model.visual_encoder.patch_embed.train()



def train(loader, val_dataloader, model, settings, accelerator, optimizer, criterion):
    best_accuracy = 0
    best_fake_accuracy = 0
    lr_decay_counter = 0
        
    for epoch in range(0, settings.num_epochs):
        set_model_mode(model, settings)
        
        # Warmup Logic
        if epoch < settings.warmup_epochs:
            warmup_factor = (epoch + 1) / settings.warmup_epochs
            for param_group in optimizer.param_groups:
                base_lr = settings.lr if param_group.get('name') == 'classifier' else settings.lr_backbone
                param_group['lr'] = base_lr * warmup_factor
            if accelerator.is_main_process:
                print(f"[Epoch {epoch}] Warmup: Learning rates set to {[pg['lr'] for pg in optimizer.param_groups]}")

        
        disable_tqdm = not accelerator.is_local_main_process
        with tqdm(loader, unit='batch', mininterval=0.5, disable=disable_tqdm) as tepoch:
            tepoch.set_description(f'Epoch {epoch}', refresh=False)
            for batch_idx, (data, label, _) in enumerate(tepoch):

                with accelerator.accumulate(model):
                    scores = model(data).squeeze(1) 

                    loss = criterion(scores, label).mean()
    
                    accelerator.backward(loss) 
                    optimizer.step()
                    optimizer.zero_grad()
                
                tepoch.set_postfix(loss=loss.item())

        if accelerator.is_main_process:
            print(f"[INFO] Starting Validation Epoch {epoch}")
        accuracy, fake_accuracy = check_accuracy(val_dataloader, model, settings, accelerator)
        if accelerator.is_main_process:
            print(f"[INFO] Validation Finished Epoch {epoch}. Accuracy: {accuracy:.4f} (Fake: {fake_accuracy:.4f})")
        
        # Save flags
        save_best = False
        save_best_fake = False

        # Check Global Accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_best = True
            lr_decay_counter = 0
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Check Fake Accuracy
        if fake_accuracy > best_fake_accuracy:
            best_fake_accuracy = fake_accuracy
            save_best_fake = True
            
        if save_best or save_best_fake:
            if accelerator.is_main_process:
                print(f"[INFO] Saving Model(s) - Best: {save_best}, Best Fake: {save_best_fake}")
            
            full_state_dict = accelerator.get_state_dict(model)
            
            classifier_state_dict = {
                k.replace('classifier.', ''): v 
                for k, v in full_state_dict.items() 
                if k.startswith('classifier.')
            }

            if accelerator.is_main_process:
                if save_best:
                    torch.save(classifier_state_dict, f'./train/{settings.name}/models/best.pt')
                    print(f'[INFO] New best model saved with accuracy {best_accuracy:.2f} \n')
                    
                    # Save LoRA adapters if enabled
                    if settings.lora_visual_encoder:
                        unwrapped_model = accelerator.unwrap_model(model)
                        vid_model = unwrapped_model.extractor_model
                        
                        if hasattr(vid_model, 'visual_encoder') and hasattr(vid_model.visual_encoder, 'save_pretrained'):
                            prefix = 'extractor_model.visual_encoder.'
                            lora_state_dict = {
                                k[len(prefix):]: v 
                                for k, v in full_state_dict.items() 
                                if k.startswith(prefix)
                            }
                            
                            vid_model.visual_encoder.save_pretrained(
                                f'./train/{settings.name}/models/best_lora',
                                state_dict=lora_state_dict
                            )
                            print(f'Saved LoRA adapters to ./train/{settings.name}/models/best_lora')
                
                if save_best_fake:
                    torch.save(classifier_state_dict, f'./train/{settings.name}/models/best_fake.pt')
                    print(f'[INFO] New best FAKE model saved with accuracy {fake_accuracy:.2f} \n')

                    # Save LoRA adapters for best fake
                    if settings.lora_visual_encoder:
                        unwrapped_model = accelerator.unwrap_model(model)
                        vid_model = unwrapped_model.extractor_model
                        
                        if hasattr(vid_model, 'visual_encoder') and hasattr(vid_model.visual_encoder, 'save_pretrained'):
                            prefix = 'extractor_model.visual_encoder.'
                            lora_state_dict = {
                                k[len(prefix):]: v 
                                for k, v in full_state_dict.items() 
                                if k.startswith(prefix)
                            }

                            vid_model.visual_encoder.save_pretrained(
                                f'./train/{settings.name}/models/best_fake_lora',
                                state_dict=lora_state_dict
                            )
                            print(f'Saved LoRA adapters to ./train/{settings.name}/models/best_fake_lora')

        # Early Stopping & Decay Logic, based on Global Accuracy
        if not save_best:
            epochs_no_improve += 1
            if accelerator.is_main_process:
                print(f"No improvement for {epochs_no_improve} epochs.")

            if epochs_no_improve >= settings.patience:
                if accelerator.is_main_process:
                    print(f"Early stopping triggered after {settings.patience} epochs without improvement.")
                break

            if settings.lr_decay_epochs > 0:
                lr_decay_counter += 1
                if lr_decay_counter == settings.lr_decay_epochs:
                    # Check if any group is above min lr
                    any_above_min = False
                    for param_group in optimizer.param_groups:
                        if param_group['lr'] > settings.lr_min:
                            param_group['lr'] *= 0.1
                            any_above_min = True
                    
                    if any_above_min:
                        if accelerator.is_main_process:
                            print('Learning rate decayed \n')
                        lr_decay_counter = 0
                    else:
                        if accelerator.is_main_process:
                            print('Learning rate already at minimum \n')
                        pass
        

if __name__ == "__main__":
    parser = get_parser()
    settings = parser.parse_args()
    
    # ccelerate Init
    # Enable mixed precision by default if not set, and handle accumulation
    accelerator = Accelerator(gradient_accumulation_steps=settings.accumulation_steps, mixed_precision='fp16')
    if accelerator.is_main_process:
        print(settings)
        # Check for SWM mismatch
        if 'swm' in settings.name.lower() and not settings.enable_swm:
            print("\n" + "#" * 60)
            print(" [WARNING] 'swm' found in run name but --enable_swm is NOT set!")
            print(" SWM Augmentation will be DISABLED.")
            print("#" * 60 + "\n")

    if accelerator.is_main_process:
        os.makedirs(f'./train/{settings.name}/models', exist_ok=True)
        for file in glob.glob(f'*.py'):
            shutil.copy(file, f'./train/{settings.name}/')
        
        with open(f'./train/{settings.name}/settings.txt', 'w') as f:
            f.write(str(settings))

    set_seed(42)

    train_dataloader = create_dataloader(settings, subdir='train', is_train=True)
    val_dataloader = create_dataloader(settings, subdir='val', is_train=False)


    model, extractor = get_network(settings)
    
    # Unfreeze Logic
    extra_params = []
    
    # LoRA for Visual Encoder
    if settings.lora_visual_encoder:
        if accelerator.is_main_process:
            print("Enabling LoRA for Visual Encoder...")
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=settings.lora_r,
            lora_alpha=settings.lora_alpha,
            target_modules=["qkv", "fc1", "fc2"], 
            lora_dropout=settings.lora_dropout,
            bias="none",
            modules_to_save=[], 
        )
        
        vid_model = extractor.model
        vid_model.visual_encoder = get_peft_model(vid_model.visual_encoder, lora_config)
        if accelerator.is_main_process:
            vid_model.visual_encoder.print_trainable_parameters()
        
        # Enable Gradient Checkpointing for memory efficiency with LoRA
        if hasattr(vid_model.visual_encoder, 'use_checkpoint'):
             vid_model.visual_encoder.use_checkpoint = True
             if accelerator.is_main_process:
                 print("Enabled Gradient Checkpointing for Visual Encoder (LoRA)")
        
        if accelerator.is_main_process:
             print("Enabling gradient checkpointing for LoRA...")
        
        # Enable on PeftModel wrapper
        if hasattr(vid_model.visual_encoder, "gradient_checkpointing_enable"):
            vid_model.visual_encoder.gradient_checkpointing_enable()
        
        # Check set model-specific flags
        base_model = getattr(vid_model.visual_encoder, "base_model", None)
        if base_model:
             inner_model = getattr(base_model, "model", None)
             if inner_model and hasattr(inner_model, "use_checkpoint"):
                 inner_model.use_checkpoint = True
                 if accelerator.is_main_process:
                     print("Enabled use_checkpoint on inner model")

        # Fallback
        if hasattr(vid_model.visual_encoder, 'use_checkpoint'):
             vid_model.visual_encoder.use_checkpoint = True
        else:
             vid_model.visual_encoder.gradient_checkpointing = True
        
    # Unfreeze Blocks
    if settings.unfreeze_qformer:
        if accelerator.is_main_process:
            print("Unfreezing Q-Former components...")
        vid_model = extractor.model
        
        for param in vid_model.Qformer.parameters():
            param.requires_grad = True
        
        # Restore original train method if it was disabled
        if "train" in vid_model.Qformer.__dict__:
            del vid_model.Qformer.train
            
        vid_model.Qformer.train()
        vid_model.query_tokens.requires_grad = True
        
        for param in vid_model.video_Qformer.parameters():
            param.requires_grad = True
        vid_model.video_Qformer.train()
        vid_model.video_query_tokens.requires_grad = True
        
        for param in vid_model.video_frame_position_embedding.parameters():
            param.requires_grad = True
            
        for param in vid_model.llama_proj.parameters():
            param.requires_grad = True
        vid_model.llama_proj.train()

    if settings.unfreeze_visual_encoder:
        if settings.lora_visual_encoder:
            if accelerator.is_main_process:
                print("[WARNING] Both LoRA and Full Unfreeze set. Full Unfreeze overrides.")
        
        if accelerator.is_main_process:
            print("Unfreezing Visual Encoder (Full)...")
        vid_model = extractor.model
        
        # Enable Gradient Checkpointing for memory efficiency
        if hasattr(vid_model.visual_encoder, 'use_checkpoint'):
             vid_model.visual_encoder.use_checkpoint = True
             if accelerator.is_main_process:
                 print("Enabled Gradient Checkpointing for Visual Encoder")

        for param in vid_model.visual_encoder.parameters():
            param.requires_grad = True
            
        # Restore original train method if it was disabled
        if "train" in vid_model.visual_encoder.__dict__:
            del vid_model.visual_encoder.train

        vid_model.visual_encoder.train()
        
        for param in vid_model.ln_vision.parameters():
            param.requires_grad = True
            
        # Restore original train method if it was disabled
        if "train" in vid_model.ln_vision.__dict__:
            del vid_model.ln_vision.train

        vid_model.ln_vision.train()

    # Unfreeze Patch Embeddings (Can be combined with LoRA or Full Unfreeze)
    if settings.unfreeze_patch_embed:
        if accelerator.is_main_process:
            print("Unfreezing Visual Encoder Patch Embeddings...")
        vid_model = extractor.model
                
        if hasattr(vid_model.visual_encoder, 'patch_embed'):
            for param in vid_model.visual_encoder.patch_embed.parameters():
                param.requires_grad = True
            
            vid_model.visual_encoder.patch_embed.train()
        else:
             if accelerator.is_main_process:
                 print("[WARNING] Could not find patch_embed in visual_encoder. Skipping.")

    
    unified_model = UnifiedModel(extractor.model, model, accelerator.device)

   
    extractor_trainable_params = [p for p in unified_model.extractor_model.parameters() if p.requires_grad]
    classifier_params = [p for p in unified_model.classifier.parameters() if p.requires_grad]
    
    # Separate parameter groups
    param_groups = [
        {'params': classifier_params, 'lr': settings.lr, 'name': 'classifier'},
        {'params': extractor_trainable_params, 'lr': settings.lr_backbone, 'name': 'backbone'}
    ]
    
    optimizer = optim.Adam(param_groups, weight_decay=settings.weight_decay)
    
    if settings.name == "source":
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    elif settings.focal_loss:
        criterion = FocalLoss(gamma=settings.focal_gamma, reduction='none')
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        unified_model, optimizer, train_dataloader, val_dataloader
    )

    train(train_dataloader, val_dataloader, model, settings, accelerator, optimizer, criterion)
