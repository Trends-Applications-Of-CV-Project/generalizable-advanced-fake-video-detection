import torch
import copy
import time
import torch.nn as nn
import os
import csv
import sys
from tqdm import tqdm
from parser import get_parser
from networks import get_network
from dataset import create_dataloader
import torch.distributed.tensor

def test_unseen():
    # Setup settings
    parser = get_parser()
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running on {device}")
    print(f"[INFO] Settings: {opt}")
    # Load Network
    model, extractor = get_network(opt)
    model.to(device)
    
    if extractor:
        # Check if extractor needs to be moved or set to eval
        extractor.to(device)
        extractor.eval()

    # 3. Load Checkpoint
    checkpoint_path = os.path.join('train', opt.name, 'models', 'best.pt')
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at {checkpoint_path}")
        return

    print(f"[INFO] Loading weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if len(missing_keys) > 0:
        print(f"[WARNING] Missing keys during loading: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"[WARNING] Unexpected keys in checkpoint (ignored): {unexpected_keys}")

    model.eval()

    # Data Loader (Test Split)
    test_dataloader = create_dataloader(opt, subdir='test', is_train=False)
    
    # Ensemble Logic
    model2 = None
    extractor2 = None
    
    if opt.enable_ensemble:
        print(f"\n[Ensemble] Loading second model: {opt.ensemble_name} (Restrav: {opt.ensemble_restrav})...")
        opt2 = copy.deepcopy(opt)
        opt2.name = opt.ensemble_name
        opt2.enable_restrav = opt.ensemble_restrav
        
        # Load Network 2
        model2, extractor2 = get_network(opt2)
        model2.to(device)
        model2.eval()
        
        if extractor2:
             extractor2.to(device)
             extractor2.eval()
             
        # Load Checkpoint 2
        checkpoint_path2 = os.path.join('train', opt2.name, 'models', 'best.pt')
        if not os.path.exists(checkpoint_path2):
             print(f"[ERROR] Ensemble checkpoint not found at {checkpoint_path2}")
             return
        
        print(f"[INFO] [Ensemble] Loading weights 2 from {checkpoint_path2}")
        state_dict2 = torch.load(checkpoint_path2, map_location=device)
        missing_keys2, unexpected_keys2 = model2.load_state_dict(state_dict2, strict=False)
        
        if len(missing_keys2) > 0:
             print(f"[WARNING] [Ensemble] Missing keys: {missing_keys2}")
             if opt2.enable_restrav and any('restrav_bn' in k for k in missing_keys2):
                  print("[Ensemble] Legacy Restrav Detected. Disabling BN for Model 2.")
                  model2.restrav_bn = nn.Identity()
                  
        print("[INFO] [Ensemble] Model 2 loaded successfully.\n")

    # Evaluation Loop
    check_accuracy(test_dataloader, extractor, model, opt, device, extractor2=extractor2, model2=model2)


def check_accuracy(loader, extractor, model, settings, device, extractor2=None, model2=None):
    model.eval()
    if model2:
        model2.eval()

    label_array = torch.empty(0, dtype=torch.int64, device=device)
    pred_array = torch.empty(0, dtype=torch.int64, device=device)
    prob_array = torch.empty(0, dtype=torch.float32, device=device) 
    
    video_metrics = {}

    with torch.no_grad():
        with tqdm(loader, unit='batch', mininterval=0.5) as tbatch:
            tbatch.set_description(f'Testing Unseen')
            for (data, label, videos) in tbatch:
                data = data.to(device)
                label = label.to(device)
                
                # Measure inference time
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                start_time = time.time()
                
                # Feature Extraction
                extractor_out = extractor.get_features(data)
                
                features = extractor_out
                frame_feats = None
                if isinstance(extractor_out, tuple):
                    features = extractor_out[0]
                    if len(extractor_out) > 1:
                        frame_feats = extractor_out[1]

                # Model Inference
                scores = model(features, frame_feats=frame_feats).squeeze(1)
                
                # Predictions
                probs = torch.sigmoid(scores)
                
                # Ensemble Inference
                if model2 is not None:
                     # Feature Extraction 2
                     extractor_out2 = extractor2.get_features(data)
                     features2 = extractor_out2
                     frame_feats2 = None
                     if isinstance(extractor_out2, tuple):
                         features2 = extractor_out2[0]
                         if len(extractor_out2) > 1:
                             frame_feats2 = extractor_out2[1]
                             
                     scores2 = model2(features2, frame_feats=frame_feats2).squeeze(1)
                     probs2 = torch.sigmoid(scores2)
                     
                     # Ensemble Strategy: Probability Averaging
                     # Average the sigmoid probabilities from both models before thresholding
                     probs = (probs + probs2) / 2.0
                
                pred = torch.round(probs).int()
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                batch_time = end_time - start_time
                time_per_sample = batch_time / len(videos)

                # Collect per-video stats
                # We aggregate clip-level predictions to form a video-level decision.
                for i, video_name in enumerate(videos):
                    if video_name not in video_metrics:
                        # Determine generator
                        gen_name = 'Unknown'
                        if 'original_size' in video_name:
                            gen_name = 'Real'
                        else:
                            for known_gen in ["Cogvideo", "SORA", "LUMA_AI", "Hunyuan", "RunawayML", "veo"]:
                                if known_gen in video_name:
                                    gen_name = known_gen
                                    break
                                    
                        video_metrics[video_name] = {'probs': [], 'labels': [], 'inference_time': 0.0, 'generator': gen_name}
                    video_metrics[video_name]['probs'].append(probs[i].item())
                    video_metrics[video_name]['labels'].append(label[i].item())
                    video_metrics[video_name]['inference_time'] += time_per_sample

                label_array = torch.cat((label_array, label))
                pred_array = torch.cat((pred_array, pred))
                prob_array = torch.cat((prob_array, probs))

                # Calculate current batch metrics (Clip-level)
                zerosamples = torch.count_nonzero(label_array==0) * (2 if settings.invert_labels else 1)
                onesamples = torch.count_nonzero(label_array==1) * (1 if settings.invert_labels else 2) 
                totalsamples = zerosamples + onesamples
                
                zerocorrect = torch.count_nonzero(pred_array[label_array==0]==0) * (2 if settings.invert_labels else 1) 
                onecorrect = torch.count_nonzero(pred_array[label_array==1]==1) * (1 if settings.invert_labels else 2) 
                totalcorrect = zerocorrect + onecorrect
                
                # Display accuracy
                acc_fake_disp = float(zerocorrect/zerosamples) if zerosamples > 0 else 0.0
                acc_real_disp = float(onecorrect/onesamples) if onesamples > 0 else 0.0
                acc_total_disp = float(totalcorrect/totalsamples) if totalsamples > 0 else 0.0
                
                if settings.invert_labels:
                    acc_fake_disp, acc_real_disp = acc_real_disp, acc_fake_disp
                
                tbatch.set_postfix(acc_tot=acc_total_disp*100, acc_fake=acc_fake_disp*100, acc_real=acc_real_disp*100)

    # Save Results to CSV
    csv_dir = os.path.join('train', settings.name, 'results')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, 'val.csv')
    print(f"Saving results to {csv_path}...")
    
    sorted_videos = sorted(video_metrics.keys())
    
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Video Name', 'Generator', 'True Label', 'Pred Label', 'Avg Score'])
            
            for vid in sorted_videos:
                probs = video_metrics[vid]['probs']
                labels = video_metrics[vid]['labels']
                
                # Aggregate
                avg_prob = sum(probs) / len(probs)
                pred_label_int = 1 if avg_prob >= 0.5 else 0
                true_label_int = labels[0]
                
                # Determine string labels
                if settings.invert_labels:
                    lbl_str_true = "Fake" if true_label_int == 1 else "Real"
                    lbl_str_pred = "Fake" if pred_label_int == 1 else "Real"
                else:
                    lbl_str_true = "Real" if true_label_int == 1 else "Fake"
                    lbl_str_pred = "Real" if pred_label_int == 1 else "Fake"
                
                vid_short = vid.split('/')[-1]
                # Write to CSV
                writer.writerow([vid_short, video_metrics[vid]['generator'], lbl_str_true, lbl_str_pred, f"{avg_prob:.4f}"])
                
        print("CSV saved successfully.")
    except Exception as e:
        print(f"Error saving CSV: {e}")

if __name__ == '__main__':
    test_unseen()
