
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys

sys.path.append(os.getcwd())

from parser import get_parser
from networks import get_network
from dataset import create_dataloader
from ensemble_networks import EnsembleModel
import argparse

def train(model, train_loader, val_loader, optimizer, criterion, device, epochs=10, save_dir='ensemble_checkpoints'):
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    print(f"[INFO] Starting training for {epochs} epochs...")
    print(f"[INFO] Saving checkpoints to {save_dir}")
    
    for epoch in range(epochs):
        model.train()
        # Keep backbones frozen
        if hasattr(model, 'model1'): model.model1.eval()
        if hasattr(model, 'model2'): model.model2.eval()
        
        train_loss = 0.0
        train_steps = 0
        
        # Train Loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", mininterval=1.0)
        for data, label, _ in pbar:
            data = data.to(device)
            label = label.to(device).float()
            
            optimizer.zero_grad()
            output = model(data).squeeze(1)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix(loss=f"{train_loss/train_steps:.4f}")
            
        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0.0
            
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for data, label, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", mininterval=1.0):
                data = data.to(device)
                label = label.to(device).float()
                
                output = model(data).squeeze(1)
                loss = criterion(output, label)
                
                val_loss += loss.item()
                val_steps += 1
                
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
        print(f"Epoch {epoch+1} finished. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save Best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(save_dir, 'best_ensemble.pt')
            torch.save(model.state_dict(), save_path)
            print(f"  [INFO] New Best Model Saved: {save_path}")

if __name__ == "__main__":
    parser = get_parser() 
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Initialize Ensemble
    model = EnsembleModel(args.model1, args.model2, args, device=device)
    model.to(device)
    
    # Optimizer only train fusion parameters
    optimizer = optim.Adam(model.fusion.parameters(), lr=args.ensemble_lr, weight_decay=args.weight_decay)
    
    # Loss
    if args.focal_loss:
             print(f"[INFO] Using Focal Loss (gamma={args.focal_gamma})")
        criterion = FocalLoss(gamma=args.focal_gamma)
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    # Dataloaders
         print("[INFO] Creating Dataloaders...")
    train_loader = create_dataloader(args, subdir='train', is_train=True)
    val_loader = create_dataloader(args, subdir='val', is_train=False)
    
    train(model, train_loader, val_loader, optimizer, criterion, device, epochs=args.num_epochs, save_dir=args.ensemble_dir)
