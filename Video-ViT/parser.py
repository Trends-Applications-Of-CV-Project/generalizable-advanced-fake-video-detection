import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    # --- Run Configuration ---
    parser.add_argument("--name", type=str, default="test", help="run name")

    # --- Model Configuration ---
    parser.add_argument("--device", type=str, default="cuda:1", help="cuda device to use")
    parser.add_argument("--model", type=str, default="vit_panda", help="architecture name")
    parser.add_argument("--freeze", action='store_true', help="Freeze all layers except the last one")
    parser.add_argument("--prototype", action='store_true', help="Use prototype layer")
    parser.add_argument("--invert_labels", action='store_true', help="Inverted labels for prototype")
    parser.add_argument("--enable_swm", action='store_true', help="Enable Seeing What Matters (SWM) augmentation")
    # --- Fine-Tuning / Unfreeze Options ---
    parser.add_argument("--unfreeze_qformer", action='store_true', help="Unfreeze Q-Former and projection layers")
    parser.add_argument("--unfreeze_visual_encoder", action='store_true', help="Unfreeze Visual Encoder layers")
    parser.add_argument("--unfreeze_patch_embed", action='store_true', help="Unfreeze patch embeddings of Visual Encoder")
    
    # --- LoRA Configuration ---
    parser.add_argument("--lora_visual_encoder", action='store_true', help="Enable LoRA for Visual Encoder")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.02, help="LoRA dropout")

    # --- Feature Engineering (Restrav) ---
    parser.add_argument("--enable_restrav", action='store_true', help="Enable Perceptual Straightening (Restrav) features")

    # --- Ensemble Configuration ---
    parser.add_argument("--enable_ensemble", action='store_true', help="Enable Ensemble Inference (2 Models)")
    parser.add_argument("--ensemble_name", type=str, help="Name of the second model run (for Ensemble)")
    parser.add_argument("--ensemble_restrav", action='store_true', help="Enable Restrav for the second model")
    
    # --- Ensemble Training Configuration ---
    parser.add_argument('--model1', type=str, help='Path to first model directory')
    parser.add_argument('--model2', type=str, help='Path to second model directory')
    parser.add_argument('--ensemble_lr', type=float, default=1e-4, help='Learning rate for ensemble head')
    parser.add_argument('--ensemble_dir', type=str, default='ensemble_results', help='Directory to save results')
    parser.add_argument('--use_lora', action='store_true', help='Attempt to load LoRA adapters from model directories')


    # --- Training Hyperparameters ---
    parser.add_argument("--num_epochs", type=int, default=200, help="# of epoches at starting learning rate")

    parser.add_argument("--lr", type=float, default=0.0001, help="initial learning rate")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma for Focal Loss")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="initial learning rate for backbone")
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Number of warmup epochs")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability for classifier")
    # Percentage of SWM application
    parser.add_argument("--swm_percentage", type=float, default=0.1, help="Percentage of SWM")

    parser.add_argument("--lr_decay_epochs",type=int, default=5, help="Number of epochs without loss reduction before lowering the learning rate by 10x")
    parser.add_argument("--lr_min",type=float, default=1e-7, help="minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience")

    # --- Dataset Configuration ---
    parser.add_argument("--split_path", type=str, help="Path to split files")
    parser.add_argument("--data_root", type=str, help="Path to dataset")
    parser.add_argument("--data_root_commercial", type=str, help="Path to dataset for commercial tools")

    parser.add_argument("--batch_size", type=int, default=32, help='Dataloader batch size')
    parser.add_argument("--num_threads", type=int, default=24, help='# threads for loading data')

    parser.add_argument("--n_frames", type=int, default=8, help='Number of frames for the ViT')
    parser.add_argument("--dilation", type=int, default=1, help='Dilation of frames sampling')
    parser.add_argument("--num_batches", type=int, default=5, help='Number of batches of frames for each video, BEWARE, this options effectively multiplies the number of samples in the dataset')
    
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")

    parser.add_argument("--focal_loss", action='store_true', help="Use Focal Loss")

    return parser