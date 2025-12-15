import os
import random
import argparse
from pathlib import Path

def read_split_file(path):
    if not path.exists():
        return []
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def main():
    parser = argparse.ArgumentParser(description="Split dataset into train, val, and test.")
    parser.add_argument('--regenerate', action='store_true', help="Regenerate all splits from scratch, overwriting existing files.")
    args = parser.parse_args()

    # Define paths
    dataset_root = Path('dataset')
    
    # List of directories to scan for video IDs
    # Standard directories (split into train/val/test)
    standard_source_dirs = [
        dataset_root / 'clips_original' / 'dataset',
        dataset_root / 'pyramidflow_whole' / 'dataset_gen',
        dataset_root / 'SWM_Gen' / 'dataset_gen'
    ]

    # Generators to be put ONLY in test set
    test_only_source_dirs = [
        dataset_root / 'Cogvideo' / 'dataset_gen',
        dataset_root / 'Hunyuan' / 'dataset_gen',
        dataset_root / 'LUMA_AI' / 'dataset_gen',
        dataset_root / 'RunawayML' / 'dataset_gen',
        dataset_root / 'SORA' / 'dataset_gen',
        dataset_root / 'veo3' / 'dataset_gen'
    ]
    
    output_split_dir = dataset_root / 'split'
    train_file = output_split_dir / 'train.txt'
    val_file = output_split_dir / 'val.txt'
    test_file = output_split_dir / 'test.txt'

    # 1. Gather Standard Video IDs
    standard_video_ids = set()
    print("Scanning standard source directories...")
    for videos_dir in standard_source_dirs:
        if not videos_dir.exists():
            print(f"Warning: Directory {videos_dir} does not exist. Skipping.")
            continue
            
        print(f"  Scanning: {videos_dir}")
        ids_in_dir = [d.name for d in videos_dir.iterdir() if d.is_dir()]
        count = len(ids_in_dir)
        print(f"    Found {count} IDs.")
        standard_video_ids.update(ids_in_dir)

    # 2. Gather Test-Only Video IDs
    test_only_video_ids = set()
    print("\nScanning test-only source directories...")
    for videos_dir in test_only_source_dirs:
        if not videos_dir.exists():
            print(f"Warning: Directory {videos_dir} does not exist. Skipping.")
            continue
            
        print(f"  Scanning: {videos_dir}")
        ids_in_dir = [d.name for d in videos_dir.iterdir() if d.is_dir()]
        count = len(ids_in_dir)
        print(f"    Found {count} IDs.")
        test_only_video_ids.update(ids_in_dir)

    total_videos_found = len(standard_video_ids) + len(test_only_video_ids)
    print(f"\nTotal videos found on disk: {total_videos_found}")
    
    if total_videos_found == 0:
        print("No videos found in any directory. Exiting.")
        return

    # Check if we need to regenerate or if we can do incremental update
    files_exist = train_file.exists() and val_file.exists() and test_file.exists()
    
    if args.regenerate or not files_exist:
        if args.regenerate:
            print("\nForce regeneration requested.")
        else:
            print("\nExisting split files not found. Creating new splits.")
        
        # --- FULL REGENERATION LOGIC ---
        standard_ids_list = sorted(list(standard_video_ids))
        test_only_ids_list = sorted(list(test_only_video_ids))
        
        # Shuffle standard IDs for random split
        random.shuffle(standard_ids_list)

        total_standard = len(standard_ids_list)

        # Percentage Splits on STANDARD videos only
        # 10% Test, 20% Val, Rest Train
        std_test_count = int(total_standard * 0.10)
        std_val_count = int(total_standard * 0.20)
        
        # Ensure nominal amounts if dataset is small
        if total_standard > 0:
            std_test_count = max(1, std_test_count)
            std_val_count = max(1, std_val_count)
            
        std_train_count = total_standard - std_val_count - std_test_count
        
        val_split = standard_ids_list[:std_val_count]
        standard_test_split = standard_ids_list[std_val_count:std_val_count+std_test_count]
        train_split = standard_ids_list[std_val_count+std_test_count:]

        # Final Test Split = Standard Test Split + All Test-Only Generators
        test_split = standard_test_split + test_only_ids_list
        
        # Create split directory if it doesn't exist
        output_split_dir.mkdir(parents=True, exist_ok=True)

        mode = 'w'
        print(f"Writing NEW splits to {output_split_dir}...")
        
    else:
        print("\nExisting splits found. Performing incremental update...")
        # --- INCREMENTAL UPDATE LOGIC ---
        
        # Read existing IDs
        existing_train = read_split_file(train_file)
        existing_val = read_split_file(val_file)
        existing_test = read_split_file(test_file)
        
        existing_ids = set(existing_train) | set(existing_val) | set(existing_test)
        print(f"  Existing IDs in splits: {len(existing_ids)}")
        
        # Identify NEW IDs
        new_standard_ids = list(standard_video_ids - existing_ids)
        new_test_only_ids = list(test_only_video_ids - existing_ids)
        
        print(f"  New Standard videos found: {len(new_standard_ids)}")
        print(f"  New Test-Only videos found: {len(new_test_only_ids)}")
        
        if len(new_standard_ids) == 0 and len(new_test_only_ids) == 0:
            print("No new videos to add.")
            return

        # Split NEW standard videos
        random.shuffle(new_standard_ids)
        total_new_std = len(new_standard_ids)
        
        new_std_test_count = int(total_new_std * 0.10)
        new_std_val_count = int(total_new_std * 0.20)
        
        # Only force nominal if we actually have some new videos, 
        # but for very small increments (<3 videos), the floor might be 0.
        # Let's trust the math: if 5 videos: 0.5->0, 1.0->1. 
        # If we really want to distribute small batches, we can.
        # But simple math is probably safer to avoid over-represented test/val in small batches.
        
        new_val_split = new_standard_ids[:new_std_val_count]
        new_std_test_split = new_standard_ids[new_std_val_count:new_std_val_count+new_std_test_count]
        new_train_split = new_standard_ids[new_std_val_count+new_std_test_count:]
        
        # Add new test-only videos to test split
        new_test_split = new_std_test_split + new_test_only_ids
        
        # Prepare data for writing (appending)
        train_split = new_train_split
        val_split = new_val_split
        test_split = new_test_split
        
        mode = 'a'
        print(f"Appending NEW videos to {output_split_dir}...")

    # Write (or Append) to files
    splits = {
        'train.txt': train_split,
        'val.txt': val_split,
        'test.txt': test_split
    }
    
    for filename, ids in splits.items():
        if not ids and mode == 'a':
            continue # Nothing to append to this file
            
        file_path = output_split_dir / filename
        with open(file_path, mode) as f:
            if mode == 'a' and ids: # Ensure newline before appending if not empty (though write usually needs \n managed)
                # Simple append row by row
                f.write('\n'.join(ids) + '\n')
            else:
                f.write('\n'.join(ids))
                
        print(f"  {filename}: {'Added' if mode == 'a' else 'Wrote'} {len(ids)} IDs")

    print("\n" + "="*50)
    print("DATASET SPLIT UPDATE COMPLETE")
    print("="*50)

if __name__ == "__main__":
    main()