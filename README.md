# Video-ViT: Fake Video Detection with Perceptual Straightening
> **Robust synthetic video detection using 3D Vision Transformers, Forensic-Oriented Augmentation, and Perceptual Straightening.**

---

## Abstract

The rapid advancement of diffusion-based generative models has necessitated robust methods for synthetic video detection. Existing detectors often overfit to specific generator artifacts, limiting their generalization to unseen architectures. This project enhances a **3D Vision Transformer (3D-ViT)** baseline by integrating forensic-oriented data augmentation (*Seeing What Matters*) and geometric consistency checks (*Perceptual Straightening*).

We construct a dataset using PyramidFlow and evaluate performance on a diverse benchmark including Sora, RunwayML, and Veo2. Our approach, culminating in a **Dual-Stream Ensemble** that integrates **Low-Rank Adaptation (LoRA)**, or **Focal Loss**, and geometric features, achieves a **34.64% improvement** in total accuracy over the baseline while maintaining high fidelity on real content. These results highlight the efficacy of targeting structural anomalies over low-level artifacts for generalized deepfake detection.

## Methodology

Our approach builds upon a strong baseline and introduces two key novelties:

### 0. Baseline - *Advance Fake Video Detection via Vision Transformers*
We utilize the **Video-ViT** architecture proposed by Waggoner et al., which leverages 3D Vision Transformers to capture spatiotemporal dependencies. This serves as our foundation, providing robust feature extraction capabilities for video-level classification.

### 1. Seeing What Matters (SWM) - *Frequency-Domain Augmentation*
SWM is a data augmentation technique that forces the model to focus on high-frequency artifacts rather than semantic content.
- It uses **Wavelet Transforms** to blend high-frequency details from reconstructed "fake" videos into real videos (and vice versa).
- This prevents the model from overfitting to the *subject* of the video (e.g., "pandas") and instead targets the *imperfections* inherent to generative models.

### 2. Perceptual Straightening (RestraV) - *Geometric Consistency*
RestraV relies on the observation that real videos have "smoother" temporal trajectories in the latent space compared to generated ones. 
- We compute **geometric features** (Curvature, Straightness, Stepwise Distance) from the frame-by-frame feature trajectory.
- These features are concatenated with the visual embedding to help the classifier distinguish stable real motion from jittery synthetic motion.

---

## Results

Our model demonstrates significant improvements in detection accuracy. Below are the performance metrics visualized:

| Total Accuracy | Fake Video Accuracy |
|:---:|:---:|
| <img src="total_accuracy.png" alt="Total Accuracy" width="300"/> | <img src="fake_accuracy.png" alt="Fake Accuracy" width="300"/> |

> **Note**: The graphs illustrate the performance gain of our method compared to baseline.

### Experiment Legend
1. **Baseline**
2. **SWM-Augmented**
3. **ReStraV-Augmented**
4. **SWM-Augmented + LoRA**
5. **Optimized SWM**
6. **Optimized ReStraV**
7. **Dual-Stream Ensemble** (Our Best Model)
---

## Environment Setup

### Prerequisites
Install dependencies using:

```bash
pip install -r requirements.txt
```
> **Note**: You can also use uv packages manager to install dependencies and run the code, there are all the dependencies in the `pyproject.toml` file.

### Data & Weights provided
- **Pre-trained Weights**: [Download Here](https://drive.google.com/file/d/1Gjp5LrgGJobcFi3AaXvLnzlY7IWXyaI5/view?usp=sharing). Place it in the `Video-ViT/checkpoints` directory.
- **Dataset**: [Download Here](https://drive.google.com/file/d/1nYTRxqZBEwIVkdDfTF8DFFqvqLewvWbH/view?usp=drive_link).
- **Massive Test Dataset**: [Download Here](https://drive.google.com/file/d/16Cm5wr6nBiJdQvL7q7BqOwdLBcEg8_qk/view?usp=sharing).

---

## Usage

The source code is located in the `Video-ViT` directory. Run commands from the root repository.

### Training
Use `train.py` to train the model. You can combine flags for SWM, LoRA, Restrav, etc.

**Basic Training:**
```bash
python train.py --name my_experiment --data_root /path/to/dataset --batch_size 8 --num_epochs 50
```

**Advanced Training (Single Stream Configuration):**
```bash
python train.py \
  --name best_model_run \
  --data_root /path/to/dataset \
  --batch_size 8 \
  --num_epochs 50 \
  --lora_visual_encoder \
  --enable_restrav \
  --enable_swm \
  --swm_percentage 0.5 \
  --focal_loss \
  --unfreeze_visual_encoder
```

### Ensemble Training
Train a lightweight fusion head to combine two pre-trained models.

```bash
python train_ensemble.py \
  --model1 ./train/run_name_1 \
  --model2 ./train/run_name_2 \
  --ensemble_dir ./ensemble_results \
  --num_epochs 10
```

### Evaluation
Evaluate on the unseen test set using `test_unseen.py`.

**Single Model:**
```bash
python test_unseen.py --name experiment_name --data_root /path/to/dataset
```

**Ensemble Evaluation:**
```bash
python test_unseen.py \
  --name run_name_1 \
  --enable_ensemble \
  --ensemble_name run_name_2 \
  --ensemble_restrav \
  --data_root /path/to/dataset
```

---

## Authors
- **Andrea Goldoni**
- **Lorenzo Arcangeli**
- **Marco Morandin**