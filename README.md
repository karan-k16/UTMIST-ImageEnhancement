# 🖼️ UTMIST Image Super-Resolution Pipeline

This repo contains a lightweight, production-oriented pipeline from UTMIST (Fall 2024–Winter 2025) that upscales 480p → 2K using a residual CNN with PixelShuffle. It includes automated data curation, patch-based training, mixed precision (AMP), tiled inference with overlap for large images, and a PSNR/SSIM benchmarking harness.

---

## 📌 Project Highlights

- 🔍 **Model**: Residual CNN (head/body/tail) with PixelShuffle upsampler  
- ⚙️ **GPU Optimization**: Mixed precision (AMP), torch.no_grad() inference, tiled 512×512 windows with ~32px overlap and feathered blending (seam-free)  
- 📊 **Evaluation Metrics**: PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index)  
- 🧪 **Training Approach**: Patch-based training with Adam, validation checkpoints, and random LR generation from HR images  
- 🎯 **Goal**: Achieve high-quality 480p → 2K super-resolution suitable for real-world applications

---

## 🧠 Core Features

- 📁 Automated dataset curation and preprocessing with OpenCV  
- 🧱 Residual CNN architecture with skip connections and sub-pixel convolution (PixelShuffle)  
- 🚀 Training with Adam optimizer, checkpoints, and validation PSNR/SSIM  
- 📈 Benchmarking vs bicubic with CSV + image exports  
- 🖼️ Tiled inference for arbitrarily large images on commodity GPUs

---

## 🧰 Tech Stack

- **Python**  
- **PyTorch**  
- **OpenCV**  
- **NumPy**  
- **Matplotlib** (optional for visualization)

---

## 📦 Setup

```bash
git clone https://github.com/karan-k16/UTMIST-Image-Enhancement.git
cd UTMIST-Image-Enhancement
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
