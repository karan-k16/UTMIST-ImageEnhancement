# 🖼️ UTMIST Image Super-Resolution Pipeline

This repo contains a lightweight, production-oriented pipeline from UTMIST (Fall 2024–Winter 2025) that upscales **480p → 2K** using a **residual CNN** with **PixelShuffle**. It includes automated data curation, **patch-based training**, **mixed precision (AMP)**, **tiled inference with overlap** for large images, and a **PSNR/SSIM** benchmarking harness.

---

## 📌 Project Highlights

- 🔍 **Model**: Residual CNN (head/body/tail) with `PixelShuffle` upsampler  
- ⚙️ **GPU Optimization**: Mixed precision (AMP), `torch.no_grad()` inference, tiled **512×512** windows with ~**32 px** overlap and feathered blending (seam-free)  
- 📊 **Metrics**: PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index)  
- 🧪 **Training**: Patch-based on HR images with on-the-fly LR generation (bicubic by default), Adam optimizer, validation checkpoints  
- 🎯 **Goal**: High-quality **2×** super-resolution for real-world 480p → 2K use cases

---

## 🧠 Core Features

- 📁 Automated dataset curation and OpenCV preprocessing  
- 🧱 Residual CNN with skip connections and sub-pixel convolution (`PixelShuffle`)  
- 🚀 Training loop with AMP, checkpoints, and per-epoch validation PSNR/SSIM  
- 📈 Benchmarking vs. bicubic with CSV exports and summary stats  
- 🖼️ Seam-free **tiled inference** for arbitrarily large images on commodity GPUs (T4-class)

---

## 🧰 Tech Stack

- **Python**  
- **PyTorch**  
- **OpenCV**  
- **NumPy**  
- **scikit-image** (PSNR/SSIM)  
- *(Matplotlib optional for visualization)*

---

## 📦 Setup

```bash
git clone https://github.com/karan-k16/UTMIST-Image-Enhancement.git
cd UTMIST-Image-Enhancement
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## 📝 Notes

- Use `--AMP` with CUDA for faster inference and lower memory.  
- If edge artifacts appear, increase `--OVERLAP` (e.g., 48–64).  
- Keep a true held-out validation set; report PSNR/SSIM on that set only.  
- PixelShuffle keeps most compute at lower spatial resolution → cheaper than early upsampling.

---

## 📊 Current Results

- **Validation set**: **+3.1 dB PSNR** and **+0.05 SSIM** vs. bicubic  
- **Inference efficiency**: Stable **2K** runs on T4 with **~40% lower VRAM** (AMP + tiled inference)

---

## 🙏 Acknowledgments

Thanks to UTMIST mentors and collaborators. Inspired by residual CNN SR literature and community baselines.

---
