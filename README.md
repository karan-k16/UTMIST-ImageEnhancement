# 🖼️ UTMIST Image Super-Resolution Pipeline

This project was developed as part of UTMIST's research initiative in Fall 2024. It implements a deep learning pipeline to upscale low-resolution images (e.g., 480p) to high-resolution (e.g., 4K) using the **Enhanced Deep Super-Resolution (EDSR)** model architecture. The pipeline also includes dataset preparation, model training, performance evaluation, and visual testing.

---

## 📌 Project Highlights

- 🔍 **Model Used**: Enhanced Deep Super-Resolution (EDSR) with residual blocks and `PixelShuffle` upsampling
- 📊 **Evaluation Metrics**: PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index)
- 🧪 **Training Approach**: Multiple image downscaling methods for robust model generalization
- 🎯 **Goal**: Achieve high-quality 4× super-resolution suitable for real-world applications

---

## 🧠 Core Features

- 📁 Loads custom image datasets and applies bicubic or Gaussian downsampling
- 🧱 Constructs EDSR model with residual blocks and sub-pixel convolution layers
- 🚀 Trains on LR-HR image pairs using MSE loss
- 📈 Evaluates model performance with PSNR and SSIM
- 🖼️ Visualizes super-resolved outputs alongside ground truth and inputs

---

## 🧰 Tech Stack

- **Python**
- **PyTorch**
- **OpenCV**
- **NumPy**
- **Matplotlib**

---

## 🖼️ Sample Results

| Low Resolution | Super-Resolved Output | Ground Truth |
|----------------|-----------------------|--------------|
| ![LR](examples/lr.png) | ![SR](examples/sr.png) | ![HR](examples/hr.png) |

> *Add screenshots or sample images to the `examples/` folder to visualize model outputs.*

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/karan-k16/UTMIST-Image-Enhancement.git
   cd UTMIST-Image-Enhancement
