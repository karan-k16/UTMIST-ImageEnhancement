import numpy as np
import torch
import cv2

def feather_mask(h, w):
    wy = np.linspace(0, 1, h, dtype=np.float32)
    wx = np.linspace(0, 1, w, dtype=np.float32)
    mask = np.minimum.outer(np.minimum(wy, wy[::-1]), np.minimum(wx, wx[::-1]))
    mask = np.clip(mask * 4.0, 0, 1)  # sharper feather towards edges
    return mask  # HxW

@torch.no_grad()
def infer_tiled(model, lr_rgb, scale=2, tile=512, overlap=32, amp=True, device="cuda"):
    """
    Seam-free tiled inference:
    - lr_rgb: uint8 RGB HxWx3
    - returns sr_rgb: uint8 RGB (scaled by `scale`)
    """
    model.eval()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    H, W, _ = lr_rgb.shape
    out_h, out_w = H * scale, W * scale

    sr_acc = np.zeros((out_h, out_w, 3), dtype=np.float32)
    w_acc  = np.zeros((out_h, out_w), dtype=np.float32)

    step = tile - overlap * 2
    if step <= 0:
        raise ValueError("tile must be > 2*overlap")

    for y in range(0, H, step):
        for x in range(0, W, step):
            y0, x0 = y, x
            y1 = min(y + tile, H)
            x1 = min(x + tile, W)

            lr_tile = lr_rgb[y0:y1, x0:x1]
            lr_tensor = torch.from_numpy(lr_tile.transpose(2,0,1)).float().unsqueeze(0) / 255.0
            lr_tensor = lr_tensor.to(device)

            if amp and device == "cuda":
                with torch.cuda.amp.autocast():
                    sr_tile = model(lr_tensor)
            else:
                sr_tile = model(lr_tensor)

            sr_tile = sr_tile.squeeze(0).clamp(0,1).cpu().numpy().transpose(1,2,0)
            th, tw = sr_tile.shape[:2]
            Y0, X0 = y0 * scale, x0 * scale
            Y1, X1 = Y0 + th, X0 + tw

            mask = feather_mask(th, tw)[..., None]
            sr_acc[Y0:Y1, X0:X1, :] += sr_tile * mask
            w_acc [Y0:Y1, X0:X1]    += mask[...,0]

    w_acc = np.maximum(w_acc, 1e-6)
    sr = sr_acc / w_acc[..., None]
    sr_u8 = np.clip(sr * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return sr_u8
