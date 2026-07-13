"""Export the SRResCNN PyTorch model to ONNX.

The exported graph uses dynamic height/width axes so the same model file
serves any input resolution (tiled or full-frame).

Usage:
    python -m ml.export_onnx --WEIGHTS_PATH runs/best.pth --OUT_PATH artifacts/sr_model.onnx
    # Or export with random-initialized weights (latency benchmarking only):
    python -m ml.export_onnx --RANDOM_INIT --OUT_PATH artifacts/sr_model.onnx
"""
import argparse
from pathlib import Path

import torch

from models.sr_residual import SRResCNN


def build_model(args, device: str = "cpu") -> SRResCNN:
    model = SRResCNN(
        channels=args.CHANNELS,
        num_blocks=args.NUM_BLOCKS,
        scale=args.UPSCALE,
        use_bn=args.USE_BN,
    ).to(device)
    if not args.RANDOM_INIT:
        state = torch.load(args.WEIGHTS_PATH, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model


def export(model: SRResCNN, out_path: Path, opset: int = 17, sample_hw: int = 128) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, sample_hw, sample_hw)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["lr"],
        output_names=["sr"],
        dynamic_axes={
            "lr": {0: "batch", 2: "height", 3: "width"},
            "sr": {0: "batch", 2: "out_height", 3: "out_width"},
        },
        opset_version=opset,
    )
    return out_path


def verify(onnx_path: Path, model: SRResCNN, sample_hw: int = 96) -> float:
    """Run the same random input through PyTorch and ONNX Runtime, return max abs diff."""
    import numpy as np
    import onnxruntime as ort

    x = torch.randn(1, 3, sample_hw, sample_hw)
    with torch.no_grad():
        ref = model(x).numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(["sr"], {"lr": x.numpy()})[0]
    return float(np.abs(ref - out).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--WEIGHTS_PATH", type=str, default=None)
    ap.add_argument("--RANDOM_INIT", action="store_true",
                    help="Export with random weights (for latency benchmarks without a checkpoint)")
    ap.add_argument("--OUT_PATH", type=str, default="artifacts/sr_model.onnx")
    ap.add_argument("--UPSCALE", type=int, default=2)
    ap.add_argument("--CHANNELS", type=int, default=64)
    ap.add_argument("--NUM_BLOCKS", type=int, default=8)
    ap.add_argument("--USE_BN", action="store_true")
    ap.add_argument("--OPSET", type=int, default=17)
    ap.add_argument("--SKIP_VERIFY", action="store_true")
    args = ap.parse_args()

    if not args.RANDOM_INIT and not args.WEIGHTS_PATH:
        ap.error("Provide --WEIGHTS_PATH or pass --RANDOM_INIT")

    model = build_model(args, device="cpu")
    out_path = export(model, Path(args.OUT_PATH), opset=args.OPSET)
    print(f"Exported ONNX model: {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")

    if not args.SKIP_VERIFY:
        diff = verify(out_path, model)
        print(f"PyTorch vs ONNX Runtime max abs diff: {diff:.2e}")
        if diff > 1e-3:
            raise SystemExit(f"Numerical mismatch too large ({diff}); export may be broken")
        print("Parity check passed.")


if __name__ == "__main__":
    main()
