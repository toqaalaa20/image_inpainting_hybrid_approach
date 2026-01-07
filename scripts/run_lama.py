import torch
import cv2
import numpy as np
from pathlib import Path
import os
import argparse
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Paths
WEIGHTS = "/workspace/weights/big-lama.pt"
RAW_DIR = Path("/workspace/data/raw")
MASK_DIR = Path("/workspace/data/masks")
OUT_DIR = Path("/workspace/results/lama")
OUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_to_multiple(img, factor):
    h, w = img.shape[:2]
    new_h = int(math.ceil(h / factor) * factor)
    new_w = int(math.ceil(w / factor) * factor)
    pad_h = new_h - h
    pad_w = new_w - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if len(img.shape) == 3:
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
    else:
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded, (top, bottom, left, right)


def unpad(img, pads):
    top, bottom, left, right = pads
    h, w = img.shape[:2]
    return img[top:h - bottom if bottom != 0 else h, left:w - right if right != 0 else w]


def load_model(weights_path, device, scripted=False):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    if scripted:
        print("🧠 Loading scripted TorchScript model...")
        model = torch.jit.load(str(weights_path), map_location=device)
        model.eval()
        return model

    # Try best-effort to load a saved module or state dict. If not a full module, raise informative error.
    print("🧠 Attempting to load weights as a saved torch module (not scripted).")
    obj = torch.load(str(weights_path), map_location=device)
    if isinstance(obj, torch.nn.Module):
        model = obj
        model.eval()
        return model.to(device)
    # Some checkpoints are dicts with a 'state_dict' - these require the original model class to load.
    if isinstance(obj, dict) and "state_dict" in obj:
        raise RuntimeError("Checkpoint contains 'state_dict' but no model class. Provide a scripted model (use --scripted) or provide a loader that constructs the model.")

    raise RuntimeError("Unsupported weight file format. Provide a scripted TorchScript model with --scripted or a saved torch.nn.Module.")


def preprocess_image(img_bgr, mask_gray):
    # img_bgr: HxWx3 uint8 BGR
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_f = img_rgb.astype(np.float32) / 255.0
    # mask: single-channel uint8
    if len(mask_gray.shape) == 3:
        mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    mask_f = (mask_bin.astype(np.float32) / 255.0)
    return img_f, mask_f


def to_tensor(img_f):
    # img_f: HxWxC (float32 0..1)
    t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).float()
    return t


def save_result(out_arr, out_path):
    # out_arr expected HxWx3 float32 or uint8 in RGB [0..1] or [0..255]
    if out_arr.dtype == np.float32 or out_arr.dtype == np.float64:
        out = np.clip(out_arr, 0.0, 1.0)
        out = (out * 255.0).round().astype(np.uint8)
    else:
        out = out_arr
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), out_bgr)


def run_inference(weights, raw_dir, mask_dir, out_dir, device_str, scripted=False, factor=8):
    device = torch.device(device_str if torch.cuda.is_available() and "cuda" in device_str else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        model = load_model(weights, device, scripted=scripted)
    except Exception as e:
        print(f"Model load error: {e}")
        return

    raw_dir = Path(raw_dir)
    mask_dir = Path(mask_dir)

    for img_path in sorted(raw_dir.glob("*.jpg")):
        stem = img_path.stem
        mask_path = mask_dir / f"{stem}_mask.png"
        if not mask_path.exists():
            print(f"⚠️  Mask missing for {stem}, skipping.")
            continue

        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Failed to read {img_path}, skipping.")
                continue
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Failed to read mask {mask_path}, skipping.")
                continue

            img_f, mask_f = preprocess_image(img, mask)

            # pad
            img_pad, pads = pad_to_multiple((img_f * 255).astype(np.uint8), factor)
            # img_pad currently uint8 in BGR if we used copyMakeBorder; convert back to float RGB
            img_pad_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # pad mask similarly
            mask_pad, _ = pad_to_multiple((mask_f * 255).astype(np.uint8), factor)
            _, mask_pad_bin = cv2.threshold(mask_pad, 127, 255, cv2.THRESH_BINARY)

            # to tensors
            img_t = torch.from_numpy(img_pad_rgb).permute(2, 0, 1).unsqueeze(0).to(device).float()
            mask_t = torch.from_numpy(mask_pad_bin.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                # Model call convention may vary; attempt common patterns
                try:
                    out = model(img_t, mask_t)
                except TypeError:
                    # some scripted models expect a dict or single tensor
                    try:
                        out = model(img_t, mask_t)
                    except Exception as e:
                        raise RuntimeError(f"Model inference failed: {e}")

            # out may be tensor or tuple/list
            if isinstance(out, (list, tuple)):
                out = out[0]
            if isinstance(out, torch.Tensor):
                res = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            else:
                raise RuntimeError("Model output type not recognized (expected torch.Tensor).")

            # ensure in 0..1
            if res.dtype != np.float32 and res.dtype != np.float64:
                res = res.astype(np.float32)
            # if model outputs 0..255 scale
            if res.max() > 2.0:
                res = res / 255.0
            res = np.clip(res, 0.0, 1.0)

            # unpad
            res_unpad = unpad((res * 255).round().astype(np.uint8), pads)
            # convert back to RGB float 0..1 for save_result
            res_unpad_rgb = cv2.cvtColor(res_unpad, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            out_path = out_dir / f"{stem}_lama.png"
            save_result(res_unpad_rgb, out_path)
            print(f"✨ AI Result Saved: {out_path.name}")

        except Exception as e:
            print(f"Error processing {stem}: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run LaMa inference (expects a scripted or saved torch module).")
    p.add_argument("--weights", default="weights/big-lama.pt", help="Path to weights (scripted TorchScript recommended)")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--mask-dir", default="data/masks")
    p.add_argument("--out-dir", default="results/lama")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to run inference on")
    p.add_argument("--scripted", action="store_true", help="Load as TorchScript via torch.jit.load")
    p.add_argument("--factor", type=int, default=8, help="Pad images so H,W are multiples of this factor")
    args = p.parse_args()

    run_inference(args.weights, args.raw_dir, args.mask_dir, args.out_dir, args.device, scripted=args.scripted, factor=args.factor)