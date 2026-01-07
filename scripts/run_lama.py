import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
import math


def pad_to_multiple(img: np.ndarray, factor: int):
    """
    Pad an image or single-channel mask so H and W are multiples of `factor`.
    Works with float or uint8 arrays and preserves channels.
    Returns (padded, (top, bottom, left, right)).
    """
    h, w = img.shape[:2]
    new_h = int(math.ceil(h / factor) * factor)
    new_w = int(math.ceil(w / factor) * factor)
    pad_h = new_h - h
    pad_w = new_w - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if img.ndim == 3:
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT)
    else:
        # single channel (mask)
        padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded, (top, bottom, left, right)


def unpad(img: np.ndarray, pads):
    top, bottom, left, right = pads
    h, w = img.shape[:2]
    bottom_idx = h - bottom if bottom != 0 else h
    right_idx = w - right if right != 0 else w
    return img[top:bottom_idx, left:right_idx]


def load_model(weights_path: str, device: torch.device, scripted: bool = False):
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    if scripted:
        # TorchScript
        model = torch.jit.load(str(weights_path), map_location=device)
        model.eval()
        return model.to(device)

    # Best-effort load: either a saved nn.Module or a checkpoint
    obj = torch.load(str(weights_path), map_location=device)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        return obj.to(device)

    if isinstance(obj, dict) and "state_dict" in obj:
        raise RuntimeError(
            "Checkpoint contains 'state_dict' but no model class. Provide a scripted model (use --scripted)"
        )

    raise RuntimeError("Unsupported weight file format. Provide a scripted TorchScript model with --scripted or a saved torch.nn.Module.")


def preprocess_image(img_bgr: np.ndarray, mask_gray: np.ndarray):
    # Convert BGR->RGB and scale to float32 0..1
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if mask_gray is None:
        raise ValueError("Mask is None")
    if mask_gray.ndim == 3:
        mask_gray = cv2.cvtColor(mask_gray, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    mask_f = (mask_bin.astype(np.float32) / 255.0)
    return img_rgb, mask_f


def save_result(out_arr: np.ndarray, out_path: Path):
    # out_arr expected HxWx3 float32 in RGB (0..1) or uint8 RGB (0..255)
    if out_arr.dtype in (np.float32, np.float64):
        out = np.clip(out_arr, 0.0, 1.0)
        out = (out * 255.0).round().astype(np.uint8)
    else:
        out = out_arr
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), out_bgr)


def run_inference(weights, raw_dir, mask_dir, out_dir, device_str, scripted=False, factor=8):
    device = torch.device("cuda" if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
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

            # pad (works with float arrays)
            img_pad, pads = pad_to_multiple(img_f, factor)
            mask_pad, _ = pad_to_multiple(mask_f, factor)
            # ensure mask is binary 0/1
            _, mask_pad_bin = cv2.threshold((mask_pad * 255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
            mask_pad_f = (mask_pad_bin.astype(np.float32) / 255.0)

            # to tensors: [B, C, H, W] for image, [B, 1, H, W] for mask
            img_t = torch.from_numpy(img_pad.transpose(2, 0, 1)).unsqueeze(0).to(device).float()
            mask_t = torch.from_numpy(mask_pad_f).unsqueeze(0).unsqueeze(0).to(device).float()

            with torch.no_grad():
                out = None
                # Try common call signatures: model(img, mask) or model({'image': img, 'mask': mask})
                try:
                    out = model(img_t, mask_t)
                except TypeError:
                    try:
                        out = model({"image": img_t, "mask": mask_t})
                    except Exception as e:
                        raise RuntimeError(f"Model inference failed (tried several call signatures): {e}")
                except Exception as e:
                    raise RuntimeError(f"Model inference failed: {e}")

            # Normalize model output to HxWx3 float 0..1
            if isinstance(out, (list, tuple)):
                out = out[0]
            if isinstance(out, torch.Tensor):
                res = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            else:
                raise RuntimeError("Model output type not recognized (expected torch.Tensor).")

            # if model outputs 0..255 scale
            if res.dtype not in (np.float32, np.float64):
                res = res.astype(np.float32)
            if res.max() > 2.0:
                res = res / 255.0
            res = np.clip(res, 0.0, 1.0)

            # unpad
            res_unpad = unpad(res, pads)

            out_path = out_dir / f"{stem}_lama.png"
            save_result(res_unpad, out_path)
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