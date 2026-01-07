import cv2
import numpy as np
from pathlib import Path
import argparse


def gaussian_lowpass_mask(h, w, cutoff):
    # cutoff: fraction of max dimension (0..0.5)
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    # choose sigma so that mask falls at cutoff * max_dim
    max_dim = max(h, w)
    sigma = max_dim * cutoff
    if sigma <= 0:
        sigma = 1.0
    mask = np.exp(-(dist ** 2) / (2 * (sigma ** 2)))
    # normalize to [0,1]
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask


def freq_blend_channel(a_chan, b_chan, lowpass_mask):
    # a_chan, b_chan: float32 arrays in 0..1
    # lowpass_mask: HxW float [0..1]
    # compute FFTs
    fa = np.fft.fftshift(np.fft.fft2(a_chan))
    fb = np.fft.fftshift(np.fft.fft2(b_chan))
    # blend in frequency domain: keep low freqs from fa (lama), high freqs from fb (gt)
    F = fa * lowpass_mask + fb * (1.0 - lowpass_mask)
    # inverse
    F_ishift = np.fft.ifftshift(F)
    img_back = np.fft.ifft2(F_ishift)
    img_back = np.real(img_back)
    return img_back


def process_image(lama_path, gt_path, out_path, mode='blend', cutoff=0.05):
    lama = cv2.imread(str(lama_path))
    gt = cv2.imread(str(gt_path))
    if lama is None:
        raise RuntimeError(f"Cannot read LaMa output: {lama_path}")
    if gt is None:
        raise RuntimeError(f"Cannot read ground truth: {gt_path}")

    # ensure same size
    if lama.shape[:2] != gt.shape[:2]:
        gt = cv2.resize(gt, (lama.shape[1], lama.shape[0]), interpolation=cv2.INTER_AREA)

    # convert to float RGB 0..1
    lama_rgb = cv2.cvtColor(lama, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    h, w = lama_rgb.shape[:2]
    lowpass_mask = gaussian_lowpass_mask(h, w, cutoff)
    # expand mask to complex multiplier shape
    lp = lowpass_mask.astype(np.complex64)

    out_channels = []
    for c in range(3):
        a = lama_rgb[:, :, c]
        b = gt_rgb[:, :, c]
        if mode == 'lowpass':
            # lowpass only from LaMa: multiply LaMa FFT by mask, inverse
            fa = np.fft.fftshift(np.fft.fft2(a))
            F = fa * lp
            F_ishift = np.fft.ifftshift(F)
            img_back = np.fft.ifft2(F_ishift)
            img_back = np.real(img_back)
        else:
            img_back = freq_blend_channel(a, b, lp)
        out_channels.append(img_back)

    out_img = np.stack(out_channels, axis=2)
    # clip and scale
    out_img = np.clip(out_img, 0.0, 1.0)
    out_u8 = (out_img * 255.0).round().astype(np.uint8)
    out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), out_bgr)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Hybrid frequency-domain postprocess for LaMa outputs')
    p.add_argument('--lama-dir', default='results/lama', help='Directory containing LaMa outputs')
    p.add_argument('--raw-dir', default='data/raw', help='Directory with ground truth images')
    p.add_argument('--out-dir', default='results/lama_hybrid', help='Output directory for hybrid results')
    p.add_argument('--mode', choices=['blend', 'lowpass'], default='blend', help='blend: low-freq from LaMa + high-freq from GT; lowpass: only low frequencies from LaMa')
    p.add_argument('--cutoff', type=float, default=0.05, help='Cutoff fraction (0..0.5) for low-frequency region')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing hybrid outputs')
    p.add_argument('--limit', type=int, default=None, help='Limit number of images to process')
    args = p.parse_args()

    lama_dir = Path(args.lama_dir)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(lama_dir.glob('*_lama.png'))
    if args.limit is not None:
        files = files[:args.limit]

    for pth in files:
        stem = pth.stem.replace('_lama', '')
        gt_p = raw_dir / f"{stem}.jpg"
        out_p = out_dir / f"{stem}_lama_hybrid.png"
        if out_p.exists() and not args.overwrite:
            print(f"Skipping existing {out_p}")
            continue
        try:
            process_image(pth, gt_p, out_p, mode=args.mode, cutoff=args.cutoff)
            print(f"Saved hybrid: {out_p}")
        except Exception as e:
            print(f"Failed {pth.name}: {e}")
