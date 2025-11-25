import os
import sys
import argparse
import cv2
import numpy as np

#!/usr/bin/env python3


def make_mask(input_path: str, output_path: str, dilate: int = 1):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: cannot read '{input_path}'", file=sys.stderr)
        sys.exit(2)

    # Ensure we have at least 3 channels (BGR)
    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img_bgr = img[:, :, :3]
    else:
        img_bgr = img

    # mask of pure black pixels (B=G=R=0)
    black_mask = np.all(img_bgr == 0, axis=2)

    # non-black mask (original, before dilation)
    non_black_orig = ~black_mask

    # compute dilated mask from original non-black
    if dilate and dilate > 0:
        ks = 2 * dilate + 1  # kernel size from radius
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        non_black_uint8 = (non_black_orig.astype(np.uint8) * 255)
        dilated_uint8 = cv2.dilate(non_black_uint8, kernel)
        dilated = dilated_uint8.astype(bool)
    else:
        dilated = non_black_orig.copy()

    # added = pixels introduced by dilation (were black, now in dilated)
    added = dilated & ~non_black_orig

    h, w = black_mask.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)

    # Set original non-black pixels to white and fully opaque
    out[non_black_orig, :3] = 255  # B,G,R = 255 (white)
    out[non_black_orig, 3] = 255   # alpha = 255

    # Set dilated-only pixels to black and fully opaque
    out[added, :3] = 0    # B,G,R = 0 (black)
    out[added, 3] = 255   # alpha = 255

    # Remaining (original black) pixels stay transparent (alpha = 0)

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not cv2.imwrite(output_path, out):
        print(f"Error: cannot write '{output_path}'", file=sys.stderr)
        sys.exit(3)