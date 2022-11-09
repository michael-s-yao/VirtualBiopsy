"""
Simple script to visualize NIfTI files using matplotlib.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import argparse
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Union


def load_nifti(fn: Union[str, Path]) -> np.ndarray:
    nii_img = nib.load(fn)
    return np.squeeze(nii_img.get_fdata()).T


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NIfTI File Explorer")
    parser.add_argument(
        "--datapath",
        type=str,
        required=True,
        help="Path to NIfTI data file. Should have .nii.gz file extension."
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default=None,
        help="Path to save png file to. Image not saved in default setting."
    )
    parser.add_argument(
        "--kspace",
        action="store_true",
        help="Visualize log of magnitude of kspace instead of image space."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    img = load_nifti(args.datapath)
    h, w = img.shape
    plt.figure()
    if args.kspace:
        plt.imshow(
            np.log(np.abs(np.fft.fftshift(np.fft.fft2(img[h-w:h, :])))),
            cmap="gray"
        )
    else:
        plt.imshow(img, cmap="gray")
    plt.axis("off")
    if args.savepath is None or len(args.savepath) == 0:
        plt.show()
    else:
        plt.savefig(
            args.savepath, dpi=600, transparent=True, bbox_inches="tight"
        )
