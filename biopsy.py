"""
Defines the virtual biopsy data explorer class.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
from twixtools import twixtools


class VirtualBiopsy:
    def __init__(self, dat_fn: Union[Path, str], num_readouts: int = 32):
        """
        Args:
            dat_fn: path to .dat file of raw Siemens virtual biopsy data.
            num_readouts: number of readouts in raw data.
        """
        self.dat_fn = dat_fn
        self.num_readouts = num_readouts
        self.kspace = self.read_image_data()
        self.cimg = self.reconstruct()

    def read_image_data(self) -> np.ndarray:
        """
        Reads raw scanner data from specified .dat file.
        Input:
            None.
        Returns:
            Raw scanner data from self.dat_fn as a 3D array with dimensions
            [acquisition_counter, num_channels, num_columns].
        """
        data = []
        for mdb in twixtools.read_twix(self.dat_fn)[-1]["mdb"]:
            if mdb.is_image_scan():
                data.append(mdb.data)
        return np.asarray(data)

    def reconstruct(self) -> np.ndarray:
        """
        Reconstructs image from kspace data.
        Input:
            None.
        Returns:
            Complex image reconstruction of self.kspace.
        """
        num_averages = self.kspace.shape[0] // self.num_readouts
        if num_averages * self.num_readouts != self.kspace.shape[0]:
            msg = f"Number of readouts {self.num_readouts} must be a "
            msg += f"multiple of {self.kspace.shape[0]}."
            raise ValueError(msg)
        _, c, d = self.kspace.shape
        kspace = self.kspace.reshape(
            (num_averages, self.num_readouts, c, d), order="C"
        )

        # Average over the num_averages acquisitions.
        kspace = np.mean(kspace, axis=0)

        # Reverse the odd count acquisitions.
        for r in range(kspace.shape[0]):
            if r % 2:
                kspace[r, :, :] = kspace[r, :, ::-1]
        # Move the coil dimension to the beginning.
        kspace = kspace.transpose((1, 0, -1))

        # Zero-pad the Fourier data.
        c, h, w = kspace.shape
        padding = np.zeros((c, (w - h) // 2, w), dtype=kspace.dtype)
        kspace = np.concatenate((padding, kspace, padding,), axis=1)

        # Inverse Fourier transform.
        recon = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1))),
            axes=(-2, -1)
        )

        # Sum over the coils.
        recon = np.sum(recon, axis=0)

        return np.flip(recon)

    def center_crop(
        self, crop_size: Optional[Tuple[int]] = None, inplace: bool = False
    ) -> Optional[np.ndarray]:
        """
        Center crops the complex biopsy image to the specified shape.
        Input:
            crop_size: crop size. Default (H/2)x(W/2).
            inplace: whether to directly modify self.cimg.
        Returns:
            Copy of self.cimg center cropped to crop_size if not inplace.
            Otherwise, return None.
        """
        if crop_size is None:
            crop_size = (self.cimg.shape[-2] // 2, self.cimg.shape[-1] // 2)
        rmin = (self.cimg.shape[-2] - crop_size[0]) // 2
        rmax = rmin + crop_size[0]
        cmin = (self.cimg.shape[-1] - crop_size[1]) // 2
        cmax = cmin + crop_size[1]
        if inplace:
            self.cimg = self.cimg[rmin:rmax, cmin:cmax]
            return None
        else:
            return np.copy(self.cimg)[rmin:rmax, cmin:cmax]

    def phase_correction(
        self,
        method: str = "Ahn and Cho",
        num_bins: int = 100,
        inplace: bool = False
    ) -> np.ndarray:
        """
        Corrects even/odd phase discrepancy in biopsy image using a specified
        algorithm.
        Input:
            method: phase correction algorithm.
            num_bins: number of bins to use in histogram for determination of
                zero-order phase correction term in Ahn and Cho (1987)
                algorithm.
            inplace: whether to directly modify self.cimg.
        Returns:
            Copy of self.cimg with even/odd phase discrepancy corrected if not
            inplace. Otherwise, return None.
        """
        def _ahn_and_cho(cimg: np.ndarray) -> np.ndarray:
            H, W = cimg.shape
            rho_x = np.zeros(W, dtype=cimg.dtype)
            for r in range(H - 1):
                rho_x = np.add(
                    rho_x,
                    np.divide(cimg[r, :] * np.conj(cimg[r + 1, :]), H - 1)
                )
            eps_1 = -np.angle(rho_x)
            eps_1 = np.mean(eps_1)
            cimg_1 = cimg * np.exp(-1j * eps_1 * np.arange(0, H))
            hist, bin_edges = np.histogram(np.angle(cimg_1), bins=num_bins)
            eps_0 = bin_edges[np.argmax(hist)]
            return cimg_1 * np.exp(-1j * eps_0)
        method_map = {
            "ahn and cho": _ahn_and_cho,
        }
        if inplace:
            self.cimg = method_map[method.lower()](self.cimg)
            return None
        else:
            return method_map[method.lower()](self.cimg)

    def plot(self, savepath: Optional[Union[Path, str]] = None) -> None:
        """
        Plots the magnitude of the virtual biopsy.
        Input:
            savepath: optional filepath to save the virtual biopsy image to.
        Returns:
            None.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(np.abs(self.cimg), cmap="gray")
        plt.axis("off")
        if savepath is None:
            plt.show()
        else:
            plt.savefig(
                savepath, dpi=600, bbox_inches="tight", transparent=True
            )
