"""
Defines the virtual biopsy data explorer class.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import convolve2d
from scipy.signal.windows import hann
from typing import Optional, List, Sequence, Tuple, Union
from twixtools import map_twix
from twixtools import twix_array


class VirtualBiopsy:
    def __init__(
        self,
        dat_fn: Union[Path, str],
        seed: int = 42
    ):
        """
        Args:
            dat_fn: path to .dat file of raw Siemens virtual biopsy data.
            seed: optional random seed.
        """
        self.dat_fn = dat_fn
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.cimg = self.reconstruct(M=20)

    def ifftnd(
        self, kspace: np.ndarray, axes: Sequence[int] = [-1]
    ) -> np.ndarray:
        """
        Performs an n-dimensional inverse fast Fourier transform.
        Input:
            kspace: kspace tensor.
            axes: axes over which to perform the iFFT operaton.
        Returns:
            The inverse fast Fourier transform of kspace.
        This function implementation was adapted from https://github.com/
        pehses/twixtools/blob/master/demo/recon_example.ipynb
        """
        if axes is None:
            axes = list(range(kspace.ndim))
        img = np.fft.fftshift(
            np.fft.ifftn(np.fft.ifftshift(kspace, axes=axes), axes=axes),
            axes=axes
        )
        return img * np.sqrt(np.prod(np.take(img.shape, axes)))

    def fftnd(
        self, img: np.ndarray, axes: Sequence[int] = [-1]
    ) -> np.ndarray:
        """
        Performs an n-dimensional fast Fourier transform.
        Input:
            img: image tensor.
            axes: axes over which to perform iFFT operaton.
        Returns:
            The fast Fourier transform of img.
        This function implementation was adapted from https://github.com/
        pehses/twixtools/blob/master/demo/recon_example.ipynb
        """
        if axes is None:
            axes = list(range(img.ndim))
        kspace = np.fft.fftshift(
            np.fft.fftn(np.fft.ifftshift(img, axes=axes), axes=axes),
            axes=axes
        )
        return np.divide(
            kspace, np.sqrt(np.prod(np.take(kspace.shape, axes)))
        )

    def phase_corr(
        self, epsi_map: List[dict], kspace: twix_array
    ) -> np.ndarray:
        """
        Calculate and apply EPI phase correction using a simple
        autocorrelation-based approach.
        Input:
            epsi_map: an object returned from calling `twixtools.map_twix`
                that represents the EPSI virtual biopsy dataset.
            kspace: the extracted image kspace dataset.
        Returns:
            The phase constructed and iFFT'ed kspace data.
        This function implementation was adapted from https://github.com/
        pehses/twixtools/blob/master/demo/recon_example.ipynb
        """
        phase_corr = epsi_map[-1]["phasecorr"]
        phase_corr.flags["remove_os"] = True
        phase_corr.flags["skip_empty_lead"] = True
        phase_corr.flags["average"]["Seg"] = False
        phase_corr.flags["regrid"] = True

        # Calculate phase correction using a simple autocorrelation-based
        # approach.
        ncol = phase_corr.shape[-1]
        pc = self.ifftnd(phase_corr[:], axes=[-1])
        slope = np.angle(
            np.sum(
                np.sum(
                    np.conj(pc[..., 1:]) * pc[..., :-1],
                    axis=-1,
                    keepdims=True
                ),
                axis=-2,
                keepdims=True
            )
        )
        pc_corr = np.exp(1j * slope * (np.arange(ncol) - (ncol // 2)))
        # Apply phase correction.
        img = self.ifftnd(kspace[:], axes=[-1])
        img = img * pc_corr
        # Remove segment dimension.
        img = np.squeeze(np.sum(img, axis=5))
        img = self.ifftnd(img, axes=[0])
        return img

    def coil_combine(self, img: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        Simple RSS coil combination implementation.
        Input:
            img: EPSI virtual biopsy image data.
            axis: axis to apply coil combination to.
        Returns:
            Coil combined image.
        """
        return np.sqrt(np.sum(np.conj(img) * img, axis=axis))

    def freq_axis(
        self,
        img: np.ndarray,
        signal_thresh: float = 0.001,
        num_sample: int = 1000
    ) -> Tuple[np.ndarray]:
        """
        Fits a quadratic model to determine the zero set point of the
        frequency axis in a 2D virtual biopsy image.
        Input:
            img: 2D EPSI virtual biopsy image.
            signal_thresh: threshold for image intensity values, above
                which an image pixel is considered part of the EPSI
                signal.
            num_sample: number of signal pixels to use in quadratic
                fitting. Default 1000. If this parameter is negative,
                then all available signal values are used.
        Returns:
            pe_locs: frequency axis locations along the frequency dimension.
            ro_locs: frequency axis locations along the spatial readout
                dimension.
        """
        # Threshold the signal in image space (empirically determined).
        w_locs, h_locs = np.where(img > signal_thresh)
        # Randomly sample a subset of the signal points in image space.
        if num_sample < 0:
            num_sample = h_locs.shape[0]
        num_sample = max(3, min(h_locs.shape[0], num_sample))
        idxs = self.rng.choice(
            np.arange(h_locs.shape[0]), size=num_sample, replace=False
        )
        h_locs, w_locs = h_locs[idxs], w_locs[idxs]

        # Fit the data to a quadratic model.
        q_model = np.poly1d(np.polyfit(h_locs, w_locs, 2))
        ro_locs = np.arange(0, img.shape[-1])
        pe_locs = q_model(ro_locs)
        return pe_locs, ro_locs

    def reconstruct(self, M: int = 20) -> np.ndarray:
        """
        Reads and reconstructs raw scanner data from specified .dat file.
        Input:
            M: Hann window size. Default 20.
            None.
        Returns:
            Raw scanner data from self.dat_fn as a 3D array with dimensions
            [acquisition_counter, num_channels, num_columns].
        """
        epsi_map = map_twix(self.dat_fn)
        kspace = epsi_map[-1]["image"]
        kspace.flags["remove_os"] = True
        # For phase-correction, we need to keep the individual segments, which
        # indicate the readout's polarity.
        kspace.flags["average"]["Seg"] = False
        kspace.flags["regrid"] = True

        img = self.phase_corr(epsi_map, kspace)
        # Coil combination.
        img = self.coil_combine(img, axis=1)

        # Zero-pad the Fourier data.
        kspace = np.squeeze(VirtualBiopsy.zero_pad(
            self.fftnd(img, axes=[0, -1])[np.newaxis, ...]
        ))

        # Apply apodization filter.
        if M > 0:
            apo_window = hann(M=M)[..., np.newaxis]
            kspace = convolve2d(kspace, apo_window, mode="same")

        img = np.flip(self.ifftnd(kspace, axes=[-2, -1]))

        x, y = self.freq_axis(np.abs(img))
        # Plot the frequency axis on the EPSI image.
        for x_loc, y_loc in zip(x, y):
            img[int(x_loc), y_loc] = np.max(img)
        self.plot(img)
        return img

    @staticmethod
    def zero_pad(kspace: np.ndarray) -> np.ndarray:
        """
        Zero-pads an input kspace data to make the dataset square.
        Input:
            kspace: kspace dataset with dimensions CHW. Padding is applied
                along the H dimension.
        Returns:
            Zero-padded kspace.
        """
        c, h, w = kspace.shape
        padding = np.zeros((c, (w - h) // 2, w), dtype=kspace.dtype)
        return np.concatenate((padding, kspace, padding,), axis=1)

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

    def plot(
        self,
        img: np.ndarray = None,
        savepath: Optional[Union[Path, str]] = None
    ) -> None:
        """
        Plots an image.
        Input:
            img: 2D image to plot. Default self.cimg.
            savepath: optional filepath to save the virtual biopsy image to.
        Returns:
            None.
        """
        plt.figure(figsize=(10, 10))
        if img is None:
            img = self.cimg
        plt.imshow(np.abs(img), cmap="gray")
        plt.axis("off")
        if savepath is None:
            plt.show()
        else:
            plt.savefig(
                savepath, dpi=600, bbox_inches="tight", transparent=True
            )
