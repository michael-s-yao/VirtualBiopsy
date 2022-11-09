"""
Defines the virtual biopsy EPSI data class.

Author(s):
    Michael S Yao
    M Dylan Tisdall

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import numpy as np
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union
from twixtools import map_twix


class VirtualBiopsy:
    def __init__(
        self,
        dat_fn: Union[Path, str],
        seed: int = 42,
        do_zero_pad: bool = True
    ):
        """
        Args:
            dat_fn: path to .dat file of raw Siemens virtual biopsy data.
            seed: optional random seed. Default 42.
            do_zero_pad: whether or not to zero-pad the kspace data to
                square dimensions. Default True.
        """
        self.dat_fn = dat_fn
        self.seed = seed
        if not self.dat_fn.endswith(".dat"):
            raise ValueError(
                f"Expecting .dat raw data file, but got {self.dat_fn} instead."
            )
        self.do_zero_pad = do_zero_pad
        self.rng = np.random.RandomState(self.seed)

        # Extract Siemens raw data.
        epsi_map = map_twix(self.dat_fn)
        kspace = epsi_map[-1]["image"]
        kspace.flags["remove_os"] = True
        kspace.flags["average"]["Seg"] = True
        kspace.flags["average"]["Ave"] = False
        kspace = np.squeeze(kspace[:])

        # Extract relevant metadata.
        metadata = epsi_map[-1]["hdr"]
        # Echo spacing has units of seconds.
        self.echo_spacing = metadata["Config"]["EchoSpacing_us"] / 1e6
        self.len_echo_train = int(metadata["Dicom"]["EchoTrainLength"])
        self.spectral_unit = 1 / (self.len_echo_train * self.echo_spacing)
        # B field strength has units of Teslas. Rounding to the nearest
        # integer field strength.
        self.B_mag = float(round(metadata["Dicom"]["flMagneticFieldStrength"]))
        self.gamma = 42.58 * 1e6  # In units of Hz / T. Assuming 1H imaging.
        self.larmor_frequency = self.gamma * self.B_mag
        # MR Imaging parameters. TE, TR, and total scan time are in units of
        # seconds. FOV and voxel size is in units of mm.
        self.TR = metadata["Config"]["TR"] / 1e6
        self.TE = metadata["Meas"]["alTE"][0] / 1e6
        self.scan_time = metadata["Config"]["TotalScanTimeFromUIinms"] / 1e3
        self.RO_FOV = metadata["Config"]["RoFOV"]
        self.RO_voxel_size = self.RO_FOV / kspace.shape[-1]

        # Phase correction.
        kspace = self.phase_correct(kspace)

        # Spectrum is the inverse Fourier transform of the kspace data.
        # Note that the kspace data is technically k-t data for EPSI.
        # Therefore, the spectrum data is in x-f space.
        spectrum = self.ifftnd(kspace, axes=[1, -1])
        # Complex average over the repetitions.
        spectrum = np.mean(spectrum, axis=0)
        # Coil combination using SVD to maximize the SNR in the final output.
        u, s, _ = np.linalg.svd(
            np.transpose(spectrum, axes=(2, 0, 1)), full_matrices=False
        )
        self.spectrum = np.flip(s[:, 0] * u[:, :, 0].T)
        self.kspace = self.fftnd(self.spectrum, axes=[0, -1])

        if self.do_zero_pad:
            self.kspace = self.zero_pad(self.kspace)
            self.spectrum = self.ifftnd(self.kspace, axes=[0, -1])

        self.res_axis = self.freq_axis(
            np.abs(self.spectrum) / np.max(np.abs(self.spectrum))
        )

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

    def phase_correct(self, kspace: np.ndarray) -> np.ndarray:
        """
        Calculate and apply EPI phase correction using the approach proposed
        by Tisdall MD. (2020). Scrambled readout polarities in 3D-encoded EPI
        markedly reduce the coherence of N/2 ghosts. Proc ISMRM.
        Input:
            kspace: the extracted EPSI kspace dataset with shape .
        Returns:
            The phase corrected kspace data.
        """
        relaxation = self.ifftnd(kspace, axes=[-1])
        mean_relaxation = np.mean(relaxation, axis=0)

        target = np.pad(
            mean_relaxation[1:],
            ((0, 1), (0, 0), (0, 0)),
            "constant",
            constant_values=0.0
        )
        target[1:] += mean_relaxation[:-1]
        target[1:-1] /= 2.0

        delay_filter = (target * np.conj(mean_relaxation)) / np.abs(
            target * mean_relaxation
        )
        delay_filter[0::2] = 1.0

        corrected_relaxation = relaxation * delay_filter
        kspace = self.fftnd(corrected_relaxation, axes=[-1])
        return kspace

    def freq_axis(
        self,
        img: np.ndarray,
        signal_thresh: float = 0.2,
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
        self.biopsy_lims = np.min(w_locs), np.max(w_locs)
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

    def zero_pad(self, kspace: np.ndarray) -> np.ndarray:
        """
        Zero-pads an input kspace data to make the dataset square.
        Input:
            kspace: kspace dataset with dimensions HW. Padding is applied
                along the H dimension.
        Returns:
            Zero-padded kspace.
        """
        h, w = kspace.shape
        padding = np.zeros(((w - h) // 2, w), dtype=kspace.dtype)
        return np.concatenate((padding, kspace, padding,), axis=0)

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
