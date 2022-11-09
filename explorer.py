"""
Defines the virtual biopsy EPSI data explorer and visualizer class.

Author(s):
    Michael S Yao
    M Dylan Tisdall

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from typing import Optional, Sequence, Union

from biopsy import VirtualBiopsy


class EPSIExplorer:
    def __init__(self, biopsy: VirtualBiopsy):
        """
        Args:
            biopsy: virtual biopsy data class.
        """
        self.plot_config()
        self.biopsy = biopsy

    def T2_spectra(
        self,
        savepath: Union[Path, str] = None,
        locs: Optional[Sequence[int]] = None,
        colors: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Plots the x-f T2-weighted EPSI spectrum.
        Input:
            savepath: optional path to save the plot. Default not saved.
            locs: optional location(s) (in pixel coordinates) at which to
                highlight with vertical lines in the spectrum.
            colors: optional colors to plot the highlighting vertical lines
                in the spectrum. If not provided, all of the spectra are
                plotted in black.
        Returns:
            None.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(np.abs(self.biopsy.spectrum), cmap="gray", aspect="auto")
        if locs is not None:
            if colors is None:
                colors = ["black"] * len(locs)
            for loc, co in zip(locs, colors):
                plt.axvline(x=loc, linestyle="-", color=co, linewidth=4)

        wticks = np.linspace(0, self.biopsy.spectrum.shape[-1], 5)
        wticklabels = [
            f"{int(self.biopsy.RO_voxel_size * x)}" for x in wticks
        ]
        plt.gca().set_xticks(wticks)
        plt.gca().set_xticklabels(wticklabels)
        plt.xlabel("Spatial Position Along Biopsy [mm]")

        hticks = np.linspace(0, self.biopsy.spectrum.shape[0], 5)
        hticklabels = [
            1e6 * f * self.biopsy.spectral_unit / self.biopsy.larmor_frequency
            for f in hticks - self.biopsy.spectrum.shape[0] // 2
        ]
        hticklabels = [f"{x:.2f}" for x in hticklabels]
        plt.gca().set_yticks(hticks)
        plt.gca().set_yticklabels(hticklabels[::-1])
        plt.ylabel("Frequency Shift [ppm]")
        if savepath is None:
            plt.show()
        else:
            plt.savefig(
                savepath,
                dpi=600,
                transparent=True,
                bbox_inches="tight",
                pad_inches=0.0
            )
        plt.close()
        return

    def peak_offresonance(
        self, savepath: Union[Path, str] = None, threshmin: float = 0.15
    ) -> None:
        """
        Plots the estimated peak off-resonance frequency as a function of
        spatial position.
        Input:
            savepath: optional path to save the plot. Default not saved.
            threshmin: signal threshold for determining biopsy signal
                versus background noise in relaxation data (normalized to
                have a maximum intensity of 1).
        Returns:
            None.
        """
        relaxation = self.biopsy.fftnd(self.biopsy.spectrum, axes=[0])
        norm_magnitude = np.abs(relaxation) / np.max(np.abs(relaxation))
        off_reson_map = np.zeros(shape=(relaxation.shape[-1],))
        for i in range(relaxation.shape[-1]):
            signal = relaxation[:, i]
            # Thresholding to only focus on biopsy with higher SNR compared
            # to background.
            if np.max(norm_magnitude[:, i]) < threshmin:
                continue
            # Use the center 80% of signal for better T2star fitting.
            frac_signal = 0.8
            frac_signal = min(max(frac_signal, 0.0), 1.0)
            imin = int((1.0 - frac_signal) * signal.shape[0] / 2)
            imax = signal.shape[0] - imin
            signal = signal[imin:imax]
            # Compute the magnitude-weighted mean of the off-resonance
            # frequency over time.
            freq = 0.0
            for t in range(signal.shape[0] - 1):
                freq += np.abs(signal[t]) * np.angle(
                    np.conj(signal[t]) * signal[t + 1]
                )
            off_reson_map[i] = freq / np.sum(np.abs(signal[:-1]))
        x = self.biopsy.RO_voxel_size * np.arange(0, relaxation.shape[-1])
        # Convert frequency units to ppm.
        off_reson_map = 1e6 * off_reson_map / self.biopsy.larmor_frequency
        off_reson_map = off_reson_map / self.biopsy.echo_spacing

        # Restrict plot to just the biopsy itself with calculated off-resonance
        # frequency values.
        signal_idxs = np.where(off_reson_map != 0)[0]
        x, off_reson_map = x[signal_idxs], off_reson_map[signal_idxs]

        _, left_ax = plt.subplots(figsize=(10, 8))
        left_ax.plot(x, off_reson_map, color="black")
        left_ax.set_xlabel("Spatial Position Along Biopsy [mm]")
        left_ax.set_ylabel("Peak Off-Resonance Frequency [ppm]")

        right_ax = left_ax.twinx()
        mn, mx = left_ax.get_ylim()
        right_ax.set_ylim(
            mn * self.biopsy.larmor_frequency / 1e6,
            mx * self.biopsy.larmor_frequency / 1e6
        )
        right_ax.set_ylabel("Peak Off-Resonance Frequency [Hz]")

        if savepath is None or len(savepath) == 0:
            plt.show()
        else:
            plt.savefig(
                savepath, dpi=600, bbox_inches="tight", transparent=True
            )
        plt.close()
        return

    def T2star_map(
        self, savepath: Union[Path, str] = None, threshmin: float = 0.15
    ) -> None:
        """
        Plots estimated T2* as a function of spatial position.
        Input:
            savepath: optional path to save the plot. Default not saved.
            threshmin: signal threshold for determining biopsy signal
                versus background noise in relaxation data (normalized to
                have a maximum intensity of 1).
        Returns:
            None.
        """
        relaxation = self.biopsy.fftnd(self.biopsy.spectrum, axes=[0])
        relaxation = np.abs(relaxation) / np.max(np.abs(relaxation))
        t2star_map = np.zeros(shape=(relaxation.shape[-1],))
        for i in range(relaxation.shape[-1]):
            signal = relaxation[:, i]
            # Thresholding to only focus on biopsy with higher SNR compared
            # to background.
            if np.max(signal) < threshmin:
                continue
            # Use the center 80% of signal for better T2star fitting.
            frac_signal = 0.8
            frac_signal = min(max(frac_signal, 0.0), 1.0)
            imin = int((1.0 - frac_signal) * signal.shape[0] / 2)
            imax = signal.shape[0] - imin
            signal = signal[imin:imax]
            signal = np.where(
                signal > np.finfo(np.float32).eps, np.log(signal), -50.0
            )
            time = np.arange(0, signal.shape[0]) * self.biopsy.echo_spacing
            r2star, _ = np.polyfit(time, signal, 1)
            t2star_map[i] = 1e3 / r2star
        x = self.biopsy.RO_voxel_size * np.arange(0, relaxation.shape[-1])

        # Restrict plot to just the biopsy itself with calculated T2* values.
        signal_idxs = np.where(t2star_map > 0)[0]
        x, t2star_map = x[signal_idxs], t2star_map[signal_idxs]

        plt.figure(figsize=(10, 6))
        plt.plot(x, t2star_map, color="black")
        plt.xlabel("Spatial Position Along Biopsy [mm]")
        plt.ylabel(r"$T_2^*$ [msec]")

        if savepath is None or len(savepath) == 0:
            plt.show()
        else:
            plt.savefig(
                savepath, dpi=600, bbox_inches="tight", transparent=True
            )
        plt.close()
        return

    def T2_map(
        self,
        other_epsi: Sequence[VirtualBiopsy],
        savepath: Union[Path, str] = None,
        threshmin: float = 0.15
    ) -> None:
        """
        Plots estimated T2 as a function of spatial position. T2 estimation
        requires at least one additional EPSI acquisition of the same sample
        at a different TE.
        Input:
            other_epsi: a list of additional VirtualBiopsy object(s)
                representing the same EPSI acquisition with only a different
                TE.
            savepath: optional path to save the plot. Default not saved.
            threshmin: signal threshold for determining biopsy signal
                versus background noise in relaxation data (normalized to
                have a maximum intensity of 1).
        Returns:
            None.
        """
        max_echos = {
            biopsy.TE: self.biopsy.fftnd(biopsy.spectrum, axes=[0])[-1, :]
            for biopsy in other_epsi
        }
        max_echos[self.biopsy.TE] = self.biopsy.fftnd(
            self.biopsy.spectrum, axes=[0]
        )[-1, :]
        norm = max([np.max(np.abs(epsi)) for _, epsi in max_echos.items()])
        max_echos = {
            TE: np.abs(epsi) / norm for TE, epsi in max_echos.items()
        }
        t2_map = np.zeros(shape=(self.biopsy.spectrum.shape[-1],))
        TE_values = sorted(list(max_echos.keys()))
        for i in range(self.biopsy.spectrum.shape[-1]):
            signal = np.array([max_echos[TE][i] for TE in TE_values])
            # Thresholding to only focus on biopsy with higher SNR compared
            # to background.
            if np.max(signal) < threshmin:
                continue
            signal = np.where(
                signal > np.finfo(np.float32).eps, np.log(signal), -50.0
            )
            minus_r2, _ = np.polyfit(TE_values, signal, 1)
            t2_map[i] = -1e3 / minus_r2
        x = self.biopsy.RO_voxel_size * np.arange(
            0, self.biopsy.spectrum.shape[-1]
        )

        # Restrict plot to just the biopsy itself with calculated T2* values.
        signal_idxs = np.where(t2_map > 0)[0]
        x, t2_map = x[signal_idxs], t2_map[signal_idxs]

        plt.figure(figsize=(10, 6))
        plt.plot(x, t2_map, color="black")
        plt.xlabel("Spatial Position Along Biopsy [mm]")
        plt.ylabel(r"$T_2$ [msec]")

        if savepath is None or len(savepath) == 0:
            plt.show()
        else:
            plt.savefig(
                savepath, dpi=600, bbox_inches="tight", transparent=True
            )
        plt.close()
        return

    def spectra1d(
        self,
        locs: Sequence[int],
        colors: Sequence[str] = None,
        savepath: Union[Path, str] = None
    ) -> np.ndarray:
        """
        Plots the frequency spectra at given spatial location along the biopsy.
        Assumes that the EPSI readout direction (ie biopsy axis) is left-right
        in the image.
        Input:
            locs: location(s) (in pixel coordinates) at which to plot the
                coresponding frequency spectra.
            colors: optional colors to plot the frequency spectra. If not
                provided, all of the spectra are plotted in black.
            savepath: optional path to save the annotated EPSI plot. Default
                not saved.
        Returns:
            None.
        """
        plt.figure(figsize=(10, 5))
        if colors is None:
            colors = ["black"] * len(locs)
        for loc, co in zip(locs, colors):
            plt.plot(
                np.abs(self.biopsy.spectrum[:, loc]), color=co, linewidth=4
            )
            # Plot formatting.
            plt.gca().get_xaxis().set_visible(False)
            plt.gca().get_yaxis().set_visible(False)
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["left"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.gca().spines["bottom"].set_linewidth(4)
            if savepath is not None:
                plt.savefig(
                    os.path.splitext(savepath)[0] + f"{loc}.png",
                    dpi=600,
                    transparent=True,
                    bbox_inches="tight"
                )
            else:
                plt.show()
            plt.cla()
        plt.close()

        self.T2_spectra(savepath, locs=locs, colors=colors)
        return

    def plot_config(self, font_size: int = 18, use_sans_serif: bool = True):
        """
        Plot configuration variables.
        Input:
            font_size: default font size. Default 18.
            use_sans_serif: whether to use Sans Serif font style.
        Returns:
            None.
        """
        matplotlib.rcParams["mathtext.fontset"] = "stix"
        if use_sans_serif:
            matplotlib.rcParams['font.family'] = "Arial"
        else:
            matplotlib.rcParams["font.family"] = "STIXGeneral"
        matplotlib.rcParams.update({"font.size": font_size})
