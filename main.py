"""
Virtual biopsy explorer program.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import argparse
import os
from analysis import QuantitativePlots
from biopsy import VirtualBiopsy


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Virtual Biopsy Explorer")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to raw Siemens data file. Should be .dat file type."
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default=None,
        help="Optional savepath to save image to. Default not saved."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed. Default 42."
    )
    parser.add_argument(
        "--M",
        type=int,
        default=20,
        help="Hann window size. Default 20."
    )
    parser.add_argument(
        "--echo_spacing",
        type=float,
        default=3.86,
        help="Echo spacing imaging parameter in msec."
    )
    parser.add_argument(
        "--spatial_resolution",
        type=float,
        default=0.2,
        help="Spatial resolution along the spatial axis in mm. Default 0.2 mm."
    )
    parser.add_argument(
        "--B", type=float, default=3.0, help="Magnet strength in Teslas."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()

    biopsy = VirtualBiopsy("./data/20221107_VirtualBiopsy_Phantom/meas_MID00045_FID05843_VB_TE16ms_TR1s_32av_PE_RL.dat", seed=args.seed, M=args.M)
    figs = QuantitativePlots(
        biopsy,
        echo_spacing=3.92,
        B=3.0,
        spatial_resolution=0.2
    )
    figs.T2_spectra(savepath=os.path.join("./docs", "phantom_T2_spectra.png"))
    figs.spectra(
        locs=[60, 230, 520, 340],
        colors=["#00A087B2", "#3C5488B2", "#E64B35B2", "#4DBBD5B2"],
        savepath=os.path.join("docs", "phantom_spectra.png")
    )

    biopsy = VirtualBiopsy(args.data_path, seed=args.seed, M=args.M)

    figs = QuantitativePlots(
        biopsy,
        echo_spacing=args.echo_spacing,
        B=args.B,
        spatial_resolution=args.spatial_resolution
    )
    figs.T2_spectra(savepath="./T2_spectra.png")
    figs.peak_offresonance(savepath="./peak_offresonance.png")
    figs.T2star_map(savepath=None)
