"""
Virtual biopsy explorer program.

Author(s):
    Michael Yao

Licensed under the MIT License.
"""
import argparse
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
        "--num_readouts",
        type=int,
        default=32,
        help="Number of readouts in raw data. Default 32."
    )
    parser.add_argument(
        "--savepath",
        type=str,
        default=None,
        help="Optional savepath to save image to. Default not saved."
    )
    parser.add_argument(
        "--even_odd_correction",
        type=str,
        default="Tisdall",
        choices=("Ahn and Cho", "Tisdall", "None"),
        help="Even-odd line correction algorithm to use. Default Tisdall 2020."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()

    biopsy = VirtualBiopsy(args.data_path, args.num_readouts, do_zero_pad=True)
    biopsy.center_crop(inplace=True)
    biopsy.phase_correction(
        method=args.even_odd_correction, num_bins=100, inplace=True
    )
    biopsy.plot(savepath=args.savepath)
