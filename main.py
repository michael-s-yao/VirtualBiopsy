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
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()

    biopsy = VirtualBiopsy(args.data_path, args.num_readouts)
    biopsy.center_crop(inplace=True)
    biopsy.phase_correction(
        method="Ahn and Cho", num_bins=100, inplace=True
    )
    biopsy.plot()
