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
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()

    biopsy = VirtualBiopsy(args.data_path, seed=args.seed)
    biopsy.center_crop(inplace=True)
    # biopsy.plot(savepath=args.savepath)
