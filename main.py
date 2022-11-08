"""
Virtual biopsy explorer program.

Author(s):
    Michael S Yao
    M Dylan Tisdall

Licensed under the MIT License. Copyright 2022 University of Pennsylvania.
"""
import argparse
import os
from pathlib import Path
from typing import Sequence, Union

from biopsy import VirtualBiopsy
from explorer import EPSIExplorer


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Virtual Biopsy Explorer")
    parser.add_argument(
        "--datapath",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to raw Siemens data file. Should be .dat file type."
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

    phantom_help = "Runs our program to reproduce our presented ISMRM "
    phantom_help += "abstract results on acquired phantom data. Turning "
    phantom_help += "this flag on will ignore other optional arguments."
    parser.add_argument(
        "--reproduce_phantom", action="store_true", help=phantom_help
    )

    exvivo_help = "Runs our program to reproduce our presented ISMSM abstract "
    exvivo_help += "results on acquired ex-vivo brain tissue data. Turning "
    exvivo_help += "this flag on will ignore other optional arguments."
    parser.add_argument(
        "--reproduce_exvivo", action="store_true", help=exvivo_help
    )

    return parser.parse_args()


def reproduce_phantom(
    datapath: Union[Path, str], seed: int = 42, do_zero_pad: bool = False
) -> None:
    """
    Function to replicate our presented ISMRM abstract results using acquired
    phantom data.
    Input:
        datapath: path to Siemens raw phantom data.
        seed: optional random seed. Default 42.
        do_zero_pad: whether or not to zero-pad the kspace data to
            square dimensions. Default False.
    Returns:
        None.
    For reproducibility, you may find our raw data at the following URL:
    https://upenn.box.com/s/dt5ma6t3yrwldmoflc1wos8sdk8s22nx
    """
    epsi = VirtualBiopsy(datapath, seed=seed, do_zero_pad=do_zero_pad)
    explorer = EPSIExplorer(epsi)

    docs = "./docs"
    phantom_savedir = os.path.join(docs, "phantom")
    if not os.path.isdir(docs):
        os.mkdir(docs)
    if not os.path.isdir(phantom_savedir):
        os.mkdir(phantom_savedir)

    explorer.spectra1d(
        locs=[60, 230, 520, 660],
        colors=["#00A087B2", "#3C5488B2", "#E64B35B2", "#4DBBD5B2"],
        savepath=os.path.join(phantom_savedir, "phantom_spectra.png")
    )
    explorer.T2_spectra(
        savepath=os.path.join(phantom_savedir, "phantom_T2_spectra.png")
    )
    return


def reproduce_exvivo(
    datapaths: Sequence[Union[Path, str]],
    seed: int = 42,
    do_zero_pad: bool = False
) -> None:
    """
    Function to replicate our presented ISMRM abstract results using acquired
    ex-vivo brain tissue data.
    Input:
        datapath: path to Siemens raw phantom data.
        seed: optional random seed. Default 42.
        do_zero_pad: whether or not to zero-pad the kspace data to
            square dimensions. Default False.
    Returns:
        None.
    For reproducibility, you may find our raw data at the following URL:
    https://upenn.box.com/s/dt5ma6t3yrwldmoflc1wos8sdk8s22nx
    """
    # Only keep unique file entries.
    datapaths = list(dict.fromkeys(datapaths))

    epsi = VirtualBiopsy(datapaths[0], seed=seed, do_zero_pad=do_zero_pad)
    explorer = EPSIExplorer(epsi)

    docs = "./docs"
    exvivo_savedir = os.path.join(docs, "exvivo")
    if not os.path.isdir(docs):
        os.mkdir(docs)
    if not os.path.isdir(exvivo_savedir):
        os.mkdir(exvivo_savedir)

    explorer.T2star_map(
        savepath=os.path.join(exvivo_savedir, "T2star_map.png")
    )
    explorer.T2_spectra(
        savepath=os.path.join(exvivo_savedir, "T2_spectra.png")
    )
    explorer.peak_offresonance(
        savepath=os.path.join(exvivo_savedir, "peak_offresonance.png")
    )

    # T2 estimation requires additional virtual biopsies.
    additional_biopsies = [
        VirtualBiopsy(path, seed=seed, do_zero_pad=do_zero_pad)
        for path in datapaths[1:]
    ]
    explorer.T2_map(
        other_epsi=additional_biopsies,
        savepath=os.path.join(exvivo_savedir, "T2_map.png"),
    )
    return


if __name__ == "__main__":
    args = build_args()

    if args.reproduce_phantom:
        print("Running Phantom Data Replication Experiment...")
        reproduce_phantom(args.datapath)
        print("Done!")
    if args.reproduce_exvivo:
        print("Running Ex-vivo Data Replication Experiment...")
        reproduce_exvivo(args.datapath)
        print("Done!")

    if not args.reproduce_phantom and not args.reproduce_exvivo:
        biopsy = VirtualBiopsy(
            args.datapath, seed=args.seed, do_zero_pad=args.do_zero_pad
        )
        explorer = EPSIExplorer(biopsy)
