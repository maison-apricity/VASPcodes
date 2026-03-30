#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import List

import numpy as np
from ase import Atoms
from ase.io import read, write


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p for p in parts]


def get_center(atoms: Atoms, mode: str = "geom") -> np.ndarray:
    if mode == "geom":
        return atoms.get_positions().mean(axis=0)
    if mode == "com":
        return atoms.get_center_of_mass()
    raise ValueError(f"Unknown center mode: {mode}")


def validate_consistency(images: List[Atoms], filenames: List[Path]) -> None:
    if not images:
        raise ValueError("No images loaded.")

    ref_natoms = len(images[0])
    ref_symbols = images[0].get_chemical_symbols()

    for atoms, fname in zip(images[1:], filenames[1:]):
        natoms = len(atoms)
        symbols = atoms.get_chemical_symbols()
        if natoms != ref_natoms:
            raise ValueError(
                f"[{fname}] atom count mismatch: {natoms} != {ref_natoms}"
            )
        if symbols != ref_symbols:
            raise ValueError(
                f"[{fname}] atomic order/species mismatch.\n"
                f"Reference: {ref_symbols}\n"
                f"Current  : {symbols}\n"
                "All NEB images must have identical atom order."
            )


def center_in_box(atoms: Atoms, box: float, center_mode: str, pbc: bool) -> Atoms:
    atoms = atoms.copy()
    cell = np.diag([box, box, box])
    box_center = np.array([box / 2.0, box / 2.0, box / 2.0])

    current_center = get_center(atoms, center_mode)
    shift = box_center - current_center
    atoms.set_positions(atoms.get_positions() + shift)
    atoms.set_cell(cell)
    atoms.set_pbc([pbc, pbc, pbc])

    pos = atoms.get_positions()
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    if np.any(mins < 0.0) or np.any(maxs > box):
        raise ValueError(
            "Molecule does not fit inside the requested box after centering.\n"
            f"min = {mins}, max = {maxs}, box = {box} Å\n"
            "Increase --box."
        )

    return atoms


def write_xdatcar(images: List[Atoms], outpath: Path) -> None:
    if not images:
        raise ValueError("No images available for XDATCAR writing.")

    first = images[0]
    cell = first.get_cell()
    symbols = first.get_chemical_symbols()

    unique_symbols: List[str] = []
    counts: List[int] = []
    for sym in symbols:
        if not unique_symbols or unique_symbols[-1] != sym:
            unique_symbols.append(sym)
            counts.append(1)
        else:
            counts[-1] += 1

    with outpath.open("w", encoding="utf-8") as f:
        f.write("Generated from XYZ images\n")
        f.write("1.0\n")
        for vec in cell:
            f.write(f"  {vec[0]:22.16f} {vec[1]:22.16f} {vec[2]:22.16f}\n")
        f.write("  " + "  ".join(unique_symbols) + "\n")
        f.write("  " + "  ".join(str(c) for c in counts) + "\n")

        for i, atoms in enumerate(images, start=1):
            scaled = atoms.get_scaled_positions(wrap=False)
            f.write(f"Direct configuration= {i:6d}\n")
            for row in scaled:
                f.write(f"  {row[0]:20.16f} {row[1]:20.16f} {row[2]:20.16f}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert multiple XYZ images into VASP NEB folders 00, 01, ..., "
            "and generate a top-level XDATCAR."
        )
    )
    parser.add_argument(
        "--pattern",
        default="*.xyz",
        help='Input XYZ pattern, e.g. "*.xyz" or "images/*.xyz" (default: *.xyz)',
    )
    parser.add_argument(
        "--box",
        type=float,
        default=20.0,
        help="Cubic lattice length in Å (default: 20.0)",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Parent directory that will contain 00, 01, ..., 25 and XDATCAR (default: current directory)",
    )
    parser.add_argument(
        "--center-mode",
        choices=["geom", "com"],
        default="geom",
        help="Center each image by geometric center or center of mass (default: geom)",
    )
    parser.add_argument(
        "--pbc",
        action="store_true",
        help="Set PBC = True for the written structures",
    )
    parser.add_argument(
        "--expect",
        type=int,
        default=26,
        help="Expected number of images (default: 26)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing 00, 01, ..., XDATCAR in outdir if present",
    )
    args = parser.parse_args()

    files = sorted(Path(".").glob(args.pattern), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No XYZ files matched pattern: {args.pattern}")
    if len(files) != args.expect:
        raise ValueError(
            f"Expected {args.expect} XYZ files, but found {len(files)}."
        )

    images = [read(str(f)) for f in files]
    validate_consistency(images, files)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    centered_images: List[Atoms] = []
    for atoms in images:
        centered_images.append(center_in_box(atoms, args.box, args.center_mode, args.pbc))

    # Create NEB folders 00..25 and write POSCAR into each.
    for idx, atoms in enumerate(centered_images):
        folder = outdir / f"{idx:02d}"
        poscar = folder / "POSCAR"
        if folder.exists() and args.force:
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)
        write(str(poscar), atoms, format="vasp", direct=True, sort=False, vasp5=True)

    # Write top-level XDATCAR outside the folders.
    xdatcar = outdir / "XDATCAR"
    if xdatcar.exists() and not args.force:
        raise FileExistsError(
            f"{xdatcar} already exists. Use --force to overwrite."
        )
    write_xdatcar(centered_images, xdatcar)

    print(f"Converted {len(files)} XYZ files.")
    print(f"Output directory : {outdir}")
    print(f"POSCAR folders   : 00 .. {len(files)-1:02d}")
    print(f"Top-level XDATCAR: {xdatcar}")


if __name__ == "__main__":
    main()
