#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GitHub backup copy
# Original file: VASP_moveCheck.py
# Suggested English filename: vasp_xdatcar_moved_atoms.py

import argparse
import numpy as np
from pathlib import Path


def is_int_list(tokens):
    try:
        [int(x) for x in tokens]
        return True
    except ValueError:
        return False


def read_xdatcar(filename):
    """
    Read VASP XDATCAR.
    Returns:
        lattice: (3,3) ndarray in Angstrom
        species: list of str
        counts: list of int
        frames: (nframe, natom, 3) ndarray of fractional coords
    Supports VASP5-like and partially VASP4-like XDATCAR.
    """
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.rstrip() for line in f if line.strip()]

    if len(lines) < 8:
        raise ValueError("XDATCAR appears too short.")

    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=float)
    lattice *= scale

    line5 = lines[5].split()
    line6 = lines[6].split()

    # VASP5: line 5 = element names, line 6 = counts
    # VASP4: line 5 = counts
    if is_int_list(line5):
        species = [f"X{i+1}" for i in range(len(line5))]
        counts = [int(x) for x in line5]
        start_idx = 6
    else:
        species = line5
        if not is_int_list(line6):
            raise ValueError("Could not parse element counts from XDATCAR header.")
        counts = [int(x) for x in line6]
        start_idx = 7

    natoms = sum(counts)

    frames = []
    i = start_idx

    while i < len(lines):
        line = lines[i].strip()

        # Typical XDATCAR frame header:
        # Direct configuration=     1
        if "configuration" in line.lower():
            i += 1
            if i + natoms - 1 >= len(lines):
                raise ValueError("Unexpected EOF while reading frame coordinates.")

            coords = []
            for _ in range(natoms):
                toks = lines[i].split()
                if len(toks) < 3:
                    raise ValueError(f"Malformed coordinate line: {lines[i]}")
                coords.append([float(toks[0]), float(toks[1]), float(toks[2])])
                i += 1

            frames.append(coords)
        else:
            # Skip any unexpected non-configuration lines
            i += 1

    if len(frames) == 0:
        raise ValueError("No frames found in XDATCAR.")

    frames = np.array(frames, dtype=float)
    return lattice, species, counts, frames


def frac_minimum_image_delta(frac2, frac1):
    """
    Minimum-image fractional displacement from frac1 -> frac2
    """
    d = frac2 - frac1
    d -= np.round(d)
    return d


def find_moved_atoms(frames, lattice, tol=1e-4, mode="cart"):
    """
    Determine atoms that moved at least once between consecutive frames.

    Parameters
    ----------
    frames : ndarray, shape (nframe, natom, 3), fractional coordinates
    lattice : ndarray, shape (3,3), Cartesian lattice vectors in Angstrom
    tol : float
        Threshold.
        - if mode='frac': tolerance in fractional coordinate norm
        - if mode='cart': tolerance in Angstrom
    mode : str
        'frac' or 'cart'

    Returns
    -------
    moved_mask : ndarray, shape (natom,), bool
    max_disp : ndarray, shape (natom,)
        Maximum displacement encountered for each atom
        in the same unit system as mode
    """
    nframe, natom, _ = frames.shape
    moved_mask = np.zeros(natom, dtype=bool)
    max_disp = np.zeros(natom, dtype=float)

    for t in range(nframe - 1):
        dfrac = frac_minimum_image_delta(frames[t + 1], frames[t])  # (natom, 3)

        if mode == "frac":
            disp = np.linalg.norm(dfrac, axis=1)
        elif mode == "cart":
            dcart = dfrac @ lattice
            disp = np.linalg.norm(dcart, axis=1)
        else:
            raise ValueError("mode must be 'frac' or 'cart'")

        moved_mask |= (disp > tol)
        max_disp = np.maximum(max_disp, disp)

    return moved_mask, max_disp


def expand_species_list(species, counts):
    atom_species = []
    for sp, n in zip(species, counts):
        atom_species.extend([sp] * n)
    return atom_species


def main():
    parser = argparse.ArgumentParser(
        description="Extract atom indices that moved at least once in a VASP XDATCAR."
    )
    parser.add_argument("xdatcar", help="Path to XDATCAR")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="Movement tolerance (default: 1e-3)")
    parser.add_argument("--mode", choices=["cart", "frac"], default="cart",
                        help="Tolerance mode: 'cart' in Angstrom or 'frac' in fractional coords (default: cart)")
    parser.add_argument("--index-base", choices=["0", "1"], default="1",
                        help="Output atom numbering base (default: 1)")
    parser.add_argument("--out", default=None,
                        help="Optional output text filename")
    parser.add_argument("--verbose", action="store_true",
                        help="Print atom index, species, and max displacement")
    args = parser.parse_args()

    xdatcar = Path(args.xdatcar)
    lattice, species, counts, frames = read_xdatcar(xdatcar)

    moved_mask, max_disp = find_moved_atoms(
        frames=frames,
        lattice=lattice,
        tol=args.tol,
        mode=args.mode
    )

    atom_species = expand_species_list(species, counts)
    moved_indices_0 = np.where(moved_mask)[0]

    if args.index_base == "1":
        moved_indices = moved_indices_0 + 1
    else:
        moved_indices = moved_indices_0

    unit = "Angstrom" if args.mode == "cart" else "fractional"

    print(f"File               : {xdatcar}")
    print(f"Number of frames   : {frames.shape[0]}")
    print(f"Number of atoms    : {frames.shape[1]}")
    print(f"Tolerance          : {args.tol} ({unit})")
    print(f"Moved atom count   : {len(moved_indices)}")
    print()

    if args.verbose:
        print("# index species max_displacement")
        for idx0, idx in zip(moved_indices_0, moved_indices):
            print(f"{idx:6d} {atom_species[idx0]:>4s} {max_disp[idx0]: .8e}")
    else:
        print("Moved atom indices:")
        print(" ".join(map(str, moved_indices.tolist())))

    if args.out is not None:
        with open(args.out, "w", encoding="utf-8") as f:
            if args.verbose:
                f.write("# index species max_displacement\n")
                for idx0, idx in zip(moved_indices_0, moved_indices):
                    f.write(f"{idx:6d} {atom_species[idx0]:>4s} {max_disp[idx0]: .8e}\n")
            else:
                f.write(" ".join(map(str, moved_indices.tolist())) + "\n")


if __name__ == "__main__":
    main()