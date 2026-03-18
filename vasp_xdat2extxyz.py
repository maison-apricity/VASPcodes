#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GitHub backup copy
# Original file: VASP_selectiveID_pos2xdat.py
# Suggested English filename: vasp_xdatcar_sd_to_extxyz.py

"""
Convert a VASP XDATCAR trajectory into a multi-frame Extended XYZ file,
using the Selective Dynamics flags stored in one POSCAR/CONTCAR file.

What it does:
- Reads atom order, element symbols, and Selective Dynamics flags from POSCAR.
- Reads all frames from XDATCAR.
- Applies the same SD mask to every frame.
- Writes a multi-frame .extxyz that OVITO can load directly.

Output properties per atom:
- species : element symbol
- pos     : Cartesian coordinates
- id      : original 1-based atom index
- sd_x    : 1 if selective flag in x is T, else 0
- sd_y    : 1 if selective flag in y is T, else 0
- sd_z    : 1 if selective flag in z is T, else 0
- sd_ttt  : 1 only for T T T atoms, else 0
- sd_id   : running ID for T T T atoms only (1..N), else 0

Example:
    python xdatcar_sd_to_extxyz.py -p POSCAR -x XDATCAR -o neb_sd.extxyz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def _is_int_list(tokens: List[str]) -> bool:
    try:
        for t in tokens:
            int(t)
        return True
    except ValueError:
        return False


def _read_nonempty_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def _parse_scale_and_lattice(lines: List[str], start: int = 0) -> Tuple[float, List[List[float]]]:
    scale = float(lines[start + 1].split()[0])
    lattice = []
    for i in range(3):
        vec = [float(x) for x in lines[start + 2 + i].split()[:3]]
        lattice.append(vec)
    # Standard VASP usage: positive scalar multiplies all lattice vectors.
    # Negative scaling (volume mode) is intentionally not handled here.
    if scale <= 0:
        raise ValueError(
            "Negative or zero POSCAR/XDATCAR scaling factor is not supported by this script. "
            f"Found scale={scale}."
        )
    lattice = [[scale * x for x in row] for row in lattice]
    return scale, lattice


def parse_poscar(poscar_path: Path):
    lines = _read_nonempty_lines(poscar_path)
    if len(lines) < 8:
        raise ValueError(f"POSCAR seems too short: {poscar_path}")

    _, lattice = _parse_scale_and_lattice(lines, 0)

    line5 = lines[5].split()
    if _is_int_list(line5):
        # VASP 4 style: no element symbols line.
        symbols = [f"X{i+1}" for i in range(len(line5))]
        counts = [int(x) for x in line5]
        idx = 6
    else:
        symbols = line5
        counts = [int(x) for x in lines[6].split()]
        idx = 7

    natoms = sum(counts)

    selective = False
    if lines[idx].strip().lower().startswith("s"):
        selective = True
        idx += 1

    coord_mode = lines[idx].strip().lower()
    if not (coord_mode.startswith("d") or coord_mode.startswith("c")):
        raise ValueError(f"Failed to locate coordinate mode in POSCAR: line='{lines[idx]}'")
    idx += 1

    coord_lines = lines[idx: idx + natoms]
    if len(coord_lines) != natoms:
        raise ValueError(f"POSCAR atom count mismatch: expected {natoms}, got {len(coord_lines)}")

    elements: List[str] = []
    for sym, n in zip(symbols, counts):
        elements.extend([sym] * n)

    sd_flags: List[Tuple[bool, bool, bool]] = []
    for line in coord_lines:
        toks = line.split()
        if selective:
            if len(toks) < 6:
                raise ValueError(f"Selective Dynamics line too short in POSCAR: '{line}'")
            flags = tuple(tok.upper().startswith("T") for tok in toks[3:6])
        else:
            flags = (True, True, True)
        sd_flags.append(flags)

    return {
        "lattice": lattice,
        "symbols": symbols,
        "counts": counts,
        "elements": elements,
        "natoms": natoms,
        "selective": selective,
        "sd_flags": sd_flags,
    }


def parse_xdatcar(xdatcar_path: Path):
    lines = _read_nonempty_lines(xdatcar_path)
    if len(lines) < 8:
        raise ValueError(f"XDATCAR seems too short: {xdatcar_path}")

    _, lattice = _parse_scale_and_lattice(lines, 0)

    line5 = lines[5].split()
    if _is_int_list(line5):
        symbols = None
        counts = [int(x) for x in line5]
        idx = 6
    else:
        symbols = line5
        counts = [int(x) for x in lines[6].split()]
        idx = 7

    natoms = sum(counts)
    frames: List[List[List[float]]] = []

    while idx < len(lines):
        header = lines[idx].strip()
        header_lower = header.lower()

        if not (
            header_lower.startswith("direct configuration")
            or header_lower.startswith("cartesian configuration")
            or header_lower.startswith("konfig")
        ):
            idx += 1
            continue

        is_direct = not header_lower.startswith("cartesian")
        idx += 1
        block = lines[idx: idx + natoms]
        if len(block) < natoms:
            raise ValueError("Incomplete frame in XDATCAR.")

        coords = []
        for line in block:
            toks = line.split()
            if len(toks) < 3:
                raise ValueError(f"Invalid coordinate line in XDATCAR: '{line}'")
            coords.append([float(toks[0]), float(toks[1]), float(toks[2])])

        frames.append((coords, is_direct))
        idx += natoms

    if not frames:
        raise ValueError("No frames found in XDATCAR.")

    return {
        "lattice": lattice,
        "symbols": symbols,
        "counts": counts,
        "natoms": natoms,
        "frames": frames,
    }


def frac_to_cart(frac: List[float], lattice: List[List[float]]) -> List[float]:
    # lattice rows are a, b, c vectors
    x = frac[0] * lattice[0][0] + frac[1] * lattice[1][0] + frac[2] * lattice[2][0]
    y = frac[0] * lattice[0][1] + frac[1] * lattice[1][1] + frac[2] * lattice[2][1]
    z = frac[0] * lattice[0][2] + frac[1] * lattice[1][2] + frac[2] * lattice[2][2]
    return [x, y, z]


def lattice_to_extxyz_string(lattice: List[List[float]]) -> str:
    flat = []
    for vec in lattice:
        flat.extend(vec)
    return " ".join(f"{x:.16f}" for x in flat)


def write_extxyz(output_path: Path, elements, sd_flags, xdat_info):
    natoms = xdat_info["natoms"]
    lattice = xdat_info["lattice"]
    frames = xdat_info["frames"]

    ttt_running_ids = []
    counter = 0
    for fx, fy, fz in sd_flags:
        if fx and fy and fz:
            counter += 1
            ttt_running_ids.append(counter)
        else:
            ttt_running_ids.append(0)

    with output_path.open("w", encoding="utf-8") as f:
        for iframe, (coords_raw, is_direct) in enumerate(frames, start=1):
            f.write(f"{natoms}\n")
            f.write(
                f'Lattice="{lattice_to_extxyz_string(lattice)}" '
                f'Properties=species:S:1:pos:R:3:id:I:1:frame:I:1:sd_x:I:1:sd_y:I:1:sd_z:I:1:sd_ttt:I:1:sd_id:I:1 '
                f'pbc="T T T" Time={iframe}\n'
            )

            for i in range(natoms):
                pos = frac_to_cart(coords_raw[i], lattice) if is_direct else coords_raw[i]
                sx, sy, sz = sd_flags[i]
                sd_ttt = int(sx and sy and sz)
                sd_id = ttt_running_ids[i]
                f.write(
                    f"{elements[i]} "
                    f"{pos[0]:.16f} {pos[1]:.16f} {pos[2]:.16f} "
                    f"{i+1} {iframe} {int(sx)} {int(sy)} {int(sz)} {sd_ttt} {sd_id}\n"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Apply Selective Dynamics flags from POSCAR to all XDATCAR frames and write OVITO-readable EXTXYZ."
    )
    parser.add_argument("-p", "--poscar", required=True, help="POSCAR or CONTCAR with Selective Dynamics flags")
    parser.add_argument("-x", "--xdatcar", required=True, help="XDATCAR trajectory file")
    parser.add_argument("-o", "--output", default="neb_sd.extxyz", help="Output EXTXYZ file")
    args = parser.parse_args()

    poscar_path = Path(args.poscar)
    xdatcar_path = Path(args.xdatcar)
    output_path = Path(args.output)

    pos = parse_poscar(poscar_path)
    xdat = parse_xdatcar(xdatcar_path)

    if pos["natoms"] != xdat["natoms"]:
        raise ValueError(
            f"Atom count mismatch: POSCAR={pos['natoms']} vs XDATCAR={xdat['natoms']}"
        )

    if pos["counts"] != xdat["counts"]:
        raise ValueError(
            f"Species count mismatch: POSCAR={pos['counts']} vs XDATCAR={xdat['counts']}"
        )

    write_extxyz(output_path, pos["elements"], pos["sd_flags"], xdat)

    n_ttt = sum(1 for flags in pos["sd_flags"] if all(flags))
    print(f"[OK] Wrote: {output_path}")
    print(f"      Frames : {len(xdat['frames'])}")
    print(f"      Atoms  : {pos['natoms']}")
    print(f"      TTT atoms assigned sd_id: {n_ttt}")
    print("      OVITO expression example: sd_ttt == 1")


if __name__ == "__main__":
    main()