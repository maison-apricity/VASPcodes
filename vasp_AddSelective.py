#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from pathlib import Path

# =========================
# User settings
# =========================
INPUT_POSCAR = r"C:\Users\Choi\Downloads\Workspace\GO\HBNH_alphaB_T1T2\POSCAR_nonselec"

# VASP-safe output
OUTPUT_POSCAR = r"C:\Users\Choi\Downloads\Workspace\GO\HBNH_alphaB_T1T2\POSCAR"

# OVITO visualization output
OUTPUT_OVITO = r"C:\Users\Choi\Downloads\Workspace\GO\HBNH_alphaB_T1T2\POSCAR.extxyz"

# Optional text report
OUTPUT_TTT_REPORT = r"C:\Users\Choi\Downloads\Workspace\GO\HBNH_alphaB_T1T2\POSCAR_TTT.txt"

# OVITO ID offset
# 0  -> OVITO id == zero-based index
# 1  -> OVITO id == one-based index
OVITO_ID_OFFSET = 0

# Zero-based atom indices
TARGET_INDICES = {
}


def is_integer_line(tokens):
    try:
        for t in tokens:
            int(t)
        return True
    except ValueError:
        return False


def parse_lattice_lines(lattice_lines):
    lattice = []
    for i, line in enumerate(lattice_lines, start=1):
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid lattice line {i}: {line}")
        lattice.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return lattice


def det3(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def get_effective_scale(scale_line, lattice_raw):
    parts = scale_line.split()
    if len(parts) != 1:
        raise ValueError(f"Unexpected scale line format: {scale_line}")

    s = float(parts[0])
    if s == 0.0:
        raise ValueError("POSCAR scale factor cannot be zero.")

    # Standard POSCAR: positive scalar
    if s > 0.0:
        return s

    # Negative scale in POSCAR means target cell volume
    target_volume = abs(s)
    raw_volume = abs(det3(lattice_raw))
    if raw_volume == 0.0:
        raise ValueError("Raw lattice volume is zero; invalid lattice vectors.")

    return (target_volume / raw_volume) ** (1.0 / 3.0)


def scale_lattice(lattice_raw, scale_factor):
    return [
        [scale_factor * x for x in lattice_raw[0]],
        [scale_factor * x for x in lattice_raw[1]],
        [scale_factor * x for x in lattice_raw[2]],
    ]


def frac_to_cart(frac, lattice_cart):
    # r = f1*a1 + f2*a2 + f3*a3
    x = frac[0] * lattice_cart[0][0] + frac[1] * lattice_cart[1][0] + frac[2] * lattice_cart[2][0]
    y = frac[0] * lattice_cart[0][1] + frac[1] * lattice_cart[1][1] + frac[2] * lattice_cart[2][1]
    z = frac[0] * lattice_cart[0][2] + frac[1] * lattice_cart[1][2] + frac[2] * lattice_cart[2][2]
    return [x, y, z]


def parse_xyz(parts, atom_index):
    if len(parts) < 3:
        raise ValueError(f"Invalid coordinate line format for atom index {atom_index}: {' '.join(parts)}")
    return [float(parts[0]), float(parts[1]), float(parts[2])]


def build_species_per_atom(species_line, counts):
    if species_line is None:
        species_names = [f"Type{i+1}" for i in range(len(counts))]
    else:
        species_names = species_line.split()
        if len(species_names) != len(counts):
            raise ValueError(
                f"Species/count mismatch: species={len(species_names)}, counts={len(counts)}"
            )

    out = []
    for sp, n in zip(species_names, counts):
        out.extend([sp] * n)
    return out


def main():
    lines = Path(INPUT_POSCAR).read_text(encoding="utf-8").splitlines()
    if len(lines) < 8:
        raise ValueError("POSCAR format appears too short.")

    # POSCAR header
    comment = lines[0]
    scale = lines[1]
    lattice_lines = lines[2:5]
    lattice_raw = parse_lattice_lines(lattice_lines)

    scale_factor = get_effective_scale(scale, lattice_raw)
    lattice_cart = scale_lattice(lattice_raw, scale_factor)

    idx = 5

    # Detect whether the file uses VASP 5 style
    tokens = lines[idx].split()
    if is_integer_line(tokens):
        # VASP 4 style: counts only
        species_line = None
        counts_line = lines[idx]
        counts = [int(x) for x in tokens]
        idx += 1
    else:
        # VASP 5 style: species then counts
        species_line = lines[idx]
        idx += 1
        counts_line = lines[idx]
        counts = [int(x) for x in counts_line.split()]
        idx += 1

    natoms = sum(counts)
    species_per_atom = build_species_per_atom(species_line, counts)

    # Check whether Selective dynamics is present
    selective_present = False
    if idx < len(lines) and lines[idx].strip().lower().startswith("s"):
        selective_present = True
        idx += 1

    # Coordinate mode
    if idx >= len(lines):
        raise ValueError("Could not find the coordinate mode line (Direct/Cartesian).")
    coord_mode = lines[idx]
    idx += 1

    coord_mode_lower = coord_mode.strip().lower()
    if not (
        coord_mode_lower.startswith("d")
        or coord_mode_lower.startswith("c")
        or coord_mode_lower.startswith("k")
    ):
        raise ValueError(f"Unexpected coordinate mode: {coord_mode}")

    # Coordinate lines
    coord_lines = lines[idx:idx + natoms]
    if len(coord_lines) != natoms:
        raise ValueError(
            f"Number of atoms ({natoms}) does not match the number of coordinate lines ({len(coord_lines)})."
        )

    max_index = natoms - 1
    bad = sorted(i for i in TARGET_INDICES if i < 0 or i > max_index)
    if bad:
        raise IndexError(
            f"Some zero-based indices are out of the valid range. "
            f"Allowed range: 0 ~ {max_index}, Problematic indices: {bad[:20]}"
            + (" ..." if len(bad) > 20 else "")
        )

    new_coord_lines = []
    ovito_atom_lines = []
    ttt_report_lines = []
    ttt_count = 0

    for i, line in enumerate(coord_lines):
        parts = line.split()
        xyz_raw = parse_xyz(parts, i)
        xyz_tokens = parts[:3]

        # If SD flags already exist, replace them and keep only trailing annotations.
        if selective_present:
            tail = parts[6:] if len(parts) >= 6 else []
        else:
            tail = parts[3:] if len(parts) > 3 else []

        selected = i in TARGET_INDICES
        flags = ["T", "T", "T"] if selected else ["F", "F", "F"]

        rebuilt = xyz_tokens + flags + tail
        new_coord_lines.append(" ".join(rebuilt))

        # Cartesian coordinates for OVITO output
        if coord_mode_lower.startswith("d"):
            cart = frac_to_cart(xyz_raw, lattice_cart)
        else:
            # POSCAR Cartesian coordinates are also scaled by the POSCAR scale factor
            cart = [scale_factor * xyz_raw[0], scale_factor * xyz_raw[1], scale_factor * xyz_raw[2]]

        atom_id = i + OVITO_ID_OFFSET
        sp = species_per_atom[i]
        selected_int = 1 if selected else 0
        sd_label = "TTT" if selected else "FFF"

        ovito_atom_lines.append(
            f"{sp} "
            f"{cart[0]:.16f} {cart[1]:.16f} {cart[2]:.16f} "
            f"{atom_id} {i} {selected_int} {sd_label}"
        )

        if selected:
            ttt_count += 1
            ttt_report_lines.append(
                f"{i}\t{atom_id}\t{sp}\t{cart[0]:.16f}\t{cart[1]:.16f}\t{cart[2]:.16f}\tTTT"
            )

    # -------------------------
    # 1) VASP-safe POSCAR output
    # -------------------------
    out = []
    out.append(comment)
    out.append(scale)
    out.extend(lattice_lines)
    if species_line is not None:
        out.append(species_line)
    out.append(counts_line)
    out.append("Selective dynamics")
    out.append(coord_mode)
    out.extend(new_coord_lines)

    Path(OUTPUT_POSCAR).write_text("\n".join(out) + "\n", encoding="utf-8")

    # -------------------------
    # 2) OVITO extxyz output
    # -------------------------
    lattice_flat = " ".join(
        f"{x:.16f}"
        for row in lattice_cart
        for x in row
    )

    ovito_header = (
        f'Lattice="{lattice_flat}" '
        f'Properties=species:S:1:pos:R:3:id:I:1:index0:I:1:selected:I:1:sd_label:S:1 '
        f'pbc="T T T"'
    )

    ovito_text = [str(natoms), ovito_header]
    ovito_text.extend(ovito_atom_lines)
    Path(OUTPUT_OVITO).write_text("\n".join(ovito_text) + "\n", encoding="utf-8")

    # -------------------------
    # 3) Text report
    # -------------------------
    report = [
        "# index0\tid\tspecies\tx(Ang)\ty(Ang)\tz(Ang)\tflags",
        *ttt_report_lines
    ]
    Path(OUTPUT_TTT_REPORT).write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"[Done] Input file              : {INPUT_POSCAR}")
    print(f"[Done] VASP-safe output        : {OUTPUT_POSCAR}")
    print(f"[Done] OVITO visualization     : {OUTPUT_OVITO}")
    print(f"[Done] TTT report              : {OUTPUT_TTT_REPORT}")
    print(f"[Info] Total number of atoms   : {natoms}")
    print(f"[Info] T T T Number of atoms   : {ttt_count}")
    print(f"[Info] F F F Number of atoms   : {natoms - ttt_count}")
    print(f"[Info] Maximum zero-based index: {max_index}")
    print(f"[Info] OVITO ID offset         : {OVITO_ID_OFFSET}")


if __name__ == "__main__":
    main()