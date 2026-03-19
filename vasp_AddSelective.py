#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

# =========================
# User settings
# =========================
INPUT_POSCAR = r"./POSCAR"
OUTPUT_POSCAR = r"./POSCAR_modified"

# Zero-based atom indices
TARGET_INDICES = {
    176, 181, 182, 183
}


def is_integer_line(tokens):
    try:
        for t in tokens:
            int(t)
        return True
    except ValueError:
        return False


def main():
    lines = Path(INPUT_POSCAR).read_text(encoding="utf-8").splitlines()
    if len(lines) < 8:
        raise ValueError("POSCAR format appears too short.")

    # POSCAR header
    comment = lines[0]
    scale = lines[1]
    lattice = lines[2:5]

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
    if not (coord_mode_lower.startswith("d") or coord_mode_lower.startswith("c") or coord_mode_lower.startswith("k")):
        raise ValueError(f"Unexpected coordinate mode: {coord_mode}")

    # Coordinate lines
    coord_lines = lines[idx:idx + natoms]
    if len(coord_lines) != natoms:
        raise ValueError(f"Number of atoms ({natoms}) does not match the number of coordinate lines ({len(coord_lines)}).")

    max_index = natoms - 1
    bad = sorted(i for i in TARGET_INDICES if i < 0 or i > max_index)
    if bad:
        raise IndexError(
            f"Some zero-based indices are out of the valid range. "
            f"Allowed range: 0 ~ {max_index}, Problematic indices: {bad[:20]}"
            + (" ..." if len(bad) > 20 else "")
        )

    new_coord_lines = []
    for i, line in enumerate(coord_lines):
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(f"Invalid coordinate line format for atom index {i}: {line}")

        xyz = parts[:3]

        # If SD flags already exist, replace them and keep only trailing annotations.
        if selective_present:
            tail = parts[6:] if len(parts) >= 6 else []
        else:
            tail = parts[3:] if len(parts) > 3 else []

        flags = ["T", "T", "T"] if i in TARGET_INDICES else ["F", "F", "F"]

        rebuilt = xyz + flags + tail
        new_coord_lines.append(" ".join(rebuilt))

    # Output reconstruction
    out = []
    out.append(comment)
    out.append(scale)
    out.extend(lattice)
    if species_line is not None:
        out.append(species_line)
    out.append(counts_line)
    out.append("Selective dynamics")
    out.append(coord_mode)
    out.extend(new_coord_lines)

    Path(OUTPUT_POSCAR).write_text("\n".join(out) + "\n", encoding="utf-8")

    print(f"[Done] Input file : {INPUT_POSCAR}")
    print(f"[Done] Output file : {OUTPUT_POSCAR}")
    print(f"[Info] Total number of atoms : {natoms}")
    print(f"[Info] T T T Number of atoms : {len(TARGET_INDICES)}")
    print(f"[Info] F F F Number of atoms : {natoms - len(TARGET_INDICES)}")
    print(f"[Info] Maximum zero-based index : {max_index}")


if __name__ == "__main__":
    main()