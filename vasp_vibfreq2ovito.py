#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VASP OUTCAR finite-difference vibrational analysis -> OVITO-friendly exports

Features
--------
- Reads a single VASP OUTCAR containing multiple POSITION/TOTAL-FORCE blocks
  from a finite-difference vibrational calculation.
- Supports NFREE = 1 and NFREE = 2 finite differences.
- Builds the Cartesian Hessian from the actual displaced geometries.
- Supports Selective Dynamics masks from POSCAR/CONTCAR and uses them as the
  primary active-DOF mask when available.
- Supports legacy ALLOWED files and legacy row-sum based fixed-DOF detection
  when no Selective Dynamics file is available.
- Handles PBC-aware displacement vectors using minimum-image mapping.
- Optional projection of translational or translational+rotational zero modes.
- Writes OVITO-friendly extxyz mode vectors and animated trajectories.
- Also writes plain XYZ animations and CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

# --- constants ---
AMU_TO_KG = 1.66053886e-27
EV_TO_J = 1.602176634e-19
ANG_TO_M = 1.0e-10
CLIGHT = 2.99792458e8
TWOPI = 2.0 * math.pi


@dataclass
class OutcarData:
    natoms: int
    nfree: int
    lattice: np.ndarray
    counts: List[int]
    symbols: List[str]
    masses_amu: np.ndarray
    positions_blocks: List[np.ndarray]
    forces_blocks: List[np.ndarray]


@dataclass
class AllowedInfo:
    path: str | None
    atom_indices_zero_based: List[int]
    directions: np.ndarray


@dataclass
class SelectiveDynamicsInfo:
    path: str | None
    selective_present: bool
    atom_flags: List[Tuple[bool, bool, bool]]
    directions: np.ndarray


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read a VASP OUTCAR from a finite-difference vibrational calculation and export "
            "vibrational modes to OVITO-friendly formats (extxyz/xyz)."
        )
    )
    p.add_argument("outcar", help="Path to OUTCAR")
    p.add_argument("--prefix", default="ovito_vib", help="Output file prefix")
    p.add_argument("--amplitude", type=float, default=0.30,
                   help="Animation amplitude in Angstrom (default: 0.30)")
    p.add_argument("--nframes", type=int, default=21,
                   help="Number of animation frames per mode (default: 21)")
    p.add_argument("--modes", default="all",
                   help="Mode selection, e.g. all | 1-12 | 1,2,5,8-10")
    p.add_argument("--project", choices=["none", "trans", "transrot"], default="none",
                   help="Project zero modes from the mass-weighted dynamical matrix")
    p.add_argument("--fixed-detector", choices=["legacy_rowsum", "norm"], default="legacy_rowsum",
                   help="Fallback fixed-DOF detector used only when no Selective Dynamics file is available")
    p.add_argument("--fixed-threshold", type=float, default=1e-3,
                   help="Threshold for the fallback fixed-DOF detector (default: 1e-3)")
    p.add_argument("--step-threshold", type=float, default=1e-5,
                   help="Finite-difference step treated as zero if below this value (default: 1e-5)")
    p.add_argument("--allowed-file", default="auto",
                   help=(
                       "ALLOWED file path. Use 'auto' to search next to OUTCAR and in the current "
                       "working directory. Use 'none' to disable ALLOWED handling."
                   ))
    p.add_argument("--sd-file", default="auto",
                   help=(
                       "POSCAR/CONTCAR file containing Selective Dynamics flags. "
                       "Use 'auto' to search next to OUTCAR. Use 'none' to disable SD handling."
                   ))
    p.add_argument("--pbc", default="T T T",
                   help='PBC flags written to extxyz, e.g. "T T T" or "T T F"')
    return p.parse_args()


# -----------------------------------------------------------------------------
# OUTCAR parsing
# -----------------------------------------------------------------------------

def _extract_nfree(lines: Sequence[str]) -> int:
    for line in lines:
        if "NFREE" in line:
            m = re.search(r'NFREE\s*=\s*(-?\d+)', line)
            if m:
                return int(m.group(1))
            toks = line.split()
            for i, tok in enumerate(toks):
                if tok == "NFREE":
                    for j in range(i + 1, min(i + 4, len(toks))):
                        try:
                            return int(float(toks[j]))
                        except ValueError:
                            pass
    raise ValueError("Could not find NFREE in OUTCAR.")


def _extract_ions_per_type(lines: Sequence[str]) -> List[int]:
    pat = re.compile(r'ions per type\s*=\s*(.+)', re.IGNORECASE)
    for line in lines:
        m = pat.search(line)
        if m:
            return [int(x) for x in m.group(1).split()]
    raise ValueError("Could not find 'ions per type' in OUTCAR.")


def _extract_symbols_and_masses(lines: Sequence[str], counts: Sequence[int]) -> Tuple[List[str], np.ndarray]:
    symbols_by_type: List[str] = []
    masses_by_type: List[float] = []

    vrh_pat = re.compile(r'VRHFIN\s*=\s*([A-Za-z]{1,3})\s*:')
    pomass_pat = re.compile(r'POMASS\s*=\s*([0-9.+\-EeDd]+)')

    for line in lines:
        m = vrh_pat.search(line)
        if m and len(symbols_by_type) < len(counts):
            symbols_by_type.append(m.group(1))
        m = pomass_pat.search(line)
        if m and len(masses_by_type) < len(counts):
            masses_by_type.append(float(m.group(1).replace("D", "E").replace("d", "e")))
        if len(symbols_by_type) >= len(counts) and len(masses_by_type) >= len(counts):
            break

    if len(symbols_by_type) != len(counts):
        titel_pat = re.compile(r'TITEL\s*=.*?([A-Z][a-z]?)')
        symbols_by_type = []
        for line in lines:
            m = titel_pat.search(line)
            if m and len(symbols_by_type) < len(counts):
                symbols_by_type.append(m.group(1))
            if len(symbols_by_type) >= len(counts):
                break

    if len(symbols_by_type) != len(counts):
        raise ValueError("Could not extract enough element symbols from OUTCAR.")
    if len(masses_by_type) != len(counts):
        raise ValueError("Could not extract enough POMASS entries from OUTCAR.")

    symbols: List[str] = []
    masses: List[float] = []
    for sym, mass, n in zip(symbols_by_type, masses_by_type, counts):
        symbols.extend([sym] * n)
        masses.extend([mass] * n)
    return symbols, np.array(masses, dtype=float)


def _extract_lattice(lines: Sequence[str]) -> np.ndarray:
    for i, line in enumerate(lines):
        if "direct lattice vectors" in line.lower():
            lat = []
            for j in range(1, 4):
                toks = lines[i + j].split()
                lat.append([float(toks[0]), float(toks[1]), float(toks[2])])
            return np.array(lat, dtype=float)
    raise ValueError("Could not find the 'direct lattice vectors' block in OUTCAR.")


def _extract_position_force_blocks(lines: Sequence[str], natoms: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    positions: List[np.ndarray] = []
    forces: List[np.ndarray] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("POSITION") and "TOTAL-FORCE" in line:
            i += 2
            pos = []
            frc = []
            for _ in range(natoms):
                toks = lines[i].split()
                if len(toks) < 6:
                    raise ValueError("Unexpected POSITION/TOTAL-FORCE block format.")
                pos.append([float(toks[0]), float(toks[1]), float(toks[2])])
                frc.append([float(toks[3]), float(toks[4]), float(toks[5])])
                i += 1
            positions.append(np.array(pos, dtype=float))
            forces.append(np.array(frc, dtype=float))
            continue
        i += 1

    if len(positions) == 0:
        raise ValueError("Could not find any POSITION/TOTAL-FORCE blocks in OUTCAR.")
    return positions, forces


def read_outcar(path: str) -> OutcarData:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()

    counts = _extract_ions_per_type(lines)
    natoms = sum(counts)
    nfree = _extract_nfree(lines)
    lattice = _extract_lattice(lines)
    symbols, masses_amu = _extract_symbols_and_masses(lines, counts)
    positions_blocks, forces_blocks = _extract_position_force_blocks(lines, natoms)

    return OutcarData(
        natoms=natoms,
        nfree=nfree,
        lattice=lattice,
        counts=counts,
        symbols=symbols,
        masses_amu=masses_amu,
        positions_blocks=positions_blocks,
        forces_blocks=forces_blocks,
    )


# -----------------------------------------------------------------------------
# POSCAR / CONTCAR Selective Dynamics handling
# -----------------------------------------------------------------------------

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


def resolve_sd_file(outcar_path: str, sd_file_arg: str) -> str | None:
    token = (sd_file_arg or "auto").strip()
    if token.lower() == "none":
        return None
    if token.lower() != "auto":
        p = Path(token).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Selective Dynamics file not found: {p}")
        return str(p)

    outcar_dir = Path(outcar_path).expanduser().resolve().parent
    candidates = [
        outcar_dir / "POSCAR",
        outcar_dir / "CONTCAR",
        outcar_dir / "POSCAR.vasp",
        outcar_dir / "CONTCAR.vasp",
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return None


def read_selective_dynamics_file(path: str | None, natoms: int) -> SelectiveDynamicsInfo:
    directions = np.ones(3 * natoms, dtype=bool)
    if path is None:
        return SelectiveDynamicsInfo(path=None, selective_present=False, atom_flags=[(True, True, True)] * natoms, directions=directions)

    poscar_path = Path(path)
    lines = _read_nonempty_lines(poscar_path)
    if len(lines) < 8:
        raise ValueError(f"Structure file seems too short: {poscar_path}")

    line5 = lines[5].split()
    if _is_int_list(line5):
        counts = [int(x) for x in line5]
        idx = 6
    else:
        counts = [int(x) for x in lines[6].split()]
        idx = 7

    nat_from_file = sum(counts)
    if nat_from_file != natoms:
        raise ValueError(
            f"Selective Dynamics file atom count ({nat_from_file}) does not match OUTCAR atom count ({natoms})."
        )

    selective = False
    if lines[idx].strip().lower().startswith("s"):
        selective = True
        idx += 1

    coord_mode = lines[idx].strip().lower()
    if not (coord_mode.startswith("d") or coord_mode.startswith("c") or coord_mode.startswith("k")):
        raise ValueError(f"Failed to locate coordinate mode in {poscar_path}: '{lines[idx]}'")
    idx += 1

    coord_lines = lines[idx: idx + natoms]
    if len(coord_lines) != natoms:
        raise ValueError(f"Atom count mismatch while reading {poscar_path}.")

    atom_flags: List[Tuple[bool, bool, bool]] = []
    if selective:
        directions = np.zeros(3 * natoms, dtype=bool)
        for ia, line in enumerate(coord_lines):
            toks = line.split()
            if len(toks) < 6:
                raise ValueError(f"Selective Dynamics line too short in {poscar_path}: '{line}'")
            flags = tuple(tok.upper().startswith("T") for tok in toks[3:6])
            atom_flags.append(flags)
            directions[3 * ia:3 * ia + 3] = np.array(flags, dtype=bool)
    else:
        atom_flags = [(True, True, True)] * natoms
        directions = np.ones(3 * natoms, dtype=bool)

    return SelectiveDynamicsInfo(path=str(poscar_path.resolve()), selective_present=selective, atom_flags=atom_flags, directions=directions)


# -----------------------------------------------------------------------------
# ALLOWED handling (legacy vibFreq.py style)
# -----------------------------------------------------------------------------

def resolve_allowed_file(outcar_path: str, allowed_file_arg: str) -> str | None:
    token = (allowed_file_arg or "auto").strip()
    if token.lower() == "none":
        return None
    if token.lower() != "auto":
        p = Path(token).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"ALLOWED file not found: {p}")
        return str(p)

    outcar_dir = Path(outcar_path).expanduser().resolve().parent
    candidates = [
        outcar_dir / "ALLOWED",
        Path.cwd() / "ALLOWED",
    ]
    for c in candidates:
        if c.is_file():
            return str(c.resolve())
    return None


def read_allowed_file(path: str | None, natoms: int) -> AllowedInfo:
    directions = np.ones(3 * natoms, dtype=bool)
    atom_indices: List[int] = []

    if path is None:
        return AllowedInfo(path=None, atom_indices_zero_based=[], directions=directions)

    directions[:] = False
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        tokens = []
        for line in fh:
            tokens.extend(line.split())

    for tok in tokens:
        try:
            idx1 = int(tok)
        except ValueError:
            continue
        if 1 <= idx1 <= natoms:
            idx0 = idx1 - 1
            if idx0 not in atom_indices:
                atom_indices.append(idx0)
                directions[3 * idx0: 3 * idx0 + 3] = True

    atom_indices.sort()
    return AllowedInfo(path=path, atom_indices_zero_based=atom_indices, directions=directions)


# -----------------------------------------------------------------------------
# Linear algebra helpers
# -----------------------------------------------------------------------------

def cart_to_frac(cart: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    return cart @ np.linalg.inv(lattice)


def frac_to_cart(frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    return frac @ lattice


def minimum_image_delta_cart(cart_a: np.ndarray, cart_b: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    fa = cart_to_frac(cart_a, lattice)
    fb = cart_to_frac(cart_b, lattice)
    df = fa - fb
    df -= np.round(df)
    return frac_to_cart(df, lattice)


def build_hessian(
    out: OutcarData,
    step_directions: np.ndarray,
    step_threshold: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    nat = out.natoms
    ndof = 3 * nat
    nblocks = len(out.positions_blocks)
    if nblocks != len(out.forces_blocks):
        raise ValueError("The number of position and force blocks does not match.")

    step_directions = np.asarray(step_directions, dtype=float).reshape(ndof)
    ref_pos = out.positions_blocks[0]
    ref_force = out.forces_blocks[0].reshape(-1)

    if out.nfree == 1:
        nrows = nblocks - 1
        if nrows <= 0:
            raise ValueError("NFREE=1 but there are not enough displacement blocks.")
        B = np.zeros((nrows, ndof), dtype=float)
        G = np.zeros((nrows, ndof), dtype=float)
        for i in range(1, nblocks):
            disp = minimum_image_delta_cart(out.positions_blocks[i], ref_pos, out.lattice).reshape(-1)
            step = float(np.sqrt(np.sum((disp * step_directions) ** 2)))
            if step > step_threshold:
                B[i - 1, :] = disp / step
                G[i - 1, :] = -(out.forces_blocks[i].reshape(-1) - ref_force) / step
    elif out.nfree == 2:
        if (nblocks - 1) % 2 != 0:
            raise ValueError("NFREE=2 but the number of displacement blocks is not an even pair count.")
        nrows = (nblocks - 1) // 2
        B = np.zeros((nrows, ndof), dtype=float)
        G = np.zeros((nrows, ndof), dtype=float)
        for i in range(nrows):
            ip = 2 * i + 1
            im = 2 * i + 2
            disp_pm = minimum_image_delta_cart(out.positions_blocks[ip], out.positions_blocks[im], out.lattice).reshape(-1)
            step = float(np.sqrt(np.sum((disp_pm * step_directions) ** 2)))
            if step > step_threshold:
                B[i, :] = disp_pm / step
                G[i, :] = (out.forces_blocks[im].reshape(-1) - out.forces_blocks[ip].reshape(-1)) / step
    else:
        raise ValueError(f"Unsupported NFREE={out.nfree}. Only 1 or 2 is supported at present.")

    H = B.T @ G
    H = 0.5 * (H + H.T)
    return H, ref_pos


def detect_fixed_dofs(H: np.ndarray, threshold: float, detector: str) -> np.ndarray:
    detector = detector.strip().lower()
    if detector == "legacy_rowsum":
        metric = np.sum(np.abs(H), axis=1)
        return metric < threshold
    if detector == "norm":
        metric = np.linalg.norm(H, axis=1)
        return metric < threshold
    raise ValueError(f"Unsupported fixed-DOF detector: {detector}")


def reduce_by_fixed_dofs(H: np.ndarray, fixed_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    active = ~fixed_mask
    return H[np.ix_(active, active)], active


def build_mass_vector(masses_amu: np.ndarray) -> np.ndarray:
    masses_kg = masses_amu * AMU_TO_KG
    return np.repeat(masses_kg, 3)


def mass_weight_hessian(H_eva2: np.ndarray, masses_amu: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    masses_kg = build_mass_vector(masses_amu)[active_mask]
    Minv_sqrt = np.diag(1.0 / np.sqrt(masses_kg))
    H_si = H_eva2 * EV_TO_J / (ANG_TO_M ** 2)
    D = Minv_sqrt @ H_si @ Minv_sqrt
    D = 0.5 * (D + D.T)
    return D


def orthonormalize_rows(vectors: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    rows = []
    for v in vectors:
        w = v.copy().astype(float)
        for q in rows:
            w -= np.dot(w, q) * q
        n = np.linalg.norm(w)
        if n > tol:
            rows.append(w / n)
    if not rows:
        return np.zeros((0, vectors.shape[1]), dtype=float)
    return np.array(rows, dtype=float)


def build_zero_mode_basis(project: str, positions: np.ndarray, masses_amu: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    ndof = 3 * len(masses_amu)
    basis = []

    if project in ("trans", "transrot"):
        for alpha in range(3):
            v = np.zeros(ndof, dtype=float)
            for i, m in enumerate(masses_amu * AMU_TO_KG):
                v[3 * i + alpha] = math.sqrt(m)
            basis.append(v[active_mask])

    if project == "transrot":
        com = np.average(positions, axis=0, weights=masses_amu)
        r = positions - com[None, :]
        for axis in range(3):
            v = np.zeros(ndof, dtype=float)
            for i, m in enumerate(masses_amu * AMU_TO_KG):
                rx, ry, rz = r[i]
                if axis == 0:
                    comp = np.array([0.0, -rz, ry])
                elif axis == 1:
                    comp = np.array([rz, 0.0, -rx])
                else:
                    comp = np.array([-ry, rx, 0.0])
                v[3 * i:3 * i + 3] = math.sqrt(m) * comp
            basis.append(v[active_mask])

    if not basis:
        return np.zeros((0, int(np.sum(active_mask))), dtype=float)
    return orthonormalize_rows(np.array(basis, dtype=float))


def project_dynamical_matrix(D: np.ndarray, zero_basis: np.ndarray) -> np.ndarray:
    if zero_basis.size == 0:
        return D
    n = D.shape[0]
    P = np.eye(n) - zero_basis.T @ zero_basis
    Dp = P @ D @ P
    Dp = 0.5 * (Dp + Dp.T)
    return Dp


def expand_eigenvectors_to_full_cart(
    evec_active: np.ndarray,
    active_mask: np.ndarray,
    masses_amu: np.ndarray,
    mass_weighted_output: bool = False,
) -> np.ndarray:
    nmodes, _ = evec_active.shape
    ndof = len(active_mask)
    out = np.zeros((nmodes, ndof), dtype=float)
    out[:, active_mask] = evec_active

    if mass_weighted_output:
        return out

    masses = build_mass_vector(masses_amu)
    conv = np.zeros(ndof, dtype=float)
    conv[active_mask] = 1.0 / np.sqrt(masses[active_mask])
    return out * conv[None, :]


def normalize_modes_per_mode(modes: np.ndarray) -> np.ndarray:
    out = modes.copy()
    for i in range(out.shape[0]):
        n = np.linalg.norm(out[i])
        if n > 0:
            out[i] /= n
    return out


def eval_to_freq_cm1(lam: float) -> float:
    if lam >= 0:
        return math.sqrt(lam) / (TWOPI * CLIGHT) * 1.0e-2
    return -math.sqrt(abs(lam)) / (TWOPI * CLIGHT) * 1.0e-2


def atom_mask_from_dof_mask(active_mask: np.ndarray) -> np.ndarray:
    return np.any(active_mask.reshape(-1, 3), axis=1)


# -----------------------------------------------------------------------------
# Mode selection parsing
# -----------------------------------------------------------------------------

def parse_mode_selection(spec: str, nmodes: int) -> List[int]:
    if spec.strip().lower() == "all":
        return list(range(nmodes))

    chosen = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            ia = int(a)
            ib = int(b)
            if ia > ib:
                ia, ib = ib, ia
            for x in range(ia, ib + 1):
                if 1 <= x <= nmodes:
                    chosen.add(x - 1)
        else:
            x = int(chunk)
            if 1 <= x <= nmodes:
                chosen.add(x - 1)
    return sorted(chosen)


# -----------------------------------------------------------------------------
# Writers
# -----------------------------------------------------------------------------

def lattice_to_extxyz_string(lattice: np.ndarray) -> str:
    a, b, c = lattice
    return (
        f"{a[0]:.10f} {a[1]:.10f} {a[2]:.10f} "
        f"{b[0]:.10f} {b[1]:.10f} {b[2]:.10f} "
        f"{c[0]:.10f} {c[1]:.10f} {c[2]:.10f}"
    )


def write_extxyz_frame(handle, symbols, coords_cart, disp_cart, atom_ids,
                       lattice, mode_id, freq_cm1, frame_id, amplitude, pbc):
    nat = len(symbols)
    handle.write(f"{nat}\n")
    handle.write(
        f'Lattice="{lattice_to_extxyz_string(lattice)}" '
        f'pbc="{pbc}" '
        f'Properties=species:S:1:pos:R:3:disp:R:3:id:I:1 '
        f'Mode={mode_id} Frequency_cm1={freq_cm1:.10f} '
        f'Frame={frame_id} Amplitude={amplitude:.10f}\n'
    )
    for i in range(nat):
        handle.write(
            f"{symbols[i]} "
            f"{coords_cart[i,0]:.10f} {coords_cart[i,1]:.10f} {coords_cart[i,2]:.10f} "
            f"{disp_cart[i,0]:.10f} {disp_cart[i,1]:.10f} {disp_cart[i,2]:.10f} "
            f"{atom_ids[i]}\n"
        )


def write_xyz_frame(handle, symbols, coords_cart, mode_id, freq_cm1, frame_id):
    nat = len(symbols)
    handle.write(f"{nat}\n")
    handle.write(f"mode={mode_id} freq_cm-1={freq_cm1:.10f} frame={frame_id}\n")
    for i in range(nat):
        handle.write(
            f"{symbols[i]} {coords_cart[i,0]:.10f} {coords_cart[i,1]:.10f} {coords_cart[i,2]:.10f}\n"
        )


def write_outputs(prefix: str,
                  symbols: List[str],
                  lattice: np.ndarray,
                  ref_positions: np.ndarray,
                  frequencies_cm1: np.ndarray,
                  modes_cart_norm: np.ndarray,
                  amplitude: float,
                  nframes: int,
                  selected_modes: Sequence[int],
                  pbc: str):
    atom_ids = np.arange(1, len(symbols) + 1, dtype=int)

    with open(f"{prefix}_modes_summary.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mode_index", "frequency_cm1", "is_imaginary"])
        for i, fcm in enumerate(frequencies_cm1, start=1):
            writer.writerow([i, f"{fcm:.10f}", int(fcm < 0.0)])

    with open(f"{prefix}_mode_vectors.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mode", "atom_index", "symbol", "dx", "dy", "dz"])
        for imode in range(modes_cart_norm.shape[0]):
            mode_mat = modes_cart_norm[imode].reshape(len(symbols), 3)
            for ia, sym in enumerate(symbols, start=1):
                dx, dy, dz = mode_mat[ia - 1]
                writer.writerow([imode + 1, ia, sym, f"{dx:.10f}", f"{dy:.10f}", f"{dz:.10f}"])

    for imode in selected_modes:
        mode_id = imode + 1
        freq = frequencies_cm1[imode]
        mode_mat = modes_cart_norm[imode].reshape(len(symbols), 3)

        with open(f"{prefix}_mode_{mode_id:03d}.extxyz", "w") as fvec:
            write_extxyz_frame(
                fvec,
                symbols,
                ref_positions,
                mode_mat,
                atom_ids,
                lattice,
                mode_id,
                freq,
                0,
                1.0,
                pbc,
            )

        with open(f"{prefix}_mode_{mode_id:03d}_anim.extxyz", "w") as fanim:
            for iframe in range(nframes):
                phase = TWOPI * iframe / float(nframes)
                scale = amplitude * math.sin(phase)
                disp = scale * mode_mat
                coords_now = ref_positions + disp
                write_extxyz_frame(
                    fanim,
                    symbols,
                    coords_now,
                    disp,
                    atom_ids,
                    lattice,
                    mode_id,
                    freq,
                    iframe,
                    amplitude,
                    pbc,
                )

        with open(f"{prefix}_mode_{mode_id:03d}_anim.xyz", "w") as fxyz:
            for iframe in range(nframes):
                phase = TWOPI * iframe / float(nframes)
                scale = amplitude * math.sin(phase)
                coords_now = ref_positions + scale * mode_mat
                write_xyz_frame(fxyz, symbols, coords_now, mode_id, freq, iframe)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    out = read_outcar(args.outcar)

    sd_path = resolve_sd_file(args.outcar, args.sd_file)
    sd_info = read_selective_dynamics_file(sd_path, out.natoms)

    allowed_path = resolve_allowed_file(args.outcar, args.allowed_file)
    allowed_info = read_allowed_file(allowed_path, out.natoms)

    # Step-length evaluation directions:
    #   - Prefer Selective Dynamics directions when available.
    #   - Intersect with ALLOWED if ALLOWED is also present.
    if sd_info.selective_present:
        step_directions = sd_info.directions.copy()
        mask_source = f"Selective Dynamics from {sd_info.path}"
        if allowed_info.path is not None:
            step_directions = np.logical_and(step_directions, allowed_info.directions)
            mask_source += f" + ALLOWED from {allowed_info.path}"
        fixed_mask = ~step_directions
    else:
        step_directions = allowed_info.directions.copy()
        H_probe, ref_positions = build_hessian(out, step_directions=step_directions, step_threshold=args.step_threshold)
        fixed_mask = detect_fixed_dofs(H_probe, args.fixed_threshold, args.fixed_detector)
        mask_source = f"fallback detector: {args.fixed_detector}"
        H = H_probe

    if sd_info.selective_present:
        H, ref_positions = build_hessian(out, step_directions=step_directions, step_threshold=args.step_threshold)

    H_red, active_mask = reduce_by_fixed_dofs(H, fixed_mask)

    if H_red.size == 0:
        raise RuntimeError(
            "All DOFs were classified as fixed. Check the Selective Dynamics file, ALLOWED file, and threshold settings."
        )

    D = mass_weight_hessian(H_red, out.masses_amu, active_mask)
    zero_basis = build_zero_mode_basis(args.project, ref_positions, out.masses_amu, active_mask)
    Dp = project_dynamical_matrix(D, zero_basis)

    evals, evecs = np.linalg.eigh(Dp)
    evals = np.real_if_close(evals)
    evecs = np.real_if_close(evecs)
    evecs = evecs.T.copy()

    freqs_cm1 = np.array([eval_to_freq_cm1(x) for x in evals], dtype=float)
    modes_cart = expand_eigenvectors_to_full_cart(evecs, active_mask, out.masses_amu, mass_weighted_output=False)
    modes_cart_norm = normalize_modes_per_mode(modes_cart)
    selected_modes = parse_mode_selection(args.modes, len(freqs_cm1))

    write_outputs(
        prefix=args.prefix,
        symbols=out.symbols,
        lattice=out.lattice,
        ref_positions=ref_positions,
        frequencies_cm1=freqs_cm1,
        modes_cart_norm=modes_cart_norm,
        amplitude=args.amplitude,
        nframes=args.nframes,
        selected_modes=selected_modes,
        pbc=args.pbc,
    )

    active_atom_mask = atom_mask_from_dof_mask(active_mask)
    n_active_dofs = int(np.sum(active_mask))
    n_fixed_dofs = int(np.sum(fixed_mask))

    print(f"[Done] OUTCAR: {args.outcar}")
    print(f"[Info] Number of atoms: {out.natoms}")
    print(f"[Info] NFREE: {out.nfree}")
    print(f"[Info] Number of POSITION/TOTAL-FORCE blocks: {len(out.positions_blocks)}")
    print(f"[Info] Selective Dynamics file: {sd_info.path if sd_info.path else 'not used'}")
    print(f"[Info] Selective Dynamics present: {'yes' if sd_info.selective_present else 'no'}")
    print(f"[Info] ALLOWED file: {allowed_info.path if allowed_info.path else 'not used'}")
    print(f"[Info] Active-mask source: {mask_source}")
    print(f"[Info] Fixed-DOF detector: {args.fixed_detector}")
    print(f"[Info] Fixed-DOF threshold: {args.fixed_threshold:.6g}")
    print(f"[Info] Step threshold: {args.step_threshold:.6g}")
    print(f"[Info] Number of fixed DOFs: {n_fixed_dofs} / {3 * out.natoms}")
    print(f"[Info] Number of active DOFs: {n_active_dofs}")
    print(f"[Info] Number of active atoms: {int(np.sum(active_atom_mask))}")
    print(f"[Info] Zero-mode projection: {args.project}")
    print(f"[Info] Output prefix: {args.prefix}")
    print()
    print(" mode    freq(cm^-1)   imag")
    print("-----  ------------   ----")
    for i, fcm in enumerate(freqs_cm1, start=1):
        print(f"{i:5d}  {fcm:12.6f}   {'Y' if fcm < 0 else 'N'}")


if __name__ == "__main__":
    main()
