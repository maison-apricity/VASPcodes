#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VASP OUTCAR finite-difference vibrational analysis -> OVITO-friendly exports

Features
--------
- Reads a single VASP OUTCAR containing multiple POSITION/TOTAL-FORCE blocks
  from a finite-difference vibrational calculation.
- Supports NFREE = 1 and NFREE = 2 finite differences.
- Builds the Cartesian Hessian using the same broad idea as the legacy vibFreq.py:
  arbitrary displacement directions are detected from the actual displaced geometries,
  then force derivatives are back-projected to the Cartesian Hessian.
- Handles PBC-aware displacement vectors using minimum-image mapping.
- Optional projection of translational or translational+rotational zero modes.
- Writes OVITO-friendly extxyz mode vectors and animated trajectories.
- Also writes plain XYZ animations and a CSV summary.

Notes
-----
- For slabs / periodic systems, keep --project none (default).
- For isolated molecules, --project transrot is often useful.
- The script does not depend on the old helper modules (takeinp, mymath, ...).

Examples
--------
python vib_outcar_to_ovito.py OUTCAR --prefix vib
python vib_outcar_to_ovito.py OUTCAR --prefix vib --amplitude 0.25 --nframes 31
python vib_outcar_to_ovito.py OUTCAR --modes 1-12 --project transrot
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dataclasses import dataclass
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
    lattice: np.ndarray              # (3,3) Cartesian lattice vectors in Angstrom
    counts: List[int]
    symbols: List[str]               # length natoms
    masses_amu: np.ndarray           # length natoms
    positions_blocks: List[np.ndarray]  # each (natoms,3), Angstrom
    forces_blocks: List[np.ndarray]     # each (natoms,3), eV/Angstrom


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
    p.add_argument("--fixed-threshold", type=float, default=1e-8,
                   help="DOF treated as fixed if Hessian row norm < threshold (default: 1e-8)")
    p.add_argument("--pbc", default="T T T",
                   help='PBC flags written to extxyz, e.g. "T T T" or "T T F"')
    return p.parse_args()


# -----------------------------------------------------------------------------
# OUTCAR parsing
# -----------------------------------------------------------------------------

def _first_int_from_line(line: str) -> int | None:
    m = re.search(r'(-?\d+)', line)
    return int(m.group(1)) if m else None


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
    raise ValueError("OUTCAR에서 NFREE를 찾지 못했습니다.")


def _extract_ions_per_type(lines: Sequence[str]) -> List[int]:
    pat = re.compile(r'ions per type\s*=\s*(.+)', re.IGNORECASE)
    for line in lines:
        m = pat.search(line)
        if m:
            return [int(x) for x in m.group(1).split()]
    raise ValueError("OUTCAR에서 'ions per type'를 찾지 못했습니다.")


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
        # fallback: TITEL lines
        titel_pat = re.compile(r'TITEL\s*=.*?([A-Z][a-z]?)')
        symbols_by_type = []
        for line in lines:
            m = titel_pat.search(line)
            if m and len(symbols_by_type) < len(counts):
                symbols_by_type.append(m.group(1))
            if len(symbols_by_type) >= len(counts):
                break

    if len(symbols_by_type) != len(counts):
        raise ValueError("OUTCAR에서 원소 기호(VRHFIN/TITEL)를 충분히 찾지 못했습니다.")
    if len(masses_by_type) != len(counts):
        raise ValueError("OUTCAR에서 POMASS를 충분히 찾지 못했습니다.")

    symbols: List[str] = []
    masses: List[float] = []
    for sym, mass, n in zip(symbols_by_type, masses_by_type, counts):
        symbols.extend([sym] * n)
        masses.extend([mass] * n)
    return symbols, np.array(masses, dtype=float)



def _extract_lattice(lines: Sequence[str]) -> np.ndarray:
    for i, line in enumerate(lines):
        if "direct lattice vectors" in line.lower():
            if i + 3 >= len(lines):
                break
            lat = []
            for j in range(1, 4):
                toks = lines[i + j].split()
                if len(toks) < 3:
                    raise ValueError("격자벡터 블록 파싱 실패")
                lat.append([float(toks[0]), float(toks[1]), float(toks[2])])
            return np.array(lat, dtype=float)
    raise ValueError("OUTCAR에서 'direct lattice vectors' 블록을 찾지 못했습니다.")



def _extract_position_force_blocks(lines: Sequence[str], natoms: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    positions: List[np.ndarray] = []
    forces: List[np.ndarray] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("POSITION") and "TOTAL-FORCE" in line:
            # 다음 줄은 보통 구분선
            i += 2
            pos = []
            frc = []
            for _ in range(natoms):
                if i >= len(lines):
                    raise ValueError("POSITION/TOTAL-FORCE 블록 도중 EOF에 도달했습니다.")
                toks = lines[i].split()
                if len(toks) < 6:
                    raise ValueError("POSITION/TOTAL-FORCE 블록 형식이 예상과 다릅니다.")
                pos.append([float(toks[0]), float(toks[1]), float(toks[2])])
                frc.append([float(toks[3]), float(toks[4]), float(toks[5])])
                i += 1
            positions.append(np.array(pos, dtype=float))
            forces.append(np.array(frc, dtype=float))
            continue
        i += 1

    if len(positions) == 0:
        raise ValueError("OUTCAR에서 POSITION/TOTAL-FORCE 블록을 하나도 찾지 못했습니다.")
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
# Linear algebra helpers
# -----------------------------------------------------------------------------

def cart_to_frac(cart: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    # cart(n,3) = frac(n,3) @ lattice(3,3)
    return cart @ np.linalg.inv(lattice)


def frac_to_cart(frac: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    return frac @ lattice


def minimum_image_delta_cart(cart_a: np.ndarray, cart_b: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Return cart_a - cart_b with minimum-image convention in Cartesian coordinates."""
    fa = cart_to_frac(cart_a, lattice)
    fb = cart_to_frac(cart_b, lattice)
    df = fa - fb
    df -= np.round(df)
    return frac_to_cart(df, lattice)


def build_hessian(out: OutcarData) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build Cartesian Hessian H (eV/Ang^2) using the same broad strategy as legacy vibFreq.py:
    arbitrary displacement directions are inferred from the actual displaced geometries,
    then back-projected to Cartesian coordinates.

    Returns
    -------
    H : (3N, 3N)
    ref_positions : (N,3) Cartesian reference geometry in Angstrom
    """
    nat = out.natoms
    ndof = 3 * nat
    nblocks = len(out.positions_blocks)
    if nblocks != len(out.forces_blocks):
        raise ValueError("positions/forces 블록 개수가 일치하지 않습니다.")

    ref_pos = out.positions_blocks[0]
    ref_force = out.forces_blocks[0].reshape(-1)

    if out.nfree == 1:
        nrows = nblocks - 1
        if nrows <= 0:
            raise ValueError("NFREE=1인데 변위 블록이 부족합니다.")
        B = np.zeros((nrows, ndof), dtype=float)
        G = np.zeros((nrows, ndof), dtype=float)
        for i in range(1, nblocks):
            disp = minimum_image_delta_cart(out.positions_blocks[i], ref_pos, out.lattice).reshape(-1)
            step = np.linalg.norm(disp)
            if step < 1e-12:
                raise ValueError(f"변위 크기가 너무 작습니다. block={i}")
            B[i - 1, :] = disp / step
            G[i - 1, :] = -(out.forces_blocks[i].reshape(-1) - ref_force) / step
    elif out.nfree == 2:
        if (nblocks - 1) % 2 != 0:
            raise ValueError("NFREE=2인데 변위 블록 수가 짝수쌍이 아닙니다.")
        nrows = (nblocks - 1) // 2
        B = np.zeros((nrows, ndof), dtype=float)
        G = np.zeros((nrows, ndof), dtype=float)
        for i in range(nrows):
            ip = 2 * i + 1
            im = 2 * i + 2
            disp_pm = minimum_image_delta_cart(out.positions_blocks[ip], out.positions_blocks[im], out.lattice).reshape(-1)
            step = np.linalg.norm(disp_pm)
            if step < 1e-12:
                raise ValueError(f"중심차분 변위 크기가 너무 작습니다. pair={i}")
            B[i, :] = disp_pm / step
            G[i, :] = (out.forces_blocks[im].reshape(-1) - out.forces_blocks[ip].reshape(-1)) / step
    else:
        raise ValueError(f"지원하지 않는 NFREE={out.nfree}. 현재는 1 또는 2만 지원합니다.")

    H = B.T @ G
    H = 0.5 * (H + H.T)
    return H, ref_pos



def detect_fixed_dofs(H: np.ndarray, threshold: float) -> np.ndarray:
    row_norm = np.linalg.norm(H, axis=1)
    return row_norm < threshold



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
    masses = build_mass_vector(masses_amu)

    basis = []

    # translational modes in mass-weighted coordinate basis
    if project in ("trans", "transrot"):
        for alpha in range(3):
            v = np.zeros(ndof, dtype=float)
            for i, m in enumerate(masses_amu * AMU_TO_KG):
                v[3 * i + alpha] = math.sqrt(m)
            basis.append(v[active_mask])

    if project == "transrot":
        # rotational modes about COM for isolated molecules; not recommended for slabs/periodic solids
        com = np.average(positions, axis=0, weights=masses_amu)
        r = positions - com[None, :]
        # rotation around x, y, z in mass-weighted coordinates
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



def expand_eigenvectors_to_full_cart(evec_active: np.ndarray, active_mask: np.ndarray, masses_amu: np.ndarray,
                                     mass_weighted_output: bool = False) -> np.ndarray:
    """
    Convert active-space eigenvectors back to full 3N Cartesian arrays.

    Parameters
    ----------
    evec_active : (nmodes, nactive) mass-weighted dynamical-matrix eigenvectors
    active_mask : length 3N
    mass_weighted_output :
        False -> convert to Cartesian displacement-like mode by M^{-1/2} q
        True  -> keep mass-weighted coordinates expanded to full 3N
    """
    nmodes, nactive = evec_active.shape
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

    # summary CSV
    with open(f"{prefix}_modes_summary.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["mode_index", "frequency_cm1", "is_imaginary"])
        for i, fcm in enumerate(frequencies_cm1, start=1):
            writer.writerow([i, f"{fcm:.10f}", int(fcm < 0.0)])

    # all modes into one CSV too
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

        # 1-frame vector file for OVITO vector display
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

        # animated extxyz
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

        # plain xyz animation
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
    H, ref_positions = build_hessian(out)

    fixed_mask = detect_fixed_dofs(H, args.fixed_threshold)
    H_red, active_mask = reduce_by_fixed_dofs(H, fixed_mask)

    D = mass_weight_hessian(H_red, out.masses_amu, active_mask)

    zero_basis = build_zero_mode_basis(args.project, ref_positions, out.masses_amu, active_mask)
    Dp = project_dynamical_matrix(D, zero_basis)

    evals, evecs = np.linalg.eigh(Dp)  # ascending
    evals = np.real_if_close(evals)
    evecs = np.real_if_close(evecs)

    # numpy.linalg.eigh returns eigenvectors as columns; convert to (nmodes, nactive)
    evecs = evecs.T.copy()

    freqs_cm1 = np.array([eval_to_freq_cm1(x) for x in evals], dtype=float)

    # Convert to full Cartesian displacement-like mode vectors and normalize per mode
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

    # Console summary
    print(f"[완료] OUTCAR: {args.outcar}")
    print(f"[정보] 원자 수: {out.natoms}")
    print(f"[정보] NFREE: {out.nfree}")
    print(f"[정보] POSITION/TOTAL-FORCE 블록 수: {len(out.positions_blocks)}")
    print(f"[정보] 고정 DOF 수: {int(np.sum(fixed_mask))} / {3*out.natoms}")
    print(f"[정보] zero-mode projection: {args.project}")
    print(f"[정보] 출력 prefix: {args.prefix}")
    print()
    print(" mode    freq(cm^-1)   imag")
    print("-----  ------------   ----")
    for i, fcm in enumerate(freqs_cm1, start=1):
        print(f"{i:5d}  {fcm:12.6f}   {'Y' if fcm < 0 else 'N'}")


if __name__ == "__main__":
    main()
