#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from ase.io import read
from ase.atoms import Atoms


def _clean_dragdrop_path(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    s = os.path.expanduser(s)
    return s.rstrip("/\\")  # Remove a trailing slash


def _exists_image_dir_any_pad(base: Path, idx: int) -> bool:
    """
    Check whether an image directory exists for the given index,
    allowing flexible zero padding such as 0, 00, 000, and so on.
    """
    for pad in (1, 2, 3, 4, 5, 6):
        name = f"{idx:0{pad}d}" if pad > 1 else str(idx)
        if (base / name).is_dir():
            return True
    return False


def _interpret_nimages(base: Path, start: int, nimages_input: int, mode: str) -> Tuple[int, str]:
    """
    Interpret nimages_input as one of the following:
      - total: total number of image directories
      - neb  : VASP IMAGES value, so total directories = IMAGES + 2
      - auto : if directory (start + nimages_input + 1) exists, treat it as neb; otherwise total
    Return (total_nimages, used_mode_str).
    """
    if nimages_input <= 0:
        raise ValueError("nimages must be a positive integer.")

    mode = (mode or "auto").lower()
    if mode not in ("auto", "total", "neb"):
        raise ValueError(f"count-mode must be one of auto/total/neb: {mode}")

    if mode == "total":
        return nimages_input, "total(forced)"
    if mode == "neb":
        return nimages_input + 2, "neb(forced, total=IMAGES+2)"

    # auto
    probe_idx = start + nimages_input + 1  # Last index if interpreted as neb
    if _exists_image_dir_any_pad(base, probe_idx):
        return nimages_input + 2, "neb(auto, total=IMAGES+2)"
    return nimages_input, "total(auto)"


def _infer_existing_pad(base: Path, start: int, nimages: int) -> Optional[int]:
    candidates = [1, 2, 3, 4, 5, 6]
    existing = {p.name for p in base.iterdir() if p.is_dir()}

    best_pad = None
    best_hits = -1
    for pad in candidates:
        hits = 0
        for i in range(start, start + nimages):
            name = f"{i:0{pad}d}" if pad > 1 else str(i)
            if name in existing:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_pad = pad

    if best_hits <= 0:
        return None
    return best_pad


def _resolve_image_dir(base: Path, idx: int, pad: Optional[int]) -> Path:
    if pad is not None:
        return base / (f"{idx:0{pad}d}" if pad > 1 else str(idx))

    for p in (2, 3, 4, 1):
        d = base / (f"{idx:0{p}d}" if p > 1 else str(idx))
        if d.is_dir():
            return d

    return base / f"{idx:02d}"


def _validate_consistency(images: List[Atoms], tol: float = 1e-8) -> Tuple[bool, List[str]]:
    msgs = []
    ok = True

    ref = images[0]
    ref_n = len(ref)
    ref_syms = ref.get_chemical_symbols()
    ref_cell = np.array(ref.cell.array, dtype=float)

    for k, a in enumerate(images[1:], start=1):
        if len(a) != ref_n:
            ok = False
            msgs.append(f"[Mismatch] frame {k}: natoms {len(a)} != {ref_n}")
        if a.get_chemical_symbols() != ref_syms:
            ok = False
            msgs.append(f"[Mismatch] frame {k}: element species/order differs from the reference frame.")
        cell = np.array(a.cell.array, dtype=float)
        if not np.allclose(cell, ref_cell, atol=tol, rtol=tol):
            msgs.append(f"[Warning] frame {k}: cell differs from the reference frame (for NEB, they are usually identical).")

    return ok, msgs


def _species_and_counts(atoms: Atoms) -> Tuple[List[str], List[int]]:
    syms = atoms.get_chemical_symbols()
    species = []
    counts = []
    if not syms:
        return species, counts

    cur = syms[0]
    cnt = 1
    for s in syms[1:]:
        if s == cur:
            cnt += 1
        else:
            species.append(cur)
            counts.append(cnt)
            cur = s
            cnt = 1
    species.append(cur)
    counts.append(cnt)
    return species, counts


def write_xdatcar(output: Path, images: List[Atoms], comment: str, wrap: bool = True) -> None:
    if not images:
        raise ValueError("images is empty.")

    ref = images[0]
    cell = np.array(ref.cell.array, dtype=float)
    species, counts = _species_and_counts(ref)
    natoms = len(ref)

    with output.open("w", encoding="utf-8") as f:
        f.write(f"{comment}\n")
        f.write("1.0\n")
        for i in range(3):
            f.write(f"{cell[i,0]: .16f} {cell[i,1]: .16f} {cell[i,2]: .16f}\n")
        f.write(" ".join(species) + "\n")
        f.write(" ".join(str(c) for c in counts) + "\n")

        for i, a in enumerate(images, start=1):
            f.write(f"Direct configuration= {i:6d}\n")
            frac = a.get_scaled_positions(wrap=wrap)
            if frac.shape != (natoms, 3):
                raise ValueError(f"frame {i}: scaled_positions shape Mismatch: {frac.shape}")

            for j in range(natoms):
                f.write(f"{frac[j,0]: .16f} {frac[j,1]: .16f} {frac[j,2]: .16f}\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Collect NEB image CONTCAR/POSCAR files into a single XDATCAR using ASE."
    )
    ap.add_argument("--path", type=str, default=None,
                    help="Path to the NEB parent directory (drag-and-drop supported). If omitted, interactive input is used.")
    ap.add_argument("--nimages", type=int, default=None,
                    help=(
                        "Input value for the number of images. In the default count-mode=auto, the script automatically decides whether this refers to the VASP IMAGES value (intermediate images, so total=IMAGES+2) or the total number of folders. Use --count-mode total or --count-mode neb to force the interpretation."
                    ))
    ap.add_argument("--count-mode", type=str, default="auto", choices=["auto", "total", "neb"],
                    help="How to interpret nimages: auto (default), total (total number of folders), or neb (treat as VASP IMAGES and add 2).")
    ap.add_argument("--start", type=int, default=0,
                    help="Starting image index (default: 0). Example: start=0 and total nimages=18 -> 00..17")
    ap.add_argument("--pad", type=int, default=None,
                    help="Zero-padding width for folder names (for example, 2 gives 00, 01, ...). If omitted, the script tries to infer it automatically.")
    ap.add_argument("--contcar", type=str, default="CONTCAR",
                    help="Primary structure filename inside each image directory (default: CONTCAR)")
    ap.add_argument("--fallback", type=str, default="POSCAR",
                    help="Fallback structure filename if CONTCAR is missing (default: POSCAR)")
    ap.add_argument("--output", type=str, default="XDATCAR",
                    help="Output XDATCAR filename (default: XDATCAR)")
    ap.add_argument("--no-wrap", action="store_true",
                    help="Do not wrap fractional coordinates into the [0, 1) interval (default: wrap)")
    ap.add_argument("--check", action="store_true",
                    help="Run consistency checks for atom count, element order, and cell (recommended)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print the actual file used for each image (CONTCAR or POSCAR)")
    return ap.parse_args()


def interactive_fill(args: argparse.Namespace) -> argparse.Namespace:
    if args.path is None:
        p = input("Enter the NEB parent directory path (drag-and-drop supported): ").strip()
        args.path = _clean_dragdrop_path(p)

    if args.nimages is None:
        n = input(
            "Enter the nimages input value.\n"
            "  - With the default --count-mode auto, if you enter the VASP IMAGES value (intermediate images), the script automatically adds 2 and reads 00..(IMAGES+1).\n"
            "  - Use --count-mode total to specify the total number of folders explicitly.\n"
            "Input: "
        ).strip()
        args.nimages = int(n)

    return args


def main():
    args = parse_args()

    # Guard against older script/editing mismatches
    if not hasattr(args, "fallback"):
        args.fallback = "POSCAR"

    # Complete missing input
    if args.path is None or args.nimages is None:
        args = interactive_fill(args)
    else:
        args.path = _clean_dragdrop_path(args.path)

    base = Path(args.path).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Could not find the NEB parent directory: {base}")

    # Interpret nimages (core logic)
    total_nimages, used_mode = _interpret_nimages(base, args.start, args.nimages, args.count_mode)

    # Infer zero padding from the interpreted total folder count (total_nimages)
    if args.pad is None:
        inferred = _infer_existing_pad(base, args.start, total_nimages)
        args.pad = inferred if inferred is not None else 2

    images: List[Atoms] = []
    missing = []
    used_files = []

    for i in range(args.start, args.start + total_nimages):
        d = _resolve_image_dir(base, i, args.pad)

        primary_path = d / args.contcar
        fallback_path = d / args.fallback

        if primary_path.exists():
            in_path = primary_path
        elif fallback_path.exists():
            in_path = fallback_path
        else:
            missing.append(f"{primary_path}  (or {fallback_path})")
            continue

        a = read(str(in_path))
        images.append(a)
        used_files.append(str(in_path))

    if missing:
        msg = "\n".join(missing[:10])
        more = "" if len(missing) <= 10 else f"\n... ({len(missing)} entries missing in total)"
        raise FileNotFoundError(f"Could not find the following structure files (CONTCAR/POSCAR):\n{msg}{more}")

    if args.verbose:
        for p in used_files:
            print(f"[READ] {p}")

    if args.check:
        ok, msgs = _validate_consistency(images)
        for m in msgs:
            print(m)
        if not ok:
            raise RuntimeError("Stopping because image consistency checks failed (atom count / element order mismatch).")

    out = Path(args.output).resolve()
    comment = (
        f"XDATCAR from NEB images under {base.name} "
        f"(start={args.start}, input_nimages={args.nimages}, mode={used_mode})"
    )
    write_xdatcar(out, images, comment=comment, wrap=(not args.no_wrap))

    print(f"[OK] {len(images)} images were combined and written:")
    print(f"     Output : {out}")
    print(f"     Base   : {base}")
    print(f"     Mode   : {used_mode}")
    print(f"     Input  : nimages={args.nimages} (count-mode={args.count_mode})")
    print(f"     Range  : {args.start}..{args.start + total_nimages - 1} (pad={args.pad})")
    print(f"     Prefer : {args.contcar}  |  Fallback : {args.fallback}")


if __name__ == "__main__":
    main()
