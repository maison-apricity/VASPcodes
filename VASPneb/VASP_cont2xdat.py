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
    return s.rstrip("/\\")  # 마지막 슬래시 제거


def _exists_image_dir_any_pad(base: Path, idx: int) -> bool:
    """
    idx에 해당하는 이미지 폴더가 존재하는지(0, 00, 000, ... 등) 폭넓게 확인.
    """
    for pad in (1, 2, 3, 4, 5, 6):
        name = f"{idx:0{pad}d}" if pad > 1 else str(idx)
        if (base / name).is_dir():
            return True
    return False


def _interpret_nimages(base: Path, start: int, nimages_input: int, mode: str) -> Tuple[int, str]:
    """
    사용자가 넣은 nimages_input을
      - total: 총 폴더 수로 해석
      - neb  : VASP IMAGES(중간 이미지 수)로 해석하여 총 폴더 수 = IMAGES + 2
      - auto : (start + nimages_input + 1) 폴더가 존재하면 neb, 아니면 total
    로 해석하여 (total_nimages, used_mode_str)을 반환.
    """
    if nimages_input <= 0:
        raise ValueError("nimages는 양의 정수여야 합니다.")

    mode = (mode or "auto").lower()
    if mode not in ("auto", "total", "neb"):
        raise ValueError(f"count-mode는 auto/total/neb 중 하나여야 합니다: {mode}")

    if mode == "total":
        return nimages_input, "total(forced)"
    if mode == "neb":
        return nimages_input + 2, "neb(forced, total=IMAGES+2)"

    # auto
    probe_idx = start + nimages_input + 1  # neb 해석일 때 마지막 인덱스
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
            msgs.append(f"[불일치] frame {k}: natoms {len(a)} != {ref_n}")
        if a.get_chemical_symbols() != ref_syms:
            ok = False
            msgs.append(f"[불일치] frame {k}: 원소/순서가 기준 프레임과 다릅니다.")
        cell = np.array(a.cell.array, dtype=float)
        if not np.allclose(cell, ref_cell, atol=tol, rtol=tol):
            msgs.append(f"[경고] frame {k}: cell이 기준 프레임과 다릅니다 (NEB에서는 보통 동일).")

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
        raise ValueError("images가 비어 있습니다.")

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
                raise ValueError(f"frame {i}: scaled_positions shape 불일치: {frac.shape}")

            for j in range(natoms):
                f.write(f"{frac[j,0]: .16f} {frac[j,1]: .16f} {frac[j,2]: .16f}\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Collect NEB image CONTCAR/POSCAR files into a single XDATCAR using ASE."
    )
    ap.add_argument("--path", type=str, default=None,
                    help="NEB 상위 폴더 경로 (드래그&드롭 가능). 미지정 시 대화형 입력.")
    ap.add_argument("--nimages", type=int, default=None,
                    help=(
                        "이미지 개수 입력값. 기본(count-mode=auto)에서는 "
                        "VASP IMAGES(중간 이미지 수)인지(총=IMAGES+2) 또는 총 폴더 수인지 자동 판별합니다. "
                        "강제하려면 --count-mode total 또는 --count-mode neb 사용."
                    ))
    ap.add_argument("--count-mode", type=str, default="auto", choices=["auto", "total", "neb"],
                    help="nimages 해석 방식: auto(기본), total(총 폴더 수), neb(VASP IMAGES로 보고 +2).")
    ap.add_argument("--start", type=int, default=0,
                    help="시작 인덱스 (기본 0). 예: start=0, (total) nimages=18 => 00..17")
    ap.add_argument("--pad", type=int, default=None,
                    help="폴더명 0패딩 자릿수 (예: 2면 00,01,...). 미지정 시 자동 추정 시도.")
    ap.add_argument("--contcar", type=str, default="CONTCAR",
                    help="각 이미지 폴더 안의 1순위 파일명 (기본 CONTCAR)")
    ap.add_argument("--fallback", type=str, default="POSCAR",
                    help="CONTCAR가 없을 때 대체로 읽을 파일명 (기본 POSCAR)")
    ap.add_argument("--output", type=str, default="XDATCAR",
                    help="출력 XDATCAR 파일명 (기본 XDATCAR)")
    ap.add_argument("--no-wrap", action="store_true",
                    help="fractional 좌표를 [0,1)로 래핑하지 않음 (기본: wrap)")
    ap.add_argument("--check", action="store_true",
                    help="원자수/원소순서/cell 일관성 검사 수행 (권장)")
    ap.add_argument("--verbose", action="store_true",
                    help="각 이미지에서 실제로 읽은 파일(CONTCAR/POSCAR)을 출력")
    return ap.parse_args()


def interactive_fill(args: argparse.Namespace) -> argparse.Namespace:
    if args.path is None:
        p = input("NEB 상위 폴더 경로를 입력하십시오 (드래그&드롭 가능): ").strip()
        args.path = _clean_dragdrop_path(p)

    if args.nimages is None:
        n = input(
            "nimages 입력값을 입력하십시오.\n"
            "  - 기본(--count-mode auto): IMAGES(중간)로 넣으면 자동으로 +2 하여 00..(IMAGES+1)까지 읽습니다.\n"
            "  - 총 폴더 수를 명시하려면 --count-mode total을 사용하십시오.\n"
            "입력: "
        ).strip()
        args.nimages = int(n)

    return args


def main():
    args = parse_args()

    # 구버전 스크립트/편집 꼬임 방지
    if not hasattr(args, "fallback"):
        args.fallback = "POSCAR"

    # 입력 보완
    if args.path is None or args.nimages is None:
        args = interactive_fill(args)
    else:
        args.path = _clean_dragdrop_path(args.path)

    base = Path(args.path).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"NEB 상위 폴더를 찾을 수 없습니다: {base}")

    # nimages 해석(핵심 수정)
    total_nimages, used_mode = _interpret_nimages(base, args.start, args.nimages, args.count_mode)

    # pad 추정은 '해석된 총 폴더 수(total_nimages)' 기준으로 수행
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
            missing.append(f"{primary_path}  (또는 {fallback_path})")
            continue

        a = read(str(in_path))
        images.append(a)
        used_files.append(str(in_path))

    if missing:
        msg = "\n".join(missing[:10])
        more = "" if len(missing) <= 10 else f"\n... (총 {len(missing)}개 누락)"
        raise FileNotFoundError(f"다음 구조 파일(CONTCAR/POSCAR)을 찾지 못했습니다:\n{msg}{more}")

    if args.verbose:
        for p in used_files:
            print(f"[READ] {p}")

    if args.check:
        ok, msgs = _validate_consistency(images)
        for m in msgs:
            print(m)
        if not ok:
            raise RuntimeError("이미지 간 일관성(원자수/원소순서) 불일치로 중단합니다.")

    out = Path(args.output).resolve()
    comment = (
        f"XDATCAR from NEB images under {base.name} "
        f"(start={args.start}, input_nimages={args.nimages}, mode={used_mode})"
    )
    write_xdatcar(out, images, comment=comment, wrap=(not args.no_wrap))

    print(f"[OK] {len(images)}개 이미지를 합쳐서 저장했습니다:")
    print(f"     Output : {out}")
    print(f"     Base   : {base}")
    print(f"     Mode   : {used_mode}")
    print(f"     Input  : nimages={args.nimages} (count-mode={args.count_mode})")
    print(f"     Range  : {args.start}..{args.start + total_nimages - 1} (pad={args.pad})")
    print(f"     Prefer : {args.contcar}  |  Fallback : {args.fallback}")


if __name__ == "__main__":
    main()