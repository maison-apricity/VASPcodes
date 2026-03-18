#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GitHub backup copy
# Original file: VASP_energyPlot.py
# Suggested English filename: vasp_neb_energy_plot.py

import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import plotly.graph_objects as go
import py4vasp
from ase.io import read as ase_read


def _clean_dragdrop_path(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    s = os.path.expanduser(s)
    return s.rstrip("/\\")  # 마지막 슬래시 제거


def _exists_image_dir_any_pad(base: Path, idx: int) -> bool:
    for pad in (1, 2, 3, 4, 5, 6):
        name = f"{idx:0{pad}d}" if pad > 1 else str(idx)
        if (base / name).is_dir():
            return True
    return False


def _interpret_nimages(base: Path, start: int, nimages_input: int, mode: str) -> Tuple[int, str]:
    if nimages_input <= 0:
        raise ValueError("nimages는 must be a positive integer.")

    mode = (mode or "auto").lower()
    if mode not in ("auto", "total", "neb"):
        raise ValueError(f"count-mode는 auto/total/neb 중 하나여야 합니다: {mode}")

    if mode == "total":
        return nimages_input, "total(forced)"
    if mode == "neb":
        return nimages_input + 2, "neb(forced, total=IMAGES+2)"

    probe_idx = start + nimages_input + 1
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


def _energy_key_from_dict(d: Dict[str, Any], prefer: Optional[str] = None) -> str:
    if prefer and prefer in d:
        return prefer

    candidates = [
        "free energy    TOTEN",
        "free energy TOTEN",
        "TOTEN",
        "energy(sigma->0)",
        "energy",
    ]
    for k in candidates:
        if k in d:
            return k

    for k in d.keys():
        if "TOTEN" in k:
            return k

    raise KeyError(f"Could not automatically detect the energy key.\nAvailable keys: {list(d.keys())}")


# ---------------- OUTCAR / OSZICAR parsers ----------------
# NOTE:
#  - "free␠␠energy ... TOTEN =" (free 뒤 공백 2개)만 잡고 싶다면 DOUBLE 패턴 사용
#  - 기존처럼 공백 무관으로 잡고 싶다면 ANY 패턴 사용

_OUTCAR_TOTEN_RE_DOUBLE = re.compile(
    r"^\s*free {2}energy\s+TOTEN\s*=\s*([-\d\.Ee+]+)",
    re.IGNORECASE
)
_OUTCAR_TOTEN_RE_ANY = re.compile(
    r"^\s*free\s+energy\s+TOTEN\s*=\s*([-\d\.Ee+]+)",
    re.IGNORECASE
)

_OSZICAR_E0_RE = re.compile(r"\bE0=\s*([-\d\.Ee+]+)")


def _read_last_toten_from_outcar(
    outcar_path: Path,
    *,
    style: str = "double",
) -> Tuple[float, str]:
    """
    style:
      - "double": 반드시 'free␠␠energy ... TOTEN =' 만 사용 (없으면 에러)
      - "prefer_double": double이 있으면 그것, 없으면 any로 fallback
      - "any": 'free\\s+energy\\s+TOTEN' 모든 변형 허용
    return: (E, tag) where tag describes which pattern was used.
    """
    if not outcar_path.is_file():
        raise FileNotFoundError(f"OUTCAR not found: {outcar_path}")

    style = (style or "double").lower()
    if style not in ("double", "prefer_double", "any"):
        raise ValueError(f"outcar toten style은 double/prefer_double/any 중 하나여야 합니다: {style}")

    last_double: Optional[float] = None
    last_any: Optional[float] = None

    with outcar_path.open("r", errors="ignore") as f:
        for line in f:
            m2 = _OUTCAR_TOTEN_RE_DOUBLE.search(line)
            if m2:
                last_double = float(m2.group(1))
                # DOUBLE은 ANY에도 포함될 수 있으므로 같이 업데이트해도 무방합니다.
                last_any = last_double
                continue

            m1 = _OUTCAR_TOTEN_RE_ANY.search(line)
            if m1:
                last_any = float(m1.group(1))

    if style == "double":
        if last_double is None:
            raise ValueError(
                "Could not parse from OUTCAR 'free␠␠energy ... TOTEN =' (free 뒤 공백 2개) pattern not found.\n"
                f"파일: {outcar_path}\n"
                "필요하면 --outcar-toten-style prefer_double 또는 any를 사용please."
            )
        return last_double, "OUTCAR_TOTEN[double-space after 'free']"

    if style == "prefer_double":
        if last_double is not None:
            return last_double, "OUTCAR_TOTEN[double-space after 'free']"
        if last_any is not None:
            return last_any, "OUTCAR_TOTEN[any-spacing fallback]"
        raise ValueError(f"Could not parse from OUTCAR TOTEN pattern not found: {outcar_path}")

    # style == "any"
    if last_any is None:
        raise ValueError(f"Could not parse from OUTCAR TOTEN pattern not found: {outcar_path}")
    return last_any, "OUTCAR_TOTEN[any-spacing]"


def _read_last_e0_from_oszicar(oszicar_path: Path) -> float:
    if not oszicar_path.is_file():
        raise FileNotFoundError(f"OSZICAR not found: {oszicar_path}")

    last_val: Optional[float] = None
    with oszicar_path.open("r", errors="ignore") as f:
        for line in f:
            m = _OSZICAR_E0_RE.search(line)
            if m:
                last_val = float(m.group(1))

    if last_val is None:
        raise ValueError(f"OSZICAR에서 E0 pattern not found: {oszicar_path}")

    return last_val


# ---------------- energy reader with fallback ----------------

def _read_energy_any(
    d: Path,
    prefer_key: Optional[str],
    *,
    allow_outcar: bool,
    allow_oszicar: bool,
    outcar_name: str,
    oszicar_name: str,
    prefer_outcar: bool,
    outcar_toten_style: str,
) -> Tuple[float, str]:
    def from_py4vasp() -> Tuple[float, str]:
        calc = py4vasp.Calculation.from_path(str(d))
        e_dict = calc.energy.read()
        key = _energy_key_from_dict(e_dict, prefer=prefer_key)
        return float(e_dict[key]), f"py4vasp[{key}]"

    def from_outcar() -> Tuple[float, str]:
        E, tag = _read_last_toten_from_outcar(d / outcar_name, style=outcar_toten_style)
        return E, f"{tag} ({outcar_name})"

    def from_oszicar() -> Tuple[float, str]:
        E = _read_last_e0_from_oszicar(d / oszicar_name)
        return E, f"OSZICAR[{oszicar_name}]"

    if prefer_outcar:
        if allow_outcar:
            try:
                return from_outcar()
            except Exception:
                pass
        if allow_oszicar:
            try:
                return from_oszicar()
            except Exception:
                pass
        return from_py4vasp()

    try:
        return from_py4vasp()
    except Exception:
        if allow_outcar:
            try:
                return from_outcar()
            except Exception:
                pass
        if allow_oszicar:
            try:
                return from_oszicar()
            except Exception:
                pass
        raise


# ---------------- structure reader with fallback ----------------

def _read_positions_any(d: Path, *, contcar_name: str, poscar_name: str) -> Tuple[np.ndarray, str]:
    # 1) py4vasp structure
    try:
        calc = py4vasp.Calculation.from_path(str(d))
        pos = np.asarray(calc.structure.cartesian_positions(), dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"cartesian_positions shape가 예상과 다릅니다: {pos.shape}")
        return pos, "py4vasp[structure]"
    except Exception:
        pass

    # 2) ASE CONTCAR
    cont = d / contcar_name
    if cont.is_file():
        atoms = ase_read(str(cont))
        pos = np.asarray(atoms.get_positions(), dtype=float)
        return pos, f"ASE[{contcar_name}]"

    # 3) ASE POSCAR
    poscar = d / poscar_name
    if poscar.is_file():
        atoms = ase_read(str(poscar))
        pos = np.asarray(atoms.get_positions(), dtype=float)
        return pos, f"ASE[{poscar_name}]"

    raise FileNotFoundError(f"Could not find structure files: {contcar_name} / {poscar_name} (dir={d})")


def _normalize_atom_index(idx: int, natoms: int) -> int:
    if natoms <= 0:
        raise ValueError("natoms가 0 이하is.")
    if idx < 0:
        idx = natoms + idx
    if idx < 0 or idx >= natoms:
        raise IndexError(f"moving-atom 인덱스 범위 오류: 0..{natoms-1} 또는 음수(-1..-{natoms}) 범위여야 합니다.")
    return idx


# ---------------- interactive helpers ----------------

def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    """
    default=False -> 엔터는 No
    default=True  -> 엔터는 Yes
    """
    if default:
        suffix = " [Y/n]: "
    else:
        suffix = " [y/N]: "

    while True:
        ans = input(prompt + suffix).strip().lower()
        if ans == "":
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  Input error: y 또는 n 으로 입력해주십시오.")


def _ask_int(prompt: str, default: Optional[int] = None) -> int:
    while True:
        if default is None:
            ans = input(prompt + ": ").strip()
        else:
            ans = input(f"{prompt} (default={default}): ").strip()
            if ans == "":
                return int(default)
        try:
            return int(ans)
        except ValueError:
            print("  Input error: Please enter an integer.")

def _compute_relative_energies(energies_abs: List[float]) -> Tuple[List[float], List[float], float, float]:
    """
    Returns:
      - rel_min  : E - min(E)
      - rel_first: E - E(first)
      - emin     : min(E)
      - e0       : first E
    """
    if not energies_abs:
        raise ValueError("energies_abs가 비어 exists.")
    emin = float(min(energies_abs))
    e0 = float(energies_abs[0])
    rel_min = [float(e) - emin for e in energies_abs]
    rel_first = [float(e) - e0 for e in energies_abs]
    return rel_min, rel_first, emin, e0


def _format_float(v: float, *, width: int = 0, prec: int = 10) -> str:
    s = f"{v:.{prec}f}"
    return s.rjust(width) if width > 0 else s


def _print_raw_table(
    labels: List[str],
    xs: List[float],
    energies_abs: List[float],
    rel_min: List[float],
    rel_first: List[float],
    sources: List[str],
    *,
    prec_x: int = 6,
    prec_e: int = 10,
) -> None:
    """터미널에서 바로 확인 가능한 고정폭(raw) 테이블을 출력합니다."""
    n = len(labels)
    if not (len(xs) == len(energies_abs) == len(rel_min) == len(rel_first) == len(sources) == n):
        raise ValueError("raw table 입력 리스트들의 길이가 서로 다릅니다.")

    # column widths
    w_i = max(3, len(str(n - 1)))
    w_lab = max(5, max(len(l) for l in labels))
    w_x = max(10, len(f"{max(xs):.{prec_x}f}") if xs else 10)
    w_e = max(14, len(f"{max(energies_abs):.{prec_e}f}") if energies_abs else 14)
    w_re = max(12, len(f"{max(rel_min):.{prec_e}f}") if rel_min else 12)

    header = (
        f"{'#':>{w_i}}  {'image':<{w_lab}}  {'x':>{w_x}}  {'E_abs(eV)':>{w_e}}  "
        f"{'dE_min(eV)':>{w_re}}  {'dE_first(eV)':>{w_re}}  source"
    )
    sep = "-" * min(180, len(header))
    print("\n[RAW DATA TABLE]")
    print(header)
    print(sep)
    for k in range(n):
        print(
            f"{k:>{w_i}d}  "
            f"{labels[k]:<{w_lab}}  "
            f"{xs[k]:>{w_x}.{prec_x}f}  "
            f"{energies_abs[k]:>{w_e}.{prec_e}f}  "
            f"{rel_min[k]:>{w_re}.{prec_e}f}  "
            f"{rel_first[k]:>{w_re}.{prec_e}f}  "
            f"{sources[k]}"
        )


def _write_raw_txt(
    out_path: Path,
    labels: List[str],
    xs: List[float],
    energies_abs: List[float],
    rel_min: List[float],
    rel_first: List[float],
    sources: List[str],
    *,
    meta: Dict[str, Any],
    prec_x: int = 6,
    prec_e: int = 10,
) -> None:
    """
    사람이 바로 읽을 수 있는 .txt raw dump를 saved합니다.
    - 상단에 메타데이터(실행 조건/폴더 범위 등)
    - 하단에 고정폭 테이블
    """
    out_path = out_path.resolve()
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# NEB energy raw dump\n")
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        f.write("#\n")
        f.write("# columns: idx, image, x, E_abs_eV, dE_min_eV, dE_first_eV, source\n")
        f.write("\n")

        n = len(labels)
        w_i = max(3, len(str(n - 1)))
        w_lab = max(5, max(len(l) for l in labels))
        w_x = max(10, len(f"{max(xs):.{prec_x}f}") if xs else 10)
        w_e = max(14, len(f"{max(energies_abs):.{prec_e}f}") if energies_abs else 14)
        w_re = max(12, len(f"{max(rel_min):.{prec_e}f}") if rel_min else 12)

        header = (
            f"{'#':>{w_i}}  {'image':<{w_lab}}  {'x':>{w_x}}  {'E_abs(eV)':>{w_e}}  "
            f"{'dE_min(eV)':>{w_re}}  {'dE_first(eV)':>{w_re}}  source\n"
        )
        sep = "-" * min(180, len(header.strip())) + "\n"
        f.write(header)
        f.write(sep)
        for k in range(n):
            f.write(
                f"{k:>{w_i}d}  "
                f"{labels[k]:<{w_lab}}  "
                f"{xs[k]:>{w_x}.{prec_x}f}  "
                f"{energies_abs[k]:>{w_e}.{prec_e}f}  "
                f"{rel_min[k]:>{w_re}.{prec_e}f}  "
                f"{rel_first[k]:>{w_re}.{prec_e}f}  "
                f"{sources[k]}\n"
            )
    print(f"[OK] TXT raw dump saved: {out_path}")


def _write_csv_extended(
    out_csv: Path,
    labels: List[str],
    xs: List[float],
    energies_abs: List[float],
    rel_min: List[float],
    rel_first: List[float],
    sources: List[str],
) -> None:
    out_csv = out_csv.resolve()
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("image,x,E_abs_eV,E_rel_min_eV,E_rel_first_eV,source\n")
        for lab, x, ea, rm, rf, src in zip(labels, xs, energies_abs, rel_min, rel_first, sources):
            f.write(f"{lab},{x:.10f},{ea:.10f},{rm:.10f},{rf:.10f},{src}\n")
    print(f"[OK] CSV(extended) saved: {out_csv}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plot NEB energies (absolute). Interactive mode chooses x-axis as image index or moving-atom distance."
    )
    ap.add_argument("--path", type=str, default=None)
    ap.add_argument("--nimages", type=int, default=None)
    ap.add_argument("--count-mode", type=str, default="auto", choices=["auto", "total", "neb"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--pad", type=int, default=None)

    # energy
    ap.add_argument("--energy-key", type=str, default=None)

    # x-axis (non-interactive override)
    ap.add_argument("--x-mode", type=str, default=None, choices=["index", "atom"],
                    help="비대화형 실행 시 x축 모드 강제: index(이미지), atom(원자 이동거리). 미지정이면 대화형 질문.")
    ap.add_argument("--moving-atom", type=int, default=None,
                    help="x-mode=atom일 때 사용할 원자 인덱스(0-based, -1 허용). 미지정이면 대화형으로 질문.")

    ap.add_argument("--sort-by-x", action="store_true",
                    help="x값 기준으로 정렬(기본은 이미지 순서대로). index 모드에서는 의미 거의 없음.")
    ap.add_argument("--save-html", type=str, default=None)
    ap.add_argument("--output-csv", type=str, default=None)
    ap.add_argument("--title", type=str, default=None)

    ap.add_argument("--no-raw-table", action="store_true",
                    help="collected (x, E) raw data를 터미널 테이블로 출력하지 않습니다.")
    ap.add_argument("--output-txt", type=str, default=None,
                    help="raw data 테이블을 사람이 읽기 쉬운 TXT로 saved합니다.")
    ap.add_argument("--output-csv-extended", type=str, default=None,
                    help="E_rel_min, E_rel_first 컬럼까지 포함한 확장 CSV를 saved합니다.")
    ap.add_argument("--no-plot", action="store_true",
                    help="Plotly 창(fig.show)을 띄우지 않습니다(SSH/헤드리스 환경용). save-html은 가능.")

    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    # energy fallback
    ap.add_argument("--no-outcar-fallback", action="store_true")
    ap.add_argument("--prefer-outcar", action="store_true")
    ap.add_argument("--outcar-name", type=str, default="OUTCAR")
    ap.add_argument("--oszicar-name", type=str, default="OSZICAR")
    ap.add_argument("--no-oszicar", action="store_true")

    # NEW: choose which OUTCAR TOTEN line to use
    ap.add_argument(
        "--outcar-toten-style",
        type=str,
        default="double",
        choices=["double", "prefer_double", "any"],
        help=(
            "Could not parse from OUTCAR TOTEN 추출 시 공백 패턴 선택. "
            "'double'은 free 뒤 공백 2개('free␠␠energy ... TOTEN=')만 허용(없으면 에러). "
            "'prefer_double'은 있으면 double, 없으면 any로 fallback. "
            "'any'는 기존처럼 공백 무관."
        )
    )

    # structure fallback
    ap.add_argument("--contcar-name", type=str, default="CONTCAR")
    ap.add_argument("--poscar-name", type=str, default="POSCAR")

    return ap.parse_args()


def interactive_fill(args: argparse.Namespace) -> argparse.Namespace:
    # 0) path/nimages는 항상 필요
    if args.path is None:
        p = input("Enter the NEB parent directory path (drag-and-drop supported): ").strip()
        args.path = _clean_dragdrop_path(p)
    else:
        args.path = _clean_dragdrop_path(args.path)

    if args.nimages is None:
        args.nimages = _ask_int(
            "nimages 입력값을 입력please (auto에서는 IMAGES(중간)일 수도, total(총 폴더 수)일 수도 exists)",
            default=None
        )

    # 1) x축 모드 결정(요청하신 3-step 로직)
    if args.x_mode is None:
        use_atom = _ask_yes_no("x축을 원자의 상대적 이동거리(초기 이미지 대비)로 사용하시겠습니까?", default=False)
        args.x_mode = "atom" if use_atom else "index"

    # 2) atom 모드면 원자 번호 질문(없으면)
    if args.x_mode == "atom" and args.moving_atom is None:
        args.moving_atom = _ask_int("Enter the atom index used to compute the displacement (0-based, -1은 마지막 원자)", default=-1)

    return args


def main():
    args = parse_args()

    need_interactive = (
        (args.path is None) or
        (args.nimages is None) or
        (args.x_mode is None) or
        (args.x_mode == "atom" and args.moving_atom is None)
    )

    if need_interactive:
        args = interactive_fill(args)
    else:
        args.path = _clean_dragdrop_path(args.path)

    base = Path(args.path).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"NEB 상위 폴더를 찾을 수 없습니다: {base}")

    total_nimages, used_mode = _interpret_nimages(base, args.start, args.nimages, args.count_mode)

    if args.pad is None:
        inferred = _infer_existing_pad(base, args.start, total_nimages)
        args.pad = inferred if inferred is not None else 2

    # collected
    xs: List[float] = []
    energies_abs: List[float] = []
    labels: List[str] = []
    sources: List[str] = []
    failures: List[str] = []

    r0: Optional[np.ndarray] = None
    moving_atom_resolved: Optional[int] = None
    prefer_key: Optional[str] = args.energy_key

    allow_fallback = not args.no_outcar_fallback
    allow_outcar = allow_fallback
    allow_oszicar = allow_fallback and (not args.no_oszicar)

    x_mode = args.x_mode  # "index" or "atom"
    if x_mode not in ("index", "atom"):
        raise ValueError(f"x_mode 오류: {x_mode}")

    for i in range(args.start, args.start + total_nimages):
        d = _resolve_image_dir(base, i, args.pad)
        label = d.name  # 보통 "00","01",...

        if not d.is_dir():
            msg = f"[{label}] 이미지 폴더 없음: {d}"
            if args.skip_missing:
                failures.append(msg)
                continue
            raise FileNotFoundError(msg)

        try:
            # Energy (absolute)
            E, srcE = _read_energy_any(
                d,
                prefer_key,
                allow_outcar=allow_outcar,
                allow_oszicar=allow_oszicar,
                outcar_name=args.outcar_name,
                oszicar_name=args.oszicar_name,
                prefer_outcar=args.prefer_outcar,
                outcar_toten_style=args.outcar_toten_style,
            )

            # X-axis
            if x_mode == "index":
                x = float(i)
                srcX = "index"
            else:
                pos, srcX = _read_positions_any(
                    d,
                    contcar_name=args.contcar_name,
                    poscar_name=args.poscar_name,
                )
                natoms = pos.shape[0]
                if moving_atom_resolved is None:
                    if args.moving_atom is None:
                        raise ValueError("x_mode=atom인데 moving_atom이 설정되지 않았습니다.")
                    moving_atom_resolved = _normalize_atom_index(args.moving_atom, natoms)

                r = pos[moving_atom_resolved]
                if r0 is None:
                    r0 = r.copy()
                x = float(np.linalg.norm(r - r0))

            xs.append(x)
            energies_abs.append(E)
            labels.append(label)
            sources.append(f"E:{srcE} | X:{srcX}")

            if args.verbose:
                extra = ""
                if x_mode == "atom":
                    extra = f" atom={args.moving_atom} -> resolved={moving_atom_resolved}"
                print(f"[{label}] x={x:.6f}  E_abs={E:.8f} eV  ({sources[-1]}){extra}")

        except Exception as e:
            msg = f"[{label}] 실패: {d}  ({type(e).__name__}: {e})"
            if args.skip_missing:
                failures.append(msg)
                continue
            raise

    if not xs:
        raise RuntimeError("유효한 데이터를 하나도 읽지 못했습니다. OUTCAR/OSZICAR 또는 구조 파일 존재 여부를 확인please.")

    if args.sort_by_x:
        order = np.argsort(np.array(xs))
        xs = [xs[j] for j in order]
        energies_abs = [energies_abs[j] for j in order]
        labels = [labels[j] for j in order]
        sources = [sources[j] for j in order]

    rel_min, rel_first, emin, e0 = _compute_relative_energies(energies_abs)
    i_min = int(np.argmin(np.array(energies_abs)))
    i_max = int(np.argmax(np.array(energies_abs)))
    barrier = float(max(rel_min))  # max(E - Emin)
    end_to_end = float(energies_abs[-1] - energies_abs[0])

    # 확장 CSV / TXT raw dump
    if args.output_csv_extended:
        _write_csv_extended(Path(args.output_csv_extended), labels, xs, energies_abs, rel_min, rel_first, sources)

    if args.output_txt:
        meta = {
            "base": str(base),
            "mode": used_mode,
            "range": f"{args.start}..{args.start + total_nimages - 1} (pad={args.pad})",
            "x_mode": ("index" if x_mode == "index" else f"atom (moving_atom={args.moving_atom} -> resolved={moving_atom_resolved})"),
            "energy_ref_first_eV": f"{e0:.10f}",
            "energy_min_eV": f"{emin:.10f}",
            "barrier_rel_to_min_eV": f"{barrier:.10f}",
            "end_to_end_dE_eV": f"{end_to_end:.10f}",
        }
        _write_raw_txt(Path(args.output_txt), labels, xs, energies_abs, rel_min, rel_first, sources, meta=meta)

    # 터미널 표 출력(기본 ON, --no-raw-table로 끔)
    if not args.no_raw_table:
        _print_raw_table(labels, xs, energies_abs, rel_min, rel_first, sources)

    print("\n[RAW DATA STATS]")
    print(f"  Emin (eV)       : {emin:.10f}   @ row={i_min} (image={labels[i_min]})")
    print(f"  Emax (eV)       : {float(energies_abs[i_max]):.10f}   @ row={i_max} (image={labels[i_max]})")
    print(f"  Barrier (eV)    : {barrier:.10f}   (max(E - Emin))")
    print(f"  End-to-end dE   : {end_to_end:.10f}   (E_last - E_first)")

    if args.output_csv:
        out_csv = Path(args.output_csv).resolve()
        with out_csv.open("w", encoding="utf-8") as f:
            f.write("image,x,E_abs_eV,source\n")
            for lab, x, ea, src in zip(labels, xs, energies_abs, sources):
                f.write(f"{lab},{x:.10f},{ea:.10f},{src}\n")
        print(f"[OK] CSV saved: {out_csv}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs,
        y=energies_abs,
        mode="markers+lines",
        text=[f"{lab} ({src})" for lab, src in zip(labels, sources)],
        hovertemplate="Image=%{text}<br>x=%{x:.6f}<br>E_abs=%{y:.6f} eV<extra></extra>",
        name="E_abs"
    ))

    if args.title:
        title = args.title
    else:
        if x_mode == "index":
            title = f"NEB energies (ABS) (base={base.name}, mode={used_mode}, x=image index)"
        else:
            title = f"NEB energies (ABS) (base={base.name}, mode={used_mode}, x=moving-atom distance)"

    fig.update_layout(
        title=title,
        xaxis_title=("NEB image index" if x_mode == "index" else "Reaction coordinate (moving atom distance, Å)"),
        yaxis_title="Absolute energy (eV)",
    )

    if args.save_html:
        out_html = Path(args.save_html).resolve()
        fig.write_html(str(out_html))
        print(f"[OK] HTML saved: {out_html}")

    if (not args.no_plot) or args.save_html:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs,
            y=energies_abs,
            mode="markers+lines",
            text=[f"{lab} ({src})" for lab, src in zip(labels, sources)],
            hovertemplate="Image=%{text}<br>x=%{x:.6f}<br>E_abs=%{y:.6f} eV<extra></extra>",
            name="E_abs"
        ))

        if args.title:
            title = args.title
        else:
            if x_mode == "index":
                title = f"NEB energies (ABS) (base={base.name}, mode={used_mode}, x=image index)"
            else:
                title = f"NEB energies (ABS) (base={base.name}, mode={used_mode}, x=moving-atom distance)"

        fig.update_layout(
            title=title,
            xaxis_title=("NEB image index" if x_mode == "index" else "Reaction coordinate (moving atom distance, Å)"),
            yaxis_title="Absolute energy (eV)",
        )

        if args.save_html:
            out_html = Path(args.save_html).resolve()
            fig.write_html(str(out_html))
            print(f"[OK] HTML saved: {out_html}")

        if not args.no_plot:
            fig.show()

    print("\n[SUMMARY]")
    print(f"  Base    : {base}")
    print(f"  Mode    : {used_mode}")
    print(f"  Range   : {args.start}..{args.start + total_nimages - 1} (pad={args.pad})")
    print(f"  Used    : {len(xs)} images")
    print(f"  Y-axis  : absolute energy (eV)")
    print(f"  X-axis  : {'NEB image index' if x_mode == 'index' else 'moving-atom distance'}")
    if x_mode == "atom":
        print(f"  Atom    : moving-atom={args.moving_atom} (0-based, -1 allowed) -> resolved={moving_atom_resolved}")
    print(f"  Fallback: {'OFF' if args.no_outcar_fallback else 'ON'} (prefer_outcar={args.prefer_outcar})")
    print(f"  OUTCAR  : outcar_toten_style={args.outcar_toten_style}  (outcar_name={args.outcar_name})")
    if failures:
        print(f"  Skipped : {len(failures)}")
        if args.verbose:
            for m in failures:
                print("   ", m)


if __name__ == "__main__":
    main()