# VASP Codes

Useful post-processing scripts for **VASP (Vienna Ab initio Simulation Package)** workflows, with a strong focus on NEB, OUTCAR/XDATCAR parsing, Selective Dynamics handling, and OVITO export.

## Included scripts

| Script | Purpose |
|---|---|
| `vasp_neb_energy_plot.py` | Plot NEB absolute energies from `py4vasp`, `OUTCAR`, or `OSZICAR`, with optional moving-atom reaction coordinate. |
| `vasp_xdatcar_sd_to_extxyz.py` | Convert `XDATCAR` to OVITO-readable `EXTXYZ` while preserving Selective Dynamics flags from `POSCAR`/`CONTCAR`. |
| `vasp_neb_contcar_to_xdatcar.py` | Collect NEB image `CONTCAR`/`POSCAR` files and assemble a single `XDATCAR`. |
| `vasp_energy_inspector.py` | Inspect energy content in `vaspout.h5` using `py4vasp`. |
| `vasp_center_of_mass.py` | Compute center of mass from `POSCAR` and print a `DIPOL`-ready line. |
| `vasp_xdatcar_moved_atoms.py` | Detect atoms that moved in `XDATCAR` above a user-defined tolerance. |
| `vasp_poscar_selective_flags.py` | Rewrite Selective Dynamics flags in `POSCAR` for selected atom indices. |
| `vasp_outcar_compare.py` | Compare two `OUTCAR` files and separate settings, metadata, and history-like fields. |
| `vasp_vibration_outcar_to_ovito.py` | Extract vibrational modes from finite-difference `OUTCAR` and export OVITO-friendly files. |

## Requirements

Some scripts use only the Python standard library, while others require additional packages.

Common packages used in this repository:

- `numpy`
- `matplotlib`
- `plotly`
- `ase`
- `py4vasp`
- `pymatgen`

Install what you need with:

```bash
pip install numpy matplotlib plotly ase py4vasp pymatgen
```

## Script overview

### 1. `vasp_neb_energy_plot.py`
Plots NEB energies and supports:
- automatic interpretation of image count
- fallback from `py4vasp` to `OUTCAR` / `OSZICAR`
- x-axis as either image index or moving-atom displacement
- raw table, TXT dump, CSV export, and HTML plot output

Example:
```bash
python vasp_neb_energy_plot.py --path . --nimages 7 --count-mode neb --save-html neb_energy.html
```

### 2. `vasp_xdatcar_sd_to_extxyz.py`
Reads Selective Dynamics flags from `POSCAR` and applies them to all frames in `XDATCAR`, then writes a multi-frame `EXTXYZ` file for OVITO.

Example:
```bash
python vasp_xdatcar_sd_to_extxyz.py -p POSCAR -x XDATCAR -o neb_sd.extxyz
```

### 3. `vasp_neb_contcar_to_xdatcar.py`
Builds a single `XDATCAR` from NEB image folders such as `00`, `01`, ..., `0N`.

Example:
```bash
python vasp_neb_contcar_to_xdatcar.py --path . --nimages 7 --count-mode neb --check
```

### 4. `vasp_energy_inspector.py`
Simple inspection script for `vaspout.h5` energy objects using `py4vasp`.

### 5. `vasp_center_of_mass.py`
Computes the Cartesian and fractional center of mass and prints a line that can be reused as:
```text
DIPOL = x y z
```

### 6. `vasp_xdatcar_moved_atoms.py`
Detects which atoms moved over a threshold between consecutive XDATCAR frames, using minimum-image displacement handling.

Example:
```bash
python vasp_xdatcar_moved_atoms.py XDATCAR --tol 1e-3 --mode cart --verbose
```

### 7. `vasp_poscar_selective_flags.py`
Applies `T T T` or `F F F` flags to chosen atom indices in `POSCAR`.

### 8. `vasp_outcar_compare.py`
Compares two OUTCAR files while separating:
- actual calculation settings
- restart/output settings
- POTCAR/system metadata
- optional history-like fields

Example:
```bash
python vasp_outcar_compare.py OUTCAR_1 OUTCAR_2 --show-history
```

### 9. `vasp_vibration_outcar_to_ovito.py`
Builds vibrational modes from a finite-difference `OUTCAR`, supports `NFREE=1` and `NFREE=2`, and exports:
- OVITO-friendly `EXTXYZ`
- plain `XYZ` animations
- CSV mode summaries

Example:
```bash
python vasp_vibration_outcar_to_ovito.py OUTCAR --prefix vib --amplitude 0.25 --nframes 31
```

## Suggested repository structure

```text
VASP-Codes/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ vasp_neb_energy_plot.py
├─ vasp_xdatcar_sd_to_extxyz.py
├─ vasp_neb_contcar_to_xdatcar.py
├─ vasp_energy_inspector.py
├─ vasp_center_of_mass.py
├─ vasp_xdatcar_moved_atoms.py
├─ vasp_poscar_selective_flags.py
├─ vasp_outcar_compare.py
└─ vasp_vibration_outcar_to_ovito.py
```

## Notes

- These scripts are provided as standalone utilities rather than a packaged Python module.
- Several scripts were originally written for personal workflow automation and have now been renamed for cleaner GitHub presentation.
- The original source files are preserved separately; this folder is a GitHub-oriented backup copy with English-based filenames and repository documentation.
