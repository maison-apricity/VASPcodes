# VASP Codes

Useful post-processing scripts for **VASP (Vienna Ab initio Simulation Package)** workflows, with a strong focus on NEB, OUTCAR/XDATCAR parsing, Selective Dynamics handling, and OVITO export.

## Included scripts

| Script | Purpose |
|---|---|
| `vasp_energy_plot.py` | Plot NEB absolute energies from `py4vasp`, `OUTCAR`, or `OSZICAR`, with optional moving-atom reaction coordinate. |
| `vasp_xdat2extxyz.py` | Convert `XDATCAR` to OVITO-readable `EXTXYZ` while preserving Selective Dynamics flags from `POSCAR`/`CONTCAR`. |
| `vasp_cont2xdat.py` | Collect NEB image `CONTCAR`/`POSCAR` files and assemble a single `XDATCAR`. |
| `vasp_Energy.py` | Inspect energy content in `vaspout.h5` using `py4vasp`. |
| `vasp_CenterOfMass.py` | Compute center of mass from `POSCAR` and print a `DIPOL`-ready line. |
| `vasp_MovedAtoms.py` | Detect atoms that moved in `XDATCAR` above a user-defined tolerance. |
| `vasp_AddSelective.py` | Rewrite Selective Dynamics flags in `POSCAR` for selected atom indices. |
| `vasp_TagCompare.py` | Compare two `OUTCAR` files and separate settings, metadata, and history-like fields. |
| `vasp_vibfreq2ovito.py` | Extract vibrational modes from finite-difference `OUTCAR` and export OVITO-friendly files. |

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

### 1. `vasp_energy_plot.py`
Plots NEB energies and supports:
- automatic interpretation of image count
- fallback from `py4vasp` to `OUTCAR` / `OSZICAR`
- x-axis as either image index or moving-atom displacement
- raw table, TXT dump, CSV export, and HTML plot output

Example:
```bash
python vasp_energy_plot.py --path . --nimages 7 --count-mode neb --save-html neb_energy.html
```

### 2. `vasp_xdat2extxyz.py`
Reads Selective Dynamics flags from `POSCAR` and applies them to all frames in `XDATCAR`, then writes a multi-frame `EXTXYZ` file for OVITO.

Example:
```bash
python vasp_xdat2extxyz.py -p POSCAR -x XDATCAR -o neb_sd.extxyz
```

### 3. `vasp_cont2xdat.py`
Builds a single `XDATCAR` from NEB image folders such as `00`, `01`, ..., `0N`.

Example:
```bash
python vasp_cont2xdat.py --path . --nimages 7 --count-mode neb --check
```

### 4. `vasp_Energy.py`
Simple inspection script for `vaspout.h5` energy objects using `py4vasp`.

### 5. `vasp_CenterOfMass.py`
Computes the Cartesian and fractional center of mass and prints a line that can be reused as:
```text
DIPOL = x y z
```

### 6. `vasp_MovedAtoms.py`
Detects which atoms moved over a threshold between consecutive XDATCAR frames, using minimum-image displacement handling.

Example:
```bash
python vasp_MovedAtoms.py XDATCAR --tol 1e-3 --mode cart --verbose
```

### 7. `vasp_AddSelective.py`
Applies `T T T` or `F F F` flags to chosen atom indices in `POSCAR`.

### 8. `vasp_TagCompare.py`
Compares two OUTCAR files while separating:
- actual calculation settings
- restart/output settings
- POTCAR/system metadata
- optional history-like fields

Example:
```bash
python vasp_TagCompare.py OUTCAR_1 OUTCAR_2 --show-history
```

### 9. `vasp_vibfreq2ovito.py`
Builds vibrational modes from a finite-difference `OUTCAR`, supports `NFREE=1` and `NFREE=2`, and exports:
- OVITO-friendly `EXTXYZ`
- plain `XYZ` animations
- CSV mode summaries

Example:
```bash
python vasp_vibfreq2ovito.py OUTCAR --prefix vib --amplitude 0.25 --nframes 31
```

## Suggested repository structure

```text
VASP-Codes/
├─ README.md
├─ .gitignore
├─ requirements.txt
├─ vasp_energy_plot.py
├─ vasp_xdat2extxyz.py
├─ vasp_cont2xdat.py
├─ vasp_Energy.py
├─ vasp_CenterOfMass.py
├─ vasp_MovedAtoms.py
├─ vasp_AddSelective.py
├─ vasp_TagCompare.py
└─ vasp_vibfreq2ovito.py
```

## Notes

- These scripts are provided as standalone utilities rather than a packaged Python module.
- Several scripts were originally written for personal workflow automation.
- The original source files are preserved separately; this folder is a GitHub-oriented backup copy.
