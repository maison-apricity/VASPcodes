# VASP Scripts

Useful Python utilities for VASP post-processing and trajectory conversion.

## Files

- `vasp_cont2xdat.py` — collect NEB image `CONTCAR`/`POSCAR` files into one `XDATCAR`.
- `vasp_energy_plot.py` — read NEB energies, build raw tables, export CSV/TXT/HTML, and plot the energy profile.
- `vasp_AddSelective.py` — rewrite a POSCAR with `Selective dynamics` flags for a chosen set of atom indices.
- `vasp_CenterOfMass.py` — compute the center of mass from `POSCAR` and print a `DIPOL` line.
- `vasp_Energy.py` — inspect energy data stored in `vaspout.h5` with `py4vasp`.
- `vasp_MovedAtoms.py` — detect atoms that moved in an `XDATCAR` trajectory.
- `vasp_TagCompare.py` — compare two `OUTCAR` files by settings and metadata.
- `vasp_vibfreq2ovito.py` — convert finite-difference vibrational data in `OUTCAR` into OVITO-friendly exports.
- `vasp_xdat2extxyz.py` — convert `XDATCAR` to multi-frame `extxyz` using `Selective dynamics` flags from `POSCAR`.

## Requirements

Install only the packages needed for the scripts you actually use.

```bash
pip install numpy ase py4vasp plotly pymatgen matplotlib
```

## Example

```bash
python vasp_cont2xdat.py --path ./VASPneb --nimages 16 --count-mode neb --check
python vasp_energy_plot.py --path ./VASPneb --nimages 16 --count-mode neb --save-html neb_energy.html
python vasp_AddSelective.py
python vasp_TagCompare.py OUTCAR_ref OUTCAR_test --show-history
```
