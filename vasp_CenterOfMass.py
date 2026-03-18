#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GitHub backup copy
# Original file: VASP_masscenter.py
# Suggested English filename: vasp_center_of_mass.py

from pymatgen.core import Structure
import numpy as np

s = Structure.from_file("POSCAR")

# 모든 원자가 B이면 질량 가중 = 단순 평균과 동일
cart = np.array([site.coords for site in s.sites])
r_cm_cart = cart.mean(axis=0)

# direct 좌표로 변환
r_cm_frac = s.lattice.get_fractional_coords(r_cm_cart)

print("Cartesian COM:", r_cm_cart)
print("Direct COM   :", r_cm_frac)
print(f"DIPOL = {r_cm_frac[0]:.6f} {r_cm_frac[1]:.6f} {r_cm_frac[2]:.6f}")