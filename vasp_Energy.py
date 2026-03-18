#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# GitHub backup copy
# Original file: VASP_energy.py
# Suggested English filename: vasp_energy_inspector.py

import py4vasp
import matplotlib.pyplot as plt
import numpy as np

# File path
file_path = r"C:\Users\dlstj\Desktop\Choi\vaspout.h5"

# Load data
data = py4vasp.Calculation.from_file(file_path)

# 1. Extract energy
energy_raw = data.energy

# 2. to_dict(): Single snapshot
try:
    energy_dict = energy_raw.to_dict()
    print("\n▶ to_dict() 출력:")
    for k, v in energy_dict.items():
        print(f"{k}: {v}")
except Exception as e:
    print("to_dict() 실패:", e)

# 3. to_numpy(): Inspect all steps
try:
    energy_np = energy_raw.to_numpy()
    print("\n▶ to_numpy() 출력:")
    print("type:", type(energy_np))
    print("value:", energy_np)
    print("shape:", getattr(energy_np, 'shape', 'shape 없음'))

    # Case handling
    if isinstance(energy_np, np.ndarray):
        if energy_np.ndim == 2:
            print("✅ 2차원 배열 → step별 에너지 존재")
            print("Total number of steps:", energy_np.shape[0])
            print("Number of energy entries:", energy_np.shape[1])
        elif energy_np.ndim == 1:
            print("⚠ Only a single ionic step is present (1차원)")
        elif energy_np.ndim == 0:
            print("⚠ Only a single scalar energy value is present (스칼라)")
    else:
        print("⚠ Return value is not an array:", energy_np)

except Exception as e:
    print("to_numpy() 실패:", e)
