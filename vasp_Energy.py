#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    print("\n▶ to_dict() output:")
    for k, v in energy_dict.items():
        print(f"{k}: {v}")
except Exception as e:
    print("to_dict() failed:", e)

# 3. to_numpy(): Inspect all steps
try:
    energy_np = energy_raw.to_numpy()
    print("\n▶ to_numpy() output:")
    print("type:", type(energy_np))
    print("value:", energy_np)
    print("shape:", getattr(energy_np, 'shape', 'shape unavailable'))

    # Case handling
    if isinstance(energy_np, np.ndarray):
        if energy_np.ndim == 2:
            print("2D array detected -> per-step energy data is available")
            print("Total number of steps:", energy_np.shape[0])
            print("Number of energy entries:", energy_np.shape[1])
        elif energy_np.ndim == 1:
            print("Only a single ionic step is present (1D)")
        elif energy_np.ndim == 0:
            print("Only a single scalar energy value is present (scalar)")
    else:
        print("⚠ Return value is not an array:", energy_np)

except Exception as e:
    print("to_numpy() failed:", e)
