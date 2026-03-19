from pymatgen.core import Structure
import numpy as np

s = Structure.from_file("POSCAR")

# If all atoms are B, the mass-weighted center is identical to the simple average
cart = np.array([site.coords for site in s.sites])
r_cm_cart = cart.mean(axis=0)

# Convert to fractional coordinates
r_cm_frac = s.lattice.get_fractional_coords(r_cm_cart)

print("Cartesian COM:", r_cm_cart)
print("Direct COM   :", r_cm_frac)
print(f"DIPOL = {r_cm_frac[0]:.6f} {r_cm_frac[1]:.6f} {r_cm_frac[2]:.6f}")