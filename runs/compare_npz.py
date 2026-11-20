"""Compare two benchmark --save-arrays files, relative to the largest baseline entry."""
import sys
import numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
for k in ("carrier", "signal", "noise", "sensitivity", "powers", "gradient"):
    d = np.abs(b[k] - a[k]).max() / np.abs(a[k]).max()
    print(f"{k:12s} {d:.3e}")
print("parameters/frequencies identical:", np.array_equal(a["parameters"], b["parameters"]), np.array_equal(a["frequencies"], b["frequencies"]))
