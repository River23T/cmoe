#!/usr/bin/env python3
"""Diagnostic script to check rsl_rl installation.
Run with: isaaclab -p check_rsl_rl.py
"""
import sys
print(f"Python: {sys.executable}")

# 1. Check metadata version
try:
    import importlib.metadata as metadata
    ver = metadata.version("rsl-rl-lib")
    print(f"rsl-rl-lib metadata version: {ver}")
except Exception as e:
    print(f"rsl-rl-lib metadata: {e}")

try:
    ver2 = metadata.version("rsl-rl")
    print(f"rsl-rl (old package) metadata version: {ver2}")
except Exception:
    print("rsl-rl (old package): not installed")

# 2. Check actual module location and contents
import rsl_rl
print(f"\nrsl_rl module location: {rsl_rl.__file__}")
print(f"rsl_rl dir: {sorted([x for x in dir(rsl_rl) if not x.startswith('_')])}")

import os
rsl_rl_dir = os.path.dirname(rsl_rl.__file__)
print(f"\nFiles/dirs in {rsl_rl_dir}:")
for item in sorted(os.listdir(rsl_rl_dir)):
    print(f"  {item}")

# 3. Check specific submodules
for mod in ['rsl_rl.models', 'rsl_rl.storage', 'rsl_rl.extensions', 'rsl_rl.utils', 'rsl_rl.runners', 'rsl_rl.algorithms']:
    try:
        __import__(mod)
        print(f"\n✓ {mod} - OK")
    except ImportError as e:
        print(f"\n✗ {mod} - MISSING: {e}")

# 4. Check if there are multiple rsl_rl installations
import importlib
spec = importlib.util.find_spec("rsl_rl")
if spec:
    print(f"\nrsl_rl spec origin: {spec.origin}")
    print(f"rsl_rl spec submodule_search_locations: {spec.submodule_search_locations}")

# 5. Check pip list for rsl
import subprocess
result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
print("\nInstalled rsl packages:")
for line in result.stdout.splitlines():
    if 'rsl' in line.lower():
        print(f"  {line}")
