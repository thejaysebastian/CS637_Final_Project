# Quick script to run both gpleiss densenet121 experiments sequentially and save results in the same directory structure as before.
# This is just for convenience and to avoid having to run each experiment separately from the command line.

import subprocess
import sys

configs = [
    "configs/gpleiss_densenet121_checkpoint.yaml",
    "configs/gpleiss_densenet121_standard.yaml"
]

for cfg in configs:
    print(f"\nRunning: {cfg}\n")
    subprocess.run([
        sys.executable,
        "run_experiment.py",
        "--config",
        cfg
    ])