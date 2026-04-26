# Quick script to run both gpleiss densenet121 experiments sequentially and save results in the same directory structure as before.
# This is just for convenience and to avoid having to run each experiment separately from the command line.

import subprocess
import sys
import datetime

configs = [
    "configs/gpleiss_densenet121_checkpoint.yaml",
    "configs/gpleiss_densenet121_standard.yaml",
    "configs/densenet121_scratch.yaml",
    "configs/densenet121_pretrained.yaml",
    "configs/resnet50_pretrained.yaml",
    "configs/googlenet_pretrained.yaml",
]

batch_run_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Starting batch run at {batch_run_time} with configs: {configs}\n")

for cfg in configs:
    print(f"\nRunning: {cfg}\n")
    subprocess.run([
        sys.executable,
        "run_experiment.py",
        "--config",
        cfg
    ])
    
batch_end_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"\nStarted batch run at {batch_run_time}")
print(f"\nCompleted batch run at {batch_end_time}.")
print(f"Total time for batch run: {batch_end_time - batch_run_time}")