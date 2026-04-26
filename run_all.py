import subprocess
import sys
import datetime

configs = [
    "configs/gpleiss_densenet121_standard.yaml",
    "configs/gpleiss_densenet121_checkpoint.yaml",
    "configs/densenet121_scratch.yaml",
    "configs/densenet121_pretrained.yaml",
    "configs/resnet50_pretrained.yaml",
    "configs/googlenet_pretrained.yaml",
]

batch_start_dt = datetime.datetime.now()
batch_start_str = batch_start_dt.strftime("%Y%m%d_%H%M%S")
print(f"Starting batch run at {batch_start_str} with {len(configs)} configs.\n")

for cfg in configs:
    print(f"\nRunning: {cfg}\n")
    subprocess.run([
        sys.executable,
        "run_experiment.py",
        "--config",
        cfg
    ])

batch_end_dt = datetime.datetime.now()
batch_end_str = batch_end_dt.strftime("%Y%m%d_%H%M%S")

elapsed = batch_end_dt - batch_start_dt
total_seconds = int(elapsed.total_seconds())
hours, remainder = divmod(total_seconds, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"\n{'='*50}")
print(f"Batch run started:   {batch_start_str}")
print(f"Batch run completed: {batch_end_str}")
print(f"Total elapsed time:  {hours:02d}h {minutes:02d}m {seconds:02d}s")
print(f"{'='*50}")
