import subprocess
import sys

# splits to test
splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#splits = [0.1, 0.2]

for frac in splits:
    print("\n" + "=" * 60)
    print(f"Running split: {int(frac * 100)}/{int((1 - frac) * 100)}")
    print("=" * 60)

    # run your experiment script
    subprocess.run([
        sys.executable,
        "run_split_experiment.py",
        "--train-fraction",
        str(frac)
    ])
