"""Plot total greyscaler runtime for each GPU block size."""

import csv
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
BLOCK_SIZES = (32, 64, 128, 256, 512, 1024)


def main() -> None:
    results = []

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "greyscale.png"
        for block_size in BLOCK_SIZES:
            command = [
                sys.executable, str(HERE / "greyscale.py"),
                "--device", "gpu",
                "--block-size", str(block_size),
                "--output", str(output),
            ]
            start = perf_counter()
            run = subprocess.run(command, capture_output=True, text=True)
            seconds = perf_counter() - start
            if run.returncode:
                raise RuntimeError(run.stderr)
            results.append((block_size, seconds))
            print(f"{block_size:4d} threads/block: {seconds:.3f} s")

    with (HERE / "block_size_benchmark.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("block_size", "seconds"))
        writer.writerows(results)

    sizes, times = zip(*results)
    plt.figure(figsize=(8, 4.5))
    positions = range(len(sizes))
    plt.plot(positions, times, marker="o")
    plt.xticks(positions, sizes)
    plt.xlabel("Threads per block")
    plt.ylabel("Total runtime (seconds)")
    plt.title("Greyscaler GPU runtime by block size")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(HERE / "block_size_benchmark.png", dpi=180)
    print("Saved block_size_benchmark.png and block_size_benchmark.csv")


if __name__ == "__main__":
    main()
