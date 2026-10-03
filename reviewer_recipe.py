"""Run the reproduction stages in dependency order."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

STEPS = (
    ("integrity check", ("check",)),
    ("configuration selection", ("train", "--stage", "selection")),
    ("selection report", ("selection-report",)),
    ("main training", ("train", "--stage", "main")),
    ("validation analysis", ("analyze", "--stage", "main")),
    ("locked test export", ("evaluate", "--stage", "main")),
    ("test analysis", ("analyze", "--stage", "main", "--split", "test")),
    ("supplementary training A", ("train", "--stage", "S1")),
    ("supplementary report A", ("supplement-report", "--stage", "S1")),
    ("supplementary training B", ("train", "--stage", "S2")),
    ("supplementary report B", ("supplement-report", "--stage", "S2")),
    ("supplementary training C", ("train", "--stage", "S3")),
    ("supplementary report C", ("supplement-report", "--stage", "S3")),
    ("efficiency analysis", ("s4", "--stage", "main", "--include-s2")),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/reviewer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Use the short interface-check recipe")
    parser.add_argument("--resume", action="store_true", help="Verify and skip completed units")
    parser.add_argument("--execute", action="store_true", help="Required for training and evaluation")
    args = parser.parse_args()
    if not args.execute:
        parser.error("add --execute to run the reviewer recipe")
    if args.timeout_seconds < 0:
        parser.error("--timeout-seconds must be nonnegative")

    common = ["--run-root", str(args.run_root.resolve()), "--device", args.device]
    if args.smoke:
        common.append("--smoke")
    if args.resume:
        common.append("--resume")
    if args.timeout_seconds:
        common.extend(["--timeout-seconds", str(args.timeout_seconds)])

    for index, (label, step) in enumerate(STEPS, 1):
        command = [sys.executable, "-B", str(ROOT / "reproduce.py"), *step]
        if args.smoke and step[0] != "check":
            # Exercise every stage with a small, dependency-complete subset.
            stage = step[step.index("--stage") + 1] if "--stage" in step else "selection"
            methods = ["cfc"] if stage == "S2" else ["p0"]
            if step[0] == "s4":
                methods = ["p0", "cfc"]
            command.extend(["--horizons", "24", "--seeds", "2021", "--methods", *methods])
        if step[0] != "check":
            command.append("--execute")
        command.extend(common)
        print(f"[{index}/{len(STEPS)}] {label}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print("Reviewer recipe completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
