#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run dune-transformercvn training with a saved command.")
    parser.add_argument(
        "--options-file",
        type=Path,
        default=Path("option_files/fdhd_beam_2018prod_aiml_tutorial_2025_04_21.json"),
        help="Path to the training options JSON file.")
    parser.add_argument(
        "--name",
        type=str,
        default="tutorial_dense",
        help="Run name used by the training script.")
    parser.add_argument(
        "--eval",
        type=int,
        default=None,
        help="Number of training steps between evaluations.")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use AMP for training.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with smaller batch size and no workers.")

    args = parser.parse_args()

    command = [
        sys.executable,
        "train.py",
        "-o",
        str(args.options_file),
        "-n",
        args.name,
    ]
    if args.eval is not None:
        command.extend(["-e", str(args.eval)])

    if args.fp16:
        command.append("-fp16")
    if args.debug:
        command.append("-d")

    print("Running command:", " ".join(command))
    process = subprocess.run(command)

    if process.returncode == 0:
        print("Training finished successfully.")
    else:
        print(f"Training exited with return code {process.returncode}.")
        sys.exit(process.returncode)


if __name__ == "__main__":
    main()
