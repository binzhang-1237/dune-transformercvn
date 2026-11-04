"""Run the README training entry point with bounded training and validation batches."""
import argparse
import os
from pathlib import Path
import runpy
import sys

import pytorch_lightning as pl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu')
    parser.add_argument('--steps', type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root))
    original_trainer = pl.Trainer

    def bounded_trainer(*positional, **kwargs):
        kwargs.update(accelerator=args.device, devices=1, strategy='auto',
                      max_epochs=1, max_steps=args.steps,
                      limit_train_batches=args.steps, limit_val_batches=2,
                      val_check_interval=args.steps, num_sanity_val_steps=0)
        return original_trainer(*positional, **kwargs)

    pl.Trainer = bounded_trainer
    sys.argv = ['train.py', '-o', str(root / 'option_files/fdhd_beam_2018prod_aiml_tutorial_2025_04_21.json'),
                '-n', f'environment_check_{args.device}_bounded',
                '--gpus', '1' if args.device == 'gpu' else '0',
                '--threads', '4', '-e', str(args.steps)]
    try:
        runpy.run_path(str(root / 'train.py'), run_name='__main__')
    finally:
        pl.Trainer = original_trainer


if __name__ == '__main__':
    main()
