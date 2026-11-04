#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source /mnt/ironwolf_14t2/users/binzhang/local/transformercvn-legacy/bin/activate
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1

exec python -u train.py -o option_files/fdhd_beam_2026prod.json -n training_2026prod
