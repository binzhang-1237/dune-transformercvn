#!/usr/bin/env bash
# Install the patched MinkowskiEngine 0.5.4 into the active Conda environment.
set -euo pipefail
: "${CONDA_PREFIX:?Activate the target Conda environment first.}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export MAX_JOBS=2
export OMP_NUM_THREADS=4
export TORCH_CUDA_ARCH_LIST=8.6

# Check prerequisites before creating build files or installing anything.
"$CONDA_PREFIX/bin/python" - <<'PY'
import os, sys, subprocess
from importlib.metadata import version
from pathlib import Path
import torch
assert Path(sys.prefix).resolve() == Path(os.environ['CONDA_PREFIX']).resolve()
assert sys.version_info[:2] == (3, 10), 'Python 3.10 is required.'
assert torch.__version__.split('+')[0] == '2.0.1', 'PyTorch 2.0.1 is required.'
assert torch.version.cuda == '11.7', 'PyTorch CUDA 11.7 is required.'
assert version('pip') == '22.0.2', 'This build command requires pip 22.0.2.'
nvcc = Path(os.environ['CUDA_HOME']) / 'bin/nvcc'
assert 'release 11.7' in subprocess.check_output([str(nvcc), '--version'], text=True)
for compiler in ['/usr/bin/gcc', '/usr/bin/g++']:
    assert subprocess.check_output([compiler, '-dumpversion'], text=True).strip().split('.')[0] == '11', 'GCC/G++ 11 is required.'
print('Target environment:', sys.prefix)
PY

build_root="$(dirname -- "$CONDA_PREFIX")/minkowskiengine-builds"
mkdir -p "$build_root"
export TMPDIR="$build_root"
build_dir="$(mktemp -d "$build_root/build-XXXXXX")"
printf 'Build directory: %s\n' "$build_dir"
cleanup_build() {
  result=$?
  if [[ "$result" == 0 ]]; then
    rm -rf -- "$build_dir"
  else
    printf 'Installation failed; build files retained at %s\n' "$build_dir" >&2
  fi
}
trap cleanup_build EXIT
git clone --depth 1 --branch v0.5.4 https://github.com/NVIDIA/MinkowskiEngine.git "$build_dir/source"
cd "$build_dir/source"
[[ "$(git rev-parse HEAD)" == 9f81ae66b33b883cd08ee4f64d08cf633608b118 ]] || {
  echo "Unexpected MinkowskiEngine source revision." >&2
  exit 1
}
"$CONDA_PREFIX/bin/python" - <<'PY'
from pathlib import Path
p = Path('src/spmm.cu')
s = p.read_text()
needle = '#include <ATen/cuda/CUDAContext.h>'
assert needle in s
p.write_text(s.replace(needle, '#include <ATen/ATen.h>\n' + needle, 1))
for p in Path('MinkowskiEngine').rglob('*.py'):
    s = p.read_text()
    patched = s.replace('from collections import Sequence, namedtuple',
                        'from collections.abc import Sequence\nfrom collections import namedtuple')
    patched = patched.replace('from collections import Sequence\n',
                              'from collections.abc import Sequence\n')
    if patched != s:
        p.write_text(patched)
PY
"$CONDA_PREFIX/bin/python" -m pip install --no-cache-dir --no-deps --no-build-isolation . \
  --install-option="--blas=openblas" --install-option="--force_cuda"
# Import the installed package rather than the source tree.
cd "$build_dir"
"$CONDA_PREFIX/bin/python" - <<'PY'
import torch
import MinkowskiEngine as ME
print('MinkowskiEngine:', ME.__version__)
print('Installed at:', ME.__file__)
assert ME.__version__ == '0.5.4'
if not torch.cuda.is_available():
    raise RuntimeError('Installation finished; GPU verification requires an available GPU.')
features = torch.randn(2, 2, device='cuda', requires_grad=True)
coordinates = torch.tensor([[0, 0, 0], [0, 1, 1]], dtype=torch.int32)
x = ME.SparseTensor(features=features, coordinates=coordinates, device='cuda')
conv = ME.MinkowskiConvolution(2, 4, kernel_size=3, dimension=2).cuda()
y = conv(x)
y.F.sum().backward()
assert tuple(y.F.shape) == (2, 4) and features.grad is not None
print('CUDA forward/backward: OK')
PY
"$CONDA_PREFIX/bin/python" -m pip check
