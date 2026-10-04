# PyTorch 2.2.1 compatibility update

This project now targets Python 3.10 and PyTorch 2.2.1. The model architecture,
HTTP response format and conditional LangGraph execution remain unchanged.

Changes:

- requirements.txt pins torch==2.2.1 instead of requiring torch>=2.6.
- NumPy 1.26.4 is pinned for the older PyTorch binary baseline.
- FastAPI 0.115.8, Pydantic 2.10.6, LangGraph 0.2.76 and Uvicorn 0.34.0 provide
  a Python-3.10-compatible application baseline.
- Artifact loading explicitly uses strict state dictionaries and weights_only=True,
  which is available in PyTorch 2.2.1. No weights_only=False fallback was added.
- Both bundled model files were loaded and re-saved using PyTorch 2.2.1 as CPU
  tensor-only state dictionaries with pickle protocol 2. Weights were not retrained
  or changed. Training now writes the same portable checkpoint format.
- check_environment.py reports interpreter/package versions and diagnoses the
  reported iJIT_NotifyEvent import failure.

## Your existing Conda environment

Your reported combination is PyTorch 2.2.1 with MKL 2025.0.0 and Intel OpenMP
2025.0.0. The known native-library incompatibility must be fixed separately:

```bash
conda activate stock_pred
conda install "mkl=2024.0.0" "intel-openmp=2024.0.0" "numpy=1.26.4"
python -c "import torch; print(torch.__version__)"
python -m pip install -r requirements.txt
python check_environment.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Conda may adjust related MKL packages to satisfy these constraints. Restart
running Python processes after the change. If the solver cannot find a consistent
environment, inspect that conflict rather than deleting packages manually.

No Python source change can fix a missing native symbol encountered while
importing torch. This release removes the project's unnecessary version-upgrade
requirement; it does not bypass the native loader.

The project does not require torchvision, torchaudio or CUDA. Your existing
CUDA-enabled torch 2.2.1 can remain installed; inference defaults to CPU.

Reference: https://github.com/pytorch/pytorch/issues/123097

## Verification

- Python 3.10.21 with PyTorch 2.2.1+cpu: all 19 project tests passed.
- Both bundled artifacts loaded successfully; the CLI returned the expected
  context and employee identifier using this environment.
- One-epoch training smoke tests for both tasks passed with PyTorch 2.2.1+cpu
  under Python 3.12.14; temporary outputs did not replace the bundled weights.
- Your specific Conda CUDA/MKL installation was not available for execution.
  Its reported native-library import error still needs the separate repair above.
