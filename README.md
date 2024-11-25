# multistream-change-detectors
This repository contains code implementing the $e$-d-BH, $e$-d-Bonferroni, and $e$-d-Holm algorithms for multi-stream change detection problems. We empirically verify that these algorithms control the inverse patience weighted rates (IPWR), provide universal control over Type I error, and analyze their detection delay.

# Usage
Run the following commands from the root directory before running any Python scripts in this repository:
```bash
export PYTHONPATH=./src
conda create -n <envname> --file requirements.txt
conda activate <envname>
```
This sets the `PYTHONPATH` for relative imports and installs all required packages into a fresh conda environment.
