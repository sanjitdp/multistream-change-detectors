# multistream-change-detectors

Code for *Multiple testing in multi-stream sequential change detection*. The
repository implements e-d-BH, e-d-Bonferroni, and e-d-Holm, and contains the
simulations used to study their error rates and detection delays.

## Paper experiments

The revised experiment scripts are in [`experiments/`](experiments/), with
fixed seeds and design specifications. See the
[experiment README](experiments/README.md) for the full reproduction sequence
and definitions of the reported metrics. Running the studies creates the
result files used by the table and figure scripts.

Create a Python 3.11 environment from the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/requirements.txt
```

| Study | Script |
| --- | --- |
| Gaussian comparisons with external identification methods | `experiments/external_identification.py` |
| Gaussian identification times | `experiments/tradeoff_benchmark.py` |
| Heavy-tailed and dependent count observations | `experiments/diverse_benchmark.py` |
| Gaussian error rates and ARL calibration | `experiments/gaussian_benchmark.py` |
| External global detectors | `experiments/global_baselines.py` |

The global detectors also require a C++17 compiler. The Python script compiles
`global_baselines.cpp` automatically. These simulation studies use generated
data and need no downloads.

Run the method checks with:

```sh
python experiments/validate_bayesian_methods.py
python experiments/external_identification.py --validate-only
python experiments/global_baselines.py --checks-only
```

After running the external identification and tradeoff simulations, generate
the main Gaussian tables with:

```sh
python experiments/external_tables.py --tables-only
python experiments/paper_tables.py
```

Full simulation runs and validation commands write to `experiments/results/`.
The optional public sensor analysis is documented separately in the experiment
README; it is exploratory and is not part of the paper's empirical validation.

## Original implementation and notebook

The original reusable implementation is in `src/multistream_algorithms/`,
and `src/simulations.ipynb` contains the original experiments. Its Conda
environment is separate from the Python environment above. From the repository
root, run:

```sh
conda create -n multistream-original --file requirements.txt
conda activate multistream-original
export PYTHONPATH=./src
```
