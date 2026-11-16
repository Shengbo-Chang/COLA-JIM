# COLA-JIM

Code and data for **Contact-aware Online Load Allocation for Composite Wing Panel Installation in Aircraft Manufacturing**, *Journal of Intelligent Manufacturing*.

## Installation

Python 3.12.

```bash
python -m pip install -r requirements.txt
```

## Files

- `cola.py`: contact-aware load allocation.
- `baselines.py`: Uniform, fixed-response, Scan-GA and COBYLA.
- `fem.py`, `model/panel.npz`: panel model and 20 gap cases.
- `data/`: CSV tables.

## Interface

`PanelModel.load(case=1)` loads a case. `acquire(forces)` accepts 18 loads (N) and returns 29 displacements (mm). Controllers average `Config.averages` acquisitions (default: 20); `Config.noise_std_mm` is the noise standard deviation per acquisition.

The FEM model is simplified; results may differ from the paper.
