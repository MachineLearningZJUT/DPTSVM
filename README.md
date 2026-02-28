# Double-Side Projection Twin SVM (DPTSVM)

This repository provides a MATLAB implementation for the paper  **Double-Side Projection Twin SVM (DPTSVM) for robust classification**. 

## Overview

Specifically, our DPTSVM  learns class-specific projection directions by minimizing within-class dispersion under $L_1$ criteria, while simultaneously enforcing a double-sided hinge loss to ensure that opposite-class projections lie outside a margin band centered around each class. We implement three solvers for our DPTSVM model using MM framework:
- **QP** (Quadratic Programming via MATLAB's `quadprog`)
- **SOR** (Successive Over-Relaxation)
- **ADMM** (Alternating Direction Method of Multipliers)

---

## Overview

The demo includes:
- data loading and preprocessing,
- model training and prediction,
- model evaluation

The main script reproduces the results on two synthetic datasets,`noisyA.mat` and `noisyB.mat`.

---

## Repository Structure

```text
.
├── DPTSVM.m            # DPTSVM class (train/test + SOR/QP/ADMM solvers)
├── MainforDPTSVM.m     # Main demo script
├── noisyA.mat          # Synthetic dataset (noisyA)
├── noisyB.mat          # Synthetic dataset (noisyB)
└── README.md           # The readme file
```

---

## Requirements

- MATLAB (recommended: R2024a or later)
- Optimization Toolbox (required if using `solver = "QP"` due to `quadprog`)

---

## Quick Start

1. Open MATLAB and change directory to this folder.
2. Run:

```matlab
MainforDPTSVM
```

3. The script will load the dataset, train the DPTSVM model and prediction on the test set.

```matlab
% For DPTSVM, set parameters:
Param.c1;   % weight for L1 within-class term
Param.c2;    % weight for double-sided hinge loss
% Param.solver;  % options: "SOR", "QP", "ADMM"

Mld = model(Param); % create DPTSVM model
Mld = Mld.train(TrainData); % train model
Res = Mld.test(TestData); % test model
```

---

## Notes

- This is a demo version. 
- The code is optimized for clarity and reproducibility.
- Computation time may vary depending on the size of simulations and hardware specifications.

---

## Citation

If you use this implementation in research, please cite the corresponding paper/project where DPTSVM is introduced.


