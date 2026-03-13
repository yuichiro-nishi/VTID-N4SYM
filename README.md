# Vanishing Theorems and Incomplete Deformations for Vacuum Data in N=4 SYM

This repository contains the LaTeX source, compiled PDF, and computational scripts for the paper.

## Abstract

We formulate the behavior of 1-point functions in the source-free, symmetric vacuum of an exact conformal theory, taking AdS_5 x S^5 and its boundary dual N=4 SU(N) super Yang-Mills theory as the reference system.

**Main results:**

1. **Unified vanishing theorem**: In the Poincare-invariant, conformally invariant, source-free vacuum of N=4 SYM, the 1-point functions of all operators in the family C_ex (local scalar primaries, conserved currents, and the energy-momentum tensor) vanish.

2. **Deformation analysis**: Under deformations (relevant deformations, finite temperature, finite chemical potential, vacuum selection, time-dependent backgrounds, etc.), we systematize the conditions under which nonzero 1-point functions are permitted.

3. **Four-layer classification**: We classify observables into four layers and prove that the vanishing theorem is specific to Layer I (local 1-point data); Layers II-IV carry nontrivial data even in the symmetric vacuum.

4. **Layer-resolved response theorem**: Representative deformations trigger systematic responses across all four layers. Of the 20 cells formed by representative deformations and the four layers, 19 are supported by literature evidence, explicit computation, or the QFT-on-de Sitter framework.

## Repository Structure

```
.
├── LICENSE
├── README.md
├── paper/
│   ├── VTID_en.tex      # LaTeX source
│   ├── VTID_en.pdf      # Compiled PDF
│   └── figures/         # Figures used in the paper
└── scripts/
    ├── sec6_wilson_loop.py
    ├── sec7_finite_density_corr.py
    ├── sec7_finite_density_info.py
    └── sec7_relevant_defect.py
```

## Requirements

- LaTeX distribution with standard packages (amsmath, amssymb, hyperref, etc.)
- Python 3.x with SymPy for running computational scripts

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use this work, please cite:

```bibtex
@article{Nishi2026VTID,
  author  = {Nishi, Yuichiro},
  title   = {Vanishing Theorems and Incomplete Deformations for Vacuum Data in {N=4} {SYM}},
  year    = {2026},
  note    = {arXiv:XXXX.XXXXX [hep-th]}
}
```

## Contact

For questions or comments, please open an issue in this repository.
