"""Simulations and high-dimensional formulas for "Silent Alignment in Two-Layer Linear Networks".

Modules:
- `paths`: folders of the repository;
- `simulator`: finite-dimensional gradient descent (needs PyTorch);
- `theory`: high-dimensional formulas (NumPy and SciPy only);
- `exact_flow`: reference solutions of the gradient flow, for the checks;
- `laws`: densities of the singular values of C and deterministic spectra;
- `configs`: configurations shared by the figure scripts and the validation runs;
- `runs`: command-line options and saving of the simulation archives;
- `animations`: GIFs of the spectra along training.

Nothing is imported here, so that the theory code can be used without PyTorch.
"""