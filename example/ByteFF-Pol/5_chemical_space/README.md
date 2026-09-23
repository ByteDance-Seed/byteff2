# Chemical Space Examples

This example demonstrates the chemical space coverage of our training set, and how to compute similarity between a test molecule and molecules in the training set.

## Usage
1. Functional group identification example:
```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} OMP_NUM_THREADS=1 python functional_group.py
```
This example will generate a folder `functional_group_examples` containing example molecules with identified functional groups.

2. Similarity computation example:
```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} OMP_NUM_THREADS=1 python compute_similarity.py
```
The result shows that the similarity between `ClC(C(Cl)(Cl)Cl)C(Cl)(Cl)Cl` and the training set is 0.9164.

3. Correlation between similarity and density error:
```bash
PYTHONPATH=$(git rev-parse --show-toplevel):${PYTHONPATH} OMP_NUM_THREADS=1 python similarity_correlation.py
```
This example will generate two figures: `sim_error_scatter.png` and `sim_error_stat.png`.