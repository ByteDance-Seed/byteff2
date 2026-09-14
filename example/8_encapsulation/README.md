# Electrolyte property examples

Each `run_*.py` builds a liquid electrolyte with `PropertiesCalculator` (or an ionic liquid with `IonicLiquidPropertiesCalculator`) and calculates its properties (density, dielectric, viscosity, conductivity). Run one with `python run_xxx.py`; results land in the script's `BASE_DIR`.

Species are given either **by name**, if they are documented in `byteff2/toolkit/properties_calculator.py`, or **by SMILES** via `custom_smiles` if they are not.

| Script | System | Input style |
|---|---|---|
| `run_by_name.py` | LiPF6 in EC:DMC 3:7 | all species by name |
| `run_by_anion_smiles.py` | LiFTA in EC:DMC 3:7 | anion by SMILES, solvents by name |
| `run_by_solvent_smiles.py` | LiPF6 in DMSO | solvent by SMILES, anion by name |
| `run_full_smiles.py` | LiFTA in EC:DMC 5:5 | undocumented species by SMILES |
| `run_full_smiles_IL.py` | EMIM/Li/BMIM + TFSI/FSI/PF6 ionic liquid (no solvent) | every ion by SMILES |
| `run_solvent_only.py` | pure EC:DMC 5:5, no ions | solvents by name, `li_count=0` |

To calculate pure-solvent properties (no ions), follow `run_solvent_only.py`: set `li_count=0`, omit `anion`, and request only `density`, `viscosity`, `dielectric` (conductivity is meaningless without ions).

ByteFF2 is trained on C, H, F, Li, N, O, P, S. Species outside the training inventory (e.g. DMSO, imidazolium cations) still run, but their parameters are extrapolated and unvalidated.
