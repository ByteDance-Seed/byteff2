# Polymer Electrolyte Simulations with ByteFF2

This directory contains examples for running MD simulations of polymer electrolyte systems
using the ByteFF2 polarizable force field.

## Quick Start

```bash
# Run with example configuration
python run_polymer_md.py --example PEO_LiTFSI --output-dir ./peo_litfsi_output

# Or use custom configuration
python run_polymer_md.py my_config.json

# Dry run to check configuration
python run_polymer_md.py --example PEO_LiTFSI --dry-run
```

## Available Example Systems

| System | Description | Temperature |
|--------|-------------|-------------|
| `PEO_LiTFSI` | Poly(ethylene oxide) + LiTFSI | 363 K |
| `PEO_LiFSI` | Poly(ethylene oxide) + LiFSI | 353 K |
| `PPO_LiTFSI` | Poly(propylene oxide) + LiTFSI | 298 K |
| `block_copolymer` | PEO-PS block copolymer + LiTFSI | 363 K |

## Configuration File Format

```json
{
  "protocol": "PolymerTransport",
  "temperature": 363,
  "components": {
    "PEO": {
      "type": "polymer",
      "count": 20,
      "monomer_smiles": "CCO",
      "degree_of_polymerization": 50
    },
    "LI": {"type": "salt_cation", "count": 100},
    "TFSI": {"type": "salt_anion", "count": 100}
  },
  "smiles": {
    "LI": "[Li+]",
    "TFSI": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F"
  },
  "box_builder": "packmol",
  "target_density": 1.2
}
```

## Polymer Component Options

| Property | Type | Description |
|----------|------|-------------|
| `type` | string | Must be "polymer" |
| `count` | int | Number of polymer chains |
| `monomer_smiles` | string | SMILES of repeat unit |
| `degree_of_polymerization` | int | Number of repeat units |
| `tacticity` | string | "isotactic", "syndiotactic", or "atactic" |
| `end_groups` | object | Left and right end group SMILES |

## Workflow

1. **Polymer chain building**: Generate polymer chains from monomer SMILES
2. **Force field parameterization**: Use ByteFF2 for polarizable parameters
3. **Box construction**: Pack polymer chains and ions using Packmol
4. **Multi-stage equilibration**:
   - Energy minimization
   - NVT with position restraints
   - Gradual restraint release
   - NPT equilibration to target density
5. **Production run**: Long NVT or NPT for transport properties
6. **Analysis**: Diffusion coefficients, conductivity, transference numbers

## Output Files

After simulation, the output directory contains:

- `config.json` - Copy of input configuration
- `system.gro` - Final system coordinates
- `topol.top` - GROMACS topology
- `production.xtc` - Production trajectory
- `polymer_analysis.json` - Polymer-specific analysis results
- `transport.json` - Transport property results

## Requirements

- ByteFF2 and dependencies
- GROMACS (2020+)
- Packmol
- MDAnalysis (for analysis)

## Notes for Polymer Systems

1. **Equilibration time**: Polymers require much longer equilibration than small molecules.
   The default settings provide 10+ ns equilibration.

2. **Production length**: For accurate diffusion coefficients, production runs of 50-100 ns
   are typically needed.

3. **Temperature**: Run above the polymer melting temperature for semi-crystalline polymers
   (e.g., 363 K for PEO with Tm ≈ 338 K).

4. **System size**: Use at least 10-20 polymer chains to avoid finite-size effects.