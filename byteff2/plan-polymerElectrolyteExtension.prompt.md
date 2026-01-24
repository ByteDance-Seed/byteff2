# Plan for Extending ByteFF2 to Polymer Electrolyte Systems

## Executive Summary

Yes, extending ByteFF2 to support polymer electrolytes is **feasible**, though it requires careful planning. The main challenges are:

1. **Box construction for polymers** - GROMACS `gmx insert-molecules` (used in `generate_system_gro`) is designed for small molecules, not polymer chains
2. **Force field parameterization** - Polymers require handling of repeating units and potentially very large molecules
3. **MD simulation protocols** - Polymers need longer equilibration and different analysis methods

---

## Phase 1: Analysis of Current Limitations

### 1.1 Current Box Building Approach

The current system uses GROMACS tools in `generate_system_gro`:

```python
# Current approach for small molecules
script.init_gro_box(f"{c.name}.gro", box)
script.insert_molecules(f"{c.name}.gro", rest_molecules)
```

**Problem**: `gmx insert-molecules` cannot handle polymer chains properly because:
- It treats each molecule as rigid during insertion
- Polymers need to be grown/relaxed into the box
- Chain entanglement and proper packing require specialized algorithms

### 1.2 PEMD's Polymer Building Approach

Looking at the PEMD directory structure, it likely uses tools like:
- **Packmol** - Can pack pre-built polymer chains
- **Polymatic/EMC** - For amorphous polymer building
- **Custom chain builders** - For specific polymer architectures

---

## Phase 2: Implementation Plan

### 2.1 Create Polymer-Aware Protocol Classes

**New file**: `byteff2/toolkit/polymer_protocol.py`

```python
# filepath: byteff2/toolkit/polymer_protocol.py
from enum import Enum
from byteff2.toolkit.protocol import Protocol, ComponentType, Component

class PolymerType(Enum):
    LINEAR = "linear"
    BRANCHED = "branched"
    CROSSLINKED = "crosslinked"
    BLOCK_COPOLYMER = "block_copolymer"

class PolymerComponent(Component):
    """Extended Component class for polymer systems."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.polymer_type = kwargs.get('polymer_type', PolymerType.LINEAR)
        self.degree_of_polymerization = kwargs.get('dp', 1)
        self.monomer_smiles = kwargs.get('monomer_smiles', None)
        self.end_groups = kwargs.get('end_groups', None)
        self.tacticity = kwargs.get('tacticity', 'atactic')  # isotactic, syndiotactic, atactic

class PolymerElectrolyteProtocol(Protocol):
    """Protocol for polymer electrolyte MD simulations."""
    
    def __init__(self, params_dir: str, output_dir: str):
        super().__init__(params_dir, output_dir)
        self.polymer_builder = None  # Will be set based on available tools
    
    def build_polymer_system(self, ...):
        """Build polymer electrolyte box using appropriate method."""
        pass
```

### 2.2 Implement Multiple Box Building Backends

**New file**: `byteff2/toolkit/box_builders.py`

```python
# filepath: byteff2/toolkit/box_builders.py
from abc import ABC, abstractmethod
import subprocess
import os

class BoxBuilder(ABC):
    """Abstract base class for system box builders."""
    
    @abstractmethod
    def build_box(self, components, box_size, output_dir) -> str:
        """Build simulation box and return path to output .gro file."""
        pass
    
    @abstractmethod
    def supports_polymers(self) -> bool:
        pass

class GMXBoxBuilder(BoxBuilder):
    """Original GROMACS-based builder for small molecules."""
    
    def supports_polymers(self) -> bool:
        return False
    
    def build_box(self, components, box_size, output_dir):
        # Existing generate_system_gro logic
        pass

class PackmolBoxBuilder(BoxBuilder):
    """Packmol-based builder supporting polymers."""
    
    def supports_polymers(self) -> bool:
        return True
    
    def build_box(self, components, box_size, output_dir):
        """
        Use Packmol for initial placement, then convert to GROMACS format.
        """
        packmol_input = self._generate_packmol_input(components, box_size)
        # Run packmol
        # Convert output to .gro format
        pass
    
    def _generate_packmol_input(self, components, box_size):
        """Generate Packmol input file."""
        lines = [
            "tolerance 2.0",
            "filetype pdb",
            f"output {output_dir}/system.pdb",
        ]
        for comp in components:
            lines.extend([
                f"structure {comp.pdb_path}",
                f"  number {comp.molar_num}",
                f"  inside box 0. 0. 0. {box_size*10} {box_size*10} {box_size*10}",
                "end structure",
            ])
        return "\n".join(lines)

class AmorphousBuilderBackend(BoxBuilder):
    """
    For building amorphous polymer systems using tools like:
    - Polymatic
    - EMC (Enhanced Monte Carlo)
    - Custom MC-based growth
    """
    
    def supports_polymers(self) -> bool:
        return True
    
    def build_box(self, components, box_size, output_dir, 
                  target_density=None, compression_steps=10000):
        """
        Build amorphous polymer box with proper chain packing.
        """
        pass
```

### 2.3 Polymer Chain Generation

**New file**: `byteff2/toolkit/polymer_builder.py`

```python
# filepath: byteff2/toolkit/polymer_builder.py
from bytemol.core import Molecule
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

class PolymerChainBuilder:
    """Build polymer chains from monomer SMILES."""
    
    def __init__(self, monomer_smiles: str, dp: int, 
                 end_group_left: str = None, end_group_right: str = None):
        self.monomer_smiles = monomer_smiles
        self.dp = dp
        self.end_groups = (end_group_left, end_group_right)
    
    def build_chain(self) -> Molecule:
        """
        Build a polymer chain by connecting monomers.
        Returns a Molecule object compatible with ByteFF2.
        """
        # 1. Parse monomer with connection points (e.g., [*] atoms)
        # 2. Connect monomers sequentially
        # 3. Add end groups
        # 4. Generate 3D conformer
        # 5. Return as Molecule object
        pass
    
    def _connect_monomers(self, mol1, mol2):
        """Connect two monomers at their reactive sites."""
        pass
    
    def generate_mapped_smiles(self) -> str:
        """Generate atom-mapped SMILES for the full polymer chain."""
        pass

class PEOBuilder(PolymerChainBuilder):
    """Specialized builder for Poly(ethylene oxide)."""
    
    def __init__(self, dp: int):
        super().__init__(
            monomer_smiles="[*]OCC[*]",  # EO monomer with connection points
            dp=dp,
            end_group_left="C",  # Methyl cap
            end_group_right="O"   # Hydroxyl cap
        )

class PolymerLibrary:
    """Pre-defined polymer builders for common systems."""
    
    POLYMERS = {
        'PEO': PEOBuilder,
        'PPO': lambda dp: PolymerChainBuilder("[*]OC(C)C[*]", dp),
        'PVDF': lambda dp: PolymerChainBuilder("[*]C(F)(F)C[*]", dp),
        'PAN': lambda dp: PolymerChainBuilder("[*]CC(C#N)[*]", dp),
    }
```

### 2.4 Modify Force Field Parameter Generation

The current `generate_ff_params` needs extension for large molecules:

```python
# filepath: byteff2/toolkit/protocol.py (modifications)

class Protocol:
    # ...existing code...
    
    def generate_ff_params(self, component_smiles: dict, force: bool = False,
                          polymer_mode: bool = False):
        """
        Generate force field parameters.
        
        For polymers, can use:
        1. Full-chain parameterization (small polymers)
        2. Fragment-based parameterization (large polymers)
        """
        model_dir = get_data_file_path('trained_models/optimal.pt', 'byteff2')
        model = load_model(os.path.dirname(model_dir))
        all_nb_params = {}

        for mol_name, smiles in component_smiles.items():
            if polymer_mode and self._is_large_molecule(smiles):
                # Fragment-based approach for large polymers
                params = self._generate_polymer_params_fragmented(
                    model, mol_name, smiles
                )
            else:
                # Original approach for small molecules
                params = self._generate_small_molecule_params(
                    model, mol_name, smiles, force
                )
            all_nb_params[mol_name] = params
        
        return all_nb_params
    
    def _generate_polymer_params_fragmented(self, model, mol_name, polymer_smiles):
        """
        Generate parameters by fragmenting polymer into representative units.
        
        Strategy:
        1. Identify unique chemical environments (monomer types, end groups)
        2. Generate parameters for each fragment
        3. Map parameters back to full chain
        """
        pass
    
    def _is_large_molecule(self, smiles: str, max_atoms: int = 500) -> bool:
        """Check if molecule is too large for direct parameterization."""
        mol = Chem.MolFromSmiles(smiles)
        return mol.GetNumAtoms() > max_atoms if mol else False
```

### 2.5 Polymer-Specific Simulation Protocols

**New file**: `byteff2/toolkit/polymer_simulation.py`

```python
# filepath: byteff2/toolkit/polymer_simulation.py
from byteff2.toolkit.protocol import TransportProtocol
from byteff2.md_utils.md_run import npt_run, nvt_run

class PolymerEquilibrationProtocol:
    """
    Multi-stage equilibration for polymer systems.
    
    Polymers require longer equilibration:
    1. Energy minimization with soft-core potentials
    2. NVT with position restraints on backbone
    3. Gradual release of restraints
    4. NPT compression to target density
    5. Production NVT/NPT
    """
    
    def __init__(self, config):
        self.config = config
        self.stages = [
            {'name': 'minimize', 'steps': 10000},
            {'name': 'nvt_restrained', 'steps': 100000, 'restraint_fc': 1000},
            {'name': 'nvt_release', 'steps': 100000, 'restraint_fc': [1000, 100, 10, 0]},
            {'name': 'npt_compress', 'steps': 500000},
            {'name': 'npt_equilibrate', 'steps': 2000000},
        ]
    
    def run_equilibration(self, top, system, positions):
        """Run multi-stage equilibration."""
        for stage in self.stages:
            if stage['name'] == 'minimize':
                self._run_minimization(system, positions)
            elif 'nvt' in stage['name']:
                self._run_nvt_stage(top, system, positions, stage)
            elif 'npt' in stage['name']:
                self._run_npt_stage(top, system, positions, stage)

class PolymerTransportProtocol(TransportProtocol):
    """
    Extended transport protocol for polymer electrolytes.
    
    Additional analyses:
    - Polymer chain diffusion (center of mass)
    - Ion hopping statistics
    - Coordination number dynamics
    - Transference number via concentrated solution theory
    """
    
    def post_process(self):
        """Extended post-processing for polymer systems."""
        super().post_process()
        
        # Additional polymer-specific analyses
        self._compute_chain_diffusion()
        self._compute_ion_coordination()
        self._compute_vehicular_vs_structural_diffusion()
    
    def _compute_chain_diffusion(self):
        """Compute polymer chain center-of-mass diffusion."""
        pass
    
    def _compute_ion_coordination(self):
        """Analyze ion-polymer coordination over trajectory."""
        pass
```

### 2.6 Configuration Schema Updates

**New file**: `byteff2/toolkit/config_schemas.py`

```python
# filepath: byteff2/toolkit/config_schemas.py
"""JSON schema definitions for configuration files."""

POLYMER_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["protocol", "temperature", "components"],
    "properties": {
        "protocol": {"enum": ["PolymerTransport", "PolymerDensity", "PolymerEquilibration"]},
        "temperature": {"type": "number"},
        "components": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "type": {"enum": ["polymer", "salt_cation", "salt_anion", "solvent"]},
                    "count": {"type": "integer"},
                    "monomer_smiles": {"type": "string"},
                    "degree_of_polymerization": {"type": "integer"},
                    "tacticity": {"enum": ["isotactic", "syndiotactic", "atactic"]},
                }
            }
        },
        "box_builder": {"enum": ["gromacs", "packmol", "amorphous"]},
        "equilibration_stages": {"type": "array"},
    }
}

# Example config for PEO/LiTFSI system
EXAMPLE_POLYMER_CONFIG = {
    "protocol": "PolymerTransport",
    "temperature": 363,  # Above Tm for PEO
    "components": {
        "PEO": {
            "type": "polymer",
            "count": 20,  # 20 chains
            "monomer_smiles": "CCO",
            "degree_of_polymerization": 50,  # ~2200 g/mol
        },
        "LI": {
            "type": "salt_cation",
            "count": 100,  # EO:Li ratio of 10:1
        },
        "TFSI": {
            "type": "salt_anion", 
            "count": 100,
        }
    },
    "smiles": {
        "LI": "[Li+]",
        "TFSI": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    },
    "box_builder": "packmol",
    "target_density": 1.1,  # g/cm³
    "npt_steps": 10000000,  # Longer for polymers
    "nvt_steps": 50000000,
}
```

---

## Phase 3: Integration with Existing Codebase

### 3.1 Modify Main Entry Points

Update `run_md.py`:

```python
# filepath: example/4_MD_simulations/run_md.py (modifications)
import json
import byteff2.toolkit.protocol as protocol
from byteff2.toolkit.polymer_protocol import PolymerElectrolyteProtocol

PROTOCOL_MAP = {
    "Density": protocol.DensityProtocol,
    "Transport": protocol.TransportProtocol,
    "HVap": protocol.HVapProtocol,
    # New polymer protocols
    "PolymerDensity": PolymerElectrolyteProtocol,
    "PolymerTransport": PolymerElectrolyteProtocol,
}

if __name__ == '__main__':
    # ...existing code...
    protocol_class = PROTOCOL_MAP.get(protocol_name)
    if protocol_class is None:
        raise ValueError(f"Unknown protocol: {protocol_name}")
    
    p = protocol_class(params_dir=config["params_dir"], output_dir=config["output_dir"])
    p.config = config
    p.run_protocol()
```

### 3.2 Update the Protocol Factory

```python
# filepath: byteff2/toolkit/protocol.py (additions at end)

def get_protocol(config: dict) -> Protocol:
    """Factory function to get appropriate protocol."""
    protocol_name = config.get("protocol", "")
    
    # Check if polymer system
    is_polymer = any(
        comp.get("type") == "polymer" 
        for comp in config.get("components", {}).values()
        if isinstance(comp, dict)
    )
    
    if is_polymer:
        from byteff2.toolkit.polymer_protocol import (
            PolymerDensityProtocol,
            PolymerTransportProtocol,
        )
        protocol_map = {
            "Density": PolymerDensityProtocol,
            "Transport": PolymerTransportProtocol,
            "PolymerDensity": PolymerDensityProtocol,
            "PolymerTransport": PolymerTransportProtocol,
        }
    else:
        protocol_map = {
            "Density": DensityProtocol,
            "Transport": TransportProtocol,
            "HVap": HVapProtocol,
        }
    
    return protocol_map.get(protocol_name)
```

---

## Phase 4: Testing Strategy

### 4.1 Unit Tests

```python
# filepath: byteff2/tests/toolkit/test_polymer_builder.py
import pytest
from byteff2.toolkit.polymer_builder import PolymerChainBuilder, PEOBuilder

class TestPolymerBuilder:
    
    def test_peo_chain_creation(self):
        """Test PEO chain building."""
        builder = PEOBuilder(dp=10)
        chain = builder.build_chain()
        
        # Check expected number of atoms
        # PEO: (CH2-CH2-O)n with end groups
        expected_atoms = 10 * 7 + 4  # Approximate
        assert abs(len(chain.atoms) - expected_atoms) < 5
    
    def test_mapped_smiles_generation(self):
        """Test that mapped SMILES is valid."""
        builder = PEOBuilder(dp=5)
        mapped_smiles = builder.generate_mapped_smiles()
        
        # Verify atom mapping is complete
        from rdkit import Chem
        mol = Chem.MolFromSmiles(mapped_smiles)
        assert mol is not None
```

### 4.2 Integration Tests

```python
# filepath: byteff2/tests/toolkit/test_polymer_protocol.py
import pytest
import tempfile
from byteff2.toolkit.polymer_protocol import PolymerElectrolyteProtocol

class TestPolymerProtocol:
    
    @pytest.fixture
    def peo_litfsi_config(self):
        return {
            "protocol": "PolymerDensity",
            "temperature": 363,
            "components": {
                "PEO": {"type": "polymer", "count": 5, "dp": 20},
                "LI": {"type": "salt_cation", "count": 10},
                "TFSI": {"type": "salt_anion", "count": 10},
            },
            "smiles": {
                "LI": "[Li+]",
                "TFSI": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
            },
            "npt_steps": 1000,  # Short for testing
        }
    
    def test_polymer_system_build(self, peo_litfsi_config):
        """Test that polymer system can be built."""
        with tempfile.TemporaryDirectory() as tmpdir:
            protocol = PolymerElectrolyteProtocol(
                params_dir=f"{tmpdir}/params",
                output_dir=f"{tmpdir}/output"
            )
            protocol.config = peo_litfsi_config
            
            # Should not raise
            protocol.build_polymer_system(...)
```

---

## Phase 5: Migration Path from PEMD

### 5.1 Identify Reusable Components from PEMD

Based on the PEMD directory, extract and adapt:

1. **Polymer building routines** → Integrate into `polymer_builder.py`
2. **Packmol interfaces** → Integrate into `box_builders.py`
3. **Equilibration protocols** → Adapt for polarizable FF
4. **Analysis scripts** → Port to use ByteFF2 trajectory formats

### 5.2 Compatibility Layer

```python
# filepath: byteff2/toolkit/pemd_compat.py
"""
Compatibility layer to use PEMD utilities with ByteFF2.
"""

def convert_pemd_topology_to_byteff2(pemd_top_path: str, output_dir: str):
    """
    Convert PEMD OPLS topology to ByteFF2 format.
    
    This allows reusing PEMD-built systems with ByteFF2 force fields.
    """
    pass

def import_pemd_polymer_structure(pemd_gro: str, pemd_top: str) -> tuple:
    """
    Import a polymer structure built with PEMD tools.
    
    Returns (positions, topology) compatible with ByteFF2.
    """
    pass
```

---

## Summary: Implementation Roadmap

| Phase | Task | Estimated Effort | Priority |
|-------|------|------------------|----------|
| 1.1 | Implement `PackmolBoxBuilder` | 1-2 days | High |
| 1.2 | Create `PolymerChainBuilder` base class | 2-3 days | High |
| 1.3 | Add PEO/PPO/common polymer builders | 2 days | High |
| 2.1 | Fragment-based parameterization | 3-4 days | Medium |
| 2.2 | Multi-stage equilibration protocol | 2-3 days | High |
| 2.3 | Polymer transport analysis | 2-3 days | Medium |
| 3.1 | Integration tests | 2 days | High |
| 3.2 | Documentation and examples | 1-2 days | Medium |
| 4.1 | PEMD compatibility layer | 2-3 days | Low |

**Total estimated time**: 3-4 weeks for core functionality

---

## Key Technical Decisions

1. **Box building**: Use **Packmol** as primary backend for polymer systems (well-tested, handles large molecules)

2. **Parameterization strategy**: For polymers > 500 atoms, use **fragment-based approach** where unique chemical environments are parameterized once and mapped to full chain

3. **Equilibration**: Implement **staged protocol** with restraint release - essential for proper polymer packing

4. **Backward compatibility**: All changes should be **additive** - existing small molecule workflows unchanged
