"""
JSON schema definitions and example configurations for polymer electrolyte systems.
"""

from typing import Dict, Any

# JSON Schema for polymer electrolyte configuration
POLYMER_ELECTROLYTE_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Polymer Electrolyte Configuration",
    "type": "object",
    "required": ["protocol", "temperature", "components", "smiles"],
    "properties": {
        "protocol": {
            "type": "string",
            "enum": [
                "PolymerDensity",
                "PolymerTransport", 
                "PolymerEquilibration",
                "Density",
                "Transport",
                "HVap"
            ],
            "description": "Simulation protocol to use"
        },
        "temperature": {
            "type": "number",
            "minimum": 0,
            "description": "Temperature in Kelvin"
        },
        "pressure": {
            "type": "number",
            "default": 1.0,
            "description": "Pressure in bar"
        },
        "components": {
            "type": "object",
            "description": "System components with their properties",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["polymer", "salt_cation", "salt_anion", "solvent", "additive"]
                    },
                    "count": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Number of molecules/chains"
                    },
                    "monomer_smiles": {
                        "type": "string",
                        "description": "SMILES of monomer unit (for polymers)"
                    },
                    "degree_of_polymerization": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Number of repeat units (for polymers)"
                    },
                    "tacticity": {
                        "type": "string",
                        "enum": ["isotactic", "syndiotactic", "atactic"],
                        "default": "atactic"
                    },
                    "end_groups": {
                        "type": "object",
                        "properties": {
                            "left": {"type": "string"},
                            "right": {"type": "string"}
                        }
                    }
                }
            }
        },
        "smiles": {
            "type": "object",
            "description": "SMILES strings for non-polymer components",
            "additionalProperties": {"type": "string"}
        },
        "box_builder": {
            "type": "string",
            "enum": ["gromacs", "packmol", "amorphous"],
            "default": "packmol",
            "description": "Tool to use for building simulation box"
        },
        "target_density": {
            "type": "number",
            "description": "Target density in g/cm³"
        },
        "box_size": {
            "type": "number",
            "description": "Initial box size in nm (if not using target_density)"
        },
        "equilibration_stages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "ensemble": {"type": "string", "enum": ["minimize", "nvt", "npt"]},
                    "steps": {"type": "integer"},
                    "restraint_fc": {"type": "number"}
                }
            }
        },
        "npt_steps": {
            "type": "integer",
            "default": 5000000,
            "description": "Number of NPT equilibration steps"
        },
        "nvt_steps": {
            "type": "integer", 
            "default": 10000000,
            "description": "Number of NVT production steps"
        },
        "params_dir": {
            "type": "string",
            "description": "Directory containing force field parameters"
        },
        "output_dir": {
            "type": "string",
            "description": "Output directory for simulation files"
        }
    }
}


# Example configurations for common polymer electrolyte systems

EXAMPLE_PEO_LITFSI: Dict[str, Any] = {
    "protocol": "PolymerTransport",
    "temperature": 363,  # Above PEO melting point (65°C)
    "pressure": 1.0,
    "components": {
        "PEO": {
            "type": "polymer",
            "count": 20,
            "monomer_smiles": "CCO",
            "degree_of_polymerization": 50,  # ~2200 g/mol
            "tacticity": "atactic",
            "end_groups": {
                "left": "C",   # Methyl
                "right": "O"   # Hydroxyl
            }
        },
        "LI": {
            "type": "salt_cation",
            "count": 100,  # EO:Li = 10:1
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
    "target_density": 1.2,  # g/cm³
    "equilibration_stages": [
        {"name": "minimize", "ensemble": "minimize", "steps": 50000},
        {"name": "nvt_restrained", "ensemble": "nvt", "steps": 100000, "restraint_fc": 1000},
        {"name": "nvt_release", "ensemble": "nvt", "steps": 100000, "restraint_fc": 100},
        {"name": "nvt_free", "ensemble": "nvt", "steps": 200000},
        {"name": "npt_equilibrate", "ensemble": "npt", "steps": 5000000},
    ],
    "npt_steps": 10000000,
    "nvt_steps": 50000000,  # Long production for polymer systems
}


EXAMPLE_PEO_LIFSI: Dict[str, Any] = {
    "protocol": "PolymerTransport",
    "temperature": 353,
    "pressure": 1.0,
    "components": {
        "PEO": {
            "type": "polymer",
            "count": 15,
            "monomer_smiles": "CCO",
            "degree_of_polymerization": 100,
            "tacticity": "atactic",
        },
        "LI": {
            "type": "salt_cation",
            "count": 150,  # EO:Li = 10:1
        },
        "FSI": {
            "type": "salt_anion",
            "count": 150,
        }
    },
    "smiles": {
        "LI": "[Li+]",
        "FSI": "[N-](S(=O)(=O)F)S(=O)(=O)F",
    },
    "box_builder": "packmol",
    "target_density": 1.3,
}


EXAMPLE_PPO_LITFSI: Dict[str, Any] = {
    "protocol": "PolymerTransport",
    "temperature": 298,  # PPO is amorphous at room temperature
    "pressure": 1.0,
    "components": {
        "PPO": {
            "type": "polymer",
            "count": 20,
            "monomer_smiles": "CC(C)O",  # Propylene oxide
            "degree_of_polymerization": 30,
            "tacticity": "atactic",
        },
        "LI": {
            "type": "salt_cation",
            "count": 60,
        },
        "TFSI": {
            "type": "salt_anion",
            "count": 60,
        }
    },
    "smiles": {
        "LI": "[Li+]",
        "TFSI": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    },
    "box_builder": "packmol",
    "target_density": 1.1,
}


EXAMPLE_BLOCK_COPOLYMER: Dict[str, Any] = {
    "protocol": "PolymerTransport",
    "temperature": 363,
    "components": {
        "PEO-PS": {
            "type": "polymer",
            "count": 10,
            "blocks": [
                {"smiles": "CCO", "dp": 50, "name": "PEO"},
                {"smiles": "CC(c1ccccc1)", "dp": 20, "name": "PS"},
            ],
            "architecture": "diblock",
        },
        "LI": {
            "type": "salt_cation",
            "count": 50,
        },
        "TFSI": {
            "type": "salt_anion",
            "count": 50,
        }
    },
    "smiles": {
        "LI": "[Li+]",
        "TFSI": "[N-](S(=O)(=O)C(F)(F)F)S(=O)(=O)C(F)(F)F",
    },
    "box_builder": "packmol",
}


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration against schema.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary to validate
    
    Returns
    -------
    bool
        True if valid, raises exception otherwise
    """
    try:
        import jsonschema
        jsonschema.validate(config, POLYMER_ELECTROLYTE_SCHEMA)
        return True
    except ImportError:
        # Basic validation without jsonschema
        required = ["protocol", "temperature", "components", "smiles"]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        return True
    except jsonschema.ValidationError as e:
        raise ValueError(f"Configuration validation failed: {e.message}")


def get_example_config(system_name: str) -> Dict[str, Any]:
    """
    Get example configuration for a named system.
    
    Parameters
    ----------
    system_name : str
        Name of the system (e.g., 'PEO_LiTFSI', 'PPO_LiTFSI')
    
    Returns
    -------
    dict
        Example configuration dictionary
    """
    examples = {
        'PEO_LiTFSI': EXAMPLE_PEO_LITFSI,
        'PEO_LiFSI': EXAMPLE_PEO_LIFSI,
        'PPO_LiTFSI': EXAMPLE_PPO_LITFSI,
        'block_copolymer': EXAMPLE_BLOCK_COPOLYMER,
    }
    
    if system_name not in examples:
        available = ', '.join(examples.keys())
        raise ValueError(f"Unknown system: {system_name}. Available: {available}")
    
    return examples[system_name].copy()