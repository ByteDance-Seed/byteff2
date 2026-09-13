# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from rdkit import Chem

from byteff2.toolkit.protocol import (
    DensityProtocol,
    DielectricProtocol,
    TransportProtocol,
    CompressibilityProtocol,
)
from byteff2.utils.mol_inventory import all_name_mapped_smiles
from bytemol.utils import setup_default_logging

logger = setup_default_logging()


def _load_inventory_databases() -> tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Split the ByteFF2 molecule inventory into cations, anions and solvents.

    Species are bucketed by net formal charge, and the inventory's atom-mapped
    SMILES are reduced to plain canonical SMILES (the form the protocols and the
    example configs consume). Deriving the tables here rather than duplicating
    them keeps byteff2.utils.mol_inventory the single source of truth.
    """
    cations: Dict[str, str] = {}
    anions: Dict[str, str] = {}
    solvents: Dict[str, str] = {}
    for name, mapped_smiles in all_name_mapped_smiles.items():
        mol = Chem.MolFromSmiles(mapped_smiles)
        if mol is None:
            raise ValueError(f"mol_inventory entry {name!r} is not valid SMILES: {mapped_smiles!r}")
        charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        plain_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        bucket = cations if charge > 0 else anions if charge < 0 else solvents
        bucket[name] = plain_smiles
    return cations, anions, solvents


_CATION_DATABASE, _ANION_DATABASE, _SOLVENT_DATABASE = _load_inventory_databases()


class PropertiesLogger:
    """Enhanced logging for PropertiesCalculator with progress tracking."""

    def __init__(self, base_dir: Path, verbose: bool = True):
        self.base_dir = Path(base_dir)
        self.log_file = self.base_dir / "properties_calculator.log"
        self.verbose = verbose
        self.progress = 0

    def setup(self):
        """Setup file and console logging."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # File handler
        file_handler = logging.FileHandler(str(self.log_file), mode='a')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def log_section(self, title: str, progress: int = 0):
        """Log a section header with progress."""
        self.progress = progress
        line = "=" * 80
        logger.info(line)
        logger.info(f"{title} [{progress}%]")
        logger.info(line)
        if self.verbose:
            print(f"\n{line}\n{title} [{progress}%]\n{line}")

    def log_step(self, step: str, details: str = ""):
        """Log a workflow step."""
        msg = f"  ✓ {step}"
        if details:
            msg += f": {details}"
        logger.info(msg)
        if self.verbose:
            print(msg)

    def log_result(self, name: str, value: float, unit: str = ""):
        """Log a calculated result."""
        msg = f"    {name:30s}: {value:12.6f} {unit}".rstrip()
        logger.info(msg)
        if self.verbose:
            print(msg)

    def log_error(self, msg: str):
        """Log an error."""
        logger.error(f"✗ {msg}")
        if self.verbose:
            print(f"✗ {msg}", file=sys.stderr)

    def log_summary(self, summary_dict: Dict):
        """Log a formatted summary."""
        line = "=" * 80
        logger.info(line)
        logger.info("CALCULATION SUMMARY")
        logger.info(line)
        for key, value in summary_dict.items():
            if isinstance(value, dict):
                logger.info(f"\n{key}:")
                for k, v in value.items():
                    logger.info(f"  {k:30s}: {v}")
            else:
                logger.info(f"{key:30s}: {value}")
        if self.verbose:
            print("\n" + line)
            print("CALCULATION SUMMARY")
            print(line)
            for key, value in summary_dict.items():
                if isinstance(value, dict):
                    print(f"\n{key}:")
                    for k, v in value.items():
                        print(f"  {k:30s}: {v}")
                else:
                    print(f"{key:30s}: {value}")


class PropertiesCalculator:
    """
    Unified interface for calculating molecular dynamics properties.

    Simplified input format:
    - solvent: one solvent name ("EC"), several names (["EC", "DMC"]) with an
      optional solvent_ratio ("3:7"), or a dict {"EC": 3, "DMC": 7}
    - anion: Name of anion (e.g., "PF6", "TFSI")
    - li_count: Number of Li ions; salt_to_solvent_ratio sets the total number
      of solvent molecules, which solvent_ratio then splits between solvents

    Default settings:
    - natoms: 5000
    - cation: LI (lithium)

    Supports:
    - Density calculation
    - Conductivity calculation
    - Viscosity calculation
    - Dielectric constant calculation
    - Compressibility calculation

    Can be called programmatically or via CLI.
    """

    # Fixed cation (always lithium)
    CATION = "LI"
    CATION_SMILES = _CATION_DATABASE["LI"]

    # Both tables are derived from byteff2.utils.mol_inventory, so they list
    # exactly the species this force field was parameterised for.
    ANION_DATABASE = _ANION_DATABASE
    SOLVENT_DATABASE = _SOLVENT_DATABASE

    def __init__(
        self,
        solvent: Union[str, Sequence[str], Dict[str, float]],
        anion: Optional[str] = None,
        li_count: int = 34,
        salt_to_solvent_ratio: Optional[float] = None,
        salt_to_solvent_ratio_str: Optional[str] = None,
        solvent_ratio: Optional[Union[str, Sequence[float]]] = None,
        total_solvent: Optional[int] = None,
        temperature: float = 298.0,
        base_dir: str = "./md_simulations",
        verbose: bool = True,
        custom_smiles: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the properties calculator with simplified input.

        Parameters
        ----------
        solvent : str, list of str, or dict
            One solvent name ("EC"), several names (["EC", "DMC"]), or a dict
            mapping names to their molar proportions ({"EC": 3, "DMC": 7}).
        anion : str, optional
            Name of anion (e.g., "PF6", "TFSI"). Required when `li_count > 0`
            (the normal, salted case); must be omitted when `li_count == 0`
            (pure-solvent mode, see `total_solvent`).
        li_count : int
            Number of Li ions (default: 34). Set to 0 for a pure-solvent, no-salt
            system -- in that mode, use `total_solvent` instead of a ratio to size
            the system, and `anion`/`salt_to_solvent_ratio(_str)` must be omitted.
        salt_to_solvent_ratio : float, optional
            Ratio of salt pairs to *total* solvent molecules (e.g., 1/10 or 0.1).
            If provided, total solvent count = li_count / ratio (rounded down).
            Only meaningful when `li_count > 0`.
        salt_to_solvent_ratio_str : str, optional
            Alternative format: "1:10" means 1 salt pair : 10 solvent molecules.
            Only meaningful when `li_count > 0`.
        solvent_ratio : str or list of float, optional
            Molar proportions between the solvents, in the order given by
            `solvent`: "3:7" or [3, 7]. Defaults to equal parts. Ignored when
            `solvent` is a dict (the dict values are the proportions).
        total_solvent : int, optional
            Total number of solvent molecules. Required (and only used) when
            `li_count == 0`: there is no salt ratio to derive the solvent count
            from, so it must be given directly. `solvent_ratio` still controls
            how this total is split between multiple solvents.
        temperature : float
            Temperature in Kelvin (default: 298.0)
        base_dir : str
            Base directory for all simulations (default: "./md_simulations")
        custom_smiles : dict, optional
            Extra {name: SMILES} entries for species absent from
            SOLVENT_DATABASE / ANION_DATABASE, e.g. {"MYSOL": "CCOCC"}. Those
            names then work anywhere a solvent or anion name is accepted. Names
            already in the inventory are rejected rather than overridden.
            ByteFF2 predicts parameters from the molecular graph, so custom
            species do run -- but they sit outside the training inventory, so
            their parameters are extrapolated and unvalidated.
        """
        self.custom_smiles = self._parse_custom_smiles(custom_smiles)
        self.solvents, self.solvent_ratio = self._parse_solvents(solvent, solvent_ratio)
        # Human-readable label, e.g. "EC" or "EC/DMC"
        self.solvent = "/".join(self.solvents)
        self.anion = anion
        self.li_count = li_count
        if li_count < 0:
            raise ValueError(f"li_count must be >= 0, got {li_count}")
        if li_count == 0:
            if total_solvent is None or total_solvent <= 0:
                raise ValueError(
                    "total_solvent (a positive int) is required when li_count=0 (pure-solvent "
                    "mode): there is no salt ratio to derive the solvent count from."
                )
            if salt_to_solvent_ratio is not None or salt_to_solvent_ratio_str is not None:
                raise ValueError(
                    "salt_to_solvent_ratio/salt_to_solvent_ratio_str don't apply when li_count=0 "
                    "(pure-solvent mode); use total_solvent instead."
                )
        else:
            if anion is None:
                raise ValueError("anion is required when li_count > 0")
            if total_solvent is not None:
                raise ValueError("total_solvent only applies when li_count=0 (pure-solvent mode)")
        self.total_solvent = total_solvent
        self.temperature = temperature
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Setup logging
        self.logger = PropertiesLogger(self.base_dir, verbose=verbose)
        self.logger.setup()

        # Log initialization
        self.logger.log_section("PropertiesCalculator Initialization", progress=0)
        solvent_desc = self.solvent
        if len(self.solvents) > 1:
            solvent_desc += " (" + ":".join(f"{r:g}" for r in self.solvent_ratio) + ")"
        self.logger.log_step(f"Solvent: {solvent_desc}, Anion: {anion}, Li count: {li_count}")

        # Validate and get SMILES
        self.solvent_smiles = self._get_solvent_smiles()
        self.anion_smiles = self._get_anion_smiles() if self.li_count > 0 else None

        # Parse salt:solvent ratio (not applicable in pure-solvent mode)
        self.salt_to_solvent_ratio = (
            self._parse_ratio(salt_to_solvent_ratio, salt_to_solvent_ratio_str) if self.li_count > 0 else None
        )

        # Build components dictionary
        self.components = self._build_components()

        # Build SMILES dictionary
        self.smiles = self._build_smiles()

        self.logger.log_step(f"Components: {self.components}")
        self.logger.log_step(f"Working directory: {self.base_dir}")

        self.results = {}

    @classmethod
    def _parse_custom_smiles(cls, custom_smiles: Optional[Dict[str, str]]) -> Dict[str, str]:
        """Validate user-supplied {name: SMILES} and canonicalise the SMILES."""
        if not custom_smiles:
            return {}

        known = set(cls.SOLVENT_DATABASE) | set(cls.ANION_DATABASE) | {cls.CATION}
        parsed = {}
        for name, smi in custom_smiles.items():
            name = str(name).strip()
            if not name:
                raise ValueError(f"Empty name in custom_smiles: {custom_smiles!r}")
            if name in known:
                raise ValueError(
                    f"custom_smiles name {name!r} already exists in the ByteFF2 inventory "
                    f"(SMILES {cls.SOLVENT_DATABASE.get(name) or cls.ANION_DATABASE.get(name)!r}). "
                    f"Pick a different name rather than overriding a parameterised species."
                )
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"custom_smiles[{name!r}] is not valid SMILES: {smi!r}")
            parsed[name] = Chem.MolToSmiles(mol)
            logger.warning(
                f"{name} is outside the ByteFF2 training inventory; its force field "
                f"parameters will be extrapolated and are unvalidated."
            )
        return parsed

    def _lookup_smiles(self, name: str, database: Dict[str, str], kind: str) -> str:
        """Resolve a species name against the inventory, then custom_smiles."""
        if name in database:
            return database[name]
        if name in self.custom_smiles:
            return self.custom_smiles[name]
        raise ValueError(
            f"Unknown {kind}: {name!r}. Known {kind}s: {sorted(database)}. "
            f"Pass custom_smiles={{{name!r}: '<SMILES>'}} to define it directly."
        )

    @staticmethod
    def _parse_solvents(
        solvent: Union[str, Sequence[str], Dict[str, float]],
        solvent_ratio: Optional[Union[str, Sequence[float]]],
    ) -> tuple[List[str], List[float]]:
        """Normalise the solvent spec into (names, molar proportions)."""
        if isinstance(solvent, dict):
            names = list(solvent.keys())
            ratio = [float(v) for v in solvent.values()]
        else:
            if isinstance(solvent, str):
                names = [s.strip() for s in solvent.split(",")]
            else:
                names = [str(s).strip() for s in solvent]
            if solvent_ratio is None:
                ratio = [1.0] * len(names)
            elif isinstance(solvent_ratio, str):
                try:
                    ratio = [float(x) for x in solvent_ratio.split(":")]
                except ValueError:
                    raise ValueError(f"Invalid solvent_ratio format: {solvent_ratio!r} (expected e.g. '3:7')")
            else:
                ratio = [float(x) for x in solvent_ratio]

        if not names or any(not n for n in names):
            raise ValueError(f"Empty solvent name in {solvent!r}")
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate solvent names in {names}")
        if len(ratio) != len(names):
            raise ValueError(f"solvent_ratio has {len(ratio)} entries but {len(names)} solvents were given: {names}")
        if any(r <= 0 for r in ratio):
            raise ValueError(f"solvent_ratio entries must be positive, got {ratio}")
        return names, ratio

    def _get_solvent_smiles(self) -> Dict[str, str]:
        """Get SMILES for every solvent, keyed by name."""
        return {s: self._lookup_smiles(s, self.SOLVENT_DATABASE, "solvent") for s in self.solvents}

    def _get_anion_smiles(self) -> str:
        """Get SMILES for anion."""
        return self._lookup_smiles(self.anion, self.ANION_DATABASE, "anion")

    def _parse_ratio(self, ratio_float: Optional[float], ratio_str: Optional[str]) -> float:
        """Parse salt:solvent ratio from float or string format."""
        if ratio_str:
            # Parse "1:10" format
            parts = ratio_str.split(":")
            if len(parts) == 2:
                try:
                    return float(parts[0]) / float(parts[1])
                except (ValueError, ZeroDivisionError):
                    raise ValueError(f"Invalid ratio format: {ratio_str}")

        if ratio_float is not None:
            return ratio_float

        # Default: 1:10 (1 salt pair : 10 solvent molecules)
        return 1.0 / 10.0

    @staticmethod
    def _apportion(total: int, weights: Sequence[float]) -> List[int]:
        """Split `total` into integers proportional to `weights` (largest remainder)."""
        wsum = float(sum(weights))
        exact = [total * w / wsum for w in weights]
        counts = [int(x) for x in exact]
        remainder = total - sum(counts)
        for i in sorted(range(len(weights)), key=lambda i: exact[i] - counts[i], reverse=True)[:remainder]:
            counts[i] += 1
        return counts

    def _build_components(self) -> Dict[str, int]:
        """Build components dictionary from salt:solvent ratio and solvent ratio."""
        if self.li_count == 0:
            # Pure solvent: no CATION/anion entries at all. protocol.py's
            # search_mixture() divides by np.min(mol_ratio), so a zero-count LI/anion
            # entry would break it -- omit the keys entirely rather than including
            # them as 0.
            solvent_counts = self._apportion(self.total_solvent, self.solvent_ratio)
            if any(c < 1 for c in solvent_counts):
                raise ValueError(
                    f"solvent_ratio {self.solvent_ratio} leaves a solvent with zero molecules "
                    f"out of {self.total_solvent} total; increase total_solvent or adjust the ratio"
                )
            components = dict(zip(self.solvents, solvent_counts))
            logger.info(f"Components: {components} (pure solvent, no salt)")
            return components

        # salt_to_solvent_ratio = salt_pairs / total solvent molecules
        # total solvent count = li_count / ratio (rounded down), then split
        # between the solvents according to solvent_ratio.
        total_solvent = int(self.li_count / self.salt_to_solvent_ratio)
        solvent_counts = self._apportion(total_solvent, self.solvent_ratio)
        if any(c < 1 for c in solvent_counts):
            raise ValueError(
                f"Solvent ratio {self.solvent_ratio} leaves a solvent with zero molecules "
                f"out of {total_solvent} total; increase li_count or adjust the ratio"
            )

        components = dict(zip(self.solvents, solvent_counts))
        components[self.CATION] = self.li_count
        components[self.anion] = self.li_count
        logger.info(f"Components: {components} (salt:solvent {self.salt_to_solvent_ratio:.4f}, "
                    f"solvent ratio {':'.join(f'{r:g}' for r in self.solvent_ratio)})")
        return components

    def _build_smiles(self) -> Dict[str, str]:
        """Build SMILES dictionary."""
        smiles = dict(self.solvent_smiles)
        if self.li_count > 0:
            smiles[self.CATION] = self.CATION_SMILES
            smiles[self.anion] = self.anion_smiles
        return smiles

    def _count_atoms(self) -> int:
        """Exact atom count of the requested composition, hydrogens included."""
        total = 0
        for name, count in self.components.items():
            mol = Chem.AddHs(Chem.MolFromSmiles(self.smiles[name]))
            total += count * mol.GetNumAtoms()
        return total

    def _create_config(self, protocol_name: str, params_dir: str, output_dir: str) -> dict:
        """Create configuration for a protocol."""
        # protocol.search_mixture treats `components` as a ratio and scales it up
        # until it reaches `natoms`. Passing the composition's exact atom count
        # makes that scale factor land on 1, preserving the requested counts.
        config = {
            "protocol": protocol_name,
            "params_dir": params_dir,
            "output_dir": output_dir,
            "working_dir": str(self.base_dir / f"{protocol_name.lower()}_working_dir"),
            "temperature": int(self.temperature),
            "natoms": self._count_atoms(),
            "components": self.components,
            "smiles": self.smiles,
        }
        return config

    def _save_config(self, config: dict, config_file: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved config to {config_file}")

    def _run_protocol(self, protocol_class, config: dict) -> dict:
        """Run a protocol and return results."""
        try:
            protocol = protocol_class(config)
            logger.info(f"Running {protocol_class.__name__}")
            protocol.run_protocol()
            protocol.post_process()

            # Read results from output file
            output_dir = config["output_dir"]
            results_file = os.path.join(output_dir, "results.json")

            # Try different result file names based on protocol
            if not os.path.exists(results_file):
                if protocol_class == DensityProtocol:
                    results_file = os.path.join(output_dir, "density_results.json")
                elif protocol_class == DielectricProtocol:
                    results_file = os.path.join(output_dir, "dielectric_results.json")

            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    return json.load(f)
            else:
                logger.warning(f"Results file not found for {protocol_class.__name__}")
                return {}
        except Exception as e:
            logger.error(f"Error running {protocol_class.__name__}: {e}")
            raise

    def calculate_density(self) -> dict:
        """Calculate density."""
        protocol_name = "Density"
        params_dir = str(self.base_dir / "density_params")
        output_dir = str(self.base_dir / "density_results")

        config = self._create_config(protocol_name, params_dir, output_dir)
        config_file = str(self.base_dir / "density_config.json")
        self._save_config(config, config_file)

        logger.info("Calculating density...")
        result = self._run_protocol(DensityProtocol, config)
        self.results["density"] = result
        return result

    def calculate_transport(self) -> dict:
        """Calculate conductivity and viscosity."""
        protocol_name = "Transport"
        params_dir = str(self.base_dir / "transport_params")
        output_dir = str(self.base_dir / "transport_results")

        config = self._create_config(protocol_name, params_dir, output_dir)
        config_file = str(self.base_dir / "transport_config.json")
        self._save_config(config, config_file)

        logger.info("Calculating conductivity and viscosity...")
        result = self._run_protocol(TransportProtocol, config)
        self.results["transport"] = result
        return result

    def calculate_dielectric(self) -> dict:
        """Calculate dielectric constant."""
        protocol_name = "Dielectric"
        params_dir = str(self.base_dir / "dielectric_params")
        output_dir = str(self.base_dir / "dielectric_results")

        config = self._create_config(protocol_name, params_dir, output_dir)
        config_file = str(self.base_dir / "dielectric_config.json")
        self._save_config(config, config_file)

        logger.info("Calculating dielectric constant...")
        result = self._run_protocol(DielectricProtocol, config)
        self.results["dielectric"] = result
        return result

    def calculate_compressibility(self) -> dict:
        """Calculate compressibility."""
        protocol_name = "Compressibility"
        params_dir = str(self.base_dir / "compressibility_params")
        output_dir = str(self.base_dir / "compressibility_results")

        config = self._create_config(protocol_name, params_dir, output_dir)
        config_file = str(self.base_dir / "compressibility_config.json")
        self._save_config(config, config_file)

        # Add compressibility-specific parameters
        config["npt_steps"] = 5000000

        logger.info("Calculating compressibility...")
        result = self._run_protocol(CompressibilityProtocol, config)
        self.results["compressibility"] = result
        return result

    def calculate(
        self,
        properties: Optional[List[str]] = None,
    ) -> dict:
        """
        Calculate specified properties.

        Parameters
        ----------
        properties : List[str], optional
            List of properties to calculate. Options:
            ["density", "conductivity", "viscosity", "dielectric", "compressibility"]
            If None, calculates all.

        Returns
        -------
        dict
            Dictionary containing all calculated results.
        """
        if properties is None:
            properties = ["density", "conductivity", "viscosity", "dielectric"]

        properties_lower = [p.lower() for p in properties]
        known = {"density", "conductivity", "viscosity", "dielectric", "compressibility"}
        unknown = sorted(set(properties_lower) - known)
        if unknown:
            raise ValueError(f"Unknown propert{'y' if len(unknown) == 1 else 'ies'}: {unknown}. "
                             f"Known: {sorted(known)}")
        all_results = {}

        self.logger.log_section(
            f"Starting Property Calculations: {', '.join(properties_lower)}",
            progress=10
        )

        # Density
        if "density" in properties_lower:
            try:
                self.logger.log_step("Calculating density...")
                all_results["density"] = self.calculate_density()
                if all_results["density"]:
                    density_val = all_results["density"].get("density", 0)
                    self.logger.log_result("Density", density_val, "g/mL")
            except Exception as e:
                self.logger.log_error(f"Failed to calculate density: {e}")

        # Transport (conductivity + viscosity)
        if any(p in properties_lower for p in ["conductivity", "viscosity"]):
            try:
                self.logger.log_step("Calculating transport properties...")
                transport_result = self.calculate_transport()
                all_results["transport"] = transport_result
                if "conductivity" in properties_lower:
                    cond = transport_result.get("conductivity_onsager", 0)
                    all_results["conductivity"] = cond
                    self.logger.log_result("Conductivity (Onsager)", cond, "mS/cm")
                if "viscosity" in properties_lower:
                    visc = transport_result.get("viscosity", 0)
                    all_results["viscosity"] = visc
                    self.logger.log_result("Viscosity", visc, "cP")
            except Exception as e:
                self.logger.log_error(f"Failed to calculate transport properties: {e}")

        # Dielectric
        if "dielectric" in properties_lower:
            try:
                self.logger.log_step("Calculating dielectric constant...")
                all_results["dielectric"] = self.calculate_dielectric()
                if all_results["dielectric"]:
                    diel_val = all_results["dielectric"].get("dielectric", 0)
                    self.logger.log_result("Dielectric Constant", diel_val)
            except Exception as e:
                self.logger.log_error(f"Failed to calculate dielectric constant: {e}")

        # Compressibility
        if "compressibility" in properties_lower:
            try:
                self.logger.log_step("Calculating compressibility...")
                all_results["compressibility"] = self.calculate_compressibility()
            except Exception as e:
                self.logger.log_error(f"Failed to calculate compressibility: {e}")

        # Save and display summary
        self.logger.log_section("Calculation Complete", progress=100)
        self._save_summary(all_results)
        self._display_summary(all_results)

        return all_results

    def _save_summary(self, results: dict) -> None:
        """Save results summary to JSON."""
        summary = {
            "solvent": self.solvent,
            "solvents": self.solvents,
            "solvent_ratio": self.solvent_ratio,
            "anion": self.anion,
            "cation": self.CATION,
            "li_count": self.li_count,
            "salt_to_solvent_ratio": self.salt_to_solvent_ratio,
            "components": self.components,
            "temperature": self.temperature,
            "custom_smiles": self.custom_smiles,
            "results": results,
        }
        summary_file = self.base_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        self.logger.log_step(f"Saved summary to {summary_file}")

    def _display_summary(self, results: dict) -> None:
        """Display formatted results summary."""
        summary_data = {
            "System Composition": {
                "Solvent": self.solvent,
                "Solvent Ratio": ":".join(f"{r:g}" for r in self.solvent_ratio),
                "Anion": self.anion,
                "Cation": self.CATION,
                "Li Count": self.li_count,
                "Salt:Solvent Ratio": (
                    f"1:{int(1/self.salt_to_solvent_ratio)}" if self.li_count > 0 else "N/A (pure solvent)"
                ),
            },
            "Simulation Parameters": {
                "Temperature": f"{self.temperature} K",
                "Components": str(self.components),
            },
        }

        # Add calculated properties
        if results:
            props_data = {}
            if "density" in results and results["density"]:
                props_data["Density (g/mL)"] = f"{results['density'].get('density', 0):.6f}"
            if "conductivity" in results:
                props_data["Conductivity (mS/cm)"] = f"{results['conductivity']:.6f}"
            if "viscosity" in results:
                props_data["Viscosity (cP)"] = f"{results['viscosity']:.6f}"
            if "dielectric" in results and results["dielectric"]:
                props_data["Dielectric Constant"] = f"{results['dielectric'].get('dielectric', 0):.6f}"

            if props_data:
                summary_data["Calculated Properties"] = props_data

        self.logger.log_summary(summary_data)

    @classmethod
    def from_simple_config(
        cls,
        solvent: Union[str, Sequence[str], Dict[str, float]],
        anion: str,
        li_count: int = 34,
        salt_to_solvent_ratio: Optional[float] = None,
        salt_to_solvent_ratio_str: Optional[str] = None,
        solvent_ratio: Optional[Union[str, Sequence[float]]] = None,
        **kwargs
    ) -> "PropertiesCalculator":
        """
        Create calculator from simple parameters.

        Parameters
        ----------
        solvent : str, list of str, or dict
            Solvent name(s); a dict maps names to molar proportions
        anion : str
            Anion name
        li_count : int
            Number of Li ions (default: 34)
        salt_to_solvent_ratio : float, optional
            Ratio of salt pairs to total solvent molecules
        salt_to_solvent_ratio_str : str, optional
            Ratio in "1:10" format
        solvent_ratio : str or list of float, optional
            Molar proportions between solvents, e.g. "3:7"
        **kwargs : dict
            Additional parameters (temperature, base_dir, custom_smiles)
        """
        return cls(
            solvent=solvent,
            anion=anion,
            li_count=li_count,
            salt_to_solvent_ratio=salt_to_solvent_ratio,
            salt_to_solvent_ratio_str=salt_to_solvent_ratio_str,
            solvent_ratio=solvent_ratio,
            **kwargs
        )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Calculate MD properties (density, conductivity, viscosity, dielectric)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1:10 LiFSI:DMC (34 Li, 340 DMC)
  python -m byteff2.toolkit.properties_calculator \\
      --solvent DMC --anion FSI --ratio 1:10

  # Custom Li count
  python -m byteff2.toolkit.properties_calculator \\
      --solvent EC --anion TFSI --li-count 50 --ratio 1:8

  # Mixed solvent: 1:10 LiPF6 in EC:DMC = 3:7 (34 Li, 102 EC, 238 DMC)
  python -m byteff2.toolkit.properties_calculator \\
      --solvent EC,DMC --solvent-ratio 3:7 --anion PF6 --ratio 1:10

  # Only density
  python -m byteff2.toolkit.properties_calculator \\
      --solvent DMC --anion PF6 --ratio 1:10 \\
      --calculate density --work-dir ./density_only
        """
    )

    parser.add_argument(
        "--solvent",
        type=str,
        required=True,
        help="Solvent name, or comma-separated names for a mixture (e.g., EC or EC,DMC)",
    )
    parser.add_argument(
        "--solvent-ratio",
        type=str,
        default=None,
        help="Molar proportions between solvents, e.g. 3:7 (default: equal parts)",
    )
    parser.add_argument(
        "--anion",
        type=str,
        required=True,
        help="Anion name (e.g., PF6, TFSI, FSI)",
    )
    parser.add_argument(
        "--li-count",
        type=int,
        default=34,
        help="Number of Li ions (default: 34)",
    )
    parser.add_argument(
        "--ratio",
        type=str,
        default="1:10",
        help="Salt:solvent ratio (default: 1:10)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=298.0,
        help="Temperature in Kelvin (default: 298.0)",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="./md_simulations",
        help="Working directory (default: ./md_simulations)",
    )
    parser.add_argument(
        "--calculate",
        type=str,
        default="density,conductivity,viscosity,dielectric",
        help="Comma-separated properties to calculate",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="properties_calculator.log",
        help="Log file (default: properties_calculator.log)",
    )
    parser.add_argument(
        "--custom-smiles",
        type=str,
        action="append",
        default=None,
        metavar="NAME=SMILES",
        help="Define a species not in the ByteFF2 inventory, e.g. MYSOL=CCOCC. "
             "Repeat the flag for several species.",
    )

    args = parser.parse_args()

    custom_smiles = {}
    for entry in args.custom_smiles or []:
        name, sep, smi = entry.partition("=")
        if not sep or not name.strip() or not smi.strip():
            parser.error(f"--custom-smiles expects NAME=SMILES, got {entry!r}")
        custom_smiles[name.strip()] = smi.strip()

    try:
        # Create calculator
        calculator = PropertiesCalculator(
            solvent=args.solvent,
            anion=args.anion,
            li_count=args.li_count,
            salt_to_solvent_ratio_str=args.ratio,
            solvent_ratio=args.solvent_ratio,
            temperature=args.temperature,
            base_dir=args.work_dir,
            custom_smiles=custom_smiles,
        )

        # Parse properties to calculate
        properties = [p.strip() for p in args.calculate.split(",")]

        # Run calculations
        results = calculator.calculate(properties=properties)

        # Print summary
        print("\n" + "="*60)
        print("CALCULATION SUMMARY")
        print("="*60)
        print(f"Solvent:  {calculator.solvent}")
        if len(calculator.solvents) > 1:
            print(f"Solvent ratio: {':'.join(f'{r:g}' for r in calculator.solvent_ratio)}")
        print(f"Anion:    {args.anion}")
        print(f"Cation:   LI (fixed)")
        print(f"Li count: {calculator.li_count}")
        print(f"Ratio:    1:{int(1/calculator.salt_to_solvent_ratio)}")
        print(f"Components: {calculator.components}")
        print(f"Temperature: {args.temperature} K")
        print("\nResults:")
        print(json.dumps(results, indent=2))
        print("="*60 + "\n")

        return 0

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
