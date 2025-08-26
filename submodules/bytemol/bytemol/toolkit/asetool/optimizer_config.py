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

import copy
import logging
import typing as T

import yaml
from ase.optimize import BFGS, FIRE, GPMin, MDMin
from ase.optimize.basin import BasinHopping
from ase.optimize.minimahopping import MinimaHopping
from ase.optimize.optimize import Optimizer
from sella import Sella

logger = logging.getLogger(__name__)


class OptimizerConfig:

    default_config = {
        "common": {
            "fmax": 0.001,  # eV/A
            "max_iterations": 10000,
            "bonds": [],
            "logfile": None,
            "working_dir": None,
        },
        "optimizer": {
            "type": "BFGS",
            "params": {
                "maxstep": 0.1,
                "alpha": 70,
            },
        },
    }

    optimizer_config = {

        # sella
        'sella': {
            # 'order': 0,  # 0 means minimization, always fixed to be zero in this code
            'eig': False,
            'delta0': 0.1,  # initial trust radius in units of Angstrom per degree of freedom
            # these settings come from the unconstrained optimization paper, probably too aggressive
            'rho_dec': 100,
            'rho_inc': 1.035,
            'sigma_dec': 0.90,
            'sigma_inc': 1.15,
            'eta': 1e-2,  # in minimization method, eta only controls the lower bound of rtrust
            # these two are not passed to Sella() class
            'no_reaction': True,  # if True, requires 'bonds' to be non-empty
            'no_reaction_scale': 1.3
        },
        # ase local
        "bfgs": {
            "maxstep": 0.2,
            "alpha": 70,
        },
        "fire": {
            "dt": 0.1,
            "dtmax": 1.0,
            "Nmin": 5,
            "finc": 1.1,
            "fdec": 0.5,
            "astart": 0.1,
            "fa": 0.99,
            "a": 0.1,
            "maxstep": 0.2,
        },
        "gpmin": {
            "scale": None,
            "noise": None,
            "weight": None,
            "bounds": None,
            "update_prior_strategy": "maximum",
            "update_hyperparams": False,
        },
        "mdmin": {
            "dt": 0.2,
        },
        # ase global
        "basinhopping": {
            "temperature": 100 * 8.61733e-5,  # 100 * kB
            "dr": 0.1,
            "adjust_cm": True,
            "global_steps": 20,
            "optimizer": "FIRE",
        },
        "minimahopping": {
            'T0': 1000.,  # K, initial MD 'temperature'
            'beta1': 1.1,  # temperature adjustment parameter
            'beta2': 1.1,  # temperature adjustment parameter
            'beta3': 0.909,  # 1/1.1, temperature adjustment parameter
            'Ediff0': 0.5,  # eV, initial energy acceptance threshold
            'alpha1': 0.98,  # energy threshold adjustment parameter
            'alpha2': 1.0204,  # 1/0.98, energy threshold adjustment parameter
            'mdmin': 2,  # criteria to stop MD simulation (no. of minima)
            'minima_threshold': 0.5,  # A, threshold for identical configs
            'timestep': 1.0,  # fs, timestep for MD simulations
            "optimizer": "FIRE",
            "global_steps": None,
        },
    }

    def __init__(self, config: T.Union[T.Dict, str] = None):
        if isinstance(config, str):
            assert config.endswith('.yaml') or config.endswith('.yml')
            with open(config, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif isinstance(config, dict):
            config_dict = config.copy()
        else:
            assert config is None
            config_dict = {}
        self.config_dict = {}
        self.init_config(config_dict)

    def init_config(self, config_dict):

        def check_and_update(origin, new):
            for k, v in new.items():
                assert k in origin, "Unrecognized config key: {}".format(k)
                origin[k] = v

        final_config = copy.deepcopy(self.default_config)

        # update config.common
        common_config = config_dict.pop("common", {})
        if "bonds" in common_config:
            common_config["bonds"] = [list(bond) for bond in common_config["bonds"]]
        check_and_update(final_config["common"], common_config)

        # update config.optimizer
        input_optimizer_config = config_dict.pop("optimizer", {})
        if input_optimizer_config:
            optimizer_type = input_optimizer_config["type"].lower()
            assert optimizer_type in self.optimizer_config.keys()
            optimizer_params = copy.copy(self.optimizer_config[optimizer_type])
            input_params = input_optimizer_config.get("params", {})
            check_and_update(optimizer_params, input_params)
            final_config["optimizer"]["type"] = optimizer_type
            final_config["optimizer"]["params"] = optimizer_params

        self.config_dict = final_config

    def to_yaml(self, yaml_file: str):
        with open(yaml_file, 'w') as f:
            yaml.dump(self.config_dict, f, default_flow_style=False)

    def set_verbose(self, verbose: bool):
        if self.config_dict["common"]["logfile"] in {None, '-'}:
            self.config_dict["common"]["logfile"] = '-' if verbose else None

    def get_optimizer_type(self) -> str:
        return self.config_dict["optimizer"]["type"]

    def is_ase_local_optimizer(self) -> bool:
        optimizer = self.config_dict["optimizer"]["type"].lower()
        return optimizer in {"bfgs", "fire", "gpmin", "mdmin"}

    def is_ase_global_optimizer(self) -> bool:
        optimizer = self.config_dict["optimizer"]["type"].lower()
        return optimizer in {"basinhopping", "minimahopping"}

    def get_ase_optimizer_config(self) -> T.Dict:
        config = copy.copy(self.config_dict["optimizer"]["params"])
        config.update({"logfile": self.config_dict["common"]["logfile"]})
        if self.is_ase_global_optimizer():
            config["optimizer"] = self.get_ase_optimizer(config["optimizer"])
            config.update({"fmax": self.config_dict["common"]["fmax"]})
            config.pop("global_steps")
            if self.get_optimizer_type() == "minimahopping" and config["logfile"] in {"-", None}:
                config["logfile"] = "hop.log"
        return config

    def get_global_steps(self) -> int:
        assert self.is_ase_global_optimizer()
        return self.config_dict["optimizer"]["params"]["global_steps"]

    def get_ase_optimizer(self, inp_optimizer=None) -> Optimizer:
        if inp_optimizer is None:
            optimizer = self.config_dict["optimizer"]["type"]
        else:
            optimizer = inp_optimizer
        optimizer = optimizer.lower()
        optimizer_map = {
            "bfgs": BFGS,
            "fire": FIRE,
            "gpmin": GPMin,
            "mdmin": MDMin,
            "basinhopping": BasinHopping,
            "minimahopping": MinimaHopping,
            "sella": Sella,
        }
        assert optimizer in optimizer_map, "Unsupported optimizer type: {}".format(optimizer)
        return optimizer_map[optimizer]

    def get_local_run_config(self):
        return {
            "fmax": self.config_dict["common"]["fmax"],
            "steps": self.config_dict["common"]["max_iterations"],
        }
