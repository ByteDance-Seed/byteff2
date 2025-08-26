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

import logging
import tempfile
import typing as T

import ase
from ase.calculators.calculator import Calculator
from ase.constraints import FixInternals
from ase.optimize.minimahopping import MHPlot, MinimaHopping
from sella import Constraints, Internals, Sella

# from bytemol.core import Topology
from bytemol.toolkit.asetool.optimizer_config import OptimizerConfig
from bytemol.utils import temporary_cd

logger = logging.getLogger(__name__)


class OptimizerNotConvergedException(Exception):
    pass


def set_ase_constraints(atoms: ase.Atoms, constraints: T.Iterable[T.Iterable[int]], epsilon: float = 1e-3):
    '''https://wiki.fysik.dtu.dk/ase/ase/constraints.html'''
    bond_list, angle_list, dihedral_list = [], [], []
    for constraint in constraints:
        if len(constraint) == 2:
            bond_list.append([atoms.get_distance(*constraint), list(constraint)])
        elif len(constraint) == 3:
            angle_list.append([atoms.get_angle(*constraint), list(constraint)])
        elif len(constraint) == 4:
            dihedral_list.append([atoms.get_dihedral(*constraint), list(constraint)])
        else:
            raise ValueError("constraint element should be tuple of indices with length 2, 3 or 4")
    constraint = FixInternals(bonds=bond_list, angles_deg=angle_list, dihedrals_deg=dihedral_list, epsilon=epsilon)
    atoms.set_constraint(constraint)


def optimize(
        input_frame: ase.Atoms,
        config: OptimizerConfig,
        *,
        calculator: Calculator = None,  # overrides input_frame.calc if not None
        constraints: T.Iterable[T.Iterable[int]] = None,
        verbose: bool = False,
        check_convergence: bool = True) -> ase.Atoms:

    assert isinstance(input_frame, ase.Atoms)
    assert isinstance(config, OptimizerConfig)
    config.set_verbose(verbose)
    optimizer_type = config.get_optimizer_type()
    if check_convergence and not config.is_ase_local_optimizer():
        logger.warning("check_convergence can only be used on ase local optimizer")

    ase_frame = input_frame.copy()  # only info, array, and constraints are copied. calculator is not copied
    ase_frame.calc = input_frame.calc
    if calculator is not None:
        ase_frame.calc = calculator
    else:
        assert ase_frame.calc is not None, 'a valid calculator must have been initialized'
    ase_frame.calc.reset()
    ase_frame.constraints.clear()

    if config.get_optimizer_type() == 'sella':
        bonds = config.config_dict["common"]['bonds']
        assert bonds is not None and len(bonds) > 0
        # topology = Topology(bonds)

        # sella automatically converts atoms.constraints to sella internal constraints
        # build constraints and IC
        sella_config = config.get_ase_optimizer_config()
        no_reaction = sella_config.pop('no_reaction', False)
        no_reaction_scale = sella_config.pop('no_reaction_scale', 1.3)

        cons = Constraints(atoms=ase_frame)  # empty constraints

        if constraints is None:
            constraints = []

        cons_indices = set()
        for c in constraints:
            assert len(c) <= 4 and len(c) >= 2
            cons_indices.add(tuple(c))

        for c in constraints:
            if len(c) == 2:
                target = ase_frame.get_distance(*c)
                logger.info('constrain bond %s to %s angstrom', c, target)
                cons.fix_bond(tuple(c), target=target, comparator='eq', replace_ok=False)
            elif len(c) == 3:
                target = ase_frame.get_angle(*c)
                logger.info('constrain angle %s to %s degree', c, target)
                cons.fix_angle(tuple(c), target=target, comparator='eq', replace_ok=False)
            elif len(c) == 4:
                target = ase_frame.get_dihedral(*c)
                target = (target + 180) % 360 - 180  # convert from ase range of [0, 360] to sella range
                logger.info('constrain dihedral %s to %s degree', c, target)
                cons.fix_dihedral(tuple(c), target=target, comparator='eq', replace_ok=False)
            else:
                raise ValueError(f'invalid constraint specified {c}')

        # build constraints and IC
        if not no_reaction:
            internals = Internals(ase_frame, cons=cons)  # all constraints will be automatically added to IC
            for i, j in bonds:
                assert i >= 0 and j >= 0 and i != j, f'invalid bond indices {(i,j)}'
                if (i, j) in cons_indices or (j, i) in cons_indices:
                    continue
                internals.add_bond((i, j))
        else:
            for i, j in bonds:
                assert i >= 0 and j >= 0 and i != j, f'invalid bond indices {(i,j)}'
                if (i, j) in cons_indices or (j, i) in cons_indices:
                    continue
                # add IC if bond (i,j) not already included in constraints
                cons.fix_bond((i, j),
                              target=no_reaction_scale * ase_frame.get_distance(i, j),
                              comparator='lt',
                              replace_ok=False)
            internals = Internals(ase_frame, cons=cons)  # all constraints will be automatically added to IC

        internals.find_all_angles()
        internals.find_all_dihedrals()

        logger.info('constructed %s bonds, %s angles, %s dihedrals in sella', internals.nbonds, internals.nangles,
                    internals.ndihedrals)

        # dump for debug
        logger.debug('-------------------------internals-------------------------')
        for k, v in internals.internals.items():
            logger.debug('%s, %s', k, v)
        logger.debug('------------------------constraints------------------------')
        for k, v in internals.cons.internals.items():
            logger.debug('%s, %s', k, v)

        logger.debug('%s', sella_config)
        sella_config.pop('order', 0)
        if hasattr(ase_frame.calc, 'calculate_hessian'):
            sella_config['hessian_function'] = ase_frame.calc.calculate_hessian

        optimizer = Sella(ase_frame, internal=internals, order=0, **sella_config)

        run_config = config.get_local_run_config()
        converged = optimizer.run(**run_config)
        if check_convergence and not converged:
            raise OptimizerNotConvergedException(f"ase optimizer is not converged in {run_config['steps']} iters")

    elif config.is_ase_local_optimizer():
        if constraints is not None:
            set_ase_constraints(ase_frame, constraints)

        optimizer_class = config.get_ase_optimizer()
        optimizer_config = config.get_ase_optimizer_config()

        optimizer = optimizer_class(ase_frame, **optimizer_config)
        run_config = config.get_local_run_config()
        converged = optimizer.run(**run_config)
        if check_convergence and not converged:
            raise OptimizerNotConvergedException(f"ase optimizer is not converged in {run_config['steps']} iters")

    elif config.is_ase_global_optimizer():
        if constraints is not None:
            set_ase_constraints(ase_frame, constraints)

        optimizer_class = config.get_ase_optimizer()
        optimizer_config = config.get_ase_optimizer_config()

        working_dir = config.config_dict["common"]["working_dir"]
        if working_dir is None:
            working_dir = tempfile.mkdtemp(prefix='optimize_tmp_')  # under /tmp/ dir
        with temporary_cd(working_dir):
            global_optimizer = optimizer_class(atoms=ase_frame, **optimizer_config)
            if isinstance(global_optimizer, MinimaHopping):
                global_optimizer(totalsteps=config.get_global_steps())
                mhplot = MHPlot(logname=optimizer_config["logfile"])
                mhplot.save_figure('minimahopping_summary.png')
            else:
                global_optimizer.run(steps=config.get_global_steps())

    else:
        raise NotImplementedError("Optimizer type {} is not supported".format(optimizer_type))

    ase_frame.info["energy"] = ase_frame.calc.results['no_restraint_energy']
    ase_frame.arrays["forces"] = ase_frame.calc.results['no_restraint_forces']
    ase_frame.calc = None
    return ase_frame.copy()
