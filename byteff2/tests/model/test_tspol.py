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

import os

import numpy as np
import torch

from byteff2.data import ClusterData, MonoData, collate_data
from byteff2.model import HybridFF
from byteff2.toolkit.openmmtool import AmoebaCalculator
from byteff2.train.utils import get_nb_params, load_model
from bytemol.core.molecule import Molecule
from bytemol.utils import get_data_file_path, setup_default_logging

setup_default_logging()

torch.set_default_dtype(torch.float32)

device_ = 'cpu'
pme_ = False
model_dir = get_data_file_path('trained_models/optimal.pt', 'byteff2')
trained_model = load_model(os.path.dirname(model_dir))
butol_xyz = get_data_file_path("tspol/butanol.xyz", "byteff2.tests.testdata")
butol_top = get_data_file_path("tspol/butanol.top", "byteff2.tests.testdata")
water_xyz = get_data_file_path("tspol/water.xyz", "byteff2.tests.testdata")
water_top = get_data_file_path("tspol/water.top", "byteff2.tests.testdata")
dimer_top = get_data_file_path("tspol/dimer.top", "byteff2.tests.testdata")


def get_amoeba_label(model: HybridFF, device: str, cluster=False, pme=False):

    dtype = torch.float32
    nonbonded_params = dict()
    nconfs = 5
    coords = []
    natoms = 0
    model = model.to('cpu')
    for name in ['butanol', 'water']:
        mol = Molecule(get_data_file_path(f"tspol/{name}.xyz", "byteff2.tests.testdata"), name=name)
        metadata, params, _, _ = get_nb_params(model, mol)
        cc = np.array([mol.conformers[i].coords for i in range(nconfs)])
        coords.append(cc)
        nonbonded_params[name] = params
        natoms += mol.natoms
    nonbonded_params['metadata'] = metadata
    model = model.to(device)

    tops = [dimer_top] if cluster else [butol_top, water_top]
    coords = [np.concatenate(coords, axis=1)] if cluster else coords
    openmm_coul_e, openmm_coul_f, openmm_vdw_e, openmm_vdw_f, openmmm_dipoles = [], [], [], [], []
    for ib, top in enumerate(tops):
        if pme:
            calculator = AmoebaCalculator(top,
                                          separate_terms=True,
                                          nonbonded_params=nonbonded_params,
                                          platform_name=device.upper(),
                                          unit_cell=[20., 20., 20.],
                                          cutoff=1.0)
        else:
            calculator = AmoebaCalculator(top,
                                          separate_terms=True,
                                          nonbonded_params=nonbonded_params,
                                          platform_name=device.upper())

        ce, cf, ve, vf, dp = [], [], [], [], []
        for i in range(nconfs):
            cc = coords[ib][i]
            calculator._calculate_without_restraint(cc)
            se, sf = calculator.get_separate_terms()
            dp.append(calculator.get_induced_dipole())

            ce.append(se['AmoebaMultipoleForce'])
            cf.append(sf['AmoebaMultipoleForce'])
            ve.append(se['CustomNonbondedForce'] + se['CustomBondForce'])
            vf.append(sf['CustomNonbondedForce'] + sf['CustomBondForce'])
        openmm_coul_e.append(np.array(ce))
        openmm_coul_f.append(np.array(cf))
        openmm_vdw_e.append(np.array(ve))
        openmm_vdw_f.append(np.array(vf))
        openmmm_dipoles.append(dp)

    openmm_coul_e = np.array(openmm_coul_e)
    openmm_coul_f = np.concatenate(openmm_coul_f, axis=1).swapaxes(0, 1)
    openmm_vdw_e = np.array(openmm_vdw_e)
    openmm_vdw_f = np.concatenate(openmm_vdw_f, axis=1).swapaxes(0, 1)
    openmmm_dipoles = np.concatenate(openmmm_dipoles, axis=1).swapaxes(0, 1)

    openmm_coul_e = torch.tensor(openmm_coul_e, dtype=dtype, device=device)
    openmm_coul_f = torch.tensor(openmm_coul_f, dtype=dtype, device=device)
    openmm_vdw_e = torch.tensor(openmm_vdw_e, dtype=dtype, device=device)
    openmm_vdw_f = torch.tensor(openmm_vdw_f, dtype=dtype, device=device)
    openmmm_dipoles = torch.tensor(openmmm_dipoles, dtype=dtype, device=device)

    return openmm_coul_e, openmm_coul_f, openmm_vdw_e, openmm_vdw_f, openmmm_dipoles


def summary_preds(preds, cluster):
    ff_parameters = preds['ff_parameters']
    surfix = '_cluster' if cluster else ''
    model_coul_e = ff_parameters['Exp6Pol.perm_chg_energy' + surfix] + ff_parameters['Exp6Pol.induction_energy' +
                                                                                     surfix]
    model_coul_f = ff_parameters['Exp6Pol.perm_chg_forces' + surfix] + ff_parameters['Exp6Pol.induction_forces' +
                                                                                     surfix]
    model_vdw_e = ff_parameters['Exp6Pol.rep_energy' + surfix] + ff_parameters['Exp6Pol.disp_energy' + surfix]
    model_vdw_f = ff_parameters['Exp6Pol.rep_forces' + surfix] + ff_parameters['Exp6Pol.disp_forces' + surfix]
    if 'Exp6Pol.ct_energy' in ff_parameters:
        model_vdw_e += ff_parameters['Exp6Pol.ct_energy' + surfix]
        model_vdw_f += ff_parameters['Exp6Pol.ct_forces' + surfix]

    model_dipoles = ff_parameters['Exp6Pol.mu_p' + surfix]
    return model_coul_e, model_coul_f, model_vdw_e, model_vdw_f, model_dipoles


def check_results(preds, target, rtol=1e-3, atol=1e-3, dtype=None, shape=None, batch_size=None):
    for j, (p, t) in enumerate(zip(preds, target)):
        if dtype is not None:
            assert p.dtype == dtype
        if batch_size is not None:
            for i in range(batch_size):
                assert torch.allclose(p[shape[j] * i:shape[j] * (i + 1)], t, rtol=rtol,
                                      atol=atol), (p[shape[j] * i:shape[j] * (i + 1)] - t).abs().max()
        else:
            assert torch.allclose(p, t, rtol=rtol, atol=atol)


def check_tspol_monomer(model, device, pme=False):
    # test batched monomers

    ind14 = 0.5
    vdw14 = 0.5
    charge14 = 0.5
    nconfs = 5

    model.ff_block.ff_layers['Exp6Pol'].charge14 = charge14
    model.ff_block.ff_layers['Exp6Pol'].vdw14 = vdw14
    model.ff_block.ff_layers['Exp6Pol'].ind14 = ind14
    model.to(device)

    mol_butol = Molecule(butol_xyz, name='butol')
    mol_water = Molecule(water_xyz, name='water')

    data_list = [
        MonoData(
            name=mol.name,
            mapped_smiles=mol.get_mapped_smiles(),
            max_n_confs=nconfs,
            confdata=dict(coords=np.array([mol._conformers[i].coords for i in range(nconfs)])),
            moldata=dict(charge=mol.get_conf_prop('charges'), volume=mol.get_conf_prop('volumes')),  # A**3
        ) for mol in [mol_butol, mol_water]
    ]
    data = collate_data(data_list).to(device)

    cluster = False
    preds = model(data, cluster=cluster)
    preds_sum = summary_preds(preds, cluster)
    targets = get_amoeba_label(model, device, cluster=cluster, pme=pme)
    check_results(preds_sum, targets, rtol=1e-3, atol=1e-5, dtype=data.coords.dtype)


def check_tspol_dimer(model, device, pme=False):

    ind14 = 0.5
    vdw14 = 0.5
    charge14 = 0.5
    nconfs = 5

    model.ff_block.ff_layers['Exp6Pol'].charge14 = charge14
    model.ff_block.ff_layers['Exp6Pol'].vdw14 = vdw14
    model.ff_block.ff_layers['Exp6Pol'].ind14 = ind14
    model.to(device)

    mol_butol = Molecule(butol_xyz, name='butol')
    mol_water = Molecule(water_xyz, name='water')
    mols = [mol_butol, mol_water]
    data = ClusterData(
        name='test',
        mapped_smiles=[mol.get_mapped_smiles() for mol in mols],
        max_n_confs=5,
        confdata=dict(
            coords=np.concatenate([np.array([mol.conformers[i].coords
                                             for i in range(nconfs)])
                                   for mol in mols], axis=1)),
        moldata=dict(charge=np.concatenate([mol.get_conf_prop('charges') for mol in mols]),
                     volume=np.concatenate([mol.get_conf_prop('volumes') for mol in mols]))  # A**3
    )
    data = data.to(device)

    # test with intermol interaction
    cluster = True
    preds = model(data, cluster=cluster)
    preds_sum = summary_preds(preds, cluster)
    targets = get_amoeba_label(model, device, cluster=cluster, pme=pme)
    check_results(preds_sum, targets, rtol=1e-3, atol=1e-4, dtype=data.coords.dtype)

    # test batch
    batch_size = 10
    data_10 = collate_data([data] * batch_size)
    preds_10 = model(data_10, cluster=True)
    preds_sum_10 = summary_preds(preds_10, cluster)
    e_shape, f_shape = targets[0].size(0), targets[1].size(0)
    check_results(preds_sum_10,
                  targets,
                  rtol=1e-3,
                  atol=1e-4,
                  dtype=data.coords.dtype,
                  batch_size=batch_size,
                  shape=[e_shape, f_shape, e_shape, f_shape, f_shape])

    # test without intermol interaction
    cluster = False
    preds_sum = summary_preds(preds, cluster)
    targets = get_amoeba_label(model, device, cluster=cluster, pme=pme)
    check_results(preds_sum, targets, rtol=1e-3, atol=1e-4, dtype=data.coords.dtype)

    preds_sum_10 = summary_preds(preds_10, cluster)
    e_shape, f_shape = targets[0].size(0), targets[1].size(0)
    check_results(preds_sum_10,
                  targets,
                  rtol=1e-3,
                  atol=1e-4,
                  dtype=data.coords.dtype,
                  batch_size=batch_size,
                  shape=[e_shape, f_shape, e_shape, f_shape, f_shape])


def test_tspol_monomer():
    check_tspol_monomer(trained_model, device_, pme=pme_)


def test_tspol_dimer():
    check_tspol_dimer(trained_model, device_, pme=pme_)
