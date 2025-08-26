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
from enum import Enum
from typing import Union

import torch

from byteff2.data import ClusterData, MonoData
from byteff2.model.ff_layers import PreMMBondedConj
from byteff2.model.ff_layers.utils import cosine_cutoff, reduce_counts

logger = logging.getLogger(__name__)

kb = 8.314 / 1000 / 4.184  # kcal/mol/K


def soft_mse(diff: torch.Tensor, max_val: float, keep_dim=False):
    la = diff.abs()
    l = diff**2
    scale = torch.tanh(la / max_val) * max_val / torch.where(la < torch.finfo().eps, torch.finfo().eps, la)
    scale = scale.detach()
    if keep_dim:
        return l * scale
    else:
        return torch.mean(l * scale)


def calc_conf_mean(src, confmask, conf_dim=-1):
    x = src * confmask  # [nmols * nconfs]
    return (torch.sum(x, dim=conf_dim) + 1e-6) / (torch.sum(confmask, dim=conf_dim) + 1e-6)


class LossType(Enum):
    MMBondedConjMSE = 1
    ParamMSE = 2
    InterEnergyMSE = 3
    InterEnergyPolMSE = 4
    InterEnergyDispMSE = 5
    InterEnergyElecPauliMSE = 6
    InterEnergyCTMSE = 7


def loss_func(preds: dict, data: Union[MonoData, ClusterData], loss_type: LossType, **kwargs):

    def get_confmask(cluster=False):
        dist_scale_args = kwargs.get('dist_scale', None)
        if dist_scale_args is None:
            dist_scale = 1.
        else:
            dist_scale = 1. - cosine_cutoff(data['min_dists'].clone(), *dist_scale_args)

        confmask = data.confmask_cluster if cluster else data.confmask
        confmask = confmask * dist_scale
        natoms = data.get_count('node', idx=None, cluster=cluster)
        confmask_na = torch.repeat_interleave(confmask, natoms, 0).unsqueeze(-1)

        force_cutoff = kwargs.pop("force_cutoff", None)
        if force_cutoff is not None:
            if 'forces_cluster' in data:
                lf = data.forces_cluster - data.forces_single
            else:
                lf = preds['forces_cluster'] - preds['forces']
            lf = lf.abs().max(dim=-1)[0]
            lf = reduce_counts(lf, natoms, reduce='max')
            ljes_scale = cosine_cutoff(lf, *force_cutoff)
            ljes_scale = (ljes_scale * confmask).detach()
            ljes_scale_na = ljes_scale.repeat_interleave(natoms, 0).unsqueeze(-1).detach()
        else:
            ljes_scale, ljes_scale_na = confmask, confmask_na

        return confmask, confmask_na, ljes_scale, ljes_scale_na

    def get_interaction_energy():
        pred_cluster = preds['energy_cluster']
        nmols = data.get_count('mol', idx=None, cluster=True)
        batches = torch.arange(nmols.shape[0], device=nmols.device).repeat_interleave(nmols).unsqueeze(-1).expand(
            -1, pred_cluster.shape[1])
        pred_single = torch.zeros_like(pred_cluster).scatter_add_(0, batches, preds['energy'])

        if 'energy_cluster' in data:
            label_cluster = data.energy_cluster
            label_single = torch.zeros_like(label_cluster).scatter_add_(0, batches, data.energy_single)
            le = label_cluster - label_single
        else:
            le = data.total_int_energy
        return pred_cluster - pred_single, le

    def get_boltzmann_weight(e_pred, e_label):
        clamp = kwargs.pop('clamp', 2)  # unit kcal/mol
        decay = kwargs.pop('decay', 2)  # unit kcal/mol
        scale = torch.exp(torch.clamp((clamp - torch.minimum(e_pred, e_label)) / decay, max=0)).detach()
        return scale

    def get_li_scale(li_scale):
        if li_scale == 1.0:
            return 1.0
        scale = [li_scale if 'LI' in n else 1.0 for n in data.name]
        scale = torch.tensor(scale, device=data.coords.device, dtype=data.coords.dtype)
        return scale

    if loss_type is LossType.MMBondedConjMSE:
        params = preds['ff_parameters']
        loss = 0.
        bk, bb = data['bond_k'], data['bond_r0']
        ak, ab = data['angle_k'], torch.deg2rad(data['angle_d0'])
        for term, values in PreMMBondedConj.param_std_mean_range.items():
            if term == 'bond_k1':
                t = bk * (PreMMBondedConj.bond_b2 - bb) / (PreMMBondedConj.bond_b2 - PreMMBondedConj.bond_b1)
            elif term == 'bond_k2':
                t = bk * (bb - PreMMBondedConj.bond_b1) / (PreMMBondedConj.bond_b2 - PreMMBondedConj.bond_b1)
            elif term == 'angle_k1':
                t = ak * (PreMMBondedConj.angle_b2 - ab) / (PreMMBondedConj.angle_b2 - PreMMBondedConj.angle_b1)
            elif term == 'angle_k2':
                t = ak * (ab - PreMMBondedConj.angle_b1) / (PreMMBondedConj.angle_b2 - PreMMBondedConj.angle_b1)
            else:
                t = data[term]
            l = torch.mean((params[f'PreMMBondedConj.{term}'] - t)**2 / values[0]**2)
            loss += l

    elif loss_type is LossType.ParamMSE:
        label_name = kwargs['label']
        param_name = kwargs['param']
        loss = torch.mean((preds['ff_parameters'][param_name].view(-1) - data[label_name].view(-1))**2)

    elif loss_type is LossType.InterEnergyMSE:
        li_scale = kwargs.get('li_scale', 1.0)
        li_scale = get_li_scale(li_scale)
        confmask, _, ljes_scale, _ = get_confmask(cluster=True)

        T = kwargs.pop("scale_by_charge_transfer", None)
        if T is not None:
            ct_scale = torch.exp(-torch.abs(data['charge_transfer_int_energy']) / kb / T)
            ljes_scale = (ljes_scale * ct_scale).detach()

        pe, le = get_interaction_energy()

        if 'clamp' in kwargs and 'decay' in kwargs:
            boltzmann_weight = get_boltzmann_weight(pe, le) * confmask
        else:
            boltzmann_weight = torch.ones_like(ljes_scale) * confmask

        if kwargs.get('mix_fe_scale', False):
            scale = (ljes_scale**2 + boltzmann_weight**2) / (ljes_scale + boltzmann_weight + 1e-6)
            loss = calc_conf_mean((pe - le)**2 * scale, confmask)
        else:
            loss = calc_conf_mean(((pe - le) * ljes_scale * boltzmann_weight)**2, confmask)

        loss = torch.mean(loss * li_scale)

    elif loss_type is LossType.InterEnergyPolMSE:
        li_scale = kwargs.get('li_scale', 1.0)
        li_scale = get_li_scale(li_scale)
        confmask, _, ljes_scale, _ = get_confmask(cluster=True)

        add_ct = kwargs.get('add_charge_transfer', False)

        T = kwargs.pop("scale_by_charge_transfer", None)
        if T is not None:
            ct_scale = torch.exp(-torch.abs(data['charge_transfer_int_energy']) / kb / T)
            ljes_scale = (ljes_scale * ct_scale).detach()

        pe = preds['ff_parameters']['POLARIZATION']
        le = data['polarization_int_energy'].clone()
        if add_ct:
            le += data['charge_transfer_int_energy']

        scale_pf6 = kwargs.get('scale_pf6', 1.0)
        pf6_scales = [scale_pf6 if ('PF6' in name and 'LI' in name) else 1. for name in data.name]
        pf6_scales = torch.tensor(pf6_scales, device=le.device, dtype=le.dtype)
        le *= pf6_scales.unsqueeze(-1)

        if 'clamp' in kwargs and 'decay' in kwargs:
            pte, lte = get_interaction_energy()
            boltzmann_weight = get_boltzmann_weight(pte, lte) * confmask
        else:
            boltzmann_weight = torch.ones_like(ljes_scale) * confmask

        if kwargs.get('mix_fe_scale', False):
            scale = (ljes_scale**2 + boltzmann_weight**2) / (ljes_scale + boltzmann_weight + 1e-6)
            loss = calc_conf_mean((pe - le)**2 * scale, confmask)
        else:
            loss = calc_conf_mean((pe - le)**2 * ljes_scale * boltzmann_weight, confmask)

        loss = torch.mean(loss * li_scale)

    elif loss_type is LossType.InterEnergyDispMSE:

        disp_scale = kwargs.get('disp_scale', 1.0)
        add_ct = kwargs.get('add_charge_transfer', False)
        li_scale = kwargs.get('li_scale', 1.0)
        li_scale = get_li_scale(li_scale)

        confmask, _, ljes_scale, _ = get_confmask(cluster=True)
        pe = preds['ff_parameters']['DISP']
        le = data['disp_int_energy'].clone()
        if add_ct:
            le += data['charge_transfer_int_energy']
        le = le * disp_scale

        if 'clamp' in kwargs and 'decay' in kwargs:
            pte, lte = get_interaction_energy()
            boltzmann_weight = get_boltzmann_weight(pte, lte) * confmask
        else:
            boltzmann_weight = torch.ones_like(ljes_scale) * confmask

        if kwargs.get('mix_fe_scale', False):
            scale = (ljes_scale**2 + boltzmann_weight**2) / (ljes_scale + boltzmann_weight + 1e-6)
            loss = calc_conf_mean((pe - le)**2 * scale, confmask)
        else:
            loss = calc_conf_mean((pe - le)**2 * ljes_scale * boltzmann_weight, confmask)
        loss = torch.mean(loss * li_scale)

    elif loss_type is LossType.InterEnergyCTMSE:

        confmask, _, ljes_scale, _ = get_confmask(cluster=True)
        pe = preds['ff_parameters']['CHARGE_TRANSFER']
        le = data['charge_transfer_int_energy'].clone()

        if 'clamp' in kwargs and 'decay' in kwargs:
            pte, lte = get_interaction_energy()
            boltzmann_weight = get_boltzmann_weight(pte, lte) * confmask
        else:
            boltzmann_weight = torch.ones_like(ljes_scale) * confmask
        loss = calc_conf_mean((pe - le)**2 * ljes_scale * boltzmann_weight, confmask)
        loss = torch.mean(loss)

    elif loss_type is LossType.InterEnergyElecPauliMSE:

        confmask, _, ljes_scale, _ = get_confmask(cluster=True)
        pe = preds['ff_parameters']['ELEC'] + preds['ff_parameters']['PAULI']
        le = data['elec_pauli_int_energy'].clone()

        if 'clamp' in kwargs and 'decay' in kwargs:
            pte, lte = get_interaction_energy()
            boltzmann_weight = get_boltzmann_weight(pte, lte) * confmask
        else:
            boltzmann_weight = torch.ones_like(ljes_scale) * confmask

        if kwargs.get('mix_fe_scale', False):
            scale = (ljes_scale**2 + boltzmann_weight**2) / (ljes_scale + boltzmann_weight + 1e-6)
            loss = calc_conf_mean((pe - le)**2 * scale, confmask)
        else:
            loss = calc_conf_mean((pe - le)**2 * ljes_scale * boltzmann_weight, confmask)
        loss = torch.mean(loss)

    else:
        raise NotImplementedError(loss_type)

    return loss
