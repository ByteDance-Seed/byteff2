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

from enum import Enum
import logging
from typing import Union

import torch

from byteff2.data import ClusterData, MonoData
from byteff2.model.ff_layers import PreMMBondedConj
from byteff2.model.ff_layers.ff_kernels import ClassicalForceField as CFF
from byteff2.model.ff_layers.utils import cosine_cutoff, get_dihedral_angle_vec, reduce_counts
from byteff2.utils.definitions import MMParam, MMTerm, MMTERM_WIDTH


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
    BondedEnergy = 8
    Partial_Hessian_MAPE = 9
    Boltzman_Soft_MSE = 10
    L1_Norm = 11
    Energy_MSE = 12
    Energy_Soft_MSE = 13
    Force_MSE = 14
    Force_Soft_MSE = 15


def loss_func(preds: dict, data: Union[MonoData, ClusterData], loss_type: LossType, **kwargs):
    remaining_kwargs = dict(kwargs)
    consumed_kwargs = set()
    missing = object()

    def pop_kwarg(name, default=missing):
        if name in remaining_kwargs:
            consumed_kwargs.add(name)
            return remaining_kwargs[name]
        if default is missing:
            raise KeyError(name)
        return default

    def has_kwarg(name):
        return name in remaining_kwargs

    def get_confmask(cluster=False):
        dist_scale_args = pop_kwarg("dist_scale", None)
        if dist_scale_args is None:
            dist_scale = 1.0
        else:
            dist_scale = 1.0 - cosine_cutoff(data["min_dists"].clone(), *dist_scale_args)

        confmask = data.confmask_cluster if cluster else data.confmask
        confmask = confmask * dist_scale
        natoms = data.get_count("node", idx=None, cluster=cluster)
        confmask_na = torch.repeat_interleave(confmask, natoms, 0).unsqueeze(-1)

        force_cutoff = pop_kwarg("force_cutoff", None)
        if force_cutoff is not None:
            if "forces_cluster" in data:
                lf = data.forces_cluster - data.forces_single
            else:
                lf = preds["forces_cluster"] - preds["forces"]
            lf = lf.abs().max(dim=-1)[0]
            lf = reduce_counts(lf, natoms, reduce="max")
            ljes_scale = cosine_cutoff(lf, *force_cutoff)
            ljes_scale = (ljes_scale * confmask).detach()
            ljes_scale_na = ljes_scale.repeat_interleave(natoms, 0).unsqueeze(-1).detach()
        else:
            ljes_scale, ljes_scale_na = confmask, confmask_na

        return confmask, confmask_na, ljes_scale, ljes_scale_na

    def get_interaction_energy():
        pred_cluster = preds["energy_cluster"]
        nmols = data.get_count("mol", idx=None, cluster=True)
        batches = (
            torch.arange(nmols.shape[0], device=nmols.device)
            .repeat_interleave(nmols)
            .unsqueeze(-1)
            .expand(-1, pred_cluster.shape[1])
        )
        pred_single = torch.zeros_like(pred_cluster).scatter_add_(0, batches, preds["energy"])

        if "energy_cluster" in data:
            label_cluster = data.energy_cluster
            label_single = torch.zeros_like(label_cluster).scatter_add_(0, batches, data.energy_single)
            le = label_cluster - label_single
        else:
            le = data.total_int_energy
        return pred_cluster - pred_single, le

    def get_boltzmann_weight(e_pred, e_label):
        clamp = pop_kwarg("clamp", 2)  # unit kcal/mol
        decay = pop_kwarg("decay", 2)  # unit kcal/mol
        scale = torch.exp(torch.clamp((clamp - torch.minimum(e_pred, e_label)) / decay, max=0)).detach()
        return scale

    def calc_circstd(coords, node_idx, counts):
        cc = [coords[node_idx[:, i]].unsqueeze(-2) for i in range(node_idx.shape[1])]
        cc = torch.concat(cc, dim=-2)
        ccs = [cc[:, :, i] for i in range(node_idx.shape[1])]
        proper_theta, *_ = get_dihedral_angle_vec(*ccs)
        confmask, *_ = get_confmask()
        confmask = torch.repeat_interleave(confmask, counts, 0)
        sin_prop = torch.sin(proper_theta) * confmask
        cos_prop = torch.cos(proper_theta) * confmask
        sin_prop = torch.sum(sin_prop, dim=-1) / torch.sum(confmask, dim=-1)
        cos_prop = torch.sum(cos_prop, dim=-1) / torch.sum(confmask, dim=-1)

        # clamp to avoid nan caused by float precision
        R2 = torch.clamp(
            torch.square(sin_prop) + torch.square(cos_prop),
            max=1.0 - torch.finfo().eps * 10.0,
            min=torch.finfo().eps * 10.0,
        )
        std = torch.sqrt(-torch.log(R2))
        return std

    def mask_by_circstd(
        std: torch.Tensor,
        param: torch.Tensor,
        threshold: float,
    ):
        p1 = param.clone().detach()
        mask = torch.where(std > threshold, 1.0, 0.0).unsqueeze(-1)
        return param * mask + p1 * (1 - mask)

    if loss_type is LossType.MMBondedConjMSE:
        params = preds["ff_parameters"]
        loss = 0.0
        bk, bb = data["bond_k"], data["bond_r0"]
        ak, ab = data["angle_k"], torch.deg2rad(data["angle_d0"])
        for term, values in PreMMBondedConj.param_std_mean_range.items():
            if term == "bond_k1":
                t = bk * (PreMMBondedConj.bond_b2 - bb) / (PreMMBondedConj.bond_b2 - PreMMBondedConj.bond_b1)
            elif term == "bond_k2":
                t = bk * (bb - PreMMBondedConj.bond_b1) / (PreMMBondedConj.bond_b2 - PreMMBondedConj.bond_b1)
            elif term == "angle_k1":
                t = ak * (PreMMBondedConj.angle_b2 - ab) / (PreMMBondedConj.angle_b2 - PreMMBondedConj.angle_b1)
            elif term == "angle_k2":
                t = ak * (ab - PreMMBondedConj.angle_b1) / (PreMMBondedConj.angle_b2 - PreMMBondedConj.angle_b1)
            else:
                t = data[term]
            l = torch.mean((params[f"PreMMBondedConj.{term}"] - t) ** 2 / values[0] ** 2)
            loss += l

    elif loss_type is LossType.ParamMSE:
        label_name = pop_kwarg("label")
        param_name = pop_kwarg("param")
        loss = torch.mean((preds["ff_parameters"][param_name].view(-1) - data[label_name].view(-1)) ** 2)

    elif loss_type is LossType.BondedEnergy:
        trained_param_name = pop_kwarg("param")
        ff_params = preds["ff_parameters"]
        if trained_param_name == "bond":
            k = ff_params["PreMMBonded.bond_k"].detach()
            r0 = ff_params["PreMMBonded.bond_r0"].clone()
            bond_params = {MMParam.bond_k: k, MMParam.bond_r0: r0}
            counts = data.get_count(
                trained_param_name,
                idx=None,
            )
            loss, _, _ = CFF.calc_bond(data.coords, bond_params, data.inc_node_bond.long(), counts)
        elif trained_param_name == "angle":
            k = ff_params["PreMMBonded.angle_k"].detach()
            d0 = ff_params["PreMMBonded.angle_d0"].clone()
            angle_params = {MMParam.angle_k: k, MMParam.angle_d0: d0}
            counts = data.get_count(
                trained_param_name,
                idx=None,
            )
            loss, _, _ = CFF.calc_angle(data.coords, angle_params, data.inc_node_angle.long(), counts)
        elif trained_param_name == "improper":
            improper_params = {MMParam.improper_k: ff_params["PreMMBonded.improper_k"]}
            counts = data.get_count(
                trained_param_name,
                idx=None,
            )
            loss, _, _ = CFF.calc_improper(data.coords, improper_params, data.inc_node_improper.long(), counts)
        else:
            raise ValueError(
                f"Unsupported trained_param_name for BondedEnergy: {trained_param_name}. "
                "Expected one of {'bond', 'angle', 'improper'}."
            )
        # When the term is empty in this batch, calc_* returns Python scalar 0.0.
        if not torch.is_tensor(loss):
            loss = torch.zeros((), device=data.coords.device, dtype=data.coords.dtype)
        else:
            loss = loss.sum() / counts.sum()

    elif loss_type is LossType.InterEnergyMSE:
        confmask, _, ljes_scale, _ = get_confmask(cluster=True)
        pe, le = get_interaction_energy()

        if has_kwarg("clamp") and has_kwarg("decay"):
            boltzmann_weight = get_boltzmann_weight(pe, le) * confmask
        else:
            boltzmann_weight = torch.ones_like(ljes_scale) * confmask
        loss = calc_conf_mean(((pe - le) * ljes_scale * boltzmann_weight) ** 2, confmask)
        loss = torch.mean(loss)

    elif loss_type is LossType.InterEnergyPolMSE:
        confmask, _, ljes_scale, _ = get_confmask(cluster=True)

        pe = preds["ff_parameters"]["POLARIZATION"]
        le = data["polarization_int_energy"].clone()
        if has_kwarg("clamp") and has_kwarg("decay"):
            pte, lte = get_interaction_energy()
            boltzmann_weight = get_boltzmann_weight(pte, lte) * confmask
        else:
            boltzmann_weight = torch.ones_like(ljes_scale) * confmask
        loss = calc_conf_mean((pe - le) ** 2 * ljes_scale * boltzmann_weight, confmask)
        loss = torch.mean(loss)

    elif loss_type is LossType.InterEnergyDispMSE:
        confmask, _, ljes_scale, _ = get_confmask(cluster=True)
        pe = preds["ff_parameters"]["DISP"]
        le = data["disp_int_energy"].clone()

        if has_kwarg("clamp") and has_kwarg("decay"):
            pte, lte = get_interaction_energy()
            boltzmann_weight = get_boltzmann_weight(pte, lte) * confmask
        else:
            boltzmann_weight = torch.ones_like(ljes_scale) * confmask

        if pop_kwarg("scale_by_min_dist", False):
            dist_scale = data["min_dists"].clone() ** 2
        else:
            dist_scale = torch.ones_like(ljes_scale)

        loss = calc_conf_mean((pe - le) ** 2 * ljes_scale * boltzmann_weight * dist_scale, confmask)
        loss = torch.mean(loss)

    elif loss_type is LossType.InterEnergyCTMSE:
        confmask, _, ljes_scale, _ = get_confmask(cluster=True)
        pe = preds["ff_parameters"]["CHARGE_TRANSFER"]
        le = data["charge_transfer_int_energy"].clone()

        if has_kwarg("clamp") and has_kwarg("decay"):
            pte, lte = get_interaction_energy()
            boltzmann_weight = get_boltzmann_weight(pte, lte) * confmask
        else:
            boltzmann_weight = torch.ones_like(ljes_scale) * confmask
        loss = calc_conf_mean((pe - le) ** 2 * ljes_scale * boltzmann_weight, confmask)
        loss = torch.mean(loss)

    elif loss_type is LossType.InterEnergyElecPauliMSE:
        confmask, _, ljes_scale, _ = get_confmask(cluster=True)
        pe = preds["ff_parameters"]["ELEC"] + preds["ff_parameters"]["PAULI"]
        le = data["elec_pauli_int_energy"].clone()

        if has_kwarg("clamp") and has_kwarg("decay"):
            pte, lte = get_interaction_energy()
            boltzmann_weight = get_boltzmann_weight(pte, lte) * confmask
        else:
            boltzmann_weight = torch.ones_like(ljes_scale) * confmask

        loss = calc_conf_mean((pe - le) ** 2 * ljes_scale * boltzmann_weight, confmask)
        loss = torch.mean(loss)

    elif loss_type is LossType.Partial_Hessian_MAPE:
        mask_threshold = pop_kwarg("mask_threshold", 1e4)

        def _to_global_idx(
            local_idx: torch.Tensor,
            shifts: torch.Tensor,
            counts: torch.Tensor,
        ):
            local_idx = local_idx.long()
            shifts = shifts.long()
            counts = counts.long()
            nedges_cumsum = torch.cumsum(shifts, 0)  # [batch_size]
            nedges_cumsum = torch.concat(
                (torch.tensor([0], device=shifts.device, dtype=shifts.dtype), nedges_cumsum[:-1]), dim=0
            )  # [batch_size]
            size = (-1,) + (1,) * (local_idx.dim() - 1)
            global_idx = local_idx + torch.repeat_interleave(nedges_cumsum, counts).view(size)  # [nterm, ...]
            return global_idx

        ff_params = preds["ff_parameters"]
        trainable_param_names = [MMParam.bond_k, MMParam.angle_k, MMParam.improper_k]
        trainable_params = {}
        for param in MMParam:
            if param in trainable_param_names:
                trainable_params[param] = ff_params[f"PreMMBonded.{param.name}"].clone()
            else:
                trainable_params[param] = ff_params[f"PreMMBonded.{param.name}"].detach()
        results = CFF.energy_force(
            data,
            trainable_params,
            [MMTerm.bond, MMTerm.angle, MMTerm.proper, MMTerm.improper],
            calc_partial_hessian=True,
        )
        partial_hessian_label = data["partial_hessian"]
        partial_hessian_pred = torch.zeros_like(partial_hessian_label)
        shifts = data.get_count("partial_hessian", idx=None)
        n_conf = partial_hessian_label.shape[1]
        for term in MMTerm:
            term_hessian = results[term.name][2]
            if term_hessian is None:
                continue
            width = MMTERM_WIDTH[term]
            for i in range(width):
                for j in range(width):
                    if i == j:
                        continue
                    if abs(i - j) > 2 and term is MMTerm.proper:
                        continue
                    rec_ij = data[f"{term.name}_rec_{i}_{j}"].long()
                    counts = data.get_count(
                        term.name,
                        idx=None,
                    )
                    if counts.sum() == 0:
                        continue
                    idx = _to_global_idx(rec_ij, shifts, counts).unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 9)
                    partial_hessian_pred.scatter_add_(0, idx, term_hessian[:, :, i * width + j])
        bad_ids = torch.where(partial_hessian_pred.abs() > mask_threshold)[0]
        mask = torch.ones_like(partial_hessian_label)
        if len(bad_ids) > 0:
            cshift = torch.cumsum(shifts, dim=0)
            bads = set()
            for idx in set(bad_ids.tolist()):
                bads.add(torch.where(idx < cshift)[0][0].item())
                mask[idx] = 0.0
        loss = (partial_hessian_pred - partial_hessian_label).abs() * mask  # [nPartialHessian, nconfs, 9]
        diag = partial_hessian_label[..., 0] + partial_hessian_label[..., 4] + partial_hessian_label[..., 8]
        loss = loss / torch.clamp(diag.abs().unsqueeze(-1), min=10.0)
        loss = torch.mean(loss)

    elif loss_type is LossType.Boltzman_Soft_MSE:
        # set trainable params
        trainable_params = {}
        ff_params = preds["ff_parameters"]
        trainable_params[MMParam.proper_k] = ff_params.get(
            "PreMMBondedConj.proper_k", ff_params["PreMMBonded.proper_k"]
        ).clone()
        if has_kwarg("mask_by_circstd_threshold"):
            proper_std = calc_circstd(data.coords, data.inc_node_proper, data.get_count("proper", idx=None))
            trainable_params[MMParam.proper_k] = mask_by_circstd(
                proper_std,
                trainable_params[MMParam.proper_k],
                threshold=pop_kwarg("mask_by_circstd_threshold"),
            )
        results = CFF.energy_force(
            data,
            trainable_params,
            [MMTerm.proper],
            calc_partial_hessian=False,
        )
        label_energy = data["energy"]
        pred_energy = preds["energy"]
        pred_energy = pred_energy.detach() - results["proper"][0].detach() + results["proper"][0]
        # set loss options
        clamp = pop_kwarg("clamp", 2)
        decay = pop_kwarg("decay", 2)  # unit kcal/mol
        max_val = pop_kwarg("max", 100.0)
        confmask, *_ = get_confmask()
        # Remove the per-molecule energy baseline before the subtractions below.
        # The loss is invariant to a per-molecule energy shift, but with large
        # absolute energies (e.g. total energies of thousands of Hartree) the
        # float32 baseline swamps the ~O(1) kcal/mol torsion signal in the
        # "large minus large" ops (alignment at *_aligned, and the main MSE).
        # Centering each molecule on its confmask-weighted mean keeps the values
        # small so the torsion differences survive. Masked confs are excluded via
        # calc_conf_mean and left untouched.
        label_energy = label_energy - calc_conf_mean(label_energy, confmask).unsqueeze(-1)
        pred_energy = pred_energy - calc_conf_mean(pred_energy.detach(), confmask).unsqueeze(-1)
        label_min = torch.min(
            torch.where(confmask > 0.5, label_energy, torch.finfo().max * torch.ones_like(label_energy)), dim=-1
        ).values
        pred_min = torch.min(
            torch.where(confmask > 0.5, pred_energy, torch.finfo().max * torch.ones_like(pred_energy)), dim=-1
        ).values
        label_energy_aligned = label_energy - label_min.unsqueeze(-1)
        pred_energy_aligned = pred_energy - pred_min.unsqueeze(-1)
        scale = torch.exp(
            torch.clamp((clamp - torch.minimum(label_energy_aligned, pred_energy_aligned)) / decay, max=0)
        ).detach()
        shift = (
            calc_conf_mean((label_energy - pred_energy) * scale, confmask) / calc_conf_mean(scale, confmask)
        ).detach()
        loss = torch.square(pred_energy - label_energy + shift.unsqueeze(-1)) * scale
        loss = calc_conf_mean(loss, confmask)
        scale = (
            torch.tanh(loss / max_val) * max_val / torch.where(abs(loss) < torch.finfo().eps, torch.finfo().eps, loss)
        )
        scale = scale.detach()
        loss = torch.mean(loss * scale)

    elif loss_type is LossType.Energy_MSE:
        confmask, *_ = get_confmask()
        pred_energy = preds["energy"]
        label_energy = data["energy"]
        shift = calc_conf_mean(pred_energy - label_energy, confmask).unsqueeze(-1)
        loss = torch.mean(calc_conf_mean((pred_energy - label_energy - shift) ** 2, confmask))

    elif loss_type is LossType.Energy_Soft_MSE:
        max_val = pop_kwarg("max", 5.0)
        confmask, *_ = get_confmask()
        pred_energy = preds["energy"]
        label_energy = data["energy"]
        shift = calc_conf_mean(pred_energy - label_energy, confmask).unsqueeze(-1)
        e_loss = soft_mse(pred_energy - label_energy - shift, max_val, keep_dim=True)
        loss = torch.mean(calc_conf_mean(e_loss, confmask))

    elif loss_type is LossType.Force_MSE:
        confmask, *_ = get_confmask()
        natoms = data.get_count("node", idx=None)
        confmask_na = torch.repeat_interleave(confmask, natoms, 0)
        pred_forces = preds["forces"]
        label_forces = data["forces"]
        f_loss = torch.mean((pred_forces - label_forces) ** 2, dim=-1)
        loss = torch.mean(calc_conf_mean(f_loss, confmask_na))

    elif loss_type is LossType.Force_Soft_MSE:
        max_val = pop_kwarg("max", 10.0)
        confmask, *_ = get_confmask()
        natoms = data.get_count("node", idx=None)
        confmask_na = torch.repeat_interleave(confmask, natoms, 0)
        pred_forces = preds["forces"]
        label_forces = data["forces"]
        f_loss = torch.mean(soft_mse(pred_forces - label_forces, max_val, keep_dim=True), dim=-1)
        loss = torch.mean(calc_conf_mean(f_loss, confmask_na))

    elif loss_type is LossType.L1_Norm:
        param_name = pop_kwarg("param")
        ff_params = preds["ff_parameters"]
        if has_kwarg("mask_by_circstd_threshold"):
            proper_std = calc_circstd(data.coords, data.inc_node_proper, data.get_count("proper", idx=None))
            ff_params[param_name] = mask_by_circstd(
                proper_std, ff_params[param_name], threshold=pop_kwarg("mask_by_circstd_threshold")
            )
        loss = torch.mean(torch.abs(ff_params[param_name]))

    else:
        raise NotImplementedError(loss_type)

    unused_kwargs = sorted(set(remaining_kwargs) - consumed_kwargs)
    if unused_kwargs:
        unused_kwargs = ", ".join(unused_kwargs)
        raise ValueError(f"Unused kwargs for {loss_type.name}: {unused_kwargs}")

    return loss
