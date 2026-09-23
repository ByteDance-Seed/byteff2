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

from typing import Union

import torch

from byteff2.data.data import ClusterData, MonoData
from byteff2.utils.definitions import (
    CHG_FACTOR,
    fudgeLJ,
    fudgeQQ,
    MMParam,
    MMTerm,
    NBMMParam,
    NBMMTerm,
    PROPERTORSION_TERMS,
)

from .utils import batch_to_atoms, get_angle_vec, get_dihedral_angle_vec, get_distance_vec, reduce_counts


class ClassicalForceField:
    @staticmethod
    def _autograd_partial_hessian(paired_coords: torch.Tensor, grad: torch.Tensor):
        npairs, nconfs, width, na = paired_coords.shape
        assert na == 3
        assert grad.shape == paired_coords.shape
        # assert paired_coords.requires_grad

        grad_outputs = (
            torch.eye(width * 3, dtype=grad.dtype, device=grad.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand((npairs, nconfs, -1, -1))
        )  # [npairs, nconfs, width*3, width*3]
        grad_outputs = grad_outputs.movedim(-1, 0).reshape(
            -1, npairs, nconfs, width, 3
        )  # [width*3, npairs, nconfs, width, 3]

        hessian = torch.autograd.grad(
            grad, paired_coords, grad_outputs=grad_outputs, create_graph=True, is_grads_batched=True
        )[0]  # [width*3, npairs, nconfs, width, 3]
        hessian = hessian.movedim(0, -3).reshape(
            npairs, nconfs, width, 3, width, 3
        )  # [npairs, nconfs, width, 3, width, 3]
        hessian = hessian.movedim(-2, -3).reshape(npairs, nconfs, width * width, 3 * 3)
        return hessian

    @classmethod
    def _calc_dihedral_energy_forces(
        cls,
        coords,
        node_idx,
        counts,
        k,
        periodicity,
        phase,
        calc_partial_hessian=False,
    ):
        n_conf = coords.shape[1]
        n_term = k.shape[-1]
        k = k.unsqueeze(1).expand(-1, n_conf, -1)  # [ndihedrals, n_conf, nterms]
        periodicity = periodicity.unsqueeze(1).expand(-1, n_conf, -1)  # [ndihedrals, n_conf, nterms]
        phase = phase.unsqueeze(1).expand(-1, n_conf, -1)  # [ndihedrals, n_conf, nterms]

        cc = [coords[node_idx[:, i]].unsqueeze(-2) for i in range(node_idx.shape[1])]
        cc = torch.concat(cc, dim=-2)
        ccs = [cc[:, :, i] for i in range(node_idx.shape[1])]
        theta, f1, f2, f3, f4 = get_dihedral_angle_vec(*ccs)
        theta_expanded = theta.unsqueeze(-1).expand(-1, -1, n_term)  # [ndihedrals, n_conf, nterms]
        dtheta = periodicity * theta_expanded - phase  # [ndihedrals, n_conf, nterms]
        # [ndihedrals, n_conf]
        dihedral_energy = torch.sum(k * (1 + torch.cos(dtheta)), dim=-1)
        energy = reduce_counts(dihedral_energy, counts)  # [batch_size, n_conf]

        forces = torch.zeros_like(coords)  # [n_atom, n_conf, 3]

        # [ndihedrals, n_conf, 1]
        force_prefac = torch.sum(k * periodicity * torch.sin(dtheta), dim=-1).unsqueeze(-1)
        force1 = force_prefac * f1  # [ndihedrals, n_conf, 3]
        force2 = force_prefac * f2  # [ndihedrals, n_conf, 3]
        force3 = force_prefac * f3  # [ndihedrals, n_conf, 3]
        force4 = force_prefac * f4  # [ndihedrals, n_conf, 3]

        # [ndihedrals, n_conf, 3]
        atom1_idxs = node_idx[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom2_idxs = node_idx[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom3_idxs = node_idx[:, 2].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom4_idxs = node_idx[:, 3].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)

        # [ndihedrals, n_conf, 3] -> [n_atom, n_conf, 3]
        forces.scatter_add_(0, atom1_idxs, force1)
        forces.scatter_add_(0, atom2_idxs, force2)
        forces.scatter_add_(0, atom3_idxs, force3)
        forces.scatter_add_(0, atom4_idxs, force4)

        if calc_partial_hessian:
            pair_forces = torch.concat(
                [force1.unsqueeze(-2), force2.unsqueeze(-2), force3.unsqueeze(-2), force4.unsqueeze(-2)], dim=-2
            )  # [ndihedrals, nconfs, width, 3]
            hessian = cls._autograd_partial_hessian(cc, -pair_forces)
        else:
            hessian = None

        return energy, forces, hessian

    @classmethod
    def _calc_nonbonded_energy_forces(
        cls,
        coords: torch.Tensor,
        node_idx: torch.LongTensor,
        counts,
        sigma,
        epsilon,
        partial_charges,
        calc_partial_hessian=False,
    ):
        nconfs = coords.shape[1]
        a1_idxs, a2_idxs = node_idx[:, 0], node_idx[:, 1]
        charge1, charge2 = partial_charges[a1_idxs], partial_charges[a2_idxs]
        sigma1, sigma2 = sigma[a1_idxs], sigma[a2_idxs]
        epsilon1, epsilon2 = epsilon[a1_idxs], epsilon[a2_idxs]

        # combination rule:
        # The combining_rules attribute (default: "none") currently only supports "Lorentz-Berthelot",
        # which specifies the geometric mean of epsilon and arithmetic mean of sigma.
        epsilon = torch.sqrt(epsilon1 * epsilon2)  # [npairs]
        sigma = 0.5 * (sigma1 + sigma2)  # [npairs]

        # [npairs, nconfs], [npairs, nconfs, 3]
        r12, r12vec = get_distance_vec(coords[a1_idxs], coords[a2_idxs])
        invr12 = 1.0 / r12  # [npairs, nconfs]

        sigma_invr12 = sigma.unsqueeze(-1) * invr12  # [npairs, nconfs]
        # U(r) = 4*epsilon*((sigma/r)^12 - (sigma/r)^6)
        u6 = torch.pow(sigma_invr12, 6)  # [npairs, nconfs]
        u12 = torch.square(u6)  # [npairs, nconfs]

        chg_energy = CHG_FACTOR * invr12 * (charge1 * charge2).unsqueeze(-1)  # [npairs, nconfs]
        lj_energy = 4 * epsilon.unsqueeze(-1) * (u12 - u6)  # [npairs, nconfs]
        chg_energy = reduce_counts(chg_energy, counts)  # [batch_size, nconfs]
        lj_energy = reduce_counts(lj_energy, counts)  # [batch_size, nconfs]

        chg_forces = torch.zeros_like(coords)
        lj_forces = torch.zeros_like(coords)

        # IMPORTANT: negative sign
        # [npairs, nconfs, 3]
        pair_force = -CHG_FACTOR * (torch.pow(invr12, 3) * (charge1 * charge2).unsqueeze(-1)).unsqueeze(-1) * r12vec
        chg_forces.scatter_add_(0, a1_idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3), pair_force)
        chg_forces.scatter_add_(0, a2_idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3), -pair_force)

        # IMPORTANT: negative sign
        pair_force = 4 * ((epsilon.unsqueeze(-1) * (-12 * u12 + 6 * u6)) * (invr12 * invr12)).unsqueeze(-1) * r12vec
        lj_forces.scatter_add_(0, a1_idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3), pair_force)
        lj_forces.scatter_add_(0, a2_idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3), -pair_force)
        if calc_partial_hessian:
            raise NotImplementedError("Partial Hessian is not implemented for nonbonded force")
        return chg_energy, chg_forces, lj_energy, lj_forces

    @classmethod
    def calc_bond(
        cls,
        mol_coords: torch.Tensor,
        ff_params: dict,
        bond_idx: torch.Tensor,
        bond_counts: torch.Tensor,
        calc_partial_hessian: bool = False,
    ):
        """
        mol_coords: (N, 3)
        bond_k: (N_bond, 1)
        bond_r0: (N_bond, 1)
        bond_idx: (N_bond, 2)
        bond_counts: (N, 1)
        """
        n_conf = mol_coords.shape[1]
        if bond_idx.shape[0] == 0:
            return 0.0, 0.0, None
        bond_k = ff_params[MMParam.bond_k]
        bond_r0 = ff_params[MMParam.bond_r0]
        cc = [mol_coords[bond_idx[:, i]].unsqueeze(-2) for i in range(bond_idx.shape[1])]
        cc = torch.concat(cc, dim=-2)

        ccs = [cc[:, :, i] for i in range(bond_idx.shape[1])]
        r12, r12vec = get_distance_vec(*ccs)

        pair_energy = 0.5 * bond_k * (r12 - bond_r0) ** 2  # [nbonds, n_conf]
        energy = reduce_counts(pair_energy, bond_counts)  # [batch_size, n_conf]

        forces = torch.zeros_like(mol_coords)  # [n_atom, n_conf, 3]
        pair_forces = (bond_k * (1 - bond_r0 / r12)).unsqueeze(-1) * r12vec  # [nbonds, n_conf, 3]
        atom1_idxs = bond_idx[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom2_idxs = bond_idx[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        # IMPORTANT: negative sign
        forces.scatter_add_(0, atom1_idxs, pair_forces)
        forces.scatter_add_(0, atom2_idxs, -pair_forces)
        if calc_partial_hessian:
            pair_forces = torch.concat(
                [pair_forces.unsqueeze(-2), -pair_forces.unsqueeze(-2)], dim=-2
            )  # [nbonds, nconfs, width, 3]
            hessian = cls._autograd_partial_hessian(cc, -pair_forces)
        else:
            hessian = None
        return energy, forces, hessian

    @classmethod
    def calc_conj_bond(
        cls,
        mol_coords: torch.Tensor,
        bond_k1: torch.Tensor,
        bond_k2: torch.Tensor,
        bond_b1: torch.Tensor,
        bond_b2: torch.Tensor,
        bond_idx: torch.Tensor,
        bond_counts: torch.Tensor,
        calc_partial_hessian: bool = False,
    ):
        if bond_idx.shape[0] == 0:
            return 0.0, 0.0, None
        n_conf = mol_coords.shape[1]
        cc = [mol_coords[bond_idx[:, i]].unsqueeze(-2) for i in range(bond_idx.shape[1])]
        cc = torch.concat(cc, dim=-2)
        ccs = [cc[:, :, i] for i in range(bond_idx.shape[1])]
        r12, r12vec = get_distance_vec(*ccs)

        pair_energy = 0.5 * bond_k1 * (r12 - bond_b1) ** 2 + 0.5 * bond_k2 * (r12 - bond_b2) ** 2  # [nbonds, n_conf]
        bond_k_sum = bond_k1 + bond_k2
        pair_energy += 0.5 * (
            -bond_k1 * bond_b1**2 - bond_k2 * bond_b2**2 + (bond_k1 * bond_b1 + bond_k2 * bond_b2) ** 2 / bond_k_sum
        )
        energy = reduce_counts(pair_energy, bond_counts)  # [batch_size, n_conf]

        forces = torch.zeros_like(mol_coords)  # [n_atom, n_conf, 3]
        pair_forces = (bond_k1 * (1 - bond_b1 / r12) + bond_k2 * (1 - bond_b2 / r12)).unsqueeze(
            -1
        ) * r12vec  # [nbonds, n_conf, 3]
        atom1_idxs = bond_idx[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom2_idxs = bond_idx[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        # IMPORTANT: negative sign
        forces.scatter_add_(0, atom1_idxs, pair_forces)
        forces.scatter_add_(0, atom2_idxs, -pair_forces)
        if calc_partial_hessian:
            pair_forces = torch.concat(
                [pair_forces.unsqueeze(-2), -pair_forces.unsqueeze(-2)], dim=-2
            )  # [nbonds, nconfs, width, 3]
            hessian = cls._autograd_partial_hessian(cc, -pair_forces)
        else:
            hessian = None
        return energy, forces, hessian

    @classmethod
    def calc_angle(
        cls,
        mol_coords: torch.Tensor,
        ff_params: dict,
        angle_idx: torch.Tensor,
        angle_counts: torch.Tensor,
        calc_partial_hessian: bool = False,
    ):
        """
        mol_coords: (N, 3)
        angle_k: (N_angle, 1)
        angle_r0: (N_angle, 1)
        angle_idx: (N_angle, 3)
        angle_counts: (N, 1)
        """
        n_conf = mol_coords.shape[1]
        if angle_idx.shape[0] == 0:
            return 0.0, 0.0, None
        angle_k = ff_params[MMParam.angle_k]
        angle_d0 = torch.deg2rad(ff_params[MMParam.angle_d0])

        cc = [mol_coords[angle_idx[:, i]].unsqueeze(-2) for i in range(angle_idx.shape[1])]
        cc = torch.concat(cc, dim=-2)
        ccs = [cc[:, :, i] for i in range(angle_idx.shape[1])]
        theta, f1, f3 = get_angle_vec(*ccs)
        pair_energy = 0.5 * angle_k * (theta - angle_d0) ** 2  # [nangles, n_conf]
        energy = reduce_counts(pair_energy, angle_counts)  # [batch_size, n_conf]

        forces = torch.zeros_like(mol_coords)  # [n_atom, n_conf, 3]
        # [nbatch, n_conf, nangles]
        atom1_idxs = angle_idx[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom2_idxs = angle_idx[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom3_idxs = angle_idx[:, 2].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        fc = -(angle_k * (theta - angle_d0)).unsqueeze(-1)  # [nangles, n_conf, 1]
        force1 = fc * f1
        force3 = fc * f3
        force2 = -force1 - force3
        forces.scatter_add_(0, atom1_idxs, force1)
        forces.scatter_add_(0, atom2_idxs, force2)
        forces.scatter_add_(0, atom3_idxs, force3)
        if calc_partial_hessian:
            pair_forces = torch.concat(
                [force1.unsqueeze(-2), force2.unsqueeze(-2), force3.unsqueeze(-2)], dim=-2
            )  # [nangles, nconfs, width, 3]
            hessian = cls._autograd_partial_hessian(cc, -pair_forces)
        else:
            hessian = None

        return energy, forces, hessian

    @classmethod
    def calc_conj_angle(
        cls,
        mol_coords: torch.Tensor,
        angle_k1: torch.Tensor,
        angle_k2: torch.Tensor,
        angle_b1: torch.Tensor,
        angle_b2: torch.Tensor,
        angle_idx: torch.Tensor,
        angle_counts: torch.Tensor,
        calc_partial_hessian: bool = False,
    ):
        if angle_idx.shape[0] == 0:
            return 0.0, 0.0, None
        n_conf = mol_coords.shape[1]
        cc = [mol_coords[angle_idx[:, i]].unsqueeze(-2) for i in range(angle_idx.shape[1])]
        cc = torch.concat(cc, dim=-2)
        ccs = [cc[:, :, i] for i in range(angle_idx.shape[1])]
        theta, f1, f3 = get_angle_vec(*ccs)
        pair_energy = (
            0.5 * angle_k1 * (theta - angle_b1) ** 2 + 0.5 * angle_k2 * (theta - angle_b2) ** 2
        )  # [nangles, n_conf]
        angle_k_sum = angle_k1 + angle_k2
        pair_energy += 0.5 * (
            -angle_k1 * angle_b1**2
            - angle_k2 * angle_b2**2
            + (angle_k1 * angle_b1 + angle_k2 * angle_b2) ** 2 / angle_k_sum
        )
        energy = reduce_counts(pair_energy, angle_counts)  # [batch_size, n_conf]

        forces = torch.zeros_like(mol_coords)  # [n_atom, n_conf, 3]
        # [nbatch, n_conf, nangles]
        atom1_idxs = angle_idx[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom2_idxs = angle_idx[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        atom3_idxs = angle_idx[:, 2].unsqueeze(-1).unsqueeze(-1).expand(-1, n_conf, 3)
        fc = -(angle_k1 * (theta - angle_b1) + angle_k2 * (theta - angle_b2)).unsqueeze(-1)  # [nangles, n_conf, 1]
        force1 = fc * f1
        force3 = fc * f3
        force2 = -force1 - force3
        forces.scatter_add_(0, atom1_idxs, force1)
        forces.scatter_add_(0, atom2_idxs, force2)
        forces.scatter_add_(0, atom3_idxs, force3)

        if calc_partial_hessian:
            pair_forces = torch.concat(
                [force1.unsqueeze(-2), force2.unsqueeze(-2), force3.unsqueeze(-2)], dim=-2
            )  # [nangles, nconfs, width, 3]
            hessian = cls._autograd_partial_hessian(cc, -pair_forces)
        else:
            hessian = None

        return energy, forces, hessian

    @classmethod
    def calc_proper(
        cls,
        mol_coords: torch.Tensor,
        proper_params: dict,
        proper_idx: torch.Tensor,
        proper_counts: torch.Tensor,
        calc_partial_hessian: bool = False,
    ):
        if proper_idx.shape[0] == 0:
            return 0.0, 0.0, None
        proper_k = proper_params[MMParam.proper_k]
        proper_n = [n + 1 for n in range(PROPERTORSION_TERMS)]
        periodicity = (
            torch.tensor(proper_n, dtype=proper_k.dtype, device=proper_k.device)
            .unsqueeze(0)
            .expand(proper_k.shape[0], -1)
        )
        phase = (
            (((torch.tensor(proper_n, device=proper_k.device, dtype=torch.int64) + 1) % 2) * torch.pi)
            .to(proper_k.dtype)
            .unsqueeze(0)
            .expand(proper_k.shape[0], -1)
        )
        return cls._calc_dihedral_energy_forces(
            mol_coords, proper_idx, proper_counts, proper_k, periodicity, phase, calc_partial_hessian
        )

    @classmethod
    def calc_improper(
        cls,
        mol_coords: torch.Tensor,
        improper_params: dict,
        improper_idx: torch.Tensor,
        improper_counts: torch.Tensor,
        calc_partial_hessian: bool = False,
    ):
        if improper_idx.shape[0] == 0:
            return 0.0, 0.0, None
        improper_k = improper_params[MMParam.improper_k]
        periodicity = torch.ones_like(improper_k) * 2.0
        phase = torch.ones_like(improper_k) * torch.pi
        return cls._calc_dihedral_energy_forces(
            mol_coords,
            improper_idx,
            improper_counts,
            improper_k,
            periodicity,
            phase,
            calc_partial_hessian,
        )

    @classmethod
    def calc_nonbonded14(
        cls,
        mol_coords: torch.Tensor,
        nonbonded14_params: dict,
        nonbonded14_idx: torch.Tensor,
        nonbonded14_counts: torch.Tensor,
        charge14_scale: float = fudgeQQ,
        lj14_scale: float = fudgeLJ,
        calc_partial_hessian: bool = False,
    ):
        sigma = nonbonded14_params[NBMMParam.sigma]
        epsilon = nonbonded14_params[NBMMParam.epsilon]
        charge = nonbonded14_params[NBMMParam.charges]
        chg_e, chg_f, lj_e, lj_f = cls._calc_nonbonded_energy_forces(
            mol_coords,
            nonbonded14_idx,
            nonbonded14_counts,
            sigma,
            epsilon,
            charge,
            calc_partial_hessian=calc_partial_hessian,
        )
        return chg_e * charge14_scale + lj_e * lj14_scale, chg_f * charge14_scale + lj_f * lj14_scale, None

    @classmethod
    def calc_nonbonded_all(
        cls,
        mol_coords: torch.Tensor,
        nonbonded_all_params: dict,
        nonbonded_all_idx: torch.Tensor,
        nonbonded_all_counts: torch.Tensor,
        calc_partial_hessian: bool = False,
    ):
        sigma = nonbonded_all_params[NBMMParam.sigma]
        epsilon = nonbonded_all_params[NBMMParam.epsilon]
        charge = nonbonded_all_params[NBMMParam.charges]
        chg_e, chg_f, lj_e, lj_f = cls._calc_nonbonded_energy_forces(
            mol_coords,
            nonbonded_all_idx,
            nonbonded_all_counts,
            sigma,
            epsilon,
            charge,
            calc_partial_hessian=calc_partial_hessian,
        )
        return chg_e + lj_e, chg_f + lj_f, None

    @classmethod
    def energy_force(
        cls,
        data: Union[MonoData, ClusterData],
        ff_params: dict,
        calc_terms: list = None,
        cluster: bool = False,
        calc_partial_hessian: bool = False,
        confmask: torch.Tensor = None,
    ):
        if calc_terms is None:
            calc_terms = list(MMTerm) + list(NBMMTerm)
        mol_coords = data.coords
        if calc_partial_hessian:
            mol_coords.requires_grad = True
        energy_force_results = {}
        total_energy = 0.0
        total_forces = torch.zeros_like(mol_coords)
        for term in calc_terms:
            node_idx = data[f"inc_node_{term.name}"].long()
            counts = data.get_count(term.name, idx=None, cluster=cluster)
            energy, forces, hessian = getattr(cls, f"calc_{term.name}")(
                mol_coords,
                ff_params,
                node_idx,
                counts,
                calc_partial_hessian,
            )
            if confmask is not None:
                energy = energy * confmask
                natoms = data.get_count("node", idx=None, cluster=cluster)
                confmask_forces = batch_to_atoms(confmask, natoms)
                forces = forces * confmask_forces
            energy_force_results[term.name] = (energy, forces, hessian)
            total_energy += energy
            total_forces += forces
        energy_force_results["total_energy"] = total_energy
        energy_force_results["total_forces"] = total_forces
        return energy_force_results
