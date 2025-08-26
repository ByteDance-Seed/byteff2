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

import ase.geometry as ag
import numpy as np

from bytemol.units import simple_unit as su

logger = logging.getLogger(__name__)

###################################################
#                    !IMPORTANT!
#       paraidx and atomidx should begin from 0 !!!
#       user should guarantee the input is valid
###################################################


def _reshape_to_two_dims(arr: np.ndarray) -> np.ndarray:
    """Flatten the first two dimensions of a numpy array.

    Args:
        arr (np.ndarray): Input numpy array.

    Returns:
        np.ndarray: Flattened array with the first two dimensions combined.
    """
    shape = arr.shape
    if len(shape) < 2:
        raise ValueError("Input array should have at least 2 dimensions")

    flattened_shape = (np.prod(shape[:-1]),) + (shape[-1],)
    flattened_arr = np.reshape(arr, flattened_shape)
    return flattened_arr


def get_coords(coords: np.ndarray, atomidx: np.ndarray):
    """ Get coords according to atom_idxs
    coords: [..., nconfs, natoms, 3]
    atomidx: [..., ngroups, n]
    """
    assert coords.shape[:-3] == atomidx.shape[:-2]
    assert coords.ndim == 4
    assert atomidx.ndim == 3
    na = atomidx.shape[-1]

    # [nbatch, nconfs, ngroups, 3]
    expand_size = list(coords.shape[:2]) + [atomidx.shape[-2]] + [3]

    coords_list = []
    for i in range(na):
        atom_idxs_i = np.expand_dims(atomidx[..., i], axis=(-3, -1))
        atom_idxs_i = np.broadcast_to(atom_idxs_i, shape=expand_size)
        atom_coords_i = np.take_along_axis(coords, atom_idxs_i, axis=-2)
        coords_list.append(atom_coords_i)
    return coords_list


def get_distance_vec(coords: np.ndarray, atomidx: np.ndarray):
    """ Compute distance between atoms

    coords: [nbatch, nconfs, natoms, 3]
    atomidx: [nbatch, npairs, 2]
    """
    assert atomidx.shape[-1] == 2
    coords_list = get_coords(coords, atomidx)
    r12vec = coords_list[1] - coords_list[0]
    r12 = np.linalg.norm(r12vec, axis=-1)
    return r12, r12vec


def get_angle_vec(coords: np.ndarray, atomidx: np.ndarray, with_vec: bool = True):
    ''' calculate angle
        coords: [..., nconfs, natoms, 3]
        atomidx: [..., nangles, 3]
    '''
    # nbatch, nconfs, ngroups
    result_shape = list(coords.shape[:2]) + [atomidx.shape[-2]]

    # [..., nconfs, nangles, 3] * 3
    coords_list = get_coords(coords, atomidx)

    # here the convention of ase and md are different
    # note the difference between this and the torch kernels in geometry.py
    v0 = coords_list[0] - coords_list[1]  # [nbatch, nconfs, nangles, 3]
    v1 = coords_list[2] - coords_list[1]  # [nbatch, nconfs, nangles, 3]
    v0 = _reshape_to_two_dims(v0)
    v1 = _reshape_to_two_dims(v1)

    angle = su.degree_to_rad(ag.get_angles(v0, v1))
    angle = np.reshape(angle, result_shape)

    if with_vec:
        grad_angle = su.degree_to_rad(ag.get_angles_derivatives(v0, v1))
        f1 = np.reshape(grad_angle[..., 0, :], result_shape + [3])
        f3 = np.reshape(grad_angle[..., 2, :], result_shape + [3])
    else:
        f1 = 0
        f3 = 0

    return angle, f1, f3


def get_dihedral_angle_vec(coords: np.ndarray, atomidx: np.ndarray, with_vec: bool = True):
    ''' calculate dihedral angle
        coords: [..., nconfs, natoms, 3]
        atomidx: [..., nangles, 4]
    '''
    # nbatch, nconfs, ngroups
    result_shape = list(coords.shape[:2]) + [atomidx.shape[-2]]

    # [nbatch, nconfs, ndihedrals, 3] * 4
    coords_list = get_coords(coords, atomidx)

    v0 = coords_list[1] - coords_list[0]  # [nbatch, nconfs, nangles, 3]
    v1 = coords_list[2] - coords_list[1]  # [nbatch, nconfs, nangles, 3]
    v2 = coords_list[3] - coords_list[2]  # [nbatch, nconfs, nangles, 3]
    v0 = _reshape_to_two_dims(v0)
    v1 = _reshape_to_two_dims(v1)
    v2 = _reshape_to_two_dims(v2)

    dihedral = su.degree_to_rad(ag.get_dihedrals(v0, v1, v2))
    # convert from ase convention to md convention
    dihedral = np.where(dihedral > np.pi, dihedral - 2 * np.pi, dihedral)
    dihedral = np.reshape(dihedral, result_shape)

    if with_vec:
        grad_dihedral = su.degree_to_rad(ag.get_dihedrals_derivatives(v0, v1, v2))
        fi = np.reshape(grad_dihedral[..., 0, :], result_shape + [3])
        fj = np.reshape(grad_dihedral[..., 1, :], result_shape + [3])
        fk = np.reshape(grad_dihedral[..., 2, :], result_shape + [3])
        fl = np.reshape(grad_dihedral[..., 3, :], result_shape + [3])
    else:
        fi = 0
        fj = 0
        fk = 0
        fl = 0

    return dihedral, fi, fj, fk, fl


def convert_to_ase_dihedral_degree(degree):
    if not np.isscalar(degree):
        degree = np.asarray(degree)
    return degree % 360  # convert to [0,360), unit degree


def convert_to_gmx_dihedral_degree(degree):
    if not np.isscalar(degree):
        degree = np.asarray(degree)
    return (degree + 180) % 360.0 - 180.0  # convert to [-180,180), unit degree


def get_dihedral_angle_degree(positions: np.ndarray, torsion_atom_idxs: np.ndarray) -> np.ndarray:
    if not isinstance(positions, np.ndarray):
        positions = np.array(positions, dtype=np.float64)
    if not isinstance(torsion_atom_idxs, np.ndarray):
        torsion_atom_idxs = np.array(torsion_atom_idxs, dtype=np.int64)

    assert positions.ndim == 2 and positions.shape[-1] == 3
    assert torsion_atom_idxs.shape[-1] == 4

    if torsion_atom_idxs.ndim == 1:
        p0, p1, p2, p3 = positions[torsion_atom_idxs, ...]
        dihedral = ag.get_dihedrals([p1 - p0], [p2 - p1], [p3 - p2])
        dihedral = dihedral[0]  # np scalar

    elif torsion_atom_idxs.ndim == 2:
        p0 = positions[torsion_atom_idxs[:, 0], ...]
        p1 = positions[torsion_atom_idxs[:, 1], ...]
        p2 = positions[torsion_atom_idxs[:, 2], ...]
        p3 = positions[torsion_atom_idxs[:, 3], ...]
        dihedral = ag.get_dihedrals(p1 - p0, p2 - p1, p3 - p2)  # 1d-array

    return convert_to_gmx_dihedral_degree(dihedral)
