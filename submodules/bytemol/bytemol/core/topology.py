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

from operator import itemgetter
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx

from bytemol.core import rkutil


class Topology:
    # each node is represented by its index
    # each node can have arbitrary node labels supported by networkx

    def __init__(self,
                 bonds: List[Tuple[int, int]],
                 node_labels: Dict[Any, Iterable] = None,
                 bond_ids: List[int] = None,
                 max_include_ring: int = 10):
        # sanitize input
        if bond_ids is not None:
            assert len(bonds) == len(bond_ids)
            assert len(bond_ids) == len(set(bond_ids))
            for idx in bond_ids:
                assert idx >= 0
        for a_ids in bonds:
            assert a_ids[0] >= 0 and a_ids[1] >= 0

        # construct graph
        # The same bond will be overwritten.
        # bonds=[[0,1],[0,1]] bond_idx=[0,1] -> edge(0, 1 ,{'idx': 1})
        self._graph = nx.Graph()
        # edges
        if bond_ids is None:
            for bond_idx, a_ids in enumerate(bonds):
                self._graph.add_edge(a_ids[0], a_ids[1], idx=bond_idx)
        else:
            for bond_idx, a_ids in zip(bond_ids, bonds):
                self._graph.add_edge(a_ids[0], a_ids[1], idx=bond_idx)

        # handle monatomic molecule
        if len(bonds) == 0:
            self._graph.add_node(0)

        # nodes
        if node_labels is not None:
            for label_name, label_data in node_labels.items():
                assert len(label_data) == len(self._graph.nodes)
                for node_idx, node_data in enumerate(label_data):
                    self._graph.nodes[node_idx][label_name] = node_data

        # initialize data members
        self._atoms = list(self._graph.nodes)
        self._bonds = list()
        for a_ids in self._graph.edges:
            assert a_ids[0] != a_ids[1]
            self._bonds.append(rkutil.sorted_atomids(a_ids))
        self._bonds.sort(key=itemgetter(0, 1))
        self._adj_list = self._cal_adj_list()
        self._angles = self._cal_angles()
        self._propers = self._cal_propers()
        self._atoms_with_three_neighbors = self._cal_atoms_with_three_neighbors()
        self._atoms_with_one_neighbor = self._cal_atoms_with_one_neighbor()
        self._rings: List[List[int]] = None
        self._nonbondedall: List[Tuple[int, int]] = None
        self._nonbonded12: List[Tuple[int, int]] = None
        self._nonbonded13: List[Tuple[int, int]] = None
        self._nonbonded14: List[Tuple[int, int]] = None
        self._nonbonded15: List[Tuple[int, int]] = None
        self._max_include_ring = max_include_ring

    def _cal_adj_list(self) -> Dict[int, List[int]]:
        adj_list = dict()
        for atom_id in self._graph.nodes:
            adj_list[atom_id] = list(sorted(self._graph.neighbors(atom_id)))
        return adj_list

    def _cal_angles(self) -> List[Tuple[int]]:
        atomsets = set()
        for atom_id in self._graph.nodes:
            neighbors = self._adj_list[atom_id]
            if len(neighbors) > 1:
                for i, a1 in enumerate(neighbors[:-1]):
                    for a2 in neighbors[i + 1:]:
                        atomsets.add(rkutil.sorted_atomids((a1, atom_id, a2)))
        angles = list(atomsets)
        angles.sort(key=itemgetter(0, 1, 2))
        return angles

    def _cal_propers(self) -> List[Tuple[int]]:
        atomsets = set()
        for a1, a2 in list(self._graph.edges):
            a1_neighbors = set(self._adj_list[a1])
            a2_neighbors = set(self._adj_list[a2])
            a1_neighbors.remove(a2)
            a2_neighbors.remove(a1)
            for a1n in a1_neighbors:
                for a2n in a2_neighbors:
                    if a1n == a2n:
                        continue
                    atomsets.add(rkutil.sorted_atomids((a1n, a1, a2, a2n)))
        propers = list(atomsets)
        propers.sort(key=itemgetter(0, 1, 2, 3))
        return propers

    def _cal_atoms_with_three_neighbors(self) -> List[Tuple[int]]:
        atomsets = set()
        for atom in self._graph.nodes:
            neighbors = self._adj_list[atom]
            if len(neighbors) > 2:
                for i, a1 in enumerate(neighbors[:-2]):
                    for j, a2 in enumerate(neighbors[i + 1:-1]):
                        for a3 in neighbors[i + j + 2:]:
                            atomsets.add(rkutil.sorted_atomids((atom, a1, a2, a3), is_improper=True))
        atoms_with_three_neighbors = list(atomsets)
        atoms_with_three_neighbors.sort(key=itemgetter(0, 1, 2, 3))
        return atoms_with_three_neighbors

    def _cal_atoms_with_one_neighbor(self) -> List[Tuple[int]]:
        atomsets = set()
        for atom_id in self._graph.nodes:
            neighbors = self._adj_list[atom_id]
            if len(neighbors) == 1:
                atomsets.add((atom_id, neighbors[0]))
        atoms_with_one_neighbor = list(atomsets)
        atoms_with_one_neighbor.sort(key=itemgetter(0, 1))
        return atoms_with_one_neighbor

    def _cal_nonbonded(self) -> Tuple[List[Tuple], List[Tuple]]:
        self._calc_nonbonded_local()
        nonbondedall = set([
            tuple(sorted([self.atoms[i], self.atoms[j]]))
            for i in range(self.natoms - 1)
            for j in range(i + 1, self.natoms)
        ])
        nonbondedall = nonbondedall - set(self._nonbonded14) - set(self._nonbonded13) - set(self._nonbonded12)
        return sorted(nonbondedall), self._nonbonded14

    @staticmethod
    def _cal_rings(G: nx.Graph, max_ring_size=8) -> list[list[int]]:
        """ Modified from from https://gist.github.com/joe-jordan/6548029,
            nearly O(n) algorithm.
            Support graph that is not connected.
        """

        def get_hashable_cycle(cycle: list[int]) -> tuple[int]:
            """cycle as a tuple in a deterministic order."""
            #cycle = [atom.idx for atom in cycle]
            m = min(cycle)
            mi = cycle.index(m)
            mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
            if cycle[mi - 1] > cycle[mi_plus_1]:
                result = cycle[mi:] + cycle[:mi]
            else:
                result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
            return tuple(result)

        output_cycles = set()
        for node in G.nodes:
            src = node
            cycle_stack = [src]
            stack = [(src, iter(G[src]))]
            indices = {src: 0}
            while stack:
                src, children = stack[-1]
                try:
                    child = next(children)

                    if child not in indices:
                        cycle_stack.append(child)
                        stack.append((child, iter(G[child])))
                        indices[child] = len(cycle_stack) - 1
                    elif child == cycle_stack[0]:
                        if len(cycle_stack) > 2:
                            output_cycles.add(get_hashable_cycle(cycle_stack))
                except StopIteration:
                    stack.pop()
                    cycle_stack.pop()
                    indices.pop(src)
                if len(cycle_stack) > max_ring_size:
                    src, _ = stack.pop()
                    cycle_stack.pop()
                    indices.pop(src)
        return [list(i) for i in output_cycles]

    @property
    def graph(self) -> nx.Graph:
        return self._graph

    @property
    def natoms(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def nbonds(self) -> int:
        return self._graph.number_of_edges()

    @property
    def atoms(self) -> list[int]:
        return self._atoms

    @property
    def bonds(self) -> List[Tuple[int]]:
        return self._bonds

    @property
    def adj_list(self) -> Dict[int, List[int]]:
        # standard adjacency list
        return self._adj_list

    @property
    def angles(self) -> List[Tuple[int]]:
        return self._angles

    @property
    def propers(self) -> List[Tuple[int]]:
        return self._propers

    @property
    def atoms_with_three_neighbors(self) -> List[Tuple[int]]:
        ''' return list of central atoms with three neighbors represented by atom ids
        '''
        return self._atoms_with_three_neighbors

    @property
    def atoms_with_one_neighbor(self) -> List[int]:
        return self._atoms_with_one_neighbor

    def _calc_nonbonded_local(self):
        nonbonded12, nonbonded13, nonbonded14, nonbonded15 = set(), set(), set(), set()
        paths = dict(nx.all_pairs_shortest_path_length(self.graph, cutoff=4))
        for k1, v in paths.items():
            for k2, distance in v.items():
                pair = tuple(sorted((k1, k2)))
                if distance == 1:
                    nonbonded12.add(pair)
                elif distance == 2:
                    nonbonded13.add(pair)
                elif distance == 3:
                    nonbonded14.add(pair)
                elif distance == 4:
                    nonbonded15.add(pair)
        self._nonbonded12 = sorted(nonbonded12)
        self._nonbonded13 = sorted(nonbonded13)
        self._nonbonded14 = sorted(nonbonded14)
        self._nonbonded15 = sorted(nonbonded15)

    @property
    def nonbonded12_pairs(self) -> List[Tuple[int]]:
        if self._nonbonded12 is None:
            self._calc_nonbonded_local()
        return self._nonbonded12.copy()

    @property
    def nonbonded13_pairs(self) -> List[Tuple[int]]:
        if self._nonbonded13 is None:
            self._calc_nonbonded_local()
        return self._nonbonded13.copy()

    @property
    def nonbonded14_pairs(self) -> List[Tuple[int]]:
        if self._nonbonded14 is None:
            self._calc_nonbonded_local()
        return self._nonbonded14.copy()

    @property
    def nonbonded15_pairs(self) -> List[Tuple[int]]:
        if self._nonbonded15 is None:
            self._calc_nonbonded_local()
        return self._nonbonded15.copy()

    @property
    def nonbondedall_pairs(self) -> List[Tuple[int]]:
        if self._nonbondedall is None:
            self._nonbondedall, self._nonbonded14 = self._cal_nonbonded()
        return self._nonbondedall.copy()

    @property
    def rings(self) -> List[List[int]]:
        if self._rings is None:
            self._rings = self._cal_rings(self.graph, max_ring_size=self._max_include_ring)
        return self._rings

    def get_nonring_dihedral_rotate_atoms(self, *indices: List[int]) -> List[int]:
        assert len(indices) == 4
        _, a2, a3, _ = indices
        scanned_list = [a3]
        to_scan_list = self.adj_list[a3].copy()
        to_scan_list.remove(a2)
        while to_scan_list:
            idx = to_scan_list.pop(0)
            scanned_list.append(idx)
            new_neighbor_set = set(self.adj_list[idx]) - set(scanned_list) - set(to_scan_list)
            assert a2 not in new_neighbor_set, f"a2({a2})-a3({a3}) is in a ring"
            to_scan_list.extend(list(new_neighbor_set))
        scanned_list.remove(a3)
        return scanned_list
