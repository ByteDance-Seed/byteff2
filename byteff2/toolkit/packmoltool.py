"""Initial liquid-box packing with Packmol.

Replaces the former ``gmx editconf`` + ``gmx insert-molecules`` box builder.
Packmol cannot read ``.gro``, so single-molecule templates are exported as
``.xyz`` (Angstrom), packed, and the resulting box is reassembled into
``solvent_salt.gro`` (nm) here.  Writing the ``.gro`` ourselves keeps the atom
and residue ordering identical to the component order, which the
``[ molecules ]`` block of ``system.top`` and ``onsager_calc`` both rely on.
"""

import os
import shutil
import subprocess

import numpy as np

from bytemol.utils import setup_default_logging

logger = setup_default_logging()

NM_TO_ANGSTROM = 10.0

# Packmol input file and the coordinate file it produces.
PACKMOL_INPUT = 'pack.inp'
PACKMOL_OUTPUT = 'solvent_salt.xyz'
TARGET_GRO = 'solvent_salt.gro'

# Minimum distance between atoms of different molecules, in Angstrom.
DEFAULT_TOLERANCE = 2.0

# Fallback location of the packmol binary, used only if it is neither pointed
# at by $PACKMOL_BIN nor found on $PATH.
FALLBACK_PACKMOL_BIN = '/home/ytong/software/packmol/packmol'


def packmol_bin() -> str:
    """Locate the packmol executable: $PACKMOL_BIN, then $PATH, then fallback."""
    env_bin = os.environ.get('PACKMOL_BIN')
    if env_bin and os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
        return env_bin
    which_bin = shutil.which('packmol')
    if which_bin:
        return which_bin
    if os.path.isfile(FALLBACK_PACKMOL_BIN) and os.access(FALLBACK_PACKMOL_BIN, os.X_OK):
        return FALLBACK_PACKMOL_BIN
    raise RuntimeError(
        'packmol executable not found. Set $PACKMOL_BIN to its full path, put '
        '`packmol` on $PATH, or install it at '
        f'{FALLBACK_PACKMOL_BIN}. See https://m3g.github.io/packmol')


class MoleculeTemplate:
    """A single-molecule conformer read back from a ``.gro`` template."""

    def __init__(self, resname: str, atom_names: list, positions: np.ndarray):
        self.resname = resname
        self.atom_names = atom_names
        self.positions = positions  # (natoms, 3) in nm

    @property
    def natoms(self) -> int:
        return len(self.atom_names)

    @classmethod
    def from_gro(cls, path: str) -> 'MoleculeTemplate':
        with open(path) as f:
            lines = f.readlines()
        natoms = int(lines[1].strip())
        atom_names, resnames, positions = [], [], []
        for line in lines[2:2 + natoms]:
            resnames.append(line[5:10].strip())
            atom_names.append(line[10:15].strip())
            positions.append([float(line[20:28]), float(line[28:36]), float(line[36:44])])
        resname = resnames[0]
        assert all(r == resname for r in resnames), \
            f'{path} holds more than one residue name, expected a single-molecule template'
        return cls(resname, atom_names, np.asarray(positions, dtype=float))

    def write_xyz(self, path: str):
        """Write the template in Angstrom for packmol to read."""
        with open(path, 'w') as f:
            f.write(f'{self.natoms}\n')
            f.write(f'{self.resname}\n')
            for name, (x, y, z) in zip(self.atom_names, self.positions * NM_TO_ANGSTROM):
                f.write(f'{name:<4s} {x:12.6f} {y:12.6f} {z:12.6f}\n')


def _xyz_name(name: str) -> str:
    return f'{name}.xyz'


def generate_packmol_input(components, working_dir: str, box: float, tolerance: float = DEFAULT_TOLERANCE):
    """Write one ``.xyz`` per component plus the packmol input file.

    ``box`` is the cubic edge in nm.  Molecules are packed into a region inset
    by ``tolerance / 2`` from each face so that nothing straddles the periodic
    boundary (packmol 20.3.5 predates the native ``pbc`` keyword).
    """
    templates = {}
    for c in components.values():
        template = MoleculeTemplate.from_gro(os.path.join(working_dir, f'{c.name}.gro'))
        template.write_xyz(os.path.join(working_dir, _xyz_name(c.name)))
        templates[c.name] = template

    edge = box * NM_TO_ANGSTROM
    lo = tolerance / 2
    hi = edge - tolerance / 2
    assert hi > lo, f'box edge {box} nm is too small to pack into'

    lines = [
        f'tolerance {tolerance}',
        'filetype xyz',
        f'output {PACKMOL_OUTPUT}',
        # clock-seeded, so each protocol packs an independent box, as
        # `gmx insert-molecules` used to
        'seed -1',
        '',
    ]
    for c in components.values():
        lines += [
            f'structure {_xyz_name(c.name)}',
            f'  number {c.molar_num}',
            f'  inside box {lo:.3f} {lo:.3f} {lo:.3f} {hi:.3f} {hi:.3f} {hi:.3f}',
            'end structure',
            '',
        ]
    with open(os.path.join(working_dir, PACKMOL_INPUT), 'w') as f:
        f.write('\n'.join(lines))
    return templates


def run_packmol(working_dir: str, timeout: int = 300) -> bool:
    """Run packmol on ``pack.inp``. Returns False if the box needs to grow.

    Packmol can exit 0 having timed out or reported ``ENDED WITHOUT PERFECT
    PACKING``, so its output is inspected as well as the return code.
    """
    # A previous attempt's output must not be mistaken for this one's.
    stale = os.path.join(working_dir, PACKMOL_OUTPUT)
    if os.path.isfile(stale):
        os.remove(stale)

    # packmol reads its input from stdin. Feeding the file directly rather than
    # going through a shell keeps `timeout` pointed at packmol itself, so a
    # stalled pack is actually killed instead of being orphaned behind a shell.
    try:
        with open(os.path.join(working_dir, PACKMOL_INPUT)) as fin:
            child = subprocess.run(
                [packmol_bin()],
                stdin=fin,
                cwd=working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired:
        logger.warning('packmol timed out after %ss', timeout)
        return False

    stdout = child.stdout or ''
    if child.returncode != 0:
        logger.warning('packmol exited with code %s:\n%s', child.returncode, (child.stderr or '')[-2000:])
        return False
    # Packmol rewrites the output file periodically as it works, and that
    # progress file already holds every atom -- it is merely still overlapping.
    # So the atom count proves nothing; only packmol's own verdict does. It
    # prints the marker below once, at the end, and flags a bad pack explicitly.
    if 'Solution written to file' not in stdout:
        logger.warning('packmol did not finish (no solution written)')
        return False
    if 'ENDED WITHOUT PERFECT PACKING' in stdout:
        logger.warning('packmol ended without perfect packing')
        return False
    if not os.path.isfile(os.path.join(working_dir, PACKMOL_OUTPUT)):
        logger.warning('packmol produced no %s', PACKMOL_OUTPUT)
        return False
    return True


def _read_xyz_positions(path: str) -> np.ndarray:
    with open(path) as f:
        lines = f.readlines()
    natoms = int(lines[0].strip())
    positions = np.empty((natoms, 3), dtype=float)
    for i, line in enumerate(lines[2:2 + natoms]):
        positions[i] = [float(v) for v in line.split()[1:4]]
    return positions


def packmol_output_to_gro(components, working_dir: str, box: float, templates: dict):
    """Reassemble the packed ``.xyz`` into ``solvent_salt.gro``.

    Packmol emits structures in input order, each replicated ``number`` times
    with atom order preserved inside a molecule, so residue numbers and names
    are recovered exactly from the templates.
    """
    positions = _read_xyz_positions(os.path.join(working_dir, PACKMOL_OUTPUT)) / NM_TO_ANGSTROM

    expected = sum(templates[c.name].natoms * c.molar_num for c in components.values())
    assert len(positions) == expected, \
        f'packmol wrote {len(positions)} atoms, expected {expected}'

    lines = ['A Gromacs structure file written by ASE\n', f'{expected:5d}\n']
    atom_idx = 0
    res_idx = 0
    for c in components.values():
        template = templates[c.name]
        for _ in range(c.molar_num):
            res_idx += 1
            for name in template.atom_names:
                x, y, z = positions[atom_idx]
                atom_idx += 1
                # gromacs .gro: %5d%-5s%5s%5d%8.3f%8.3f%8.3f
                lines.append(f'{res_idx % 100000:>5d}{template.resname:<5s}{name:>5s}'
                             f'{atom_idx % 100000:>5d}{x:>8.3f}{y:>8.3f}{z:>8.3f}\n')
    lines.append(f'{box:10.5f}{box:10.5f}{box:10.5f}\n')

    with open(os.path.join(working_dir, TARGET_GRO), 'w') as f:
        f.writelines(lines)


def build_box(components, working_dir: str, box: float, timeout: int = 300) -> bool:
    """Pack ``components`` into a cubic box of edge ``box`` nm.

    Writes ``working_dir/solvent_salt.gro`` on success.  Returns False if
    packmol could not fit everything, in which case the caller should retry
    with a larger box.
    """
    templates = generate_packmol_input(components, working_dir, box)
    if not run_packmol(working_dir, timeout=timeout):
        return False
    packmol_output_to_gro(components, working_dir, box, templates)
    return True
