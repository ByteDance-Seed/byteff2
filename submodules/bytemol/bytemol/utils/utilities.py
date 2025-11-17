# Copyright (c) 2021 The Open Force Field Initiative
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025.08.25
#
# Original file was released under MIT, with the full license text
# available at https://github.com/openforcefield/openff-utilities/blob/main/LICENSE.
#
# This modified file is released under the same license.

# The file is modified from https://github.com/openforcefield/openff-utilities/blob/main/openff/utilities/utilities.py

import dataclasses
import errno
import importlib
import logging
import math
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import PosixPath
from tempfile import TemporaryDirectory
from typing import Any, Callable, Generator, List, Optional, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def has_package(package_name: str) -> bool:
    """
    Helper function to generically check if a Python package is installed.
    Intended to be used to check for optional dependencies.

    Parameters
    ----------
    package_name : str
        The name of the Python package to check the availability of

    Returns
    -------
    package_available : bool
        Boolean indicator if the package is available or not

    Examples
    --------
    >>> has_numpy = has_package('numpy')
    >>> has_numpy
    True
    >>> has_foo = has_package('other_non_installed_package')
    >>> has_foo
    False
    """
    try:
        importlib.import_module(package_name)
    except ModuleNotFoundError:
        return False
    return True


@contextmanager
def temporary_cd(directory_path: Optional[str] = None) -> Generator[None, None, None]:
    """Temporarily move the current working directory to the path
    specified. If no path is given, a temporary directory will be
    created, moved into, and then destroyed when the context manager
    is closed.

    Parameters
    ----------
    directory_path: str, optional

    Returns
    -------

    """
    if isinstance(directory_path, PosixPath):
        directory_path = directory_path.as_posix()

    if directory_path is not None and len(directory_path) == 0:
        yield
        return

    old_directory = os.getcwd()

    try:

        if directory_path is None:

            with TemporaryDirectory() as new_directory:
                os.chdir(new_directory)
                yield

        else:

            os.makedirs(directory_path, exist_ok=True)
            os.chdir(directory_path)
            yield

    finally:
        os.chdir(old_directory)


def is_file_and_not_empty(file_path):
    """Checks that a file both exists at the specified ``path`` and is not empty.

    Parameters
    ----------
    file_path: str
        The file path to check.

    Returns
    -------
    bool
        That a file both exists at the specified ``path`` and is not empty.
    """
    return os.path.isfile(file_path) and (os.path.getsize(file_path) != 0)


def get_data_file_path(relative_path: str, package_name: str) -> str:
    """Get the full path to one of the files in the data directory.

    If no file is found at `relative_path`, a second attempt will be made
    with `data/` preprended. If no files exist at either path, a FileNotFoundError
    is raised.

    Parameters
    ----------
    relative_path : str
        The relative path of the file to load.
    package_name : str
        The name of the package in which a file is to be loaded, i.e.

    Returns
    -------
        The absolute path to the file.

    Raises
    ------
    FileNotFoundError
    """
    from importlib.resources import files

    file_path = files(package_name) / relative_path

    if not file_path.is_file():
        try_path = files(package_name) / f"data/{relative_path}"
        if try_path.is_file():
            file_path = try_path
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_path)

    return file_path.as_posix()  # type: ignore


def split_array_evenly(array: List, num_parts: int) -> List[List]:
    """
    Split an array into num_parts subarrays of roughly equal size.

    Args:
        array: An array to be split.
        num_parts: The number of subarrays to split the array into.

    Returns:
        A list of lists, where each sublist contains a roughly equal portion of the original array.
        The number of sublists equals num_parts, unless the length of the array is not evenly divisible by num_parts.
    """
    num_elements_per_part = math.ceil(len(array) / num_parts)
    vacancy = num_parts * num_elements_per_part - len(array)
    subarrays = []
    start_index = 0

    for i in range(num_parts):
        end_index = start_index + num_elements_per_part - 1 if i >= num_parts - vacancy else start_index + num_elements_per_part
        subarrays.append(array[start_index:end_index])
        start_index = end_index

    return subarrays


def get_current_time_str():
    return datetime.now().strftime("%y_%m_%d_%H_%M_%S")


def convert_keys_to_string(obj):
    if dataclasses.is_dataclass(obj):
        return convert_keys_to_string(dataclasses.asdict(obj))
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [convert_keys_to_string(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): convert_keys_to_string(value) for key, value in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
