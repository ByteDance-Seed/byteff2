# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates

import os
from pathlib import Path


ASSET_ROOT_ENV = "BYTEFF2_ASSET_ROOT"


def get_asset_path(asset_relative_path: str, asset_root: str | os.PathLike | None = None) -> str:
    """Resolve an existing file or directory below the configured asset root."""
    root_value = asset_root if asset_root is not None else os.environ.get(ASSET_ROOT_ENV)
    if not root_value:
        raise FileNotFoundError(f"ByteFF2 asset root is not configured. Pass asset_root or set {ASSET_ROOT_ENV}.")

    relative = Path(asset_relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"asset_relative_path must stay below the asset root: {asset_relative_path}")

    root = Path(root_value).expanduser().resolve()
    path = root.joinpath(relative)
    if not path.exists():
        raise FileNotFoundError(f"ByteFF2 asset not found: {path} (asset root: {root})")
    return str(path)
