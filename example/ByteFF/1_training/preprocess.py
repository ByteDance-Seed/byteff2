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

import argparse
from pathlib import Path

from byteff2.bytemol.utils import setup_default_logging
from byteff2.data import DatasetConfig, IMDataset
from byteff2.utils import get_asset_path


logger = setup_default_logging()


def main():
    parser = argparse.ArgumentParser(description="process data and save to pkl")
    parser.add_argument("--conf", type=str, help="config yaml file")
    parser.add_argument("--asset-root", help="ByteFF2 assets root; overrides BYTEFF2_ASSET_ROOT")
    args = parser.parse_args()

    config = DatasetConfig(args.conf)
    config_dir = Path(args.conf).resolve().parent
    meta_asset = config._config.pop("meta_asset", None)
    if meta_asset:
        config.set("meta_fp", get_asset_path(meta_asset, args.asset_root))
    elif config.get("meta_fp"):
        config.set("meta_fp", str((config_dir / config.get("meta_fp")).resolve()))
    config.set("save_dir", str((config_dir / config.get("save_dir")).resolve()))
    logger.info(str(config))

    for i in range(config.get("shards")):
        IMDataset.process(config._config, shard_id=i)


if __name__ == "__main__":
    main()
