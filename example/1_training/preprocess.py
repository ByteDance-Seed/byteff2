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

from byteff2.data import DatasetConfig, IMDataset
from bytemol.utils import setup_default_logging

logger = setup_default_logging()

parser = argparse.ArgumentParser(description="process data and save to pkl")
parser.add_argument("--conf", type=str, help='config yaml file')

args = parser.parse_args()

if __name__ == "__main__":

    config = DatasetConfig(args.conf)
    logger.info(str(config))

    for i in range(config.get('shards')):
        dataset = IMDataset.process(args.conf, shard_id=i)
