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
import os

from byteff2.train import FFTrainer
from bytemol.utils import setup_default_logging

logger = setup_default_logging()

parser = argparse.ArgumentParser(description='train local')
parser.add_argument('--conf', type=str, default='config.yaml')
parser.add_argument('--timestamp', type=bool, default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--restart', type=bool, default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()


def main():

    assert os.path.exists(args.conf) and args.conf.endswith('.yaml'), f'yaml config {args.conf} not found.'
    trainer = FFTrainer(args.conf, timestamp=args.timestamp, restart=args.restart)
    trainer.train_loop()
    logger.info("Training finished!")


if __name__ == "__main__":
    main()
