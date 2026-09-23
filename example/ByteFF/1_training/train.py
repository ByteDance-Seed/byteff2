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

from byteff2.bytemol.utils import setup_default_logging
from byteff2.train import FFJointTrainer, FFTrainer
from byteff2.train.utils import load_training_config


logger = setup_default_logging()

parser = argparse.ArgumentParser(description="train local")
parser.add_argument("--conf", type=str, default="train.yaml")
parser.add_argument("--timestamp", type=bool, default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--restart", type=bool, default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--asset-root", help="ByteFF2 assets root; overrides BYTEFF2_ASSET_ROOT")
args = parser.parse_args()


def main():

    assert os.path.exists(args.conf) and args.conf.endswith(".yaml"), f"yaml config {args.conf} not found."

    # Use FFJointTrainer if meta.is_joint is set, otherwise FFTrainer
    raw = load_training_config(args.conf, args.asset_root)
    is_joint = raw.get("meta", {}).get("is_joint", False)

    trainer_cls = FFJointTrainer if is_joint else FFTrainer
    logger.info(f"Using {'FFJointTrainer' if is_joint else 'FFTrainer'}")

    trainer = trainer_cls(raw, timestamp=args.timestamp, ddp=False, restart=args.restart, use_amp=True)
    trainer.train_loop()
    logger.info("Training finished!")


if __name__ == "__main__":
    main()
