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

import json

import byteff2.toolkit.protocol as protocol

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")

    args = parser.parse_args()
    config_file = args.config
    with open(config_file, "r") as config_f:
        config = json.load(config_f)
    protocol_name = config["protocol"]
    md_protocol = getattr(protocol, f'{protocol_name}Protocol')(config)
    md_protocol.run_protocol()
    md_protocol.post_process()
