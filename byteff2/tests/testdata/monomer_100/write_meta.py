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
from glob import glob
from pathlib import Path

meta_name = 'meta.txt'
lines = []
for json_file in glob('*.json'):
    with open(json_file) as file:
        name_mps = json.load(file)
    dataset_name = Path(json_file).stem
    for name in name_mps:
        lines.append(f'{dataset_name},{name}\n')
with open(meta_name, 'w') as file:
    file.writelines(lines)
