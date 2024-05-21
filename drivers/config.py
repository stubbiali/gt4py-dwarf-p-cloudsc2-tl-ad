# -*- coding: utf-8 -*-
#
# Copyright 2022-2024 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import numpy as np
from os.path import dirname, join, normpath
from typing import Literal

from ifs_physics_common.config import DataTypes, GT4PyConfig, IOConfig, PythonConfig


DATA_DIR = normpath(join(dirname(__file__), "../data"))


class Config(PythonConfig):
    reference_file: str

    def with_precision(self, precision: Literal["double", "single"]) -> Config:
        args = super().with_precision(precision).dict()
        args["reference_file"] = join(DATA_DIR, f"reference_{precision}.h5")
        return Config(**args)


DEFAULT_CONFIG = Config(
    num_cols=1,
    enable_validation=True,
    input_file=join(DATA_DIR, "input.h5"),
    reference_file="",
    num_runs=1,
    precision="double",
    data_types=DataTypes(bool=bool, float=np.float64, int=np.int64),
    gt4py_config=GT4PyConfig(backend="numpy", rebuild=False, validate_args=True, verbose=True),
    sympl_enable_checks=True,
)
DEFAULT_IO_CONFIG = IOConfig(output_csv_file=None, host_name="")
