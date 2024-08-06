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
from functools import cached_property
from typing import TYPE_CHECKING

from ifs_physics_common.components import DiagnosticComponent
from ifs_physics_common.grid import I, J, K

if TYPE_CHECKING:
    from ifs_physics_common.typingx import NDArrayLikeDict, PropertyDict


class EtaLevels(DiagnosticComponent):
    """Diagnose reference eta-levels."""

    @cached_property
    def input_grid_properties(self) -> PropertyDict:
        return {
            "f_ap": {"grid_dims": (I, J, K), "units": "Pa"},
            "f_aph": {"grid_dims": (I, J, K - 1 / 2), "units": "Pa"},
        }

    @cached_property
    def diagnostic_grid_properties(self) -> PropertyDict:
        return {"f_eta": {"grid_dims": (K,), "units": ""}}

    def array_call(self, state: NDArrayLikeDict, out: NDArrayLikeDict) -> None:
        nz = self.computational_grid.grids[I, J, K].shape[2]
        for k in range(nz):
            out["f_eta"][k] = state["f_ap"][0, 0, k] / state["f_aph"][0, 0, nz]
