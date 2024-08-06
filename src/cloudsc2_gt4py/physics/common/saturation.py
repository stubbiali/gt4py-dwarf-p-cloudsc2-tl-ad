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
    from gt4py.cartesian import StencilObject

    from cloudsc2_gt4py.iox import YoethfParams, YomcstParams
    from ifs_physics_common.config import GT4PyConfig
    from ifs_physics_common.grid import ComputationalGrid
    from ifs_physics_common.typingx import NDArrayLikeDict, PropertyDict


class Saturation(DiagnosticComponent):
    """Perform the moist saturation adjustment."""

    saturation: StencilObject

    def __init__(
        self,
        computational_grid: ComputationalGrid,
        kflag: int,
        lphylin: bool,
        yoethf_params: YoethfParams,
        yomcst_params: YomcstParams,
        *,
        enable_checks: bool = True,
        gt4py_config: GT4PyConfig,
    ) -> None:
        super().__init__(computational_grid, enable_checks=enable_checks, gt4py_config=gt4py_config)

        externals = {"KFLAG": kflag, "LPHYLIN": lphylin, "QMAX": 0.5}
        externals.update(yoethf_params.dict())
        externals.update(yomcst_params.dict())
        self.saturation = self.compile_stencil("saturation", externals)

    @cached_property
    def input_grid_properties(self) -> PropertyDict:
        return {
            "f_ap": {"grid_dims": (I, J, K), "units": "Pa"},
            "f_t": {"grid_dims": (I, J, K), "units": "K"},
        }

    @cached_property
    def diagnostic_grid_properties(self) -> PropertyDict:
        return {"f_qsat": {"grid_dims": (I, J, K), "units": "g g^-1"}}

    def array_call(self, state: NDArrayLikeDict, out: NDArrayLikeDict) -> None:
        self.saturation(
            in_ap=state["f_ap"],
            in_t=state["f_t"],
            out_qsat=out["f_qsat"],
            origin=(0, 0, 0),
            domain=self.computational_grid.grids[I, J, K].shape,
            validate_args=self.gt4py_config.validate_args,
            exec_info=self.gt4py_config.exec_info,
        )
