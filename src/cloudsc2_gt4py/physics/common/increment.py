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

    from ifs_physics_common.config import GT4PyConfig
    from ifs_physics_common.grid import ComputationalGrid
    from ifs_physics_common.typingx import NDArrayLikeDict, PropertyDict


class StateIncrement(DiagnosticComponent):
    f: float
    increment: StencilObject

    def __init__(
        self,
        computational_grid: ComputationalGrid,
        factor: float,
        ignore_supsat: bool = False,
        *,
        enable_checks: bool = True,
        gt4py_config: GT4PyConfig,
    ):
        super().__init__(computational_grid, enable_checks=enable_checks, gt4py_config=gt4py_config)
        self.f = gt4py_config.dtypes.float(factor)
        self.increment = self.compile_stencil(
            "state_increment", externals={"IGNORE_SUPSAT": ignore_supsat}
        )

    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "f_aph": {"grid": (I, J, K - 1 / 2), "units": "Pa"},
            "f_ap": {"grid": (I, J, K), "units": "Pa"},
            "f_q": {"grid": (I, J, K), "units": "g g^-1"},
            "f_qsat": {"grid": (I, J, K), "units": "g g^-1"},
            "f_t": {"grid": (I, J, K), "units": "K"},
            "f_ql": {"grid": (I, J, K), "units": "g g^-1"},
            "f_qi": {"grid": (I, J, K), "units": "g g^-1"},
            "f_lude": {"grid": (I, J, K), "units": "kg m^-3 s^-1"},
            "f_lu": {"grid": (I, J, K), "units": "g g^-1"},
            "f_mfu": {"grid": (I, J, K), "units": "kg m^-2 s^-1"},
            "f_mfd": {"grid": (I, J, K), "units": "kg m^-2 s^-1"},
            "f_tnd_cml_t": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_q": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_ql": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_qi": {"grid": (I, J, K), "units": "K s^-1"},
            "f_supsat": {"grid": (I, J, K), "units": "g g^-1"},
        }

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        return {
            "f_aph_i": {"grid": (I, J, K - 1 / 2), "units": "Pa"},
            "f_ap_i": {"grid": (I, J, K), "units": "Pa"},
            "f_q_i": {"grid": (I, J, K), "units": "g g^-1"},
            "f_qsat_i": {"grid": (I, J, K), "units": "g g^-1"},
            "f_t_i": {"grid": (I, J, K), "units": "K"},
            "f_ql_i": {"grid": (I, J, K), "units": "g g^-1"},
            "f_qi_i": {"grid": (I, J, K), "units": "g g^-1"},
            "f_lude_i": {"grid": (I, J, K), "units": "kg m^-3 s^-1"},
            "f_lu_i": {"grid": (I, J, K), "units": "g g^-1"},
            "f_mfu_i": {"grid": (I, J, K), "units": "kg m^-2 s^-1"},
            "f_mfd_i": {"grid": (I, J, K), "units": "kg m^-2 s^-1"},
            "f_tnd_cml_t_i": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_q_i": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_ql_i": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_qi_i": {"grid": (I, J, K), "units": "K s^-1"},
            "f_supsat_i": {"grid": (I, J, K), "units": "g g^-1"},
        }

    def array_call(self, state: NDArrayLikeDict, out: NDArrayLikeDict) -> None:
        self.increment(
            in_aph=state["f_aph"],
            in_ap=state["f_ap"],
            in_q=state["f_q"],
            in_qsat=state["f_qsat"],
            in_t=state["f_t"],
            in_ql=state["f_ql"],
            in_qi=state["f_qi"],
            in_lude=state["f_lude"],
            in_lu=state["f_lu"],
            in_mfu=state["f_mfu"],
            in_mfd=state["f_mfd"],
            in_tnd_cml_t=state["f_tnd_cml_t"],
            in_tnd_cml_q=state["f_tnd_cml_q"],
            in_tnd_cml_ql=state["f_tnd_cml_ql"],
            in_tnd_cml_qi=state["f_tnd_cml_qi"],
            in_supsat=state["f_supsat"],
            out_aph_i=out["f_aph_i"],
            out_ap_i=out["f_ap_i"],
            out_q_i=out["f_q_i"],
            out_qsat_i=out["f_qsat_i"],
            out_t_i=out["f_t_i"],
            out_ql_i=out["f_ql_i"],
            out_qi_i=out["f_qi_i"],
            out_lude_i=out["f_lude_i"],
            out_lu_i=out["f_lu_i"],
            out_mfu_i=out["f_mfu_i"],
            out_mfd_i=out["f_mfd_i"],
            out_tnd_cml_t_i=out["f_tnd_cml_t_i"],
            out_tnd_cml_q_i=out["f_tnd_cml_q_i"],
            out_tnd_cml_ql_i=out["f_tnd_cml_ql_i"],
            out_tnd_cml_qi_i=out["f_tnd_cml_qi_i"],
            out_supsat_i=out["f_supsat_i"],
            f=self.f,
            origin=(0, 0, 0),
            domain=self.computational_grid.grids[I, J, K - 1 / 2].shape,
            validate_args=self.gt4py_config.validate_args,
            exec_info=self.gt4py_config.exec_info,
        )


class PerturbedState(DiagnosticComponent):
    def __init__(
        self,
        computational_grid: ComputationalGrid,
        factor: float,
        *,
        enable_checks: bool = True,
        gt4py_config: GT4PyConfig,
    ):
        super().__init__(computational_grid, enable_checks=enable_checks, gt4py_config=gt4py_config)
        self.f = gt4py_config.dtypes.float(factor)
        self.perturbed_state = self.compile_stencil("perturbed_state")

    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "f_aph": {"grid": (I, J, K - 1 / 2), "units": "Pa"},
            "f_aph_i": {"grid": (I, J, K - 1 / 2), "units": "Pa"},
            "f_ap": {"grid": (I, J, K), "units": "Pa"},
            "f_ap_i": {"grid": (I, J, K), "units": "Pa"},
            "f_q": {"grid": (I, J, K), "units": "g g^-1"},
            "f_q_i": {"grid": (I, J, K), "units": "g g^-1"},
            "f_qsat": {"grid": (I, J, K), "units": "g g^-1"},
            "f_qsat_i": {"grid": (I, J, K), "units": "g g^-1"},
            "f_t": {"grid": (I, J, K), "units": "K"},
            "f_t_i": {"grid": (I, J, K), "units": "K"},
            "f_ql": {"grid": (I, J, K), "units": "g g^-1"},
            "f_ql_i": {"grid": (I, J, K), "units": "g g^-1"},
            "f_qi": {"grid": (I, J, K), "units": "g g^-1"},
            "f_qi_i": {"grid": (I, J, K), "units": "g g^-1"},
            "f_lude": {"grid": (I, J, K), "units": "kg m^-3 s^-1"},
            "f_lude_i": {"grid": (I, J, K), "units": "kg m^-3 s^-1"},
            "f_lu": {"grid": (I, J, K), "units": "g g^-1"},
            "f_lu_i": {"grid": (I, J, K), "units": "g g^-1"},
            "f_mfu": {"grid": (I, J, K), "units": "kg m^-2 s^-1"},
            "f_mfu_i": {"grid": (I, J, K), "units": "kg m^-2 s^-1"},
            "f_mfd": {"grid": (I, J, K), "units": "kg m^-2 s^-1"},
            "f_mfd_i": {"grid": (I, J, K), "units": "kg m^-2 s^-1"},
            "f_tnd_cml_t": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_t_i": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_q": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_q_i": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_ql": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_ql_i": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_qi": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_qi_i": {"grid": (I, J, K), "units": "K s^-1"},
            "f_supsat": {"grid": (I, J, K), "units": "g g^-1"},
            "f_supsat_i": {"grid": (I, J, K), "units": "g g^-1"},
        }

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        return {
            "f_aph": {"grid": (I, J, K - 1 / 2), "units": "Pa"},
            "f_ap": {"grid": (I, J, K), "units": "Pa"},
            "f_q": {"grid": (I, J, K), "units": "g g^-1"},
            "f_qsat": {"grid": (I, J, K), "units": "g g^-1"},
            "f_t": {"grid": (I, J, K), "units": "K"},
            "f_ql": {"grid": (I, J, K), "units": "g g^-1"},
            "f_qi": {"grid": (I, J, K), "units": "g g^-1"},
            "f_lude": {"grid": (I, J, K), "units": "kg m^-3 s^-1"},
            "f_lu": {"grid": (I, J, K), "units": "g g^-1"},
            "f_mfu": {"grid": (I, J, K), "units": "kg m^-2 s^-1"},
            "f_mfd": {"grid": (I, J, K), "units": "kg m^-2 s^-1"},
            "f_tnd_cml_t": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_q": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_ql": {"grid": (I, J, K), "units": "K s^-1"},
            "f_tnd_cml_qi": {"grid": (I, J, K), "units": "K s^-1"},
            "f_supsat": {"grid": (I, J, K), "units": "g g^-1"},
        }

    def array_call(self, state: NDArrayLikeDict, out: NDArrayLikeDict) -> None:
        self.perturbed_state(
            in_aph=state["f_aph"],
            in_aph_i=state["f_aph_i"],
            in_ap=state["f_ap"],
            in_ap_i=state["f_ap_i"],
            in_q=state["f_q"],
            in_q_i=state["f_q_i"],
            in_qsat=state["f_qsat"],
            in_qsat_i=state["f_qsat_i"],
            in_t=state["f_t"],
            in_t_i=state["f_t_i"],
            in_ql=state["f_ql"],
            in_ql_i=state["f_ql_i"],
            in_qi=state["f_qi"],
            in_qi_i=state["f_qi_i"],
            in_lude=state["f_lude"],
            in_lude_i=state["f_lude_i"],
            in_lu=state["f_lu"],
            in_lu_i=state["f_lu_i"],
            in_mfu=state["f_mfu"],
            in_mfu_i=state["f_mfu_i"],
            in_mfd=state["f_mfd"],
            in_mfd_i=state["f_mfd_i"],
            in_tnd_cml_t=state["f_tnd_cml_t"],
            in_tnd_cml_t_i=state["f_tnd_cml_t_i"],
            in_tnd_cml_q=state["f_tnd_cml_q"],
            in_tnd_cml_q_i=state["f_tnd_cml_q_i"],
            in_tnd_cml_ql=state["f_tnd_cml_ql"],
            in_tnd_cml_ql_i=state["f_tnd_cml_ql_i"],
            in_tnd_cml_qi=state["f_tnd_cml_qi"],
            in_tnd_cml_qi_i=state["f_tnd_cml_qi_i"],
            in_supsat=state["f_supsat"],
            in_supsat_i=state["f_supsat_i"],
            out_aph=out["f_aph"],
            out_ap=out["f_ap"],
            out_q=out["f_q"],
            out_qsat=out["f_qsat"],
            out_t=out["f_t"],
            out_ql=out["f_ql"],
            out_qi=out["f_qi"],
            out_lude=out["f_lude"],
            out_lu=out["f_lu"],
            out_mfu=out["f_mfu"],
            out_mfd=out["f_mfd"],
            out_tnd_cml_t=out["f_tnd_cml_t"],
            out_tnd_cml_q=out["f_tnd_cml_q"],
            out_tnd_cml_ql=out["f_tnd_cml_ql"],
            out_tnd_cml_qi=out["f_tnd_cml_qi"],
            out_supsat=out["f_supsat"],
            f=self.f,
            origin=(0, 0, 0),
            domain=self.computational_grid.grids[I, J, K - 1 / 2].shape,
            validate_args=self.gt4py_config.validate_args,
            exec_info=self.gt4py_config.exec_info,
        )
