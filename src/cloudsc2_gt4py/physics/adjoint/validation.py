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
import sys
from typing import TYPE_CHECKING

from cloudsc2_gt4py.physics.adjoint.microphysics import Cloudsc2AD
from cloudsc2_gt4py.physics.common.increment import StateIncrement
from cloudsc2_gt4py.physics.common.saturation import Saturation
from cloudsc2_gt4py.physics.tangent_linear.microphysics import Cloudsc2TL
from ifs_physics_common.utils.numpyx import to_numpy

if TYPE_CHECKING:
    from datetime import timedelta
    from numpy.typing import NDArray
    from typing import Optional

    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid
    from ifs_physics_common.utils.typingx import DataArrayDict, ParameterDict


class SymmetryTest:
    cloudsc2_ad: Cloudsc2AD
    cloudsc2_tl: Cloudsc2TL
    diags_ad: DataArrayDict
    diags_sat: DataArrayDict
    diags_tl: DataArrayDict
    f: float
    saturation: Saturation
    state_i: DataArrayDict
    state_increment: StateIncrement
    tends_ad: DataArrayDict
    tends_tl: DataArrayDict

    def __init__(
        self,
        computational_grid: ComputationalGrid,
        factor: float,
        kflag: int,
        lphylin: bool,
        ldrain1d: bool,
        yoethf_parameters: Optional[ParameterDict] = None,
        yomcst_parameters: Optional[ParameterDict] = None,
        yrecld_parameters: Optional[ParameterDict] = None,
        yrecldp_parameters: Optional[ParameterDict] = None,
        yrephli_parameters: Optional[ParameterDict] = None,
        yrncl_parameters: Optional[ParameterDict] = None,
        yrphnc_parameters: Optional[ParameterDict] = None,
        *,
        enable_checks: bool = True,
        gt4py_config: GT4PyConfig,
    ) -> None:
        self.f = factor

        # saturation
        self.saturation = Saturation(
            computational_grid,
            kflag,
            lphylin,
            yoethf_parameters,
            yomcst_parameters,
            enable_checks=enable_checks,
            gt4py_config=gt4py_config,
        )

        # microphysics
        self.cloudsc2_tl = Cloudsc2TL(
            computational_grid,
            lphylin,
            ldrain1d,
            yoethf_parameters,
            yomcst_parameters,
            yrecld_parameters,
            yrecldp_parameters,
            yrephli_parameters,
            yrncl_parameters,
            yrphnc_parameters,
            enable_checks=enable_checks,
            gt4py_config=gt4py_config,
        )
        self.cloudsc2_ad = Cloudsc2AD(
            computational_grid,
            lphylin,
            ldrain1d,
            yoethf_parameters,
            yomcst_parameters,
            yrecld_parameters,
            yrecldp_parameters,
            yrephli_parameters,
            yrncl_parameters,
            yrphnc_parameters,
            enable_checks=enable_checks,
            gt4py_config=gt4py_config,
        )

        # perturbation
        self.state_increment = StateIncrement(
            computational_grid,
            factor,
            ignore_supsat=True,
            enable_checks=enable_checks,
            gt4py_config=gt4py_config,
        )

        # auxiliary dictionaries
        self.diags_sat: DataArrayDict = {}
        self.state_i: DataArrayDict = {}
        self.tends_tl: DataArrayDict = {}
        self.diags_tl: DataArrayDict = {}
        self.tends_ad: DataArrayDict = {}
        self.diags_ad: DataArrayDict = {}

    def __call__(
        self, state: DataArrayDict, timestep: timedelta, enable_validation: bool = True
    ) -> None:
        self.diags_sat = self.saturation(state, out=self.diags_sat)
        state.update(self.diags_sat)

        self.state_i = self.state_increment(state, out=self.state_i)
        state.update(self.state_i)
        self.tends_tl, self.diags_tl = self.cloudsc2_tl(
            state, timestep, out_tendencies=self.tends_tl, out_diagnostics=self.diags_tl
        )

        if enable_validation:
            norm1 = self.get_norm1(self.tends_tl, self.diags_tl)
        else:
            norm1 = None

        self.add_tendencies_to_state(state, self.tends_tl)
        state.update(self.diags_tl)
        self.tends_ad, self.diags_ad = self.cloudsc2_ad(
            state, timestep, out_tendencies=self.tends_ad, out_diagnostics=self.diags_ad
        )

        if enable_validation:
            norm2 = self.get_norm2(self.state_i, self.tends_ad, self.diags_ad)
            norm3 = np.where(
                norm2 == 0,
                abs(norm1 - norm2) / sys.float_info.epsilon,
                abs(norm1 - norm2) / (sys.float_info.epsilon * norm2),
            )
            if norm3.max() < 1e4:
                print("The symmetry test passed. HOORAY!")
            else:
                print("The symmetry test failed.")
            print(f"The maximum error is {norm3.max():.10e} times the zero of the machine.")

    def get_norm1(self, tends_tl: DataArrayDict, diags_tl: DataArrayDict) -> NDArray:
        out: NDArray = None  # type: ignore[assignment]

        tend_names = ("f_t_i", "f_q_i", "f_ql_i", "f_qi_i")
        for name in tend_names:
            field = self.get_field(name, tends_tl)
            out = np.zeros(field.shape[0]) if out is None else out
            out += np.sum(field**2, axis=1)

        diag_names = ("f_clc_i", "f_fhpsl_i", "f_fhpsn_i", "f_fplsl_i", "f_fplsn_i", "f_covptot_i")
        for name in diag_names:
            field = self.get_field(name, diags_tl)
            out += np.sum(field**2, axis=1)

        return out

    def get_norm2(
        self, state_i: DataArrayDict, tends_ad: DataArrayDict, diags_ad: DataArrayDict
    ) -> NDArray:
        out: NDArray = None  # type: ignore[assignment]

        tend_names = ("f_cml_t_i", "f_cml_q_i", "f_cml_ql_i", "f_cml_qi_i")
        for name in tend_names:
            field_a = self.get_field("f_tnd_" + name[2:], state_i)
            field_b = self.get_field(name, tends_ad)
            out = np.zeros(field_a.shape[0]) if out is None else out
            out += np.sum(field_a * field_b, axis=1)

        diag_names = (
            "f_ap_i",
            "f_aph_i",
            "f_t_i",
            "f_q_i",
            "f_qsat_i",
            "f_ql_i",
            "f_qi_i",
            "f_lu_i",
            "f_lude_i",
            "f_mfd_i",
            "f_mfu_i",
            "f_supsat_i",
        )
        for name in diag_names:
            field_a = self.get_field(name, state_i)
            field_b = self.get_field(name, diags_ad)
            out = np.zeros(field_a.shape[0]) if out is None else out
            out += np.sum(field_a * field_b, axis=1)

        return out

    @staticmethod
    def get_field(name: str, dct: DataArrayDict) -> np.ndarray:
        field = to_numpy(dct[name].data)[:, 0, :]
        return field

    @staticmethod
    def add_tendencies_to_state(state: DataArrayDict, tends_tl: DataArrayDict) -> None:
        state["f_tnd_t"] = tends_tl["f_t"]
        state["f_tnd_t_i"] = tends_tl["f_t_i"]
        state["f_tnd_q"] = tends_tl["f_q"]
        state["f_tnd_q_i"] = tends_tl["f_q_i"]
        state["f_tnd_ql"] = tends_tl["f_ql"]
        state["f_tnd_ql_i"] = tends_tl["f_ql_i"]
        state["f_tnd_qi"] = tends_tl["f_qi"]
        state["f_tnd_qi_i"] = tends_tl["f_qi_i"]
