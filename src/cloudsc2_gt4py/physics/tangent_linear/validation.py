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

from cloudsc2_gt4py.physics.common.increment import PerturbedState, StateIncrement
from cloudsc2_gt4py.physics.common.saturation import Saturation
from cloudsc2_gt4py.physics.nonlinear.microphysics import Cloudsc2NL
from cloudsc2_gt4py.physics.tangent_linear.microphysics import Cloudsc2TL
from ifs_physics_common.timing import timing

if TYPE_CHECKING:
    from datetime import timedelta
    from numpy.typing import NDArray
    from typing import Optional

    from ifs_physics_common.config import GT4PyConfig
    from ifs_physics_common.grid import ComputationalGrid
    from ifs_physics_common.typingx import DataArrayDict, ParameterDict


class TaylorTest:
    cloudsc2_nl: Cloudsc2NL
    cloudsc2_tl: Cloudsc2TL
    diags_nl: DataArrayDict
    diags_nl_p: DataArrayDict
    diags_sat: DataArrayDict
    diags_tl: DataArrayDict
    f1: float
    f2s: tuple[float, ...]
    perturbed_states: list[PerturbedState]
    saturation: Saturation
    state_i: DataArrayDict
    state_increment: StateIncrement
    state_p: DataArrayDict
    tends_nl: DataArrayDict
    tends_nl_p: DataArrayDict
    tends_tl: DataArrayDict

    def __init__(
        self,
        computational_grid: ComputationalGrid,
        factor1: float,
        factor2s: tuple[float, ...],
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
        self.f1 = factor1
        self.f2s = factor2s

        # no regularization in Taylor test
        yrncl_parameters: ParameterDict = yrncl_parameters or {}  # type: ignore[no-redef]
        yrncl_parameters["LREGCL"] = False  # type: ignore[index]

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
        self.cloudsc2_nl = Cloudsc2NL(
            computational_grid,
            lphylin,
            ldrain1d,
            yoethf_parameters,
            yomcst_parameters,
            yrecld_parameters,
            yrecldp_parameters,
            yrephli_parameters,
            yrphnc_parameters,
            enable_checks=enable_checks,
            gt4py_config=gt4py_config,
        )
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

        # perturbation
        self.state_increment = StateIncrement(
            computational_grid, factor1, enable_checks=enable_checks, gt4py_config=gt4py_config
        )
        self.perturbed_states = [
            PerturbedState(
                computational_grid, factor2, enable_checks=enable_checks, gt4py_config=gt4py_config
            )
            for factor2 in factor2s
        ]

        # auxiliary dicts
        self.diags_nl: DataArrayDict = {}
        self.diags_nl_p: DataArrayDict = {}
        self.diags_sat: DataArrayDict = {}
        self.diags_tl: DataArrayDict = {}
        self.state_i: DataArrayDict = {}
        self.state_p: DataArrayDict = {}
        self.tends_nl: DataArrayDict = {}
        self.tends_nl_p: DataArrayDict = {}
        self.tends_tl: DataArrayDict = {}

    def __call__(self, state: DataArrayDict, timestep: timedelta) -> None:
        self.validate(self.run(state, timestep))

    def run(self, state: DataArrayDict, timestep: timedelta) -> np.ndarray:
        with timing("run"):
            self.diags_sat = self.saturation(state, out=self.diags_sat)
            state.update(self.diags_sat)

            self.tends_nl, self.diags_nl = self.cloudsc2_nl(
                state, timestep, out_tendencies=self.tends_tl, out_diagnostics=self.diags_nl
            )

            self.state_i = self.state_increment(state, out=self.state_i)
            state.update(self.state_i)
            self.tends_tl, self.diags_tl = self.cloudsc2_tl(
                state, timestep, out_tendencies=self.tends_tl, out_diagnostics=self.diags_tl
            )

        norms = np.zeros(len(self.f2s))
        for i, perturbed_state in enumerate(self.perturbed_states):
            with timing("run"):
                self.state_p = perturbed_state(state, out=self.state_p)
                self.state_p["time"] = state["time"]
                self.state_p["f_eta"] = state["f_eta"]
                self.tends_nl_p, self.diags_nl_p = self.cloudsc2_nl(
                    self.state_p,
                    timestep,
                    out_tendencies=self.tends_nl_p,
                    out_diagnostics=self.diags_nl_p,
                )

            with timing("norms"):
                norms[i] = self.get_norm(i)

        return norms

    def validate(self, norms: np.ndarray) -> None:
        print(">>> Taylor test: Start")
        start = -1
        for i in range(norms.size):
            print(
                f"  factor1 = {self.f1:.3e}, factor2 = {self.f2s[i]:.3e}, "
                f"norm = {norms[i]:.10f}"
            )
            norms[i] = np.abs(1 - norms[i])
            if start == -1 and norms[i] < 0.5:
                start = i

        if start == -1 or start > 3:
            log = "The test failed with error 13."
        else:
            test = -10
            negat = 1
            for i in range(start, norms.size - 1):
                tmp_negat = int(norms[i + 1] < norms[i])
                if negat > tmp_negat:
                    test += 10
                negat = tmp_negat
            if test == -10:
                test = 11
            if np.min(norms[start:]) > 1e-5:
                test += 7
            if np.min(norms[start:]) > 1e-6:
                test += 5
            if test > 5:
                log = f"The test failed with error {test}."
            else:
                log = f"The test passed with penalty {test}. HOORAY!"

        print("<<< Taylor test: End")
        print(log)

    def get_norm(self, i: int) -> float:
        assert self.diags_nl is not None, "Did you execute the run() method?"

        total_count = 0
        total_norm = 0.0

        tend_names = ("f_t", "f_q", "f_ql", "f_qi")
        for name in tend_names:
            field_nl, field_nl_p, field_tl = self.get_fields(name, "tends")
            norm = self.get_field_norm(i, field_nl, field_nl_p, field_tl)
            total_count += norm > 0
            total_norm += norm

        diag_names = ("f_clc", "f_fhpsl", "f_fhpsn", "f_fplsl", "f_fplsn", "f_covptot")
        for name in diag_names:
            field_nl, field_nl_p, field_tl = self.get_fields(name, "diags")
            norm = self.get_field_norm(i, field_nl, field_nl_p, field_tl)
            total_count += norm > 0
            total_norm += norm

        return total_norm / total_count if total_count > 0 else 0

    def get_fields(self, name: str, dct_name: str) -> tuple[NDArray, NDArray, NDArray]:
        dct_nl = getattr(self, dct_name + "_nl")
        field_nl = dct_nl[name].data[:, 0, :]

        dct_nl_p = getattr(self, dct_name + "_nl_p")
        field_nl_p = dct_nl_p[name].data[:, 0, :]

        dct_tl = getattr(self, dct_name + "_tl")
        field_tl = dct_tl[name + "_i"].data[:, 0, :]

        return field_nl, field_nl_p, field_tl

    def get_field_norm(
        self, i: int, field_nl: NDArray, field_nl_p: NDArray, field_tl: NDArray
    ) -> float:
        den = np.abs(self.f2s[i] * np.sum(field_tl))
        if den > sys.float_info.epsilon:
            norm = np.abs(np.sum(field_nl_p - field_nl)) / den
        else:
            norm = 0
        return norm  # type: ignore[no-any-return]
