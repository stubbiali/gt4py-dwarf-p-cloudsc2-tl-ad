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
from itertools import repeat
from typing import TYPE_CHECKING

from ifs_physics_common.framework.components import ImplicitTendencyComponent
from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import managed_temporary_storage

if TYPE_CHECKING:
    from datetime import timedelta
    from typing import Dict, Optional, Union

    from gt4py.cartesian import StencilObject

    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid
    from ifs_physics_common.utils.typingx import NDArrayLikeDict, ParameterDict, PropertyDict


class Cloudsc2NL(ImplicitTendencyComponent):
    cloudsc2: StencilObject

    def __init__(
        self,
        computational_grid: ComputationalGrid,
        lphylin: bool,
        ldrain1d: bool,
        yoethf_parameters: Optional[ParameterDict] = None,
        yomcst_parameters: Optional[ParameterDict] = None,
        yrecld_parameters: Optional[ParameterDict] = None,
        yrecldp_parameters: Optional[ParameterDict] = None,
        yrephli_parameters: Optional[ParameterDict] = None,
        yrphnc_parameters: Optional[ParameterDict] = None,
        *,
        enable_checks: bool = True,
        gt4py_config: GT4PyConfig,
    ) -> None:
        super().__init__(computational_grid, enable_checks=enable_checks, gt4py_config=gt4py_config)

        externals: Dict[str, Union[bool, float, int]] = {}
        externals.update(yoethf_parameters or {})
        externals.update(yomcst_parameters or {})
        externals.update(yrecld_parameters or {})
        externals.update(yrecldp_parameters or {})
        externals.update(yrephli_parameters or {})
        externals.update(yrphnc_parameters or {})
        externals.update(
            {
                "ICALL": 0,
                "LPHYLIN": lphylin,
                "LDRAIN1D": ldrain1d,
                "ZEPS1": 1e-12,
                "ZEPS2": 1e-10,
                "ZQMAX": 0.5,
                "ZSCAL": 0.9,
            }
        )
        self.cloudsc2 = self.compile_stencil("cloudsc2_nl", externals)

    @cached_property
    def _input_properties(self) -> PropertyDict:
        return {
            "f_eta": {"grid": (K,), "units": ""},
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
    def _tendency_properties(self) -> PropertyDict:
        return {
            "f_t": {"grid": (I, J, K), "units": "K s^-1"},
            "f_q": {"grid": (I, J, K), "units": "g g^-1 s^-1"},
            "f_ql": {"grid": (I, J, K), "units": "g g^-1 s^-1"},
            "f_qi": {"grid": (I, J, K), "units": "g g^-1 s^-1"},
        }

    @cached_property
    def _diagnostic_properties(self) -> PropertyDict:
        return {
            "f_clc": {"grid": (I, J, K), "units": ""},
            "f_fhpsl": {"grid": (I, J, K - 1 / 2), "units": "J m^-2 s^-1"},
            "f_fhpsn": {"grid": (I, J, K - 1 / 2), "units": "J m^-2 s^-1"},
            "f_fplsl": {"grid": (I, J, K - 1 / 2), "units": "Kg m^-2 s^-1"},
            "f_fplsn": {"grid": (I, J, K - 1 / 2), "units": "Kg m^-2 s^-1"},
            "f_covptot": {"grid": (I, J, K), "units": ""},
        }

    def array_call(
        self,
        state: NDArrayLikeDict,
        timestep: timedelta,
        out_tendencies: NDArrayLikeDict,
        out_diagnostics: NDArrayLikeDict,
        overwrite_tendencies: dict[str, bool],
    ) -> None:
        with managed_temporary_storage(
            self.computational_grid, *repeat(((I, J), "float"), 5), gt4py_config=self.gt4py_config
        ) as (aph_s, rfl, sfl, covptot, trpaus):
            aph_s[...] = state["f_aph"][..., -1]
            self.cloudsc2(
                in_ap=state["f_ap"],
                in_aph=state["f_aph"],
                in_eta=state["f_eta"],
                in_lu=state["f_lu"],
                in_lude=state["f_lude"],
                in_mfd=state["f_mfd"],
                in_mfu=state["f_mfu"],
                in_q=state["f_q"],
                in_qi=state["f_qi"],
                in_ql=state["f_ql"],
                in_qsat=state["f_qsat"],
                in_supsat=state["f_supsat"],
                in_t=state["f_t"],
                in_tnd_cml_q=state["f_tnd_cml_q"],
                in_tnd_cml_qi=state["f_tnd_cml_qi"],
                in_tnd_cml_ql=state["f_tnd_cml_ql"],
                in_tnd_cml_t=state["f_tnd_cml_t"],
                out_clc=out_diagnostics["f_clc"],
                out_covptot=out_diagnostics["f_covptot"],
                out_fhpsl=out_diagnostics["f_fhpsl"],
                out_fhpsn=out_diagnostics["f_fhpsn"],
                out_fplsl=out_diagnostics["f_fplsl"],
                out_fplsn=out_diagnostics["f_fplsn"],
                out_tnd_q=out_tendencies["f_q"],
                out_tnd_qi=out_tendencies["f_qi"],
                out_tnd_ql=out_tendencies["f_ql"],
                out_tnd_t=out_tendencies["f_t"],
                tmp_aph_s=aph_s,
                tmp_covptot=covptot,
                tmp_rfl=rfl,
                tmp_sfl=sfl,
                tmp_trpaus=trpaus,
                dt=self.gt4py_config.dtypes.float(timestep.total_seconds()),
                origin=(0, 0, 0),
                domain=self.computational_grid.grids[I, J, K - 1 / 2].shape,
                validate_args=self.gt4py_config.validate_args,
                exec_info=self.gt4py_config.exec_info,
            )
