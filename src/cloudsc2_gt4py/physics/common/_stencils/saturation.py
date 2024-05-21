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

from gt4py.cartesian import gtscript

from cloudsc2_gt4py.physics.common._stencils.fcttre import f_foealfa, f_foeewm, f_foeewmcu
from ifs_physics_common.stencil import stencil_collection


@stencil_collection("saturation")
def saturation(
    in_ap: gtscript.Field["float"], in_t: gtscript.Field["float"], out_qsat: gtscript.Field["float"]
):
    from __externals__ import KFLAG, LPHYLIN, QMAX, R2ES, R3IES, R3LES, R4IES, R4LES, RETV, RTT

    with computation(PARALLEL), interval(...):
        if LPHYLIN:
            alfa = f_foealfa(in_t)
            foeewl = R2ES * exp(R3LES * (in_t[0, 0, 0] - RTT) / (in_t[0, 0, 0] - R4LES))
            foeewi = R2ES * exp(R3IES * (in_t[0, 0, 0] - RTT) / (in_t[0, 0, 0] - R4IES))
            foeew = alfa * foeewl + (1.0 - alfa) * foeewi
            qs = min(foeew / in_ap[0, 0, 0], QMAX)
        else:
            if KFLAG == 1:
                ew = f_foeewmcu(in_t)
            else:
                ew = f_foeewm(in_t)
            qs = min(ew / in_ap[0, 0, 0], QMAX)
        out_qsat[0, 0, 0] = qs / (1.0 - RETV * qs)
