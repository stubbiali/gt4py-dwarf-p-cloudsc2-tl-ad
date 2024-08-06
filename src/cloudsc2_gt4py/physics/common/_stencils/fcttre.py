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

from ifs_physics_common.stencil import function_collection


@function_collection("f_foealfa")
@gtscript.function
def f_foealfa(t):
    from __externals__ import RTICE, RTWAT, RTWAT_RTICE_R

    return min(1.0, ((max(RTICE, min(RTWAT, t)) - RTICE) * RTWAT_RTICE_R) ** 2.0)


@function_collection("f_foealfcu")
@gtscript.function
def f_foealfcu(t):
    from __externals__ import RTICECU, RTWAT, RTWAT_RTICECU_R

    return min(1.0, ((max(RTICECU, min(RTWAT, t)) - RTICECU) * RTWAT_RTICECU_R) ** 2.0)


@function_collection("f_foeewm")
@gtscript.function
def f_foeewm(t):
    from __externals__ import R2ES, R3IES, R3LES, R4IES, R4LES, RTT

    return R2ES * (
        f_foealfa(t) * exp(R3LES * (t - RTT) / (t - R4LES))
        + (1.0 - f_foealfa(t)) * (exp(R3IES * (t - RTT) / (t - R4IES)))
    )


@function_collection("f_foeewmcu")
@gtscript.function
def f_foeewmcu(t):
    from __externals__ import R2ES, R3IES, R3LES, R4IES, R4LES, RTT

    return R2ES * (
        f_foealfcu(t) * exp(R3LES * (t - RTT) / (t - R4LES))
        + (1.0 - f_foealfcu(t)) * (exp(R3IES * (t - RTT) / (t - R4IES)))
    )
