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

from ifs_physics_common.framework.stencil import function_collection


@function_collection("f_cuadjtqs_nl_0")
@gtscript.function
def f_cuadjtqs_nl_0(ap, t, q, z3es, z4es, z5alcp, zaldcp):
    from __externals__ import R2ES, RETV, RTT, ZQMAX

    foeew = R2ES * exp(z3es * (t - RTT) / (t - z4es))
    qsat = min(foeew / ap, ZQMAX)
    cor = 1.0 / (1.0 - RETV * qsat)
    qsat *= cor
    z2s = z5alcp / (t - z4es) ** 2.0
    cond = (q - qsat) / (1.0 + qsat * cor * z2s)
    t += zaldcp * cond
    q -= cond
    return t, q


@function_collection("f_cuadjtqs_nl")
@gtscript.function
def f_cuadjtqs_nl(ap, t, q):
    from __externals__ import (
        ICALL,
        R3IES,
        R3LES,
        R4IES,
        R4LES,
        R5ALSCP,
        R5ALVCP,
        RALSDCP,
        RALVDCP,
        RTT,
    )

    if t > RTT:
        z3es = R3LES
        z4es = R4LES
        z5alcp = R5ALVCP
        zaldcp = RALVDCP
    else:
        z3es = R3IES
        z4es = R4IES
        z5alcp = R5ALSCP
        zaldcp = RALSDCP

    if ICALL == 0:
        t, q = f_cuadjtqs_nl_0(ap, t, q, z3es, z4es, z5alcp, zaldcp)
        t, q = f_cuadjtqs_nl_0(ap, t, q, z3es, z4es, z5alcp, zaldcp)
        return t, q
