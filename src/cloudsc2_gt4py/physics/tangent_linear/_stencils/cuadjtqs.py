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


@function_collection("f_cuadjtqs_tl_0")
@gtscript.function
def f_cuadjtqs_tl_0(ap, ap_i, t, t_i, q, q_i, z3es, z4es, z5alcp, zaldcp):
    from __externals__ import R2ES, RETV, RTT, ZQMAX

    qp = 1.0 / ap
    qp_i = -ap_i / ap**2.0
    foeew = R2ES * exp(z3es * (t - RTT) / (t - z4es))
    foeew_i = foeew * z3es * t_i * (RTT - z4es) / (t - z4es) ** 2
    qsat = qp * foeew
    qsat_i = qp_i * foeew + qp * foeew_i
    if qsat > ZQMAX:
        qsat = ZQMAX
        qsat_i = 0.0
    cor = 1.0 / (1.0 - RETV * qsat)
    cor_i = RETV * qsat_i / (1.0 - RETV * qsat) ** 2.0
    qsat_i = qsat_i * cor + qsat * cor_i
    qsat *= cor
    z2s = z5alcp / (t - z4es) ** 2.0
    z2s_i = -2.0 * z5alcp * t_i / (t - z4es) ** 3.0
    cond = (q - qsat) / (1.0 + qsat * cor * z2s)
    cond_i = (q_i - qsat_i) / (1.0 + qsat * cor * z2s) - (q - qsat) * (
        qsat_i * cor * z2s + qsat * cor_i * z2s + qsat * cor * z2s_i
    ) / (1.0 + qsat * cor * z2s) ** 2.0
    t += zaldcp * cond
    t_i += zaldcp * cond_i
    q -= cond
    q_i -= cond_i

    return t, t_i, q, q_i


@function_collection("f_cuadjtqs_tl")
@gtscript.function
def f_cuadjtqs_tl(ap, ap_i, t, t_i, q, q_i):
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
        t, t_i, q, q_i = f_cuadjtqs_tl_0(ap, ap_i, t, t_i, q, q_i, z3es, z4es, z5alcp, zaldcp)
        t, t_i, q, q_i = f_cuadjtqs_tl_0(ap, ap_i, t, t_i, q, q_i, z3es, z4es, z5alcp, zaldcp)
        return t, t_i, q, q_i
