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


@function_collection("f_cuadjtqs_ad")
@gtscript.function
def f_cuadjtqs_ad(ap, ap_i, t, t_i, q, q_i):
    from __externals__ import (
        ICALL,
        R2ES,
        R3IES,
        R3LES,
        R4IES,
        R4LES,
        R5ALSCP,
        R5ALVCP,
        RALSDCP,
        RALVDCP,
        RETV,
        RTT,
        ZQMAX,
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
        targ = t
        foeew = R2ES * exp(z3es * (targ - RTT) / (targ - z4es))
        foeew_b = foeew
        qsat = foeew / ap
        ltest2 = qsat > ZQMAX
        if ltest2:
            qsat = ZQMAX
        cor = 1.0 / (1.0 - RETV * qsat)
        qsat_d = qsat
        qsat *= cor
        targ_b = targ
        z2s = z5alcp / (targ - z4es) ** 2.0
        qsat_b = qsat
        cor_b = cor
        z2s_b = z2s
        q_b = q
        cond1 = (q - qsat) / (1.0 + qsat * cor * z2s)
        t += zaldcp * cond1
        q -= cond1

        targ = t
        foeew = R2ES * exp(z3es * (targ - RTT) / (targ - z4es))
        foeew_a = foeew
        qsat = foeew / ap
        ltest1 = qsat > ZQMAX
        if ltest1:
            qsat = ZQMAX
        cor = 1.0 / (1.0 - RETV * qsat)
        qsat_c = qsat
        qsat *= cor
        targ_a = targ
        z2s = z5alcp / (targ - z4es) ** 2.0
        qsat_a = qsat
        cor_a = cor
        z2s_a = z2s
        q_a = q
        cond1 = (q - qsat) / (1.0 + qsat * cor * z2s)
        t += zaldcp * cond1
        q -= cond1

        cond1_i = -q_i + zaldcp * t_i
        qsat = qsat_a
        cor = cor_a
        z2s = z2s_a
        q_i += cond1_i / (1.0 + qsat * cor * z2s)
        qsat_i = (
            -cond1_i / (1.0 + qsat * cor * z2s)
            - cond1_i * (q_a - qsat) * cor * z2s / (1.0 + qsat * cor * z2s) ** 2.0
        )
        cor_i = -cond1_i * (q_a - qsat) * qsat * z2s / (1.0 + qsat * cor * z2s) ** 2.0
        z2s_i = -cond1_i * (q_a - qsat) * qsat * cor / (1.0 + qsat * cor * z2s) ** 2.0
        targ = targ_a
        targ_i = -2.0 * z2s_i * z5alcp / (targ - z4es) ** 3.0
        qsat = qsat_c
        cor_i += qsat_i * qsat
        qsat_i *= cor
        qsat_i += cor_i * RETV / (1.0 - RETV * qsat) ** 2.0
        if ltest1:
            qsat_i = 0.0
        foeew_i = qsat_i / ap
        foeew = foeew_a
        qp_i = qsat_i * foeew
        targ_i += (
            foeew_i
            * R2ES
            * z3es
            * (RTT - z4es)
            * exp(z3es * (targ - RTT) / (targ - z4es))
            / (targ - z4es) ** 2.0
        )
        t_i += targ_i

        cond1_i = -q_i + zaldcp * t_i
        qsat = qsat_b
        cor = cor_b
        z2s = z2s_b
        q_i += cond1_i / (1.0 + qsat * cor * z2s)
        qsat_i = (
            -cond1_i / (1.0 + qsat * cor * z2s)
            - cond1_i * (q_b - qsat) * cor * z2s / (1.0 + qsat * cor * z2s) ** 2.0
        )
        cor_i = -cond1_i * (q_b - qsat) * qsat * z2s / (1.0 + qsat * cor * z2s) ** 2.0
        z2s_i = -cond1_i * (q_b - qsat) * qsat * cor / (1.0 + qsat * cor * z2s) ** 2.0
        targ = targ_b
        targ_i = -2.0 * z2s_i * z5alcp / (targ - z4es) ** 3.0
        qsat = qsat_d
        cor_i += qsat_i * qsat
        qsat_i *= cor
        qsat_i += cor_i * RETV / (1.0 - RETV * qsat) ** 2.0
        if ltest2:
            qsat_i = 0.0
        foeew_i = qsat_i / ap
        foeew = foeew_b
        qp_i += qsat_i * foeew
        targ_i += (
            foeew_i
            * R2ES
            * z3es
            * (RTT - z4es)
            * exp(z3es * (targ - RTT) / (targ - z4es))
            / (targ - z4es) ** 2.0
        )
        t_i += targ_i
        ap_i -= qp_i / ap**2.0

        return ap_i, t, t_i, q, q_i
