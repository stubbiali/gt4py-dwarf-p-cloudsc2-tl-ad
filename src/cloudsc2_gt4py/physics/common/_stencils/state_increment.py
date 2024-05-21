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

from ifs_physics_common.stencil import stencil_collection


@stencil_collection("state_increment")
def state_increment(
    in_aph: gtscript.Field["float"],
    in_ap: gtscript.Field["float"],
    in_q: gtscript.Field["float"],
    in_qsat: gtscript.Field["float"],
    in_t: gtscript.Field["float"],
    in_ql: gtscript.Field["float"],
    in_qi: gtscript.Field["float"],
    in_lude: gtscript.Field["float"],
    in_lu: gtscript.Field["float"],
    in_mfu: gtscript.Field["float"],
    in_mfd: gtscript.Field["float"],
    in_tnd_cml_t: gtscript.Field["float"],
    in_tnd_cml_q: gtscript.Field["float"],
    in_tnd_cml_ql: gtscript.Field["float"],
    in_tnd_cml_qi: gtscript.Field["float"],
    in_supsat: gtscript.Field["float"],
    out_aph_i: gtscript.Field["float"],
    out_ap_i: gtscript.Field["float"],
    out_q_i: gtscript.Field["float"],
    out_qsat_i: gtscript.Field["float"],
    out_t_i: gtscript.Field["float"],
    out_ql_i: gtscript.Field["float"],
    out_qi_i: gtscript.Field["float"],
    out_lude_i: gtscript.Field["float"],
    out_lu_i: gtscript.Field["float"],
    out_mfu_i: gtscript.Field["float"],
    out_mfd_i: gtscript.Field["float"],
    out_tnd_cml_t_i: gtscript.Field["float"],
    out_tnd_cml_q_i: gtscript.Field["float"],
    out_tnd_cml_ql_i: gtscript.Field["float"],
    out_tnd_cml_qi_i: gtscript.Field["float"],
    out_supsat_i: gtscript.Field["float"],
    *,
    f: "float",
):
    from __externals__ import IGNORE_SUPSAT

    with computation(PARALLEL), interval(...):
        out_aph_i[0, 0, 0] = f * in_aph[0, 0, 0]
        out_ap_i[0, 0, 0] = f * in_ap[0, 0, 0]
        out_q_i[0, 0, 0] = f * in_q[0, 0, 0]
        out_qsat_i[0, 0, 0] = f * in_qsat[0, 0, 0]
        out_t_i[0, 0, 0] = f * in_t[0, 0, 0]
        out_ql_i[0, 0, 0] = f * in_ql[0, 0, 0]
        out_qi_i[0, 0, 0] = f * in_qi[0, 0, 0]
        out_lude_i[0, 0, 0] = f * in_lude[0, 0, 0]
        out_lu_i[0, 0, 0] = f * in_lu[0, 0, 0]
        out_mfu_i[0, 0, 0] = f * in_mfu[0, 0, 0]
        out_mfd_i[0, 0, 0] = f * in_mfd[0, 0, 0]
        out_tnd_cml_t_i[0, 0, 0] = f * in_tnd_cml_t[0, 0, 0]
        out_tnd_cml_q_i[0, 0, 0] = f * in_tnd_cml_q[0, 0, 0]
        out_tnd_cml_ql_i[0, 0, 0] = f * in_tnd_cml_ql[0, 0, 0]
        out_tnd_cml_qi_i[0, 0, 0] = f * in_tnd_cml_qi[0, 0, 0]
        if not IGNORE_SUPSAT:
            out_supsat_i[0, 0, 0] = f * in_supsat[0, 0, 0]
        else:
            out_supsat_i[0, 0, 0] = 0.0
