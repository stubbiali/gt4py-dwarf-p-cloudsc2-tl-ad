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

from ifs_physics_common.framework.stencil import stencil_collection


@stencil_collection(name="diagnose_eta")
def diagnose_eta(
    in_ap: gtscript.Field["float"], out_eta: gtscript.Field[gtscript.K, "float"], *, ap_top: "float"
):
    with computation(FORWARD), interval(...):
        out_eta[0] = in_ap[0, 0, 0] / ap_top
