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
from typing import TYPE_CHECKING

from cloudsc2_gt4py.setup import IJK_ARGS, IJKD5_ARGS, REFERENCE_TIME
from ifs_physics_common.grid import K

if TYPE_CHECKING:
    from ifs_physics_common.iox import HDF5GridOperator
    from ifs_physics_common.typingx import DataArrayDict


def get_reference_tendencies(hdf5_grid_operator: HDF5GridOperator) -> DataArrayDict:
    field_properties = {
        "f_qi": IJKD5_ARGS(h5_name="TENDENCY_LOC_CLD", units="g g^-1 s^-1", index=1),
        "f_ql": IJKD5_ARGS(h5_name="TENDENCY_LOC_CLD", units="g g^-1 s^-1", index=0),
        "f_qv": IJK_ARGS(h5_name="TENDENCY_LOC_Q", units="g g^-1 s^-1"),
        "f_t": IJK_ARGS(h5_name="TENDENCY_LOC_T", units="K s^-1"),
    }
    tends = {
        name: hdf5_grid_operator.get_field(**props) for name, props in field_properties.items()
    }
    tends["time"] = REFERENCE_TIME
    return tends


def get_reference_diagnostics(hdf5_grid_operator: HDF5GridOperator) -> DataArrayDict:
    field_properties = {
        "f_clc": IJK_ARGS(h5_name="PCLC", units="1"),
        "f_covptot": IJK_ARGS(h5_name="PCOVPTOT", units="1"),
        "f_fhpsl": IJK_ARGS(h5_name="PFHPSL", units="J m^-2 s^-1", dim_k=K - 1 / 2),
        "f_fhpsn": IJK_ARGS(h5_name="PFHPSN", units="J m^-2 s^-1", dim_k=K - 1 / 2),
        "f_fplsl": IJK_ARGS(h5_name="PFPLSL", units="kg m^-2 s^-1", dim_k=K - 1 / 2),
        "f_fplsn": IJK_ARGS(h5_name="PFPLSN", units="kg m^-2 s^-1", dim_k=K - 1 / 2),
    }
    diags = {
        name: hdf5_grid_operator.get_field(**props) for name, props in field_properties.items()
    }
    diags["time"] = REFERENCE_TIME
    return diags
