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
from datetime import datetime
from typing import TYPE_CHECKING

from ifs_physics_common.grid import D5, ExpandedDim, I, IJ, J, K

if TYPE_CHECKING:
    from ifs_physics_common.iox import HDF5GridOperator
    from ifs_physics_common.typingx import DataArrayDict


IJK_ARGS = lambda h5_name, dim_k=K, units="": {
    "grid_dims": (I, J, dim_k),
    "dtype_name": "float",
    "units": units,
    "h5_name": h5_name,
    "h5_dims": (dim_k, IJ),
    "h5_dims_map": (IJ, ExpandedDim, dim_k),
}
IJKD5_ARGS = lambda h5_name, index, units="": {
    "grid_dims": (I, J, K),
    "dtype_name": "float",
    "units": units,
    "h5_name": h5_name,
    "h5_dims": (D5, K, IJ),
    "h5_dims_map": (IJ, ExpandedDim, K, D5[index]),
}
REFERENCE_TIME = datetime(year=1970, month=1, day=1)


def get_state(hdf5_grid_operator: HDF5GridOperator) -> DataArrayDict:
    field_properties = {
        "f_a": IJK_ARGS(h5_name="PA", units="1"),
        "f_ap": IJK_ARGS(h5_name="PAP", units="Pa"),
        "f_aph": IJK_ARGS(h5_name="PAPH", units="Pa", dim_k=K - 1 / 2),
        "f_lu": IJK_ARGS(h5_name="PLU", units="g g^-1"),
        "f_lude": IJK_ARGS(h5_name="PLUDE", units="kg m^-3 s^-1"),
        "f_mfd": IJK_ARGS(h5_name="PMFD", units="kg m^-2 s^-1"),
        "f_mfu": IJK_ARGS(h5_name="PMFU", units="kg m^-2 s^-1"),
        "f_qi": IJKD5_ARGS(h5_name="PCLV", units="g g^-1", index=1),
        "f_ql": IJKD5_ARGS(h5_name="PCLV", units="g g^-1", index=0),
        "f_q": IJK_ARGS(h5_name="PQ", units="g g^-1"),
        "f_supsat": IJK_ARGS(h5_name="PSUPSAT", units="g g^-1"),
        "f_t": IJK_ARGS(h5_name="PT", units="K"),
        "f_tnd_cml_qi": IJKD5_ARGS(h5_name="TENDENCY_CML_CLD", units="g g^-1 s^-1", index=1),
        "f_tnd_cml_ql": IJKD5_ARGS(h5_name="TENDENCY_CML_CLD", units="g g^-1 s^-1", index=0),
        "f_tnd_cml_q": IJK_ARGS(h5_name="TENDENCY_CML_Q", units="g g^-1 s^-1"),
        "f_tnd_cml_t": IJK_ARGS(h5_name="TENDENCY_CML_T", units="K s^-1"),
    }
    state = {
        name: hdf5_grid_operator.get_field(**props) for name, props in field_properties.items()
    }
    state["time"] = REFERENCE_TIME
    return state
