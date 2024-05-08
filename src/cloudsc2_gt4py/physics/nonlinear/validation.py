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

from cloudsc2_gt4py.utils.iox import HDF5Reader
from ifs_physics_common.utils.numpyx import to_numpy
from ifs_physics_common.utils.validation import validate_field

if TYPE_CHECKING:
    from typing import Optional

    from ifs_physics_common.utils.typingx import DataArrayDict


def validate(
    hdf5_reader: HDF5Reader,
    tendencies: DataArrayDict,
    diagnostics: DataArrayDict,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> None:
    # tendencies
    trg_keys = {
        "f_q": "TENDENCY_LOC_Q",
        "f_qi": "TENDENCY_LOC_CLD",
        "f_ql": "TENDENCY_LOC_CLD",
        "f_t": "TENDENCY_LOC_T",
    }
    data_indices = {"f_q": None, "f_qi": 1, "f_ql": 0, "f_t": None}
    for src_key, trg_key in trg_keys.items():
        src_field = to_numpy(tendencies[src_key].data[:, 0, :])
        trg_field = hdf5_reader.get_field(trg_key)
        if trg_field.ndim == 3 and data_indices[src_key] is not None:
            trg_field = trg_field[..., data_indices[src_key]]
        slc = tuple(
            slice(0, min(s_src, s_trg)) for s_src, s_trg in zip(src_field.shape, trg_field.shape)
        )
        validate_field(src_key, src_field[slc], trg_field[slc], atol=atol, rtol=rtol)

    # diagnostics
    trg_keys = {
        "f_covptot": "PCOVPTOT",
        "f_fhpsl": "PFHPSL",
        "f_fhpsn": "PFHPSN",
        "f_fplsl": "PFPLSL",
        "f_fplsn": "PFPLSN",
    }
    for src_key, trg_key in trg_keys.items():
        src_field = to_numpy(diagnostics[src_key].data[:, 0, :])
        trg_field = hdf5_reader.get_field(trg_key)
        slc = tuple(
            slice(0, min(s_src, s_trg)) for s_src, s_trg in zip(src_field.shape, trg_field.shape)
        )
        validate_field(src_key, src_field[slc], trg_field[slc], atol=atol, rtol=rtol)
