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
import numpy as np
from typing import TYPE_CHECKING

from ifs_physics_common.framework.grid import I, J, K
from ifs_physics_common.framework.storage import allocate_data_array
from ifs_physics_common.utils.numpyx import assign

if TYPE_CHECKING:
    from cloudsc2_gt4py.utils.iox import HDF5Reader
    from ifs_physics_common.framework.config import GT4PyConfig
    from ifs_physics_common.framework.grid import ComputationalGrid
    from ifs_physics_common.utils.typingx import DataArray, DataArrayDict


def allocate_state(
    computational_grid: ComputationalGrid, *, gt4py_config: GT4PyConfig
) -> dict[str, DataArray]:
    def allocator(units: str) -> DataArray:
        return allocate_data_array(
            computational_grid, (I, J, K), units, gt4py_config=gt4py_config, dtype="float"
        )

    def allocator_zh(units: str) -> DataArray:
        return allocate_data_array(
            computational_grid, (I, J, K - 1 / 2), units, gt4py_config=gt4py_config, dtype="float"
        )

    def allocator_d(units: str) -> DataArray:
        return allocate_data_array(
            computational_grid,
            (I, J, K),
            units,
            data_shape=(5,),
            data_dims=("d",),
            gt4py_config=gt4py_config,
            dtype="float",
        )

    return {
        "f_t": allocator("K"),
        "f_q": allocator("g g^-1"),
        "f_ql": allocator("g g^-1"),
        "f_qi": allocator("g g^-1"),
        "f_ap": allocator("Pa"),
        "f_aph": allocator_zh("Pa"),
        "f_lu": allocator("g g^-1"),
        "f_lude": allocator("kg m^-3 s^-1"),
        "f_mfu": allocator("kg m^-2 s^-1"),
        "f_mfd": allocator("kg m^-2 s^-1"),
        "f_a": allocator("1"),
        "f5_clv": allocator_d("g g^-1"),
        "f_supsat": allocator("g g^-1"),
    }


def allocate_tendencies(
    computational_grid: ComputationalGrid, *, gt4py_config: GT4PyConfig
) -> dict[str, DataArray]:
    def allocator_ik(units: str) -> DataArray:
        return allocate_data_array(
            computational_grid, (I, J, K), units, gt4py_config=gt4py_config, dtype="float"
        )

    def allocator_ikd(units: str) -> DataArray:
        return allocate_data_array(
            computational_grid,
            (I, J, K),
            units,
            data_shape=(5,),
            data_dims=("d",),
            gt4py_config=gt4py_config,
            dtype="float",
        )

    return {
        "f_t": allocator_ik("K s^-1"),
        "f_a": allocator_ik("s^-1"),
        "f_q": allocator_ik("g g^-1 s^-1"),
        "f_ql": allocator_ik("g g^-1 s^-1"),
        "f_qi": allocator_ik("g g^-1 s^-1"),
        "f5_cld": allocator_ikd("s^-1"),
    }


def initialize(
    data_array_dict: dict[str, DataArray],
    hdf5_reader: HDF5Reader,
    data_array_dict_key: str,
    hdf5_reader_key: str,
) -> None:
    field = data_array_dict[data_array_dict_key].data
    ni, _, nk = field.shape[:3]
    buffer = hdf5_reader.get_field(hdf5_reader_key)
    mi, mk = buffer.shape[:2]

    nk = min(nk, mk)

    nb = ni // mi
    for b in range(nb):
        assign(field[b * mi : (b + 1) * mi, 0:1, :nk], buffer[:, np.newaxis, :nk])
    assign(field[nb * mi :, 0:1, :nk], buffer[: ni - nb * mi, np.newaxis, :nk])


def initialize_state(state: dict[str, DataArray], hdf5_reader: HDF5Reader) -> None:
    for key in state:
        hdf5_reader_key = "P" + key.split("_", maxsplit=1)[1].upper()
        initialize(state, hdf5_reader, key, hdf5_reader_key)


def initialize_tendencies(tendencies: dict[str, DataArray], hdf5_reader: HDF5Reader) -> None:
    for key in tendencies:
        hdf5_reader_key = "TENDENCY_CML_" + key.split("_", maxsplit=1)[1].upper()
        initialize(tendencies, hdf5_reader, key, hdf5_reader_key)


def get_accumulated_tendencies(
    computational_grid: ComputationalGrid, hdf5_reader: HDF5Reader, *, gt4py_config: GT4PyConfig
) -> dict[str, DataArray]:
    tendencies = allocate_tendencies(computational_grid, gt4py_config=gt4py_config)
    out = {key: tendencies[key] for key in tendencies if key not in ("f_ql", "f_qi")}
    initialize_tendencies(out, hdf5_reader)
    out["f_ql"] = tendencies["f_ql"]
    out["f_qi"] = tendencies["f_qi"]
    out["f_ql"][...] = out["f5_cld"][..., 0]
    out["f_qi"][...] = out["f5_cld"][..., 1]
    return out


def get_initial_state(
    computational_grid: ComputationalGrid, hdf5_reader: HDF5Reader, *, gt4py_config: GT4PyConfig
) -> DataArrayDict:
    state = allocate_state(computational_grid, gt4py_config=gt4py_config)

    out = {key: state[key] for key in state if key not in ("f_ql", "f_qi")}
    initialize_state(out, hdf5_reader)
    out["f_ql"] = state["f_ql"]
    out["f_qi"] = state["f_qi"]
    out["f_ql"][...] = out["f5_clv"][..., 0]
    out["f_qi"][...] = out["f5_clv"][..., 1]

    tendencies = get_accumulated_tendencies(
        computational_grid, hdf5_reader, gt4py_config=gt4py_config
    )
    out["f_tnd_cml_t"] = tendencies["f_t"]
    out["f_tnd_cml_q"] = tendencies["f_q"]
    out["f_tnd_cml_ql"] = tendencies["f_ql"]
    out["f_tnd_cml_qi"] = tendencies["f_qi"]

    out["time"] = datetime(1970, 1, 1)

    return out
