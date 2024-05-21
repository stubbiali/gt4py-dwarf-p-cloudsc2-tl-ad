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
from datetime import timedelta
from functools import lru_cache
import h5py
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing import Union

    from ifs_physics_common.config import DataTypes


class HDF5Reader:
    default_dataset_b = [True]
    default_dataset_f = [0.0]
    default_dataset_i = [0]

    f: h5py.File
    data_types: DataTypes

    def __init__(self, filename: str, data_types: DataTypes) -> None:
        self.f = h5py.File(filename)
        self.data_types = data_types

    def __del__(self) -> None:
        self.f.close()

    def get_field(self, name: str) -> NDArray:
        ds = self.f.get(name, None)
        if ds is None:
            raise RuntimeError(f"Unknown field `{name}`.")

        if ds.ndim == 1:
            return self._get_field_1d(ds, name)
        elif ds.ndim == 2:
            return self._get_field_2d(ds, name)
        elif ds.ndim == 3:
            return self._get_field_3d(ds, name)
        else:
            raise RuntimeError(f"The field `{name}` has unexpected shape {ds.shape}.")

    @lru_cache
    def get_nlev(self) -> int:
        return self.f["KLEV"][0]  # type: ignore[no-any-return]

    @lru_cache
    def get_nlon(self) -> int:
        return self.f["KLON"][0]  # type: ignore[no-any-return]

    def get_timestep(self) -> timedelta:
        return timedelta(seconds=float(self._get_parameter_f("PTSPHY")))

    def get_yoethf_parameters(self) -> dict[str, float]:
        out = {
            "R2ES": self._get_parameter_f("R2ES"),
            "R3LES": self._get_parameter_f("R3LES"),
            "R3IES": self._get_parameter_f("R3IES"),
            "R4LES": self._get_parameter_f("R4LES"),
            "R4IES": self._get_parameter_f("R4IES"),
            "R5LES": self._get_parameter_f("R5LES"),
            "R5IES": self._get_parameter_f("R5IES"),
            "R5ALVCP": self._get_parameter_f("R5ALVCP"),
            "R5ALSCP": self._get_parameter_f("R5ALSCP"),
            "RALVDCP": self._get_parameter_f("RALVDCP"),
            "RALSDCP": self._get_parameter_f("RALSDCP"),
            "RALFDCP": self._get_parameter_f("RALFDCP"),
            "RTWAT": self._get_parameter_f("RTWAT"),
            "RTICE": self._get_parameter_f("RTICE"),
            "RTICECU": self._get_parameter_f("RTICECU"),
            "RTWAT_RTICE_R": self._get_parameter_f("RTWAT_RTICE_R"),
            "RTWAT_RTICECU_R": self._get_parameter_f("RTWAT_RTICECU_R"),
            "RKOOP1": self._get_parameter_f("RKOOP1"),
            "RKOOP2": self._get_parameter_f("RKOOP2"),
            "RVTMP2": 0.0,
        }
        return out

    def get_yomcst_parameters(self) -> dict[str, float]:
        out = {
            "RG": self._get_parameter_f("RG"),
            "RD": self._get_parameter_f("RD"),
            "RCPD": self._get_parameter_f("RCPD"),
            "RETV": self._get_parameter_f("RETV"),
            "RLVTT": self._get_parameter_f("RLVTT"),
            "RLSTT": self._get_parameter_f("RLSTT"),
            "RLMLT": self._get_parameter_f("RLMLT"),
            "RTT": self._get_parameter_f("RTT"),
            "RV": self._get_parameter_f("RV"),
        }
        return out

    def get_yrecld_parameters(self) -> dict[str, Union[bool]]:
        return {}

    def get_yrecldp_parameters(self) -> dict[str, Union[bool, float, int]]:
        out = {
            "RAMID": self._get_parameter_f("YRECLDP_RAMID"),
            "RCLDIFF": self._get_parameter_f("YRECLDP_RCLDIFF"),
            "RCLDIFF_CONVI": self._get_parameter_f("YRECLDP_RCLDIFF_CONVI"),
            "RCLCRIT": self._get_parameter_f("YRECLDP_RCLCRIT"),
            "RCLCRIT_SEA": self._get_parameter_f("YRECLDP_RCLCRIT_SEA"),
            "RCLCRIT_LAND": self._get_parameter_f("YRECLDP_RCLCRIT_LAND"),
            "RKCONV": self._get_parameter_f("YRECLDP_RKCONV"),
            "RPRC1": self._get_parameter_f("YRECLDP_RPRC1"),
            "RPRC2": self._get_parameter_f("YRECLDP_RPRC2"),
            "RCLDMAX": self._get_parameter_f("YRECLDP_RCLDMAX"),
            "RPECONS": self._get_parameter_f("YRECLDP_RPECONS"),
            "RVRFACTOR": self._get_parameter_f("YRECLDP_RVRFACTOR"),
            "RPRECRHMAX": self._get_parameter_f("YRECLDP_RPRECRHMAX"),
            "RTAUMEL": self._get_parameter_f("YRECLDP_RTAUMEL"),
            "RAMIN": self._get_parameter_f("YRECLDP_RAMIN"),
            "RLMIN": self._get_parameter_f("YRECLDP_RLMIN"),
            "RKOOPTAU": self._get_parameter_f("YRECLDP_RKOOPTAU"),
            "RCLDTOPP": self._get_parameter_f("YRECLDP_RCLDTOPP"),
            "RLCRITSNOW": self._get_parameter_f("YRECLDP_RLCRITSNOW"),
            "RSNOWLIN1": self._get_parameter_f("YRECLDP_RSNOWLIN1"),
            "RSNOWLIN2": self._get_parameter_f("YRECLDP_RSNOWLIN2"),
            "RICEHI1": self._get_parameter_f("YRECLDP_RICEHI1"),
            "RICEHI2": self._get_parameter_f("YRECLDP_RICEHI2"),
            "RICEINIT": self._get_parameter_f("YRECLDP_RICEINIT"),
            "RVICE": self._get_parameter_f("YRECLDP_RVICE"),
            "RVRAIN": self._get_parameter_f("YRECLDP_RVRAIN"),
            "RVSNOW": self._get_parameter_f("YRECLDP_RVSNOW"),
            "RTHOMO": self._get_parameter_f("YRECLDP_RTHOMO"),
            "RCOVPMIN": self._get_parameter_f("YRECLDP_RCOVPMIN"),
            "RCCN": self._get_parameter_f("YRECLDP_RCCN"),
            "RNICE": self._get_parameter_f("YRECLDP_RNICE"),
            "RCCNOM": self._get_parameter_f("YRECLDP_RCCNOM"),
            "RCCNSS": self._get_parameter_f("YRECLDP_RCCNSS"),
            "RCCNSU": self._get_parameter_f("YRECLDP_RCCNSU"),
            "RCLDTOPCF": self._get_parameter_f("YRECLDP_RCLDTOPCF"),
            "RDEPLIQREFRATE": self._get_parameter_f("YRECLDP_RDEPLIQREFRATE"),
            "RDEPLIQREFDEPTH": self._get_parameter_f("YRECLDP_RDEPLIQREFDEPTH"),
            "RCL_KKAac": self._get_parameter_f("YRECLDP_RCL_KKAac"),
            "RCL_KKBac": self._get_parameter_f("YRECLDP_RCL_KKBac"),
            "RCL_KKAau": self._get_parameter_f("YRECLDP_RCL_KKAau"),
            "RCL_KKBauq": self._get_parameter_f("YRECLDP_RCL_KKBauq"),
            "RCL_KKBaun": self._get_parameter_f("YRECLDP_RCL_KKBaun"),
            "RCL_KK_cloud_num_sea": self._get_parameter_f("YRECLDP_RCL_KK_cloud_num_sea"),
            "RCL_KK_cloud_num_land": self._get_parameter_f("YRECLDP_RCL_KK_cloud_num_land"),
            "RCL_AI": self._get_parameter_f("YRECLDP_RCL_AI"),
            "RCL_BI": self._get_parameter_f("YRECLDP_RCL_BI"),
            "RCL_CI": self._get_parameter_f("YRECLDP_RCL_CI"),
            "RCL_DI": self._get_parameter_f("YRECLDP_RCL_DI"),
            "RCL_X1I": self._get_parameter_f("YRECLDP_RCL_X1I"),
            "RCL_X2I": self._get_parameter_f("YRECLDP_RCL_X2I"),
            "RCL_X3I": self._get_parameter_f("YRECLDP_RCL_X3I"),
            "RCL_X4I": self._get_parameter_f("YRECLDP_RCL_X4I"),
            "RCL_CONST1I": self._get_parameter_f("YRECLDP_RCL_CONST1I"),
            "RCL_CONST2I": self._get_parameter_f("YRECLDP_RCL_CONST2I"),
            "RCL_CONST3I": self._get_parameter_f("YRECLDP_RCL_CONST3I"),
            "RCL_CONST4I": self._get_parameter_f("YRECLDP_RCL_CONST4I"),
            "RCL_CONST5I": self._get_parameter_f("YRECLDP_RCL_CONST5I"),
            "RCL_CONST6I": self._get_parameter_f("YRECLDP_RCL_CONST6I"),
            "RCL_APB1": self._get_parameter_f("YRECLDP_RCL_APB1"),
            "RCL_APB2": self._get_parameter_f("YRECLDP_RCL_APB2"),
            "RCL_APB3": self._get_parameter_f("YRECLDP_RCL_APB3"),
            "RCL_AS": self._get_parameter_f("YRECLDP_RCL_AS"),
            "RCL_BS": self._get_parameter_f("YRECLDP_RCL_BS"),
            "RCL_CS": self._get_parameter_f("YRECLDP_RCL_CS"),
            "RCL_DS": self._get_parameter_f("YRECLDP_RCL_DS"),
            "RCL_X1S": self._get_parameter_f("YRECLDP_RCL_X1S"),
            "RCL_X2S": self._get_parameter_f("YRECLDP_RCL_X2S"),
            "RCL_X3S": self._get_parameter_f("YRECLDP_RCL_X3S"),
            "RCL_X4S": self._get_parameter_f("YRECLDP_RCL_X4S"),
            "RCL_CONST1S": self._get_parameter_f("YRECLDP_RCL_CONST1S"),
            "RCL_CONST2S": self._get_parameter_f("YRECLDP_RCL_CONST2S"),
            "RCL_CONST3S": self._get_parameter_f("YRECLDP_RCL_CONST3S"),
            "RCL_CONST4S": self._get_parameter_f("YRECLDP_RCL_CONST4S"),
            "RCL_CONST5S": self._get_parameter_f("YRECLDP_RCL_CONST5S"),
            "RCL_CONST6S": self._get_parameter_f("YRECLDP_RCL_CONST6S"),
            "RCL_CONST7S": self._get_parameter_f("YRECLDP_RCL_CONST7S"),
            "RCL_CONST8S": self._get_parameter_f("YRECLDP_RCL_CONST8S"),
            "RDENSWAT": self._get_parameter_f("YRECLDP_RDENSWAT"),
            "RDENSREF": self._get_parameter_f("YRECLDP_RDENSREF"),
            "RCL_AR": self._get_parameter_f("YRECLDP_RCL_AR"),
            "RCL_BR": self._get_parameter_f("YRECLDP_RCL_BR"),
            "RCL_CR": self._get_parameter_f("YRECLDP_RCL_CR"),
            "RCL_DR": self._get_parameter_f("YRECLDP_RCL_DR"),
            "RCL_X1R": self._get_parameter_f("YRECLDP_RCL_X1R"),
            "RCL_X2R": self._get_parameter_f("YRECLDP_RCL_X2R"),
            "RCL_X4R": self._get_parameter_f("YRECLDP_RCL_X4R"),
            "RCL_KA273": self._get_parameter_f("YRECLDP_RCL_KA273"),
            "RCL_CDENOM1": self._get_parameter_f("YRECLDP_RCL_CDENOM1"),
            "RCL_CDENOM2": self._get_parameter_f("YRECLDP_RCL_CDENOM2"),
            "RCL_CDENOM3": self._get_parameter_f("YRECLDP_RCL_CDENOM3"),
            "RCL_SCHMIDT": self._get_parameter_f("YRECLDP_RCL_SCHMIDT"),
            "RCL_DYNVISC": self._get_parameter_f("YRECLDP_RCL_DYNVISC"),
            "RCL_CONST1R": self._get_parameter_f("YRECLDP_RCL_CONST1R"),
            "RCL_CONST2R": self._get_parameter_f("YRECLDP_RCL_CONST2R"),
            "RCL_CONST3R": self._get_parameter_f("YRECLDP_RCL_CONST3R"),
            "RCL_CONST4R": self._get_parameter_f("YRECLDP_RCL_CONST4R"),
            "RCL_CONST5R": self._get_parameter_f("YRECLDP_RCL_CONST5R"),
            "RCL_CONST6R": self._get_parameter_f("YRECLDP_RCL_CONST6R"),
            "RCL_FAC1": self._get_parameter_f("YRECLDP_RCL_FAC1"),
            "RCL_FAC2": self._get_parameter_f("YRECLDP_RCL_FAC2"),
            "RCL_FZRAB": self._get_parameter_f("YRECLDP_RCL_FZRAB"),
            "RCL_FZRBB": self._get_parameter_f("YRECLDP_RCL_FZRBB"),
            "LCLDEXTRA": self._get_parameter_b("YRECLDP_LCLDEXTRA"),
            "LCLDBUDGET": self._get_parameter_b("YRECLDP_LCLDBUDGET"),
            "NSSOPT": self._get_parameter_i("YRECLDP_NSSOPT"),
            "NCLDTOP": self._get_parameter_i("YRECLDP_NCLDTOP"),
            "NAECLBC": self._get_parameter_i("YRECLDP_NAECLBC"),
            "NAECLDU": self._get_parameter_i("YRECLDP_NAECLDU"),
            "NAECLOM": self._get_parameter_i("YRECLDP_NAECLOM"),
            "NAECLSS": self._get_parameter_i("YRECLDP_NAECLSS"),
            "NAECLSU": self._get_parameter_i("YRECLDP_NAECLSU"),
            "NCLDDIAG": self._get_parameter_i("YRECLDP_NCLDDIAG"),
            "NAERCLD": self._get_parameter_i("YRECLDP_NAERCLD"),
            "LAERLIQAUTOLSP": self._get_parameter_b("YRECLDP_LAERLIQAUTOLSP"),
            "LAERLIQAUTOCP": self._get_parameter_b("YRECLDP_LAERLIQAUTOCP"),
            "LAERLIQAUTOCPB": self._get_parameter_b("YRECLDP_LAERLIQAUTOCPB"),
            "LAERLIQCOLL": self._get_parameter_b("YRECLDP_LAERLIQCOLL"),
            "LAERICESED": self._get_parameter_b("YRECLDP_LAERICESED"),
            "LAERICEAUTO": self._get_parameter_b("YRECLDP_LAERICEAUTO"),
            "NSHAPEP": self._get_parameter_i("YRECLDP_NSHAPEP"),
            "NSHAPEQ": self._get_parameter_i("YRECLDP_NSHAPEQ"),
            "NBETA": self._get_parameter_i("YRECLDP_NBETA"),
        }
        return out

    def get_yrephli_parameters(self) -> dict[str, Union[bool, float]]:
        out = {
            "LTLEVOL": self._get_parameter_b("YREPHLI_LTLEVOL"),
            "LPHYLIN": self._get_parameter_b("YREPHLI_LPHYLIN"),
            "LENOPERT": self._get_parameter_b("YREPHLI_LENOPERT"),
            "LEPPCFLS": self._get_parameter_b("YREPHLI_LEPPCFLS"),
            "LRAISANEN": self._get_parameter_b("YREPHLI_LRAISANEN"),
            "RLPTRC": self._get_parameter_f("YREPHLI_RLPTRC"),
            "RLPAL1": self._get_parameter_f("YREPHLI_RLPAL1"),
            "RLPAL2": self._get_parameter_f("YREPHLI_RLPAL2"),
            "RLPBB": self._get_parameter_f("YREPHLI_RLPBB"),
            "RLPCC": self._get_parameter_f("YREPHLI_RLPCC"),
            "RLPDD": self._get_parameter_f("YREPHLI_RLPDD"),
            "RLPMIXL": self._get_parameter_f("YREPHLI_RLPMIXL"),
            "RLPBETA": self._get_parameter_f("YREPHLI_RLPBETA"),
            "RLPDRAG": self._get_parameter_f("YREPHLI_RLPDRAG"),
            "RLPEVAP": self._get_parameter_f("YREPHLI_RLPEVAP"),
            "RLPP00": self._get_parameter_f("YREPHLI_RLPP00"),
        }
        return out

    def get_yrncl_parameters(self) -> dict[str, bool]:
        return {"LREGCL": True}

    def get_yrphnc_parameters(self) -> dict[str, bool]:
        return {"LEVAPLS2": False}

    def _get_field_1d(self, ds: h5py.Dataset, name: str) -> NDArray:
        nlon = self.get_nlon()
        nlev = self.get_nlev()
        if nlon <= ds.shape[0] <= nlon + 1 or nlev <= ds.shape[0] <= nlev + 1:
            return ds[:]  # type: ignore[no-any-return]
        else:
            raise RuntimeError(
                f"The field `{name}` is expected to have shape ({nlon}(+1),) or "
                f"({nlev}(+1),), but has shape {ds.shape}."
            )

    def _get_field_2d(self, ds: h5py.Dataset, name: str) -> NDArray:
        nlon = self.get_nlon()
        nlev = self.get_nlev()
        if nlon <= ds.shape[0] <= nlon + 1 and nlev <= ds.shape[1] <= nlev + 1:
            return ds[...]  # type: ignore[no-any-return]
        elif nlon <= ds.shape[1] <= nlon + 1 and nlev <= ds.shape[0] <= nlev + 1:
            return np.transpose(ds[...])
        else:
            raise RuntimeError(
                f"The field `{name}` is expected to have shape "
                f"({nlon}(+1), {nlev}(+1)) or ({nlev}(+1), {nlon}(+1)), "
                f"but has shape {ds.shape}."
            )

    def _get_field_3d(self, ds: h5py.Dataset, name: str) -> NDArray:
        nlon = self.get_nlon()
        nlev = self.get_nlev()

        if nlon in ds.shape:
            axes = [ds.shape.index(nlon)]
        elif nlon + 1 in ds.shape:
            axes = [ds.shape.index(nlon + 1)]
        else:
            raise RuntimeError(f"The field `{name}` has unexpected shape {ds.shape}.")

        if nlev in ds.shape:
            axes += [ds.shape.index(nlev)]
        elif nlev + 1 in ds.shape:
            axes += [ds.shape.index(nlev + 1)]
        else:
            raise RuntimeError(f"The field `{name}` has unexpected shape {ds.shape}.")

        axes += tuple({0, 1, 2} - set(axes))

        return np.transpose(ds[...], axes=axes)

    def _get_parameter_b(self, name: str) -> bool:
        return self.data_types.bool(self.f.get(name, [True])[0])  # type: ignore[no-any-return]

    def _get_parameter_f(self, name: str) -> float:
        return self.data_types.float(self.f.get(name, [0.0])[0])  # type: ignore[no-any-return]

    def _get_parameter_i(self, name: str) -> int:
        return self.data_types.int(self.f.get(name, [0])[0])  # type: ignore[no-any-return]
