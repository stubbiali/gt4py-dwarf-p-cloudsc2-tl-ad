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
from pydantic import BaseModel

from ifs_physics_common.iox import HDF5Operator as BaseHDF5Operator


class YoethfParams(BaseModel):
    R2ES: float
    R3IES: float
    R3LES: float
    R4IES: float
    R4LES: float
    R5ALSCP: float
    R5ALVCP: float
    R5IES: float
    R5LES: float
    RALFDCP: float
    RALSDCP: float
    RALVDCP: float
    RKOOP1: float
    RKOOP2: float
    RTICE: float
    RTICECU: float
    RTWAT: float
    RTWAT_RTICECU_R: float
    RTWAT_RTICE_R: float
    RVTMP2: float = 0.0


class YomcstParams(BaseModel):
    RCPD: float
    RD: float
    RETV: float
    RG: float
    RLMLT: float
    RLSTT: float
    RLVTT: float
    RTT: float
    RV: float


class YrecldpParams(BaseModel):
    LAERICEAUTO: bool
    LAERICESED: bool
    LAERLIQAUTOCP: bool
    LAERLIQAUTOCPB: bool
    LAERLIQAUTOLSP: bool
    LAERLIQCOLL: bool
    LCLDBUDGET: bool
    LCLDEXTRA: bool
    NAECLBC: int
    NAECLDU: int
    NAECLOM: int
    NAECLSS: int
    NAECLSU: int
    NAERCLD: int
    NBETA: int
    NCLDDIAG: int
    NCLDTOP: int
    NSHAPEP: int
    NSHAPEQ: int
    NSSOPT: int
    RAMID: float
    RAMIN: float
    RCCN: float
    RCCNOM: float
    RCCNSS: float
    RCCNSU: float
    RCLCRIT: float
    RCLCRIT_LAND: float
    RCLCRIT_SEA: float
    RCLDIFF: float
    RCLDIFF_CONVI: float
    RCLDMAX: float
    RCLDTOPCF: float
    RCLDTOPP: float
    RCL_AI: float
    RCL_APB1: float
    RCL_APB2: float
    RCL_APB3: float
    RCL_AR: float
    RCL_AS: float
    RCL_BI: float
    RCL_BR: float
    RCL_BS: float
    RCL_CDENOM1: float
    RCL_CDENOM2: float
    RCL_CDENOM3: float
    RCL_CI: float
    RCL_CONST1I: float
    RCL_CONST1R: float
    RCL_CONST1S: float
    RCL_CONST2I: float
    RCL_CONST2R: float
    RCL_CONST2S: float
    RCL_CONST3I: float
    RCL_CONST3R: float
    RCL_CONST3S: float
    RCL_CONST4I: float
    RCL_CONST4R: float
    RCL_CONST4S: float
    RCL_CONST5I: float
    RCL_CONST5R: float
    RCL_CONST5S: float
    RCL_CONST6I: float
    RCL_CONST6R: float
    RCL_CONST6S: float
    RCL_CONST7S: float
    RCL_CONST8S: float
    RCL_CR: float
    RCL_CS: float
    RCL_DI: float
    RCL_DR: float
    RCL_DS: float
    RCL_DYNVISC: float
    RCL_FAC1: float
    RCL_FAC2: float
    RCL_FZRAB: float
    RCL_FZRBB: float
    RCL_KA273: float
    RCL_KKAac: float
    RCL_KKAau: float
    RCL_KKBac: float
    RCL_KKBaun: float
    RCL_KKBauq: float
    RCL_KK_cloud_num_land: float
    RCL_KK_cloud_num_sea: float
    RCL_SCHMIDT: float
    RCL_X1I: float
    RCL_X1R: float
    RCL_X1S: float
    RCL_X2I: float
    RCL_X2R: float
    RCL_X2S: float
    RCL_X3I: float
    RCL_X3S: float
    RCL_X41: float
    RCL_X4R: float
    RCL_X4S: float
    RCOVPMIN: float
    RDENSREF: float
    RDENSWAT: float
    RDEPLIQREFDEPTH: float
    RDEPLIQREFRATE: float
    RICEHI1: float
    RICEHI2: float
    RICEINIT: float
    RKCONV: float
    RKOOPTAU: float
    RLCRITSNOW: float
    RLMIN: float
    RNICE: float
    RPECONS: float
    RPRC1: float
    RPRC2: float
    RPRECRHMAX: float
    RSNOWLIN1: float
    RSNOWLIN2: float
    RTAUMEL: float
    RTHOMO: float
    RVICE: float
    RVRAIN: float
    RVRFACTOR: float
    RVSNOW: float


class YrephliParams(BaseModel):
    LTLEVOL: bool
    LPHYLIN: bool
    LENOPERT: bool
    LEPPCFLS: bool
    LRAISANEN: bool
    RLPTRC: float
    RLPAL1: float
    RLPAL2: float
    RLPBB: float
    RLPCC: float
    RLPDD: float
    RLPMIXL: float
    RLPBETA: float
    RLPDRAG: float
    RLPEVAP: float
    RLPP00: float


class YrnclParams(BaseModel):
    LREGCL: bool = True


class YrphncParams(BaseModel):
    LEVAPLS2: bool = False


class HDF5Operator(BaseHDF5Operator):
    @lru_cache
    def get_nlev(self) -> int:
        return self.f["KLEV"][0]  # type: ignore[no-any-return]

    @lru_cache
    def get_nlon(self) -> int:
        return self.f["KLON"][0]  # type: ignore[no-any-return]

    def get_timestep(self) -> timedelta:
        return timedelta(seconds=float(self.f.get("PTSPHY", [0.0])[0]))

    def get_yoethf_params(self) -> YoethfParams:
        return self.get_params(YoethfParams)  # type: ignore[return-value]

    def get_yomcst_params(self) -> YomcstParams:
        return self.get_params(YomcstParams)  # type: ignore[return-value]

    def get_yrecldp_params(self) -> YrecldpParams:
        return self.get_params(  # type: ignore[return-value]
            YrecldpParams, get_param_name=lambda attr_name: "YRECLDP_" + attr_name
        )

    def get_yrephli_params(self) -> YrephliParams:
        return self.get_params(  # type: ignore[return-value]
            YrephliParams, get_param_name=lambda attr_name: "YREPHLI_" + attr_name
        )

    def get_yrncl_params(self) -> YrnclParams:
        return self.get_params(YrnclParams)  # type: ignore[return-value]

    def get_yrphnc_params(self) -> YrphncParams:
        return self.get_params(YrphncParams)  # type: ignore[return-value]
