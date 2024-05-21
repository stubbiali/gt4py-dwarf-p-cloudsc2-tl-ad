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

from cloudsc2_gt4py.physics.adjoint._stencils.cuadjtqs import f_cuadjtqs_ad
from cloudsc2_gt4py.physics.nonlinear._stencils.cuadjtqs import f_cuadjtqs_nl
from ifs_physics_common.stencil import stencil_collection


@stencil_collection("cloudsc2_ad")
def cloudsc2_ad(
    in_ap: gtscript.Field["float"],
    in_aph: gtscript.Field["float"],
    in_clc_i: gtscript.Field["float"],
    in_covptot_i: gtscript.Field["float"],
    in_eta: gtscript.Field[gtscript.K, "float"],
    in_fhpsl_i: gtscript.Field["float"],
    in_fhpsn_i: gtscript.Field["float"],
    in_fplsl_i: gtscript.Field["float"],
    in_fplsn_i: gtscript.Field["float"],
    in_lu: gtscript.Field["float"],
    in_lude: gtscript.Field["float"],
    in_mfd: gtscript.Field["float"],
    in_mfu: gtscript.Field["float"],
    in_q: gtscript.Field["float"],
    in_qi: gtscript.Field["float"],
    in_ql: gtscript.Field["float"],
    in_qsat: gtscript.Field["float"],
    in_supsat: gtscript.Field["float"],
    in_t: gtscript.Field["float"],
    in_tnd_cml_q: gtscript.Field["float"],
    in_tnd_cml_qi: gtscript.Field["float"],
    in_tnd_cml_ql: gtscript.Field["float"],
    in_tnd_cml_t: gtscript.Field["float"],
    in_tnd_q_i: gtscript.Field["float"],
    in_tnd_qi_i: gtscript.Field["float"],
    in_tnd_ql_i: gtscript.Field["float"],
    in_tnd_t_i: gtscript.Field["float"],
    out_ap_i: gtscript.Field["float"],
    out_aph_i: gtscript.Field["float"],
    out_clc: gtscript.Field["float"],
    out_covptot: gtscript.Field["float"],
    out_fhpsl: gtscript.Field["float"],
    out_fhpsn: gtscript.Field["float"],
    out_fplsl: gtscript.Field["float"],
    out_fplsn: gtscript.Field["float"],
    out_lu_i: gtscript.Field["float"],
    out_lude_i: gtscript.Field["float"],
    out_mfd_i: gtscript.Field["float"],
    out_mfu_i: gtscript.Field["float"],
    out_q_i: gtscript.Field["float"],
    out_qi_i: gtscript.Field["float"],
    out_ql_i: gtscript.Field["float"],
    out_qsat_i: gtscript.Field["float"],
    out_supsat_i: gtscript.Field["float"],
    out_t_i: gtscript.Field["float"],
    out_tnd_cml_q_i: gtscript.Field["float"],
    out_tnd_cml_qi_i: gtscript.Field["float"],
    out_tnd_cml_ql_i: gtscript.Field["float"],
    out_tnd_cml_t_i: gtscript.Field["float"],
    out_tnd_q: gtscript.Field["float"],
    out_tnd_qi: gtscript.Field["float"],
    out_tnd_ql: gtscript.Field["float"],
    out_tnd_t: gtscript.Field["float"],
    tmp_aph_s: gtscript.Field[gtscript.IJ, "float"],
    tmp_aph_s_i: gtscript.Field[gtscript.IJ, "float"],
    tmp_covptotp: gtscript.Field[gtscript.IJ, "float"],
    tmp_klevel: gtscript.Field[gtscript.K, "int"],
    tmp_rfln: gtscript.Field[gtscript.IJ, "float"],
    tmp_rfln_i: gtscript.Field[gtscript.IJ, "float"],
    tmp_sfln: gtscript.Field[gtscript.IJ, "float"],
    tmp_sfln_i: gtscript.Field[gtscript.IJ, "float"],
    tmp_trpaus: gtscript.Field[gtscript.IJ, "float"],
    *,
    dt: "float",
):
    from __externals__ import (
        LDRAIN1D,
        LEVAPLS2,
        LREGCL,
        NLEV,
        R2ES,
        R3IES,
        R3LES,
        R4IES,
        R4LES,
        R5IES,
        R5LES,
        RCLCRIT,
        RCPD,
        RD,
        RETV,
        RG,
        RKCONV,
        RLMIN,
        RLMLT,
        RLPTRC,
        RLSTT,
        RLVTT,
        RPECONS,
        RTICE,
        RTT,
        RVTMP2,
        ZEPS1,
        ZEPS2,
        ZQMAX,
        ZSCAL,
    )

    # set to zero precipitation fluxes at the top
    with computation(FORWARD), interval(0, 1):
        tmp_covptotp[0, 0] = 0.0
        tmp_rfln[0, 0] = 0.0
        tmp_sfln[0, 0] = 0.0

    with computation(PARALLEL), interval(0, -1):
        # first guess values for T
        t = in_t[0, 0, 0] + dt * in_tnd_cml_t[0, 0, 0]
        # store trajectory arrays for adjoint
        t2 = t

    # eta value at tropopause
    with computation(FORWARD), interval(0, 1):
        tmp_trpaus[0, 0] = 0.1
    with computation(FORWARD), interval(0, -2):
        if in_eta[0] > 0.1 and in_eta[0] < 0.4 and t[0, 0, 0] > t[0, 0, 1]:
            tmp_trpaus[0, 0] = in_eta[0]

    with computation(FORWARD):
        with interval(0, -1):
            # assign data computed within the previous level
            rfl = tmp_rfln[0, 0]
            sfl = tmp_sfln[0, 0]

            # first guess values for q, ql and qi
            q = in_q[0, 0, 0] + dt * in_tnd_cml_q[0, 0, 0] + in_supsat[0, 0, 0]
            ql = in_ql[0, 0, 0] + dt * in_tnd_cml_ql[0, 0, 0]
            qi = in_qi[0, 0, 0] + dt * in_tnd_cml_qi[0, 0, 0]
            # store trajectory arrays for adjoint
            q2 = q

            # set up constants required
            ckcodtl = 2.0 * RKCONV * dt
            ckcodti = 5.0 * RKCONV * dt
            cons2 = 1.0 / (RG * dt)
            cons3 = RLVTT / RCPD
            meltp2 = RTT + 2.0

            # parameter for cloud formation
            scalm = ZSCAL * max(in_eta[0] - 0.2, ZEPS1) ** 0.2

            # thermodynamic constants
            dp = in_aph[0, 0, 1] - in_aph[0, 0, 0]
            zz = RCPD + RCPD * RVTMP2 * q
            lfdcp = RLMLT / zz
            lsdcp = RLSTT / zz
            lvdcp = RLVTT / zz

            # clear cloud and freezing arrays
            out_clc[0, 0, 0] = 0.0
            out_covptot[0, 0, 0] = 0.0

            # calculate dqs/dT correction factor
            if t < RTT:
                fwat = 0.545 * (tanh(0.17 * (t2 - RLPTRC)) + 1.0)
                z3es = R3IES
                z4es = R4IES
            else:
                fwat = 1.0
                z3es = R3LES
                z4es = R4LES
            foeew = R2ES * exp(z3es * (t2 - RTT) / (t2 - z4es))
            esdp1 = foeew / in_ap[0, 0, 0]
            esdp = min(esdp1, ZQMAX)
            facw = R5LES / (t2 - R4LES) ** 2.0
            faci = R5IES / (t2 - R4IES) ** 2.0
            fac = fwat * facw + (1.0 - fwat) * faci
            cor = 1.0 / (1.0 - RETV * esdp)
            dqsdtemp = fac * cor * in_qsat[0, 0, 0]
            corqs = 1.0 + cons3 * dqsdtemp

            # use clipped state
            qlim = min(q2, in_qsat[0, 0, 0])

            # set up critical value of humidity
            rh1 = 1.0
            rh2 = (
                0.35
                + 0.14 * ((tmp_trpaus[0, 0] - 0.25) / 0.15) ** 2.0
                + 0.04 * min(tmp_trpaus[0, 0] - 0.25, 0.0) / 0.15
            )
            rh3 = 1.0
            if in_eta[0] < tmp_trpaus[0, 0]:
                crh2 = rh3
            else:
                deta2 = 0.3
                bound1 = tmp_trpaus[0, 0] + deta2
                if in_eta[0] < bound1:
                    crh2 = rh3 + (rh2 - rh3) * (in_eta[0] - tmp_trpaus[0, 0]) / deta2
                else:
                    deta1 = 0.09 + 0.16 * (0.4 - tmp_trpaus[0, 0]) / 0.3
                    bound2 = 1.0 - deta1
                    if in_eta[0] < bound2:
                        crh2 = rh2
                    else:
                        crh2 = rh1 + (rh2 - rh1) * sqrt((1.0 - in_eta[0]) / deta1)

            # allow ice supersaturation at cold temperatures
            if t2 < RTICE:
                supsat = 1.8 - 0.003 * t2
            else:
                supsat = 1.0
            qsat = in_qsat[0, 0, 0] * supsat
            qcrit = crh2 * qsat

            # simple uniform distribution of total water from Leutreut & Li (1990)
            qt = q + ql + qi
            if qt <= qcrit:
                clc = 0.0
                qc1 = 0.0
                qcd = 0.0
                qpd = 0.0
                tmp3 = 0.0
            elif qt >= qsat:
                clc = 1.0
                qc1 = (1.0 - scalm) * (qsat - qcrit)
                qcd = 0.0
                qpd = 0.0
                tmp3 = 0.0
            else:
                qcd = qsat - qcrit
                qpd = qsat - qt
                tmp3 = sqrt(qpd / (qcd - scalm * (qt - qcrit)))
                clc = 1.0 - tmp3
                qc1 = (scalm * qpd + (1.0 - scalm) * qcd) * clc**2.0

            # add convective component
            gdp = RG / (in_aph[0, 0, 1] - in_aph[0, 0, 0])
            lude = dt * in_lude[0, 0, 0] * gdp
            lo1 = lude >= RLMIN and in_lu[0, 0, 1] >= ZEPS2
            if lo1:
                out_clc[0, 0, 0] = clc + (1.0 - clc) * (1.0 - exp(-lude / in_lu[0, 0, 1]))
                qc2 = qc1 + lude
            else:
                out_clc[0, 0, 0] = clc
                qc2 = qc1

            # add compensating subsidence component
            fac1 = 1.0 / (RD * t2)
            rho = in_ap[0, 0, 0] * fac1
            fac2 = 1.0 / (in_ap[0, 0, 0] - RETV * foeew)
            rodqsdp = -rho * in_qsat[0, 0, 0] * fac2
            ldcp = fwat * lvdcp + (1.0 - fwat) * lsdcp
            fac3 = 1.0 / (1.0 + ldcp * dqsdtemp)
            dtdzmo = RG * (1.0 / RCPD - ldcp * rodqsdp) * fac3
            dqsdz = dqsdtemp * dtdzmo - RG * rodqsdp
            fac4 = 1.0 / rho
            lo3 = dt * dqsdz * (in_mfu[0, 0, 0] + in_mfd[0, 0, 0]) * fac4 < qc2
            dqc = min(dt * dqsdz * (in_mfu[0, 0, 0] + in_mfd[0, 0, 0]) * fac4, qc2)
            qc3 = qc2 - dqc

            # new cloud liquid/ice contents and condensation rates (liquid/ice)
            qlwc1 = qc3 * fwat
            qiwc1 = qc3 * (1.0 - fwat)
            condl1 = (qlwc1 - ql) / dt
            condi1 = (qiwc1 - qi) / dt

            # calculate precipitation overlap
            # simple form based on Maximum Overlap
            covptot1 = max(tmp_covptotp[0, 0], out_clc[0, 0, 0])
            covptot = covptot1
            covpclr1 = covptot - out_clc[0, 0, 0]
            covpclr = max(covpclr1, 0.0)

            # melting of incoming snow
            if sfl != 0.0:
                cons = cons2 * dp / lfdcp
                z2s = cons * max(t2 - meltp2, 0.0)
                snmlt = min(sfl, z2s)
                tmp_rfln[0, 0] = rfl + snmlt
                tmp_sfln[0, 0] = sfl - snmlt
                t = t2 - snmlt / cons
            else:
                tmp_rfln[0, 0] = rfl
                tmp_sfln[0, 0] = sfl

            # diagnostic calculation of rain production from cloud liquid water
            if out_clc[0, 0, 0] > ZEPS2:
                if LEVAPLS2 or LDRAIN1D:
                    lcrit = 1.9 * RCLCRIT
                else:
                    lcrit = 2.0 * RCLCRIT
                cldl = qlwc1 / out_clc[0, 0, 0]
                ltmp1 = exp(-((cldl / lcrit) ** 2.0))
                dl = ckcodtl * (1.0 - ltmp1)
                ltmp2 = exp(-dl)
                qlnew = out_clc[0, 0, 0] * cldl * ltmp2
                prr = qlwc1 - qlnew
                qlwc = qlwc1 - prr
            else:
                prr = 0.0
                qlwc = qlwc1

            # diagnostic calculation of snow production from cloud ice
            if out_clc[0, 0, 0] > ZEPS2:
                if LEVAPLS2 or LDRAIN1D:
                    icrit = 0.0001
                else:
                    icrit = 2.0 * RCLCRIT
                cldi = qiwc1 / out_clc[0, 0, 0]
                itmp11 = exp(-((cldi / icrit) ** 2.0))
                itmp12 = exp(0.025 * (t - RTT))
                di = ckcodti * itmp12 * (1.0 - itmp11)
                itmp2 = exp(-di)
                qinew = out_clc[0, 0, 0] * cldi * itmp2
                prs = qiwc1 - qinew
                qiwc = qiwc1 - prs
            else:
                prs = 0.0
                qiwc = qiwc1

            # new precipitation (rain + snow)
            dr1 = cons2 * dp * (prr + prs)

            # rain fraction (different from cloud liquid water fraction!)
            if t < RTT:
                rfreeze1 = cons2 * dp * prr
                fwatr1 = 0.0
            else:
                rfreeze1 = 0.0
                fwatr1 = 1.0
            tmp_rfln[0, 0] += fwatr1 * dr1
            tmp_sfln[0, 0] += (1.0 - fwatr1) * dr1
            # store trajectory for adjoint
            rfln2 = tmp_rfln[0, 0]
            sfln2 = tmp_sfln[0, 0]

            # precipitation evaporation
            prtot = tmp_rfln[0, 0] + tmp_sfln[0, 0]
            if prtot > ZEPS2 and covpclr > ZEPS2 and (LEVAPLS2 or LDRAIN1D):
                preclr1 = prtot * covpclr / covptot1

                # this is the humidity in the moistest zcovpclr region
                qe = (
                    in_qsat[0, 0, 0]
                    - (in_qsat[0, 0, 0] - qlim) * covpclr / (1.0 - out_clc[0, 0, 0]) ** 2.0
                )
                beta = (
                    RG
                    * RPECONS
                    * (sqrt(in_ap[0, 0, 0] / tmp_aph_s[0, 0]) / 0.00509 * preclr1 / covpclr)
                    ** 0.5777
                )

                # implicit solution
                b = dt * beta * (in_qsat[0, 0, 0] - qe) / (1.0 + dt * beta * corqs)

                dtgdp = dt * RG / (in_aph[0, 0, 1] - in_aph[0, 0, 0])
                dpr1 = covpclr * b / dtgdp
                dpr = min(dpr1, preclr1)

                # take away from clear sky flux
                preclr = preclr1 - dpr
                if preclr <= 0.0:
                    covptot = out_clc[0, 0, 0]
                out_covptot[0, 0, 0] = covptot

                # warm proportion
                evapr = dpr * rfln2 / prtot
                tmp_rfln[0, 0] -= evapr

                # ice proportion
                evaps = dpr * sfln2 / prtot
                tmp_sfln[0, 0] -= evaps
            else:
                evapr = 0.0
                evaps = 0.0

            # update of T and Q tendencies due to:
            # - condensation/evaporation of cloud water/ice
            # - detrainment of convective cloud condensate
            # - evaporation of precipitation
            # - freezing of rain (impact on T only)
            dqdt = -(condl1 + condi1) + (in_lude[0, 0, 0] + evapr + evaps) * gdp
            dtdt = (
                lvdcp * condl1
                + lsdcp * condi1
                - (
                    lvdcp * evapr
                    + lsdcp * evaps
                    + in_lude[0, 0, 0] * (fwat * lvdcp + (1.0 - fwat) * lsdcp)
                    - (lsdcp - lvdcp) * rfreeze1
                )
                * gdp
            )

            # first guess T and Q
            t3 = t + dt * dtdt
            q = q2 + dt * dqdt
            told = t3
            qold = q
            qold1 = q

            # clipping of final qv
            t, q = f_cuadjtqs_nl(in_ap, t3, q)

            # update rain fraction and freezing
            dq = max(qold1 - q, 0.0)
            dr2 = cons2 * dp * dq
            if t3 < RTT:
                rfreeze2 = fwat * dr2
                fwatr2 = 0.0
            else:
                rfreeze2 = 0.0
                fwatr2 = 1.0
            rn = fwatr2 * dr2
            sn = (1.0 - fwatr2) * dr2
            condl2 = condl1 + fwatr2 * dq / dt
            condi2 = condi1 + (1.0 - fwatr2) * dq / dt
            tmp_rfln[0, 0] += rn
            tmp_sfln[0, 0] += sn
            rfreeze3 = rfreeze1 + rfreeze2

            # calculate output tendencies
            out_tnd_q[0, 0, 0] = -(condl2 + condi2) + (in_lude[0, 0, 0] + evapr + evaps) * gdp
            out_tnd_t[0, 0, 0] = (
                lvdcp * condl2
                + lsdcp * condi2
                - (
                    lvdcp * evapr
                    + lsdcp * evaps
                    + in_lude[0, 0, 0] * (fwat * lvdcp + (1.0 - fwat) * lsdcp)
                    - (lsdcp - lvdcp) * rfreeze3
                )
                * gdp
            )
            out_tnd_ql[0, 0, 0] = (qlwc - ql) / dt
            out_tnd_qi[0, 0, 0] = (qiwc - qi) / dt

            # record data for next level
            tmp_covptotp[0, 0] = covptot
        with interval(-1, None):
            # assign data computed within the previous level
            rfl = tmp_rfln[0, 0]
            sfl = tmp_sfln[0, 0]

    # enthalpy fluxes due to precipitation
    with computation(FORWARD):
        with interval(0, 1):
            out_fplsl[0, 0, 0] = 0.0
            out_fplsn[0, 0, 0] = 0.0
            out_fhpsl[0, 0, 0] = 0.0
            out_fhpsn[0, 0, 0] = 0.0
        with interval(1, None):
            out_fplsl[0, 0, 0] = rfl
            out_fplsn[0, 0, 0] = sfl
            out_fhpsl[0, 0, 0] = -out_fplsl[0, 0, 0] * RLVTT
            out_fhpsn[0, 0, 0] = -out_fplsn[0, 0, 0] * RLSTT

    # === adjoint computations

    with computation(BACKWARD), interval(...):
        # enthalpy fluxes due to precipitation
        in_fplsn_i[0, 0, 0] -= in_fhpsn_i[0, 0, 0] * RLSTT
        in_fhpsn_i[0, 0, 0] = 0.0
        in_fplsl_i[0, 0, 0] -= in_fhpsl_i[0, 0, 0] * RLVTT
        in_fhpsl_i[0, 0, 0] = 0.0

    with computation(BACKWARD):
        with interval(-1, None):
            covptot_i = 0.0
            rfl_i = 0.0
            sfl_i = 0.0
            tmp_aph_s_i[0, 0] = 0.0
            tmp_rfln_i[0, 0] = 0.0
            tmp_sfln_i[0, 0] = 0.0
        with interval(0, -1):
            # set up constants
            ckcodtla = ckcodtl / 100.0
            ckcodtia = ckcodti / 100.0

            # incrementation of T and q, and fluxes swap
            tmp_rfln_i[0, 0] += rfl_i[0, 0, 1] + in_fplsl_i[0, 0, 1]
            tmp_sfln_i[0, 0] += sfl_i[0, 0, 1] + in_fplsn_i[0, 0, 1]

            # qice tendency
            out_qi_i[0, 0, 0] = -in_tnd_qi_i[0, 0, 0] / dt
            qiwc_i = in_tnd_qi_i[0, 0, 0] / dt
            in_tnd_qi_i[0, 0, 0] = 0.0

            # qliq tendency
            out_ql_i[0, 0, 0] = -in_tnd_ql_i[0, 0, 0] / dt
            qlwc_i = in_tnd_ql_i[0, 0, 0] / dt
            in_tnd_ql_i[0, 0, 0] = 0.0

            # T tendency
            gdp_i = -in_tnd_t_i[0, 0, 0] * (
                lvdcp * evapr
                + lsdcp * evaps
                + in_lude[0, 0, 0] * (fwat * lvdcp + (1.0 - fwat) * lsdcp)
                - (lsdcp - lvdcp) * rfreeze3
            )
            condl_i = in_tnd_t_i[0, 0, 0] * lvdcp
            condi_i = in_tnd_t_i[0, 0, 0] * lsdcp
            evapr_i = -in_tnd_t_i[0, 0, 0] * lvdcp * gdp
            evaps_i = -in_tnd_t_i[0, 0, 0] * lsdcp * gdp
            lvdcp_i = in_tnd_t_i[0, 0, 0] * (condl2 - evapr * gdp)
            lsdcp_i = in_tnd_t_i[0, 0, 0] * (condi2 - evaps * gdp)
            out_lude_i[0, 0, 0] -= in_tnd_t_i[0, 0, 0] * gdp * (fwat * lvdcp + (1.0 - fwat) * lsdcp)
            lvdcp_i -= in_tnd_t_i[0, 0, 0] * in_lude[0, 0, 0] * gdp * fwat
            lsdcp_i -= in_tnd_t_i[0, 0, 0] * in_lude[0, 0, 0] * gdp * (1.0 - fwat)
            fwat_i = -in_tnd_t_i[0, 0, 0] * in_lude[0, 0, 0] * gdp * (lvdcp - lsdcp)
            lvdcp_i -= in_tnd_t_i[0, 0, 0] * rfreeze3 * gdp
            lsdcp_i += in_tnd_t_i[0, 0, 0] * rfreeze3 * gdp
            rfreeze_i = in_tnd_t_i[0, 0, 0] * (lsdcp - lvdcp) * gdp
            in_tnd_t_i[0, 0, 0] = 0.0

            # q tendency
            gdp_i += in_tnd_q_i[0, 0, 0] * (in_lude[0, 0, 0] + evapr + evaps)
            out_lude_i[0, 0, 0] += in_tnd_q_i[0, 0, 0] * gdp
            evapr_i += in_tnd_q_i[0, 0, 0] * gdp
            evaps_i += in_tnd_q_i[0, 0, 0] * gdp
            condl_i -= in_tnd_q_i[0, 0, 0]
            condi_i -= in_tnd_q_i[0, 0, 0]
            in_tnd_q_i[0, 0, 0] = 0.0

            # # T and q tendency
            # gdp_i = -in_tnd_t_i[0, 0, 0] * (
            #     lvdcp * evapr
            #     + lsdcp * evaps
            #     + in_lude[0, 0, 0] * (fwat * lvdcp + (1.0 - fwat) * lsdcp)
            #     - (lsdcp - lvdcp) * rfreeze3
            # ) + in_tnd_q_i[0, 0, 0] * (in_lude[0, 0, 0] + evapr + evaps)
            # condl_i = in_tnd_t_i[0, 0, 0] * lvdcp - in_tnd_q_i[0, 0, 0]
            # condi_i = in_tnd_t_i[0, 0, 0] * lsdcp - in_tnd_q_i[0, 0, 0]
            # evapr_i = -in_tnd_t_i[0, 0, 0] * lvdcp * gdp + in_tnd_q_i[0, 0, 0] * gdp
            # evaps_i = -in_tnd_t_i[0, 0, 0] * lsdcp * gdp + in_tnd_q_i[0, 0, 0] * gdp
            # lvdcp_i = in_tnd_t_i[0, 0, 0] * (condl2 - gdp * (evapr + in_lude[0, 0, 0] * fwat + rfreeze3))
            # lsdcp_i = in_tnd_t_i[0, 0, 0] * (condi2 - gdp * (evaps + in_lude[0, 0, 0] * (1.0 - fwat) - rfreeze3))
            # out_lude_i[0, 0, 0] = gdp * (
            #     -in_tnd_t_i[0, 0, 0] * (fwat * lvdcp + (1.0 - fwat) * lsdcp) + in_tnd_q_i[0, 0, 0]
            # )
            # fwat_i = -in_tnd_t_i[0, 0, 0] * in_lude[0, 0, 0] * gdp * (lvdcp - lsdcp)
            # rfreeze_i = in_tnd_t_i[0, 0, 0] * (lsdcp - lvdcp) * gdp
            # in_tnd_q_i[0, 0, 0] = 0.0
            # in_tnd_t_i[0, 0, 0] = 0.0

            # clipping of final qv
            rn_i = tmp_rfln_i[0, 0]
            sn_i = tmp_sfln_i[0, 0]

            # note: the extra condensation due to the adjustment goes directly
            # to precipitation
            dq_i = (fwatr2 * condl_i + (1.0 - fwatr2) * condi_i) / dt
            # fwatr_i = (condl_i - condi_i) * dq / dt + dr2 * (rn_i - sn_i)
            dr2_i = fwatr2 * rn_i + (1.0 - fwatr2) * sn_i

            # update rain fraction and freezing
            # note: impact of new temperature out_t_i on fwat_i is neglected here
            if t3 < RTT:
                fwat_i += dr2 * rfreeze_i
                dr2_i += fwat * rfreeze_i

            dq_i += cons2 * dp * dr2_i
            dp_i = cons2 * dq * dr2_i

            if qold1 >= q:
                # regularization
                if LREGCL:
                    dq_i *= 0.7
                qold_i = dq_i
                out_q_i[0, 0, 0] = -dq_i
            else:
                qold_i = 0.0
                out_q_i[0, 0, 0] = 0.0

            out_ap_i[0, 0, 0] = 0.0
            out_t_i[0, 0, 0] = 0.0
            out_ap_i[0, 0, 0], told, out_t_i[0, 0, 0], qold, out_q_i[0, 0, 0] = f_cuadjtqs_ad(
                in_ap, out_ap_i, told, out_t_i, qold, out_q_i
            )

            # first guess T and q
            out_q_i[0, 0, 0] += qold_i
            dqdt_i = dt * out_q_i[0, 0, 0]
            dtdt_i = dt * out_t_i[0, 0, 0]

            # incrementation of T and q
            # T tendency
            gdp_i -= dtdt_i * (
                lvdcp * evapr
                + lsdcp * evaps
                + in_lude[0, 0, 0] * (fwat * lvdcp + (1.0 - fwat) * lsdcp)
                - (lsdcp - lvdcp) * rfreeze1
            )
            condl_i += dtdt_i * lvdcp
            condi_i += dtdt_i * lsdcp
            evapr_i -= dtdt_i * lvdcp * gdp
            evaps_i -= dtdt_i * lsdcp * gdp
            lvdcp_i += dtdt_i * (condl1 - evapr * gdp)
            lsdcp_i += dtdt_i * (condi1 - evaps * gdp)
            out_lude_i[0, 0, 0] -= dtdt_i * gdp * (fwat * lvdcp + (1.0 - fwat) * lsdcp)
            lvdcp_i -= dtdt_i * in_lude[0, 0, 0] * gdp * fwat
            lsdcp_i -= dtdt_i * in_lude[0, 0, 0] * gdp * (1.0 - fwat)
            fwat_i -= dtdt_i * in_lude[0, 0, 0] * gdp * (lvdcp - lsdcp)
            lvdcp_i -= dtdt_i * rfreeze1 * gdp
            lsdcp_i += dtdt_i * rfreeze1 * gdp
            rfreeze_i += dtdt_i * (lsdcp - lvdcp) * gdp

            # q tendency
            gdp_i += dqdt_i * (in_lude[0, 0, 0] + evapr + evaps)
            out_lude_i[0, 0, 0] += dqdt_i * gdp
            evapr_i += dqdt_i * gdp
            evaps_i += dqdt_i * gdp
            condl_i -= dqdt_i
            condi_i -= dqdt_i

            if prtot > ZEPS2 and covpclr > ZEPS2 and (LEVAPLS2 or LDRAIN1D):
                # ice proportion
                evaps_i -= tmp_sfln_i[0, 0]
                tmp_sfln_i[0, 0] += dpr * evaps_i / prtot
                dpr_i = sfln2 * evaps_i / prtot
                prtot_i = -dpr * sfln2 * evaps_i / prtot**2.0

                # warm proportion
                evapr_i -= tmp_rfln_i[0, 0]
                tmp_rfln_i[0, 0] += dpr * evapr_i / prtot
                dpr_i += rfln2 * evapr_i / prtot
                prtot_i -= dpr * rfln2 * evapr_i / prtot**2.0

                # take away from clear sky flux
                covptot_i = covptot_i[0, 0, 1] + in_covptot_i[0, 0, 0]
                in_covptot_i[0, 0, 0] = 0.0
                if preclr <= 0:
                    in_clc_i[0, 0, 0] += covptot_i
                    covptot_i = 0.0

                if dpr1 > preclr1:
                    preclr_i = dpr_i
                    dpr_i = 0.0
                else:
                    preclr_i = 0.0

                b_i = covpclr * dpr_i / dtgdp
                covpclr_i = b * dpr_i / dtgdp
                dtgdp_i = -covpclr * b * dpr_i / dtgdp**2.0
                daph_i = dt * RG * dtgdp_i / (in_aph[0, 0, 1] - in_aph[0, 0, 0])

                # implicit solution
                tmp1 = 1.0 + dt * beta * corqs
                beta_i = (
                    dt * (in_qsat[0, 0, 0] - qe) * b_i / tmp1
                    - (dt**2.0) * beta * (in_qsat[0, 0, 0] - qe) * corqs * b_i / tmp1**2.0
                )
                out_qsat_i[0, 0, 0] = dt * beta * b_i / tmp1
                qe_i = -dt * beta * b_i / tmp1
                corqs_i = -(dt**2.0) * beta * (in_qsat[0, 0, 0] - qe) * beta * b_i / tmp1**2.0

                # this is the humidity in the moistest covpclr region
                xx = (
                    0.5777
                    * (RG * RPECONS / 0.00509)
                    * (0.00509 * covpclr / (preclr1 * sqrt(in_ap[0, 0, 0] / tmp_aph_s[0, 0])))
                    ** 0.4223
                )
                preclr_i += xx * sqrt(in_ap[0, 0, 0] / tmp_aph_s[0, 0]) * beta_i / covpclr
                out_ap_i[0, 0, 0] += (
                    0.5 * xx * preclr1 * beta_i / (covpclr * sqrt(in_ap[0, 0, 0] * tmp_aph_s[0, 0]))
                )
                tmp_aph_s_i[0, 0] -= (
                    0.5
                    * xx
                    * preclr1
                    * sqrt(in_ap[0, 0, 0] / tmp_aph_s[0, 0])
                    * beta_i
                    / (covpclr * tmp_aph_s[0, 0])
                )
                covpclr_i += (
                    -(xx * preclr1 * sqrt(in_ap[0, 0, 0] / tmp_aph_s[0, 0]) * beta_i / covpclr**2.0)
                    - (in_qsat[0, 0, 0] - qlim) * qe_i / (1.0 - out_clc[0, 0, 0]) ** 2.0
                ) + prtot * preclr_i / covptot1
                out_qsat_i[0, 0, 0] += qe_i - covpclr * qe_i / (1.0 - out_clc[0, 0, 0]) ** 2.0
                qlim_i = covpclr * qe_i / (1.0 - out_clc[0, 0, 0]) ** 2.0
                in_clc_i[0, 0, 0] -= (
                    2.0
                    * (in_qsat[0, 0, 0] - qlim)
                    * covpclr
                    * qe_i
                    / (1.0 - out_clc[0, 0, 0]) ** 3.0
                )
                prtot_i += covpclr * preclr_i / covptot1
                covptot_i -= prtot * covpclr * preclr_i / covptot1**2.0
            else:
                corqs_i = 0.0
                covpclr_i = 0.0
                covptot_i = 0.0
                in_covptot_i[0, 0, 0] = 0.0
                daph_i = 0.0
                out_aph_i[0, 0, 0] = 0.0
                out_qsat_i[0, 0, 0] = 0.0
                prtot_i = 0.0
                qlim_i = 0.0

            # new precipitation
            tmp_rfln_i[0, 0] += prtot_i
            tmp_sfln_i[0, 0] += prtot_i
            # fwatr_i = dr1 * (tmp_rfln_i[0, 0] - tmp_sfln_i[0, 0])
            dr_i = fwatr1 * tmp_rfln_i[0, 0] + (1.0 - fwatr1) * tmp_sfln_i[0, 0]

            # update rain fraction and freezing
            # note: impact of new temperature out_t_i on fwat_i in neglected here
            if t < RTT:
                dp_i += rfreeze_i * cons2 * prr
                prr_i = rfreeze_i * cons2 * dp
            else:
                prr_i = 0.0
            prr_i += cons2 * dp * dr_i
            prs_i = cons2 * dp * dr_i
            dp_i += cons2 * (prr + prs) * dr_i

            if out_clc[0, 0, 0] > ZEPS2:
                # diagnostic calculation of rain production from cloud ice
                if LEVAPLS2 or LDRAIN1D:
                    icrit = 0.0001
                else:
                    icrit = 2.0 * RCLCRIT
                prs_i -= qiwc_i
                qiwc_i += prs_i
                qinew_i = -prs_i
                in_clc_i[0, 0, 0] += qinew_i * cldi * itmp2
                cldi_i = qinew_i * out_clc[0, 0, 0] * itmp2
                di_i = -qinew_i * out_clc[0, 0, 0] * cldi * itmp2

                # regularization
                if LREGCL:
                    itmp4 = ckcodtia
                else:
                    itmp4 = ckcodti
                out_t_i[0, 0, 0] += 0.025 * itmp4 * itmp12 * (1.0 - itmp11) * di_i
                cldi_i += 2.0 * itmp4 * itmp12 * itmp11 * cldi * di_i / icrit**2.0

                qiwc_i += cldi_i / out_clc[0, 0, 0]
                in_clc_i[0, 0, 0] -= qiwc1 * cldi_i / out_clc[0, 0, 0] ** 2.0

                # diagnostic calculation of rain production from cloud liquid water
                if LEVAPLS2 or LDRAIN1D:
                    lcrit = 1.9 * RCLCRIT
                else:
                    lcrit = 2.0 * RCLCRIT
                prr_i -= qlwc_i
                qlwc_i += prr_i
                qlnew_i = -prr_i
                in_clc_i[0, 0, 0] += qlnew_i * cldl * ltmp2
                cldl_i = qlnew_i * out_clc[0, 0, 0] * ltmp2
                dl_i = -qlnew_i * out_clc[0, 0, 0] * cldl * ltmp2

                # regularization
                if LREGCL:
                    ltmp4 = ckcodtla
                else:
                    ltmp4 = ckcodtl
                cldl_i += 2.0 * ltmp4 * ltmp1 * cldl * dl_i / lcrit**2.0

                qlwc_i += cldl_i / out_clc[0, 0, 0]
                in_clc_i[0, 0, 0] -= qlwc1 * cldl_i / out_clc[0, 0, 0] ** 2.0

            # melting of incoming snow
            if sfl != 0.0:
                snmlt_i = -out_t_i[0, 0, 0] / cons + tmp_rfln_i[0, 0] - tmp_sfln_i[0, 0]
                cons_i = out_t_i[0, 0, 0] * snmlt / cons**2.0
                rfl_i = tmp_rfln_i[0, 0]
                tmp_rfln_i[0, 0] = 0.0
                sfl_i = tmp_sfln_i[0, 0]
                tmp_sfln_i[0, 0] = 0.0

                if sfl <= z2s:
                    sfl_i += snmlt_i
                    z2s_i = 0.0
                else:
                    z2s_i = snmlt_i

                if t2 > meltp2:
                    out_t_i[0, 0, 0] += cons * z2s_i
                    cons_i += (t2 - meltp2) * z2s_i

                dp_i += cons2 * cons_i / lfdcp
                lfdcp_i = -cons2 * dp * cons_i / lfdcp**2.0
            else:
                lfdcp_i = 0.0

            # calculate precipitation overlap
            # simple form based on Maximum Overlap
            if covpclr1 < 0.0:
                covpclr_i = 0.0
            covptot_i += covpclr_i
            in_clc_i[0, 0, 0] -= covpclr_i

            if out_clc[0, 0, 0] > covptot:
                in_clc_i[0, 0, 0] += covptot_i
                covptot_i = 0.0

            # new cloud liquid/ice contents and condensation rates (liquid/ice)
            qiwc_i += condi_i / dt
            out_qi_i[0, 0, 0] -= condi_i / dt
            qlwc_i += condl_i / dt
            out_ql_i[0, 0, 0] -= condl_i / dt
            qc_i = fwat * qlwc_i + (1.0 - fwat) * qiwc_i
            fwat_i += qc3 * (qlwc_i - qiwc_i)

            # add compensating subsidence component
            dqc_i = -qc_i
            if lo3:
                if LREGCL:
                    # regularization
                    dqc_i *= 0.1
                dqsdz_i = dt * dqc_i * (in_mfd[0, 0, 0] + in_mfu[0, 0, 0]) * fac4
                out_mfd_i[0, 0, 0] = dt * dqc_i * dqsdz * fac4
                out_mfu_i[0, 0, 0] = dt * dqc_i * dqsdz * fac4
                rho_i = -dqc_i * dqc * fac4
            else:
                qc_i += dqc_i
                dqsdz_i = 0.0
                out_mfd_i[0, 0, 0] = 0.0
                out_mfu_i[0, 0, 0] = 0.0
                rho_i = 0.0

            dtdzmo_i = dqsdz_i * dqsdtemp
            dqsdtemp_i = dqsdz_i * dtdzmo - dtdzmo * dtdzmo_i * ldcp * fac3
            rodqsdp_i = -RG * (dqsdz_i + dtdzmo_i * ldcp * fac3)
            ldcp_i = -dtdzmo_i * (RG * rodqsdp + dtdzmo * dqsdtemp) * fac3
            fwat_i += ldcp_i * (lvdcp - lsdcp)
            lvdcp_i += fwat * ldcp_i
            lsdcp_i += (1.0 - fwat) * ldcp_i
            rho_i -= rodqsdp_i * in_qsat[0, 0, 0] * fac2
            out_qsat_i[0, 0, 0] -= rodqsdp_i * rho * fac2
            out_ap_i[0, 0, 0] += rodqsdp_i * rho * in_qsat[0, 0, 0] * fac2**2.0 + rho_i * fac1
            foeew_i = -RETV * rodqsdp_i * rho * in_qsat[0, 0, 0] * fac2**2.0
            out_t_i[0, 0, 0] -= rho_i * in_ap[0, 0, 0] * fac1 / t2

            # add convective component
            if tmp_klevel[0] < NLEV - 1 and lude >= RLMIN and in_lu[0, 0, 1] >= ZEPS2:
                lude_i = (
                    qc_i
                    + (1.0 - clc) / in_lu[0, 0, 1] * exp(-lude / in_lu[0, 0, 1]) * in_clc_i[0, 0, 0]
                )
                dlu_i = (
                    (1.0 - clc)
                    * lude
                    / in_lu[0, 0, 1] ** 2.0
                    * exp(-lude / in_lu[0, 0, 1])
                    * in_clc_i[0, 0, 0]
                )
                in_clc_i[0, 0, 0] *= 1.0 - (1.0 - exp(-lude / in_lu[0, 0, 1]))
            else:
                lude_i = 0.0
                dlu_i = 0.0

            out_lude_i[0, 0, 0] += dt * gdp * lude_i
            gdp_i += dt * in_lude[0, 0, 0] * lude_i
            daph_i += RG * gdp_i / (in_aph[0, 0, 1] - in_aph[0, 0, 0]) ** 2.0

            # simple uniform distribution of total water from Letreut & Li (1990)
            qt_i = 0.0
            if qt < qcrit:
                qpd_i = 0.0
                qcd_i = 0.0
                qsat_i = 0.0
                qcrit_i = 0.0
            elif qt >= qsat:
                qpd_i = 0.0
                qcd_i = 0.0
                qsat_i = (1.0 - scalm) * qc_i
                qcrit_i = -(1.0 - scalm) * qc_i
            else:
                qpd_i = scalm * qc_i * clc**2.0
                qcd_i = (1.0 - scalm) * qc_i * clc**2.0
                in_clc_i[0, 0, 0] += 2.0 * (scalm * qpd + (1.0 - scalm) * qcd) * clc * qc_i

                if LREGCL:
                    # regularization of cloud fraction
                    rat = qpd / qcd
                    yyy = min(
                        0.3, 3.5 * sqrt(rat * (1.0 - scalm * (1.0 - rat)) ** 3.0) / (1.0 - scalm)
                    )
                    in_clc_i[0, 0, 0] *= yyy

                qpd_i -= 0.5 / tmp3 * in_clc_i[0, 0, 0] / (qcd - scalm * (qt - qcrit))
                qcd_i += 0.5 / tmp3 * qpd * in_clc_i[0, 0, 0] / (qcd - scalm * (qt - qcrit)) ** 2.0
                qt_i = (
                    -0.5
                    / tmp3
                    * (qpd * scalm * in_clc_i[0, 0, 0])
                    / (qcd - scalm * (qt - qcrit)) ** 2.0
                ) - qpd_i
                qcrit_i = (
                    0.5
                    / tmp3
                    * (qpd * scalm * in_clc_i[0, 0, 0])
                    / (qcd - scalm * (qt - qcrit)) ** 2.0
                ) - qcd_i
                qsat_i = qcd_i + qpd_i

            in_clc_i[0, 0, 0] = 0.0
            out_q_i[0, 0, 0] += qt_i
            out_ql_i[0, 0, 0] += qt_i
            out_qi_i[0, 0, 0] += qt_i

            # set up critical value of humidity
            qsat_i += qcrit_i * crh2
            out_qsat_i[0, 0, 0] += qsat_i * supsat
            supsat_i = qsat_i * in_qsat[0, 0, 0]

            # allow ice supersaturation at cold temperatures
            if t2 < RTICE:
                out_t_i[0, 0, 0] -= 0.003 * supsat_i

            # use clipped state
            if q2 > in_qsat[0, 0, 0]:
                out_qsat_i[0, 0, 0] += qlim_i
            else:
                out_q_i[0, 0, 0] += qlim_i

            # calculate dqs/dT correction factor
            dqsdtemp_i += cons3 * corqs_i
            out_qsat_i[0, 0, 0] += fac * cor * dqsdtemp_i
            cor_i = fac * in_qsat[0, 0, 0] * dqsdtemp_i
            fac_i = cor * in_qsat[0, 0, 0] * dqsdtemp_i
            esdp_i = RETV * cor_i * cor**2.0
            facw_i = fwat * fac_i
            faci_i = (1.0 - fwat) * fac_i
            fwat_i += (facw - faci) * fac_i
            out_t_i[0, 0, 0] -= 2.0 * (
                R5IES * faci_i / (t2 - R4IES) ** 3.0 + R5LES * facw_i / (t2 - R4LES) ** 3.0
            )

            if esdp1 > ZQMAX:
                esdp_i = 0.0
            foeew_i += esdp_i / in_ap[0, 0, 0]
            out_ap_i[0, 0, 0] -= esdp_i * foeew / in_ap[0, 0, 0] ** 2.0

            if t2 < RTT:
                z3es = R3IES
                z4es = R4IES
            else:
                z3es = R3LES
                z4es = R4LES
            out_t_i[0, 0, 0] += z3es * (RTT - z4es) * foeew_i * foeew / (t2 - z4es) ** 2.0

            if t2 < RTT:
                out_t_i[0, 0, 0] += 0.545 * 0.17 * fwat_i / cosh(0.17 * (t2 - RLPTRC)) ** 2.0

    # apply corrections to staggered fields
    with computation(BACKWARD):
        with interval(-1, None):
            in_fplsl_i[0, 0, 0] = 0.0
            in_fplsn_i[0, 0, 0] = 0.0
            tmp_aph_s_i[0, 0] += -daph_i[0, 0, -1] + dp_i[0, 0, -1]
            out_aph_i[0, 0, 0] = tmp_aph_s_i[0, 0]
            out_lu_i[0, 0, 0] = -dlu_i[0, 0, -1]
        with interval(1, -1):
            in_fplsl_i[0, 0, 0] = 0.0
            in_fplsn_i[0, 0, 0] = 0.0
            out_aph_i[0, 0, 0] = daph_i[0, 0, 0] - daph_i[0, 0, -1] - dp_i[0, 0, 0] + dp_i[0, 0, -1]
            out_lu_i[0, 0, 0] = -dlu_i[0, 0, -1]
        with interval(0, 1):
            in_fplsl_i[0, 0, 0] = 0.0
            in_fplsn_i[0, 0, 0] = 0.0
            out_aph_i[0, 0, 0] = daph_i - dp_i
            out_lu_i[0, 0, 0] = 0.0

    with computation(BACKWARD):
        with interval(0, -1):
            zz = RLVTT * lvdcp_i + RLSTT * lsdcp_i + RLMLT * lfdcp_i
            out_q_i[0, 0, 0] += -zz * RCPD * RVTMP2 / (RCPD + RCPD * RVTMP2 * q) ** 2.0
            out_supsat_i[0, 0, 0] = dt * out_q_i[0, 0, 0]
            out_tnd_cml_t_i[0, 0, 0] = dt * out_t_i[0, 0, 0]
            out_tnd_cml_q_i[0, 0, 0] = dt * out_q_i[0, 0, 0]
            out_tnd_cml_ql_i[0, 0, 0] = dt * out_ql_i[0, 0, 0]
            out_tnd_cml_qi_i[0, 0, 0] = dt * out_qi_i[0, 0, 0]
