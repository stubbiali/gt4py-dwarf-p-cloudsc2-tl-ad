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

from cloudsc2_gt4py.physics.common._stencils.fcttre import f_foealfa, f_foeewm
from cloudsc2_gt4py.physics.nonlinear._stencils.cuadjtqs import f_cuadjtqs_nl
from ifs_physics_common.stencil import stencil_collection


@stencil_collection("cloudsc2_nl")
def cloudsc2_nl(
    in_ap: gtscript.Field["float"],
    in_aph: gtscript.Field["float"],
    in_eta: gtscript.Field[gtscript.K, "float"],
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
    out_clc: gtscript.Field["float"],
    out_covptot: gtscript.Field["float"],
    out_fhpsl: gtscript.Field["float"],
    out_fhpsn: gtscript.Field["float"],
    out_fplsl: gtscript.Field["float"],
    out_fplsn: gtscript.Field["float"],
    out_tnd_q: gtscript.Field["float"],
    out_tnd_qi: gtscript.Field["float"],
    out_tnd_ql: gtscript.Field["float"],
    out_tnd_t: gtscript.Field["float"],
    tmp_aph_s: gtscript.Field[gtscript.IJ, "float"],
    tmp_covptot: gtscript.Field[gtscript.IJ, "float"],
    tmp_rfl: gtscript.Field[gtscript.IJ, "float"],
    tmp_sfl: gtscript.Field[gtscript.IJ, "float"],
    tmp_trpaus: gtscript.Field[gtscript.IJ, "float"],
    *,
    dt: "float",
):
    from __externals__ import (
        LDRAIN1D,
        LEVAPLS2,
        LPHYLIN,
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

    with computation(FORWARD), interval(-1, None):
        # set to zero precipitation fluxes at the top
        tmp_rfl[0, 0] = 0.0
        tmp_sfl[0, 0] = 0.0
        tmp_covptot[0, 0] = 0.0

        # set surface pressure
        tmp_aph_s[0, 0] = in_aph[0, 0, 0]

    with computation(PARALLEL), interval(0, -1):
        # first guess values for T
        t = in_t[0, 0, 0] + dt * in_tnd_cml_t[0, 0, 0]

    # eta value at tropopause
    with computation(FORWARD), interval(0, 1):
        tmp_trpaus[0, 0] = 0.1
    with computation(FORWARD), interval(0, -2):
        if in_eta[0] > 0.1 and in_eta[0] < 0.4 and t[0, 0, 0] > t[0, 0, 1]:
            tmp_trpaus[0, 0] = in_eta[0]

    with computation(FORWARD), interval(0, -1):
        # first guess values for q, ql and qi
        q = in_q[0, 0, 0] + dt * in_tnd_cml_q[0, 0, 0] + in_supsat[0, 0, 0]
        ql = in_ql[0, 0, 0] + dt * in_tnd_cml_ql[0, 0, 0]
        qi = in_qi[0, 0, 0] + dt * in_tnd_cml_qi[0, 0, 0]

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
        if LPHYLIN or LDRAIN1D:
            if t < RTT:
                fwat = 0.545 * (tanh(0.17 * (t - RLPTRC)) + 1.0)
                z3es = R3IES
                z4es = R4IES
            else:
                fwat = 1.0
                z3es = R3LES
                z4es = R4LES
            foeew = R2ES * exp(z3es * (t - RTT) / (t - z4es))
            esdp = min(foeew / in_ap[0, 0, 0], ZQMAX)
        else:
            fwat = f_foealfa(t)
            foeew = f_foeewm(t)
            esdp = foeew / in_ap[0, 0, 0]
        facw = R5LES / ((t - R4LES) ** 2.0)
        faci = R5IES / ((t - R4IES) ** 2.0)
        fac = fwat * facw + (1.0 - fwat) * faci
        dqsdtemp = fac * in_qsat[0, 0, 0] / (1.0 - RETV * esdp)
        corqs = 1.0 + cons3 * dqsdtemp

        # use clipped state
        qlim = min(q, in_qsat[0, 0, 0])

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
        if t < RTICE:
            qsat = in_qsat[0, 0, 0] * (1.8 - 0.003 * t)
        else:
            qsat = in_qsat[0, 0, 0]
        qcrit = crh2 * qsat

        # simple uniform distribution of total water from Leutreut & Li (1990)
        qt = q + ql + qi
        if qt < qcrit:
            out_clc[0, 0, 0] = 0.0
            qc = 0.0
        elif qt >= qsat:
            out_clc[0, 0, 0] = 1.0
            qc = (1.0 - scalm) * (qsat - qcrit)
        else:
            qpd = qsat - qt
            qcd = qsat - qcrit
            out_clc[0, 0, 0] = 1.0 - sqrt(qpd / (qcd - scalm * (qt - qcrit)))
            qc = (scalm * qpd + (1.0 - scalm) * qcd) * (out_clc[0, 0, 0] ** 2.0)

        # add convective component
        gdp = RG / (in_aph[0, 0, 1] - in_aph[0, 0, 0])
        lude = dt * in_lude[0, 0, 0] * gdp
        lo1 = lude >= RLMIN and in_lu[0, 0, 1] >= ZEPS2
        if lo1:
            out_clc[0, 0, 0] += (1.0 - out_clc[0, 0, 0]) * (1.0 - exp(-lude / in_lu[0, 0, 1]))
            qc += lude

        # add compensating subsidence component
        rho = in_ap[0, 0, 0] / (RD * t)
        rodqsdp = -rho * in_qsat[0, 0, 0] / (in_ap[0, 0, 0] - RETV * foeew)
        ldcp = fwat * lvdcp + (1.0 - fwat) * lsdcp
        dtdzmo = RG * (1.0 / RCPD - ldcp * rodqsdp) / (1.0 + ldcp * dqsdtemp)
        dqsdz = dqsdtemp * dtdzmo - RG * rodqsdp
        dqc = min(dt * dqsdz * (in_mfu[0, 0, 0] + in_mfd[0, 0, 0]) / rho, qc)
        qc -= dqc

        # new cloud liquid/ice contents and condensation rates (liquid/ice)
        qlwc = qc * fwat
        qiwc = qc * (1.0 - fwat)
        condl = (qlwc - ql) / dt
        condi = (qiwc - qi) / dt

        # calculate precipitation overlap
        # simple form based on Maximum Overlap
        tmp_covptot[0, 0] = max(tmp_covptot[0, 0], out_clc[0, 0, 0])
        covpclr = max(tmp_covptot[0, 0] - out_clc[0, 0, 0], 0.0)

        # melting of incoming snow
        if tmp_sfl[0, 0] != 0.0:
            cons = cons2 * dp / lfdcp
            snmlt = min(tmp_sfl[0, 0], cons * max(t - meltp2, 0.0))
            rfln = tmp_rfl[0, 0] + snmlt
            sfln = tmp_sfl[0, 0] - snmlt
            t -= snmlt / cons
        else:
            rfln = tmp_rfl[0, 0]
            sfln = tmp_sfl[0, 0]

        # diagnostic calculation of rain production from cloud liquid water
        if out_clc[0, 0, 0] > ZEPS2:
            if LEVAPLS2 or LDRAIN1D:
                lcrit = 1.9 * RCLCRIT
            else:
                lcrit = 2.0 * RCLCRIT
            cldl = qlwc / out_clc[0, 0, 0]
            dl = ckcodtl * (1.0 - exp(-((cldl / lcrit) ** 2.0)))
            prr = qlwc - out_clc[0, 0, 0] * cldl * exp(-dl)
            qlwc -= prr
        else:
            prr = 0.0

        # diagnostic calculation of snow production from cloud ice
        if out_clc[0, 0, 0] > ZEPS2:
            if LEVAPLS2 or LDRAIN1D:
                icrit = 0.0001
            else:
                icrit = 2.0 * RCLCRIT
            cldi = qiwc / out_clc[0, 0, 0]
            di = ckcodti * exp(0.025 * (t - RTT)) * (1.0 - exp(-((cldi / icrit) ** 2.0)))
            prs = qiwc - out_clc[0, 0, 0] * cldi * exp(-di)
            qiwc -= prs
        else:
            prs = 0.0

        # new precipitation (rain + snow)
        dr = cons2 * dp * (prr + prs)

        # rain fraction (different from cloud liquid water fraction!)
        if t < RTT:
            rfreeze = cons2 * dp * prr
            fwatr = 0.0
        else:
            rfreeze = 0.0
            fwatr = 1.0
        rfln += fwatr * dr
        sfln += (1.0 - fwatr) * dr

        # precipitation evaporation
        prtot = rfln + sfln
        if prtot > ZEPS2 and covpclr > ZEPS2 and (LEVAPLS2 or LDRAIN1D):
            preclr = prtot * covpclr / tmp_covptot[0, 0]

            # this is the humidity in the moisest zcovpclr region
            qe = in_qsat[0, 0, 0] - (in_qsat[0, 0, 0] - qlim) * covpclr / (
                (1.0 - out_clc[0, 0, 0]) ** 2.0
            )
            beta = (
                RG
                * RPECONS
                * (sqrt(in_ap[0, 0, 0] / tmp_aph_s[0, 0]) / 0.00509 * preclr / covpclr) ** 0.5777
            )

            # implicit solution
            b = dt * beta * (in_qsat[0, 0, 0] - qe) / (1.0 + dt * beta * corqs)

            dtgdp = dt * RG / (in_aph[0, 0, 1] - in_aph[0, 0, 0])
            dpr = min(covpclr * b / dtgdp, preclr)
            preclr -= dpr
            if preclr <= 0.0:
                tmp_covptot[0, 0] = out_clc[0, 0, 0]
            out_covptot[0, 0, 0] = tmp_covptot[0, 0]

            # warm proportion
            evapr = dpr * rfln / prtot
            rfln -= evapr

            # ice proportion
            evaps = dpr * sfln / prtot
            sfln -= evaps
        else:
            evapr = 0.0
            evaps = 0.0

        # update of T and Q tendencies due to:
        # - condensation/evaporation of cloud water/ice
        # - detrainment of convective cloud condensate
        # - evaporation of precipitation
        # - freezing of rain (impact on T only)
        dqdt = -(condl + condi) + (in_lude[0, 0, 0] + evapr + evaps) * gdp
        dtdt = (
            lvdcp * condl
            + lsdcp * condi
            - (
                lvdcp * evapr
                + lsdcp * evaps
                + in_lude[0, 0, 0] * (fwat * lvdcp + (1.0 - fwat) * lsdcp)
                - (lsdcp - lvdcp) * rfreeze
            )
            * gdp
        )

        # first guess T and Q
        t += dt * dtdt
        q += dt * dqdt
        qold = q

        # clipping of final qv
        t, q = f_cuadjtqs_nl(in_ap, t, q)

        # update rain fraction and freezing
        dq = max(qold - q, 0.0)
        dr2 = cons2 * dp * dq
        if t < RTT:
            rfreeze2 = fwat * dr2
            fwatr = 0.0
        else:
            rfreeze2 = 0.0
            fwatr = 1.0
        rn = fwatr * dr2
        sn = (1.0 - fwatr) * dr2
        condl += fwatr * dq / dt
        condi += (1.0 - fwatr) * dq / dt
        rfln += rn
        sfln += sn
        rfreeze += rfreeze2

        # calculate output tendencies
        out_tnd_q[0, 0, 0] = -(condl + condi) + (in_lude[0, 0, 0] + evapr + evaps) * gdp
        out_tnd_t[0, 0, 0] = (
            lvdcp * condl
            + lsdcp * condi
            - (
                lvdcp * evapr
                + lsdcp * evaps
                + in_lude[0, 0, 0] * (fwat * lvdcp + (1.0 - fwat) * lsdcp)
                - (lsdcp - lvdcp) * rfreeze
            )
            * gdp
        )
        out_tnd_ql[0, 0, 0] = (qlwc - ql) / dt
        out_tnd_qi[0, 0, 0] = (qiwc - qi) / dt

        # these fluxes will later be shifted one level downward
        fplsl = rfln
        fplsn = sfln

        # record rain flux for next level
        tmp_rfl[0, 0] = rfln
        tmp_sfl[0, 0] = sfln

    # enthalpy fluxes due to precipitation
    with computation(FORWARD):
        with interval(0, 1):
            out_fhpsl[0, 0, 0] = 0.0
            out_fhpsn[0, 0, 0] = 0.0
        with interval(1, None):
            out_fplsl[0, 0, 0] = fplsl[0, 0, -1]
            out_fplsn[0, 0, 0] = fplsn[0, 0, -1]
            out_fhpsl[0, 0, 0] = -out_fplsl[0, 0, 0] * RLVTT
            out_fhpsn[0, 0, 0] = -out_fplsn[0, 0, 0] * RLSTT
