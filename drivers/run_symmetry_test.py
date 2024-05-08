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
import click
from typing import TYPE_CHECKING

from cloudsc2_gt4py.physics.common.diagnostics import EtaLevels
from cloudsc2_gt4py.physics.adjoint.validation import SymmetryTest
from cloudsc2_gt4py.state import get_initial_state
from cloudsc2_gt4py.utils.iox import HDF5Reader
from ifs_physics_common.framework.grid import ComputationalGrid
from ifs_physics_common.utils.output import (
    write_performance_to_csv,
    write_stencils_performance_to_csv,
)
from ifs_physics_common.utils.timing import timing

if TYPE_CHECKING:
    from typing import Literal, Optional

    from ifs_physics_common.framework.config import IOConfig, PythonConfig

    from .config import DEFAULT_CONFIG, DEFAULT_IO_CONFIG
else:
    from config import DEFAULT_CONFIG, DEFAULT_IO_CONFIG


def core(config: PythonConfig, io_config: IOConfig) -> PythonConfig:
    # input file
    hdf5_reader = HDF5Reader(config.input_file, config.data_types)

    # grid
    nx = config.num_cols or hdf5_reader.get_nlon()
    config = config.with_num_cols(nx)
    nz = hdf5_reader.get_nlev()
    computational_grid = ComputationalGrid(nx, 1, nz)

    # state and accumulated tendencies
    state = get_initial_state(computational_grid, hdf5_reader, gt4py_config=config.gt4py_config)

    # timestep
    dt = hdf5_reader.get_timestep()

    # parameters
    yoethf_params = hdf5_reader.get_yoethf_parameters()
    yomcst_params = hdf5_reader.get_yomcst_parameters()
    yrecld_params = hdf5_reader.get_yrecld_parameters()
    yrecldp_params = hdf5_reader.get_yrecldp_parameters()
    yrephli_params = hdf5_reader.get_yrephli_parameters()
    yrncl_params = hdf5_reader.get_yrncl_parameters()
    yrphnc_params = hdf5_reader.get_yrphnc_parameters()

    # diagnose reference eta-levels
    eta_levels = EtaLevels(
        computational_grid,
        enable_checks=config.sympl_enable_checks,
        gt4py_config=config.gt4py_config,
    )
    state.update(eta_levels(state))

    # taylor test
    st = SymmetryTest(
        computational_grid,
        factor=0.01,
        kflag=1,
        lphylin=True,
        ldrain1d=False,
        yoethf_parameters=yoethf_params,
        yomcst_parameters=yomcst_params,
        yrecld_parameters=yrecld_params,
        yrecldp_parameters=yrecldp_params,
        yrephli_parameters=yrephli_params,
        yrncl_parameters=yrncl_params,
        yrphnc_parameters=yrphnc_params,
        enable_checks=config.sympl_enable_checks,
        gt4py_config=config.gt4py_config,
    )
    st(state, dt, enable_validation=True)

    config.gt4py_config.reset_exec_info()

    runtime_l = []
    for i in range(config.num_runs):
        with timing(f"run_{i}") as timer:
            st(state, dt, enable_validation=False)
        runtime_l.append(timer.get_time(f"run_{i}", units="ms"))

    runtime_mean = sum(runtime_l) / config.num_runs
    runtime_stddev = (
        sum((runtime - runtime_mean) ** 2 for runtime in runtime_l)
        / (config.num_runs - 1 if config.num_runs > 1 else config.num_runs)
    ) ** 0.5
    print(f"\nThe test completed in {runtime_mean:.3f} \u00B1 {runtime_stddev:.3f} ms.")

    if io_config.output_csv_file is not None:
        write_performance_to_csv(
            io_config.output_csv_file,
            io_config.host_name,
            config.precision,
            "ad-" + config.gt4py_config.backend,
            nx,
            config.num_threads,
            1,
            config.num_runs,
            runtime_mean,
            runtime_stddev,
            0,
            0,
        )

    return config


@click.command()
@click.option(
    "--backend",
    type=str,
    default="numpy",
    help="GT4Py backend (options: cuda, dace:cpu, dace:gpu, gt:cpu_ifirst, gt:cpu_kfirst, gt:gpu, "
    "numpy; default: numpy).",
)
@click.option(
    "--enable-checks/--disable-checks",
    is_flag=True,
    type=bool,
    default=False,
    help="Enable/disable sanity checks performed by Sympl and GT4Py (default: enabled).",
)
@click.option("--num-cols", type=int, default=1, help="Number of domain columns (default: 1).")
@click.option("--num-runs", type=int, default=1, help="Number of executions (default: 1).")
@click.option(
    "--precision",
    type=str,
    default="double",
    help="Select either `double` (default) or `single` precision.",
)
@click.option("--host-alias", type=str, default=None, help="Name of the host machine (optional).")
@click.option(
    "--output-csv-file",
    type=str,
    default=None,
    help="Path to the CSV file where writing performance counters (optional).",
)
@click.option(
    "--output-csv-file-stencils",
    type=str,
    default=None,
    help="Path to the CSV file where writing performance counters for each stencil (optional).",
)
def main(
    backend: Optional[str],
    enable_checks: bool,
    num_cols: int,
    num_runs: int,
    precision: Literal["double", "single"],
    host_alias: Optional[str],
    output_csv_file: Optional[str],
    output_csv_file_stencils: Optional[str],
) -> None:
    config = (
        DEFAULT_CONFIG.with_precision(precision)
        .with_backend(backend)
        .with_checks(enable_checks)
        .with_num_cols(num_cols)
        .with_num_runs(num_runs)
        .with_precision(precision)
    )
    io_config = DEFAULT_IO_CONFIG.with_host_name(host_alias).with_output_csv_file(output_csv_file)
    config = core(config, io_config)

    if output_csv_file_stencils is not None:
        write_stencils_performance_to_csv(
            output_csv_file_stencils,
            io_config.host_name,
            config.precision,
            "ad-" + config.gt4py_config.backend,
            config.num_cols,
            config.num_threads,
            config.num_runs,
            config.gt4py_config.exec_info,
            key_patterns=["cloudsc", "increment", "saturation"],
        )


if __name__ == "__main__":
    main()
