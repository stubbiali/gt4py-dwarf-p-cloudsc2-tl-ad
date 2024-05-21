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

from cloudsc2_gt4py.iox import HDF5Operator
from cloudsc2_gt4py.physics.common.diagnostics import EtaLevels
from cloudsc2_gt4py.physics.adjoint.validation import SymmetryTest
from cloudsc2_gt4py.setup import get_state
from ifs_physics_common.config import GridConfig
from ifs_physics_common.grid import ComputationalGrid
from ifs_physics_common.iox import HDF5GridOperator
from ifs_physics_common.output import write_performance_to_csv, write_stencils_performance_to_csv
from ifs_physics_common.timing import timing

if TYPE_CHECKING:
    from typing import Literal, Optional

    from ifs_physics_common.config import IOConfig, PythonConfig

    from .config import DEFAULT_CONFIG, DEFAULT_IO_CONFIG
else:
    from config import DEFAULT_CONFIG, DEFAULT_IO_CONFIG


def core(config: PythonConfig, io_config: IOConfig) -> PythonConfig:
    # grid
    hdf5_operator = HDF5Operator(config.input_file, gt4py_config=config.gt4py_config)
    nx = config.num_cols or hdf5_operator.get_nlon()
    config = config.with_num_cols(nx)
    nz = hdf5_operator.get_nlev()
    computational_grid = ComputationalGrid(GridConfig(nx=nx, ny=1, nz=nz))

    # state and accumulated tendencies
    hdf5_grid_operator = HDF5GridOperator(
        config.input_file, computational_grid, gt4py_config=config.gt4py_config
    )
    state = get_state(hdf5_grid_operator)

    # timestep
    dt = hdf5_operator.get_timestep()

    # params
    yoethf_params = hdf5_operator.get_yoethf_params()
    yomcst_params = hdf5_operator.get_yomcst_params()
    yrecldp_params = hdf5_operator.get_yrecldp_params()
    yrephli_params = hdf5_operator.get_yrephli_params()
    yrncl_params = hdf5_operator.get_yrncl_params()
    yrphnc_params = hdf5_operator.get_yrphnc_params()

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
        yoethf_params=yoethf_params,
        yomcst_params=yomcst_params,
        yrecldp_params=yrecldp_params,
        yrephli_params=yrephli_params,
        yrncl_params=yrncl_params,
        yrphnc_params=yrphnc_params,
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
