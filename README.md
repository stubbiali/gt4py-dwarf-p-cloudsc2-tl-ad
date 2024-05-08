[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# `cloudsc2_gt4py`: GT4Py-based implementation of the ECMWF CLOUDSC2 dwarf

This repository contains the Python rewrite of the
[CLOUDSC2 microphysics dwarf](https://github.com/ecmwf-ifs/dwarf-p-cloudsc2-tl-ad) based on
[GT4Py](https://github.com/GridTools/gt4py.git). The code is bundled as an installable
package called `cloudsc2_gt4py`, whose source code is placed under `src/`.

We strongly recommend installing the package in an isolated virtual environment:

```shell
# create the virtual environment under `venv/`
$ python -m venv venv

# activate the virtual environment
$ . venv/bin/activate

# upgrade base packages
(venv) $ pip install --upgrade pip setuptools wheel

# install cloudsc2_gt4py in editable mode
(venv) $ pip install -e .[<optional dependencies>]
```

`<optional dependencies>` can be any of the following strings, or a comma-separated list of them:

* `dev`: get a full-fledged development installation;
* `gpu`: enable GPU support by installing CuPy from source;
* `gpu-cuda11x`: enable GPU support for NVIDIA GPUs using CUDA 11.x;
* `gpu-cuda12x`: enable GPU support for NVIDIA GPUs using CUDA 12.x;
* `gpu-rocm`: enable GPU support for AMD GPUs using ROCm; the following additional environment
variables must be set:
    ```shell
    (venv) $ export CUPY_INSTALL_USE_HIP=1
    (venv) $ export ROCM_HOME=<path to ROCm installation>
    (venv) $ export HCC_AMDGPU_TARGET=<string denoting the Instruction Set Architecture (ISA) supported by the target GPU>
    ```

The CLOUDSC2 microphysics scheme comes in three formulations: nonlinear (NL), tangent-linear TL) and
adjoint (AD). The easiest way to run the dwarf is through the scripts contained in `drivers/`:

* `run_nonlinear.py` executes CLOUDSC2-NL and validates the results against reference data available
in `data/`;
* `run_taylor_test.py` performs the Taylor test (also known as "V-shape test") for CLOUDSC2-TL;
* `run_symmetry_test.py` performs the symmetry test for CLOUDSC2-AD.
*
Run the scripts with the `--help` option to get the full list of command-line options.
