# tf-mpc [![Py Versions][py-versions.svg]][pypi-project] [![PyPI version][pypi-version.svg]][pypi-version] [![Build Status][travis.svg]][travis-project] [![License: GPL v3][license.svg]][license]


# Quickstart

**tfmpc** is a Python3.6+ package available in PyPI.

```text
$ pip3 install -U tfmpc
```


# Usage

```bash
$ tfmpc ilqr --help

Usage: tfmpc ilqr [OPTIONS] ENV

  Run iLQR for a given environment and horizon.

  Args:

      ENV: Path to the environment's config JSON file.

Options:
  --online                        Online mode flag.  [default: False]
  --ignore-final-cost             Ignore state-dependent final cost.
                                  [default: False]
  -hr, --horizon INTEGER RANGE    The number of timesteps.  [default: 10]
  --atol FLOAT RANGE              Absolute tolerance for convergence.
                                  [default: 0.005]
  -miter, --max-iterations INTEGER RANGE
                                  Maximum number of iterations.  [default:
                                  100]
  --logdir PATH                   Directory used for logging results.
                                  [default: /tmp/ilqr/]
  -ns, --num-samples INTEGER RANGE
                                  Number of runs.  [default: 1]
  -nw, --num-workers INTEGER RANGE
                                  Number of worker processes (min=1, max=12).
                                  [default: 1]
  -v, --verbose                   Verbosity level flag.
  --help                          Show this message and exit.
```

# Examples


## LQR

```bash
$ python examples/lqr.py

Trajectory(init=[-0.9436722 -5.9413767 -9.7090645], final=[-6.831274    3.5397437   0.79844564], total=-34.2876)

Steps |             States             |            Actions             |  Costs  
===== | ============================== | ============================== | ========
  0   | [-29.6400,  12.4868,  -6.1247] | [ 12.0202,   6.2650,   2.7019] |   9.9491
  1   | [  1.1229,  -1.0781,  -0.9041] | [ 24.8006,  16.6294, -10.9740] |  49.6677
  2   | [ -8.8750,   2.3962,  -4.4266] | [  3.7858,   3.3769,  -1.8138] |  -1.6455
  3   | [ -9.3617,   3.2755,  -3.5806] | [ 11.8333,   7.8142,  -3.6503] | -11.4392
  4   | [ -6.6389,   2.0026,  -3.2240] | [ 11.3348,   7.6663,  -4.2552] | -11.8703
  5   | [ -7.7849,   2.3658,  -3.6332] | [  9.6319,   6.4642,  -3.2991] | -12.2632
  6   | [ -7.5215,   2.4822,  -3.0080] | [ 10.1523,   6.7136,  -3.4948] | -12.7255
  7   | [ -6.2336,   1.5849,  -2.9592] | [  9.6488,   6.2573,  -3.1976] | -12.8830
  8   | [ -8.7144,   2.0473,  -4.4850] | [ 10.1518,   6.4578,  -2.9710] | -11.6011
  9   | [ -6.8313,   3.5397,   0.7984] | [  8.3644,   5.6785,  -3.5642] | -12.9032

```

## Linear Navigation

```bash
$ python examples/navigation_lqr.py

Goal = [[-17.498825073242188], [-55.275390625]]

Trajectory(init=[0. 0.], final=[-17.498783 -55.275257], total=-32385.3555)

Steps |        States        |       Actions        |   Costs   
===== | ==================== | ==================== | ==========
  0   | [-12.8100, -40.4644] | [-12.8100, -40.4644] |  900.7320 
  1   | [-16.2425, -51.3068] | [ -3.4324, -10.8424] | -3055.5571
  2   | [-17.1622, -54.2120] | [ -0.9197,  -2.9052] | -3339.6064
  3   | [-17.4086, -54.9905] | [ -0.2464,  -0.7784] | -3360.0002
  4   | [-17.4747, -55.1990] | [ -0.0660,  -0.2086] | -3361.4644
  5   | [-17.4924, -55.2549] | [ -0.0177,  -0.0559] | -3361.5696
  6   | [-17.4971, -55.2699] | [ -0.0047,  -0.0150] | -3361.5774
  7   | [-17.4984, -55.2739] | [ -0.0013,  -0.0040] | -3361.5776
  8   | [-17.4987, -55.2750] | [ -0.0003,  -0.0011] | -3361.5774
  9   | [-17.4988, -55.2753] | [ -0.0001,  -0.0003] | -3361.5776

```


# Documentation

Please refer to [https://tfmpc.readthedocs.io/](https://tfmpc.readthedocs.io/) for the code documentation.


# License

Copyright (c) 2020- Thiago P. Bueno All Rights Reserved.

tfmpc is free software: you can redistribute it and/or modify it
under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at
your option) any later version.

tfmpc is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with tfmpc. If not, see http://www.gnu.org/licenses/.


[py-versions.svg]: https://img.shields.io/pypi/pyversions/tfmpc.svg?logo=python&logoColor=white
[pypi-project]: https://pypi.org/project/tfmpc

[pypi-version.svg]: https://badge.fury.io/py/tfmpc.svg
[pypi-version]: https://badge.fury.io/py/tfmpc

[travis.svg]: https://img.shields.io/travis/thiagopbueno/tf-mpc/master.svg?logo=travis
[travis-project]: https://travis-ci.org/thiagopbueno/tf-mpc

[license.svg]: https://img.shields.io/badge/License-GPL%20v3-blue.svg
[license]: https://github.com/thiagopbueno/tf-mpc/blob/master/LICENSE