#!/usr/bin/env python
# coding: utf-8

import os

import click
import gym
import psutil
import tensorflow.compat.v1.logging as tf_logging
import tuneconfig

from tfmpc.launchers import online_ilqr_run
from tfmpc.launchers import ilqr_run


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

gym.logger.set_level(gym.logger.ERROR)
tf_logging.set_verbosity(tf_logging.ERROR)


@click.group()
def cli():
    pass


@cli.command()
@click.argument("env", type=click.Path(exists=True))
@click.option(
    "--online",
    is_flag=True,
    help="Online mode flag.",
    show_default=True)
@click.option(
    "--ignore-final-cost",
    is_flag=True,
    help="Ignore state-dependent final cost.",
    show_default=True)
@click.option(
    "--horizon", "-hr",
    type=click.IntRange(min=1),
    default=10,
    help="The number of timesteps.",
    show_default=True)
@click.option(
    "--atol",
    type=click.FloatRange(min=0.0),
    default=5e-3,
    help="Absolute tolerance for convergence.",
    show_default=True)
@click.option(
    "--max-iterations", "-miter",
    type=click.IntRange(min=1),
    default=100,
    help="Maximum number of iterations.",
    show_default=True)
@click.option(
    "--logdir",
    type=click.Path(),
    default="/tmp/ilqr/",
    help="Directory used for logging results.",
    show_default=True)
@click.option(
    "--num-samples", "-ns",
    type=click.IntRange(min=1),
    default=1,
    help="Number of runs.",
    show_default=True)
@click.option(
    "--num-workers", "-nw",
    type=click.IntRange(min=1, max=psutil.cpu_count()),
    default=1,
    help=f"Number of worker processes (min=1, max={psutil.cpu_count()}).",
    show_default=True)
@click.option(
    "--verbose", "-v",
    count=True,
    help="Verbosity level flag.")
def ilqr(**kwargs):
    """Run iLQR for a given environment and horizon.

    Args:

        ENV: Path to the environment's config JSON file.
    """
    verbose = kwargs["verbose"]

    if verbose == 1:
        level = tf_logging.INFO
    elif verbose == 2:
        level = tf_logging.DEBUG
    else:
        level = tf_logging.ERROR

    tf_logging.set_verbosity(level)

    def format_fn(param):
        fmt = {
            "env": None,
            "logdir": None,
            "num_samples": None,
            "num_workers": None,
            "verbose": None
        }
        return fmt.get(param, param)

    config_it = tuneconfig.ConfigFactory(kwargs, format_fn)

    runner = tuneconfig.Experiment(config_it, kwargs["logdir"])
    runner.start()

    exec_func = online_ilqr_run if kwargs["online"] else ilqr_run

    results = runner.run(
        exec_func, kwargs["num_samples"], kwargs["num_workers"])

    for trial_id, runs in results.items():
        for _, trajectory in runs:
            print(repr(trajectory))
            print(str(trajectory))
