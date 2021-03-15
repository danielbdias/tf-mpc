#! /usr/bin/env python3

import os
import sys

import click
import gym
import psutil
import tensorflow.compat.v1.logging as tf_logging
import tuneconfig
import wandb

from tfmpc.launchers import online_ilqr_run


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

gym.logger.set_level(gym.logger.ERROR)
tf_logging.set_verbosity(tf_logging.ERROR)


def setup(logdir):
    if not os.path.exists(logdir):
        os.makedirs(logdir)


def solve(config):
    run_id = config["run_id"]
    rddl = config["planner"]["rddl"]

    name = f"{rddl}-ilqr-run{run_id}"
    group = config.get("group") or rddl
    job_type = f"{config['job_type']}".format(**config)

    config["show_progress"] = False

    run = wandb.init(
        project="online-tfplan",
        name=name,
        config=config,
        tags=["baseline", "online-ilqr"],
        group=group,
        job_type=job_type,
        reinit=True
    )

    with run:
        online_ilqr_run(config)


@click.command()
@click.argument("envs", nargs=-1)
@click.option(
    "--group", "-g",
    help="Experiment group name."
)
@click.option(
    "--job-type", "-jt",
    help="Experiment job type."
)
@click.option(
    "--num-samples", "-ns",
    type=int,
    default=1,
    help="Number of runs.",
    show_default=True,
)
@click.option(
    "--num-workers", "-nw",
    type=click.IntRange(min=1, max=psutil.cpu_count()),
    default=1,
    help=f"Number of worker processes (min=1, max={psutil.cpu_count()}).",
    show_default=True,
)
def cli(envs, **kwargs):
    """Run iLQR on the RDDL problem."""
    num_samples = kwargs.pop("num_samples")
    num_workers = kwargs.pop("num_workers")

    logdir = "results"

    def format_fn(param):
        fmt = {
            "horizon": "hr",
            "ignore_final_cost": None,
            "logdir": None,
            "logger": None,
            "num_samples": None,
            "num_workers": None,
            "group": None
        }
        return fmt.get(param, param)

    for env in envs:

        setup(logdir=logdir)

        config = {
            "planner": {
                "planner": "online-ilqr",
                "rddl": env.replace(".config.json", ""),
            },

            # === task ===
            "env": env,
            "horizon": 40,
            "ignore_final_cost": True,

            # === solve ===
            "atol": 5e-3,
            "max_iterations": tuneconfig.grid_search([500]),

            # === forward ===
            "c1": tuneconfig.grid_search([0.0, 0.5, 0.75]),

            "num_samples": num_samples,
            "num_workers": num_workers,

            # === logging ===
            "logdir": logdir,

            # === wandb ===
            "logger": "wandb",
            "group": env.replace(".json", ""),
            "job_type": "ilqr-c1={c1}-maxiter={max_iterations}"
        }

        config_iterator = tuneconfig.ConfigFactory(config, format_fn=format_fn)

        runner = tuneconfig.Experiment(config_iterator, config["logdir"])
        runner.start()

        runner.run(solve, num_samples, num_workers)


if __name__ == "__main__":
    cli()
