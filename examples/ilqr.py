import json
import os
from pathlib import Path

import fire
import gym
import tensorflow as tf
import tensorflow.compat.v1.logging as tf_logging

from tfmpc.envs.lqr import LQR
from tfmpc.envs.lqr import make_random_lqr_problem
from tfmpc.envs.lqr.navigation import make_lqr_navigation_problem
from tfmpc.solvers.ilqr import iLQR
from tfmpc.utils.viz import TracePlotter


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
gym.logger.set_level(gym.logger.ERROR)
tf_logging.set_verbosity(tf_logging.ERROR)


def lqr_navigation():
    goal = [[10.], [10.0]]
    beta = 0.5
    task = make_lqr_navigation_problem(goal, beta)
    x0 = tf.random.uniform(minval=-10., maxval=10., shape=(len(goal), 1))
    return task, x0


def lqr(n=3, m=3):
    task = make_random_lqr_problem(n, m)
    x0 = tf.random.uniform(minval=-10., maxval=10., shape=(n, 1))
    return task


def benchmark(dirpath, horizon=100):
    dirpath = Path(dirpath)

    for path in dirpath.iterdir():
        if path.is_dir():
            print(path)
            run(path, horizon)


def run(path, horizon=10):
    path = Path(path)
    configpath = path / "config.json"

    config = json.loads(configpath.read_text())
    task = LQR(**config["FfCc"])
    x0 = tf.expand_dims(config["initial_state"], axis=-1)

    solver = iLQR(task)
    trajectory, num_iterations = solver.solve(x0, horizon)

    tracepath = path.parent / path.stem / "ilqr" / f"trace-hr={horizon}.csv"
    trajectory.save(tracepath)

    fname = tracepath.parent / f"plot-hr={horizon}.png"
    plot_results(
        trajectory, suptitle=f"iLQR (horizon={horizon})", fname=fname)


def plot_results(trajectory, suptitle=None, show=True, fname=None):
    plotter = TracePlotter(trajectory)

    with plotter.display(
        nrows=2, ncols=2, suptitle=suptitle, show=show, fname=fname
    ) as axes:
        plotter.plot_cumulative_cost(axes[0][0])
        plotter.plot_cost_history(axes[1][0])
        plotter.plot_states_history(axes[0][1])
        plotter.plot_actions_history(axes[1][1])


if __name__ == "__main__":
    fire.Fire({
        "run": run,
        "benchmark": benchmark
    })
