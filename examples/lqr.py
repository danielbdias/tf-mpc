import os

import fire
import tensorflow as tf
import tensorflow.compat.v1.logging as tf_logging

from tfmpc.envs.lqr import make_random_lqr_problem
from tfmpc.solvers.lqr import LQR


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf_logging.set_verbosity(tf_logging.ERROR)


def run(state_size=3, action_size=3, horizon=10):
    """Generate and solve a random LQR problem."""

    task = make_random_lqr_problem(state_size, action_size)

    solver = LQR(task)

    x0 = tf.random.uniform(minval=-10., maxval=10., shape=(state_size, 1))
    trajectory = solver.solve(x0, horizon)
    trajectory.summary()


if __name__ == "__main__":
    fire.Fire(run)
