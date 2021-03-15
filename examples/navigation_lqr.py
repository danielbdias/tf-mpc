import os

import fire
import tensorflow as tf
import tensorflow.compat.v1.logging as tf_logging

from tfmpc.envs.lqr.navigation import make_lqr_navigation_problem
from tfmpc.solvers.lqr import LQR


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf_logging.set_verbosity(tf_logging.ERROR)


def run(state_size=2, beta=0.5, horizon=10):
    """Generate and solve a linear navigation problem."""

    goal = tf.random.uniform(minval=-100., maxval=100., shape=(state_size, 1))
    print(f'Goal = {goal.numpy().tolist()}\n')

    task = make_lqr_navigation_problem(goal, beta)

    solver = LQR(task)

    x0 = tf.zeros_like(goal)
    trajectory = solver.solve(x0, horizon)
    trajectory.summary()



if __name__ == "__main__":
    fire.Fire(run)
