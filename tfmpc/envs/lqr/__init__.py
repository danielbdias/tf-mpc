import json

import gym
import numpy as np
from sklearn.datasets import make_spd_matrix
import tensorflow as tf

from tfmpc.envs.diffenv import DiffEnv
from tfmpc.envs.gymenv import GymEnv


def make_random_lqr_problem(state_size, action_size):
    n_dim = state_size + action_size

    F = np.random.normal(size=(state_size, n_dim))
    f = np.random.normal(size=(state_size, 1))

    C = make_spd_matrix(n_dim)
    c = np.random.normal(size=(n_dim, 1))

    return LQR(F, f, C, c)


class LQR(DiffEnv, GymEnv):
    """
    Linear Quadratic Regulator (LQR)

    For notation and more details on LQR please check out:
    - http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf
    """

    def __init__(self, F, f, C, c, name=None):
        self.name = name

        self.F = tf.Variable(F, trainable=False, dtype=tf.float32, name="F")
        self.f = tf.Variable(f, trainable=False, dtype=tf.float32, name="f")
        self.C = tf.Variable(C, trainable=False, dtype=tf.float32, name="C")
        self.c = tf.Variable(c, trainable=False, dtype=tf.float32, name="c")

        self.obs_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(self.state_size, 1))
        self.action_space = gym.spaces.Box(
            -1., 1., shape=(self.action_size, 1))

    @property
    def config(self):
        return {
            "F": self.F.numpy().tolist(),
            "f": self.f.numpy().tolist(),
            "C": self.C.numpy().tolist(),
            "c": self.c.numpy().tolist()
        }

    @property
    def n_dim(self):
        return self.F.shape[1]

    @property
    def state_size(self):
        return self.F.shape[0]

    @property
    def action_size(self):
        return self.n_dim - self.state_size

    @tf.function
    def transition(self, x, u):
        inputs = tf.concat([x, u], axis=0)
        return self.F @ inputs + self.f

    @tf.function
    def cost(self, x, u):
        inputs = tf.concat([x, u], axis=0)
        inputs_T = tf.transpose(inputs)
        c1 = 1 / 2 * inputs_T @ self.C @ inputs
        c2 = inputs_T @ self.c
        c = tf.squeeze(c1 + c2)
        return c

    @tf.function
    def final_cost(self, x):
        n = self.state_size
        C_xx = self.C[:n, :n]
        c_x = self.c[:n]
        x_T = tf.transpose(x)
        c1 = 1 / 2 * x_T @ C_xx @ x
        c2 = x_T @ c_x
        c = tf.squeeze(c1 + c2)
        return c

    def dump(self, filepath):
        config = self.config
        with open(filepath, "w") as file:
            json.dump(self.config, file)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r") as file:
            config = json.load(file)
        config = {k: np.array(v).astype("f") for k, v in config.items()}
        return cls(**config)

    def __repr__(self):
        name = self.name or "LQR"
        return f"{name}(n={self.state_size},m={self.action_size})"
