import json

import numpy as np
from sklearn.datasets import make_spd_matrix
import tensorflow as tf


def make_random_lqr_problem(state_size, action_size):
    n_dim = state_size + action_size

    F = np.random.normal(size=(state_size, n_dim))
    f = np.random.normal(size=(state_size, 1))

    C = make_spd_matrix(n_dim)
    c = np.random.normal(size=(n_dim, 1))

    return LQR(F, f, C, c)


class LQR:
    """
    Linear Quadratic Regulator (LQR)

    For notation and more details on LQR please check out:
    - http://rail.eecs.berkeley.edu/deeprlcourse/static/slides/lec-10.pdf
    """

    def __init__(self, F, f, C, c):
        self.F = tf.Variable(F, trainable=False, dtype=tf.float32, name="F")
        self.f = tf.Variable(f, trainable=False, dtype=tf.float32, name="f")
        self.C = tf.Variable(C, trainable=False, dtype=tf.float32, name="C")
        self.c = tf.Variable(c, trainable=False, dtype=tf.float32, name="c")

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
        return tf.matmul(self.F, inputs) + self.f

    @tf.function
    def cost(self, x, u):
        inputs = tf.concat([x, u], axis=0)
        inputs_transposed = tf.transpose(inputs)
        c1 = 1 / 2 * tf.matmul(tf.matmul(inputs_transposed, self.C), inputs)
        c2 = tf.matmul(inputs_transposed, self.c)
        return tf.squeeze(c1 + c2)

    @tf.function
    def final_cost(self, x):
        state_size = self.state_size
        C_xx = self.C[:state_size, :state_size]
        c_x = self.c[:state_size]
        x_transposed = tf.transpose(x)
        c1 = 1 / 2 * tf.matmul(tf.matmul(x_transposed, C_xx), x)
        c2 = tf.matmul(x_transposed, c_x)
        return tf.squeeze(c1 + c2)

    def dump(self, file):
        config = {
            "F": self.F.numpy().tolist(),
            "f": self.f.numpy().tolist(),
            "C": self.C.numpy().tolist(),
            "c": self.c.numpy().tolist()
        }
        json.dump(config, file)

    @classmethod
    def load(cls, file):
        config = json.load(file)
        config = {k: np.array(v).astype("f") for k, v in config.items()}
        return cls(**config)
