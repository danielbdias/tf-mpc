import gym
import numpy as np
import tensorflow as tf

from tfmpc.envs.diffenv import DiffEnv
from tfmpc.envs.gymenv import GymEnv
from tfmpc.envs.lqr import LQR


def make_lqr_navigation_problem(goal, beta):
    state_size = action_size = goal.shape[0]

    F = np.concatenate([np.identity(state_size)] * action_size, axis=1).astype("f")
    f = np.zeros((state_size, 1)).astype("f")

    C = np.diag([2.0] * state_size + [2.0 * beta] * action_size).astype("f")
    c = np.concatenate([-2.0 * goal, np.zeros((action_size, 1))], axis=0).astype("f")

    return LQR(F, f, C, c)


class NavigationLQR(DiffEnv, GymEnv):

    def __init__(self, goal, beta, low=None, high=None):
        self.goal = goal
        self.beta = beta

        if low is None:
            low = -np.inf
        if high is None:
            high = np.inf

        self.obs_space = gym.spaces.Box(-np.inf, np.inf, shape=tf.shape(goal))
        self.action_space = gym.spaces.Box(low, high, shape=tf.shape(goal))

    @property
    def action_size(self):
        return self.state_size

    @property
    def state_size(self):
        return self.goal.shape[0]

    @tf.function
    def transition(self, state, action):
        return state + action

    @tf.function
    def cost(self, state, action):
        state = tf.squeeze(state, axis=-1)
        goal = tf.squeeze(self.goal, axis=-1)
        action = tf.squeeze(action)
        c1 = tf.reduce_sum((state - goal) ** 2, axis=-1)
        c2 = tf.reduce_sum(action ** 2, axis=-1)
        return c1 + self.beta * c2

    @tf.function
    def final_cost(self, state):
        state = tf.squeeze(state, axis=-1)
        goal = tf.squeeze(self.goal, axis=-1)
        return tf.reduce_sum((state - goal) ** 2, axis=-1)

    @classmethod
    def load(cls, config):
        goal = config["goal"]
        state_dim = len(goal)
        goal = tf.constant(goal, shape=(state_dim, 1), dtype=tf.float32)
        beta = tf.constant(config["beta"], dtype=tf.float32)
        low = config.get("low")
        high = config.get("high")

        return cls(goal, beta, low, high)

    def __repr__(self):
        goal = self.goal.numpy().squeeze().tolist()
        beta = self.beta
        bounds = ""
        if self.action_space.is_bounded():
            low = self.action_space.low.squeeze().tolist()
            high = self.action_space.high.squeeze().tolist()
            bounds = f", bounds=[{low}, {high}]"
        return f"NavigationLQR(goal={goal}, beta={beta}{bounds})"
