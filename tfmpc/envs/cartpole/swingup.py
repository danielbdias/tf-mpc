import math

import gym
import numpy as np
import tensorflow as tf

from tfmpc.envs.diffenv import DiffEnv
from tfmpc.envs.gymenv import GymEnv


class CartPoleSwingUp(DiffEnv, GymEnv):

    def __init__(self, m1=1.0, m2=0.3, l=0.5, g=9.81, tau=0.02):
        self.m1 = m1
        self.m2 = m2
        self.l = l
        self.g = g
        self.tau = tau

        self.obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,1))
        self.action_space = gym.spaces.Box(low=-20.0, high=20.0, shape=(1,1))

    @property
    def state_size(self):
        return 4

    @property
    def action_size(self):
        return 1

    @property
    def initial_state(self):
        return tf.zeros(self.obs_space.shape)

    def transition(self, state, action):
        x, theta, x_dot, theta_dot = tf.split(state, 4, axis=1)
        x = tf.squeeze(x, axis=1)
        theta = tf.squeeze(theta, axis=1)
        x_dot = tf.squeeze(x_dot, axis=1)
        theta_dot = tf.squeeze(theta_dot, axis=1)
        u = tf.squeeze(action, axis=1)

        x = x + self.tau * x_dot
        theta = theta + self.tau * theta_dot

        x_acc = self.l * self.m2 * tf.sin(theta) * theta_dot ** 2
        x_acc += self.m2 * self.g * tf.cos(theta) * tf.sin(theta)
        x_acc += u
        x_acc /= self.m1 + self.m2 * (1 - tf.cos(theta) ** 2)

        x_dot = x_dot + self.tau * x_acc

        theta_acc = self.l * self.m2 * tf.cos(theta) * tf.sin(theta) * theta_dot ** 2
        theta_acc += u * tf.cos(theta)
        theta_acc += (self.m1 + self.m2) * self.g * tf.sin(theta)
        theta_acc /= - (self.l * self.m1 + self.l + self.m2 * (1 - tf.cos(theta)**2))

        theta_dot = theta_dot + self.tau * theta_acc

        next_state = tf.concat([x, theta, x_dot, theta_dot], axis=1)
        next_state = tf.expand_dims(next_state, axis=-1)
        return next_state

    def cost(self, state, action):
        del state
        return tf.reduce_sum(0.0001 * action ** 2, axis=[1, 2])

    def final_cost(self, state):
        goal = tf.constant([[2.0], [math.pi], [0.0], [0.0]])
        return 100 * tf.reduce_sum((state - goal) ** 2)
