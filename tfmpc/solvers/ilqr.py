import logging
from collections import namedtuple
import os
import time

import numpy as np
import tensorflow as tf
from tuneconfig import experiment

from tfmpc.envs import diffenv
from tfmpc.utils import optimization
from tfmpc.utils import trajectory
from tfmpc import loggers


Trajectory = namedtuple("Trajectory", "states actions costs")
Controller = namedtuple("Controller", "K k")


class iLQR:
    """
    Iterative Linear Quadratic Regulator (iLQR)

    For details please see:

    >> Synthesis and stabilization of complex behaviors through online trajectory optimization
    >> Tassa, Erez, and Todorov (2012)
    """

    def __init__(
            self,
            env,
            max_iterations=100,
            reg_factor=10.,
            reg_min=1e-9,
            reg_max=1e9,
            step_length_factor=0.5,
            step_length_num=10,
            step_length_min=0.01,
            step_length_max=0.5,
            c1=0.0,
            c2=0.5,
            residual_threshold=5e-3,
            g_norm_threshold=1e-9,
            **kwargs
    ):
        self.env = env

        self.max_iterations = max_iterations

        # state trajectory regularization
        self.reg_factor = reg_factor
        self.reg_min = reg_min
        self.reg_max = reg_max

        # line-search parameters
        self.step_length_factor = step_length_factor
        self.step_length_num = step_length_num
        self.step_length_min = step_length_min
        self.step_length_max = step_length_max

        # acceptance criteria
        self.c1 = c1
        self.c2 = c2

        # stopping criteria
        self.residual_threshold = residual_threshold
        self.g_norm_threshold = g_norm_threshold

        self._reg = reg_min

        self._candidate = None
        self._current_total_cost = None

    def start(self, x0, T):
        """Returns an initial feasible candidate trajectory."""
        states = [x0]
        costs = []
        actions = []

        action = tf.zeros_like(self.env.action_space.sample())

        state = x0

        for t in tf.range(T):
            state = tf.expand_dims(state, axis=0)
            action = tf.expand_dims(action, axis=0)
            next_state = self.env.transition(state, action)
            cost = self.env.cost(state, action)

            state = tf.squeeze(next_state, axis=0)
            action = tf.squeeze(action, axis=0)
            cost = tf.squeeze(cost, axis=0)

            actions.append(action)
            states.append(state)
            costs.append(cost)

        final_cost = self.env.final_cost(state)
        costs.append(final_cost)

        states = tf.stack(states, axis=0)
        actions = tf.stack(actions, axis=0)
        costs = tf.stack(costs, axis=0)

        return Trajectory(states, actions, costs)

    def derivatives(self, states, actions):
        """Returns the locally-approximated dynamics and cost model.

        Args:
            states: the sequence of states in a candidate trajectory.
            actions: the sequence of actions in a candidate trajectory.

        Returns:
            (transition_model, cost_model, final_cost_model): the local approximation model.
        """
        transition_model = self.env.get_linear_transition(states[:-1], actions)
        cost_model = self.env.get_quadratic_cost(states[:-1], actions)
        final_cost_model = self.env.get_quadratic_final_cost(states[-1])

        return transition_model, cost_model, final_cost_model

    def backward(self, f_model, l_model, l_f_model):
        """Implements the DDP backward pass.

        Args:
            f_model: dynamics approximation model.
            l_model: running cost approximation model.
            l_f_model: final cost approximation model.

        Returns:
            (controller, dV1, dV2, g_norm): the controller, the corresponding
            expected value improvement terms, and the total gradient norm.
        """

    def forward(self, controller, alpha):
        """Implements the DDP forward pass.

        Args:
            controller: the (K, k) gain matrices of the recent controller.
            alpha: the line-search parameter.

        Returns:
            (trajectory, total_cost, residual): A new candidate trajectory and
            its total cost and total residual.
        """

    def solve(self, x0, T, show_progress=True):
        """Solves the task by iteratively running backward-forward DDP passes.

        Initializes a feasible solution and incrementally improves it
        until convergence or the maximum number of iterations is achieved.

        Args:
            x0: the initial state.
            T: the horizon of the task.
            show_progress: the boolean flag for progress monitoring.

        Returns:
            (trajectory, num_iterations): the best candidate trajectory and
            the number of iterations used.
        """

    def _hamiltonian(self, f_model, l_model, V_model):
        """Returns the Hamiltonian model.

        Args:
            f_model: the dynamics approximated model.
            l_model: the running cost approximated model.
            V_model: the next state Value function model.

        Returns:
            (Q_x, Q_u, Q_xx, Q_uu, Q_ux): the components of the Hamiltonin.
        """

    def _controller(self, Q_uu, Q_u, Q_ux):
        """Returns the controller for the Hamiltonian action-based components.

        Args:
            Q_uu: the action Hessian of the Q-function approximation.
            Q_u: the action gradient of the Q-function approximation.
            Q_ux: the action-state Hessian of the Q-function approximation.

        Returns:
            (controller): the gains (K, k) of the controller.
        """

    def _increase_regularization(self):
        """Increases the current regularization parameter."""

    def _decrease_regularization(self):
        """Decreases the current regularization parameter."""

    def _expected_improvement(self, dV1, dV2, alpha):
        """Returns the expected value improvement."""

    def _accept_criteria(self, total_cost, dV):
        """Returns True if the candidate acceptance criteria is met. Otherwise, False."""

    def _stopping_criteria(self, g_norm, residual):
        """Returns True if stopping criteria is met. Otherwise, False."""
