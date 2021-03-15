# pylint: disable=missing-docstring

import numpy as np
import pytest
import tensorflow as tf

from tfmpc.envs.lqr.navigation import make_lqr_navigation_problem
from tfmpc.solvers.lqr import LQR


@pytest.fixture(scope="module")
def nav():
    goal = np.array([[8.32], [-5.5]])
    beta = 10.0
    return make_lqr_navigation_problem(goal, beta)


@pytest.fixture
def solver(nav):
    return LQR(nav)


def test_make_lqr_linear_navigation(nav):
    state_size = nav.state_size
    action_size = nav.action_size
    n_dim = state_size + action_size

    F = nav.F
    assert isinstance(F, tf.Variable)
    assert F.shape == (state_size, n_dim)

    f = nav.f
    assert isinstance(f, tf.Variable)
    assert f.shape == (state_size, 1)

    C = nav.C
    assert isinstance(C, tf.Variable)
    assert C.shape == (n_dim, n_dim)

    C = C.numpy()
    assert np.allclose(C.T, C, atol=1e-4)
    np.linalg.cholesky(C)
    assert np.all(np.linalg.eigvals(C) > 0)

    c = nav.c
    assert isinstance(c, tf.Variable)
    assert c.shape == (n_dim, 1)


def test_backward(solver):
    T = 10
    policy, value_fn = solver.backward(T)
    assert len(policy) == len(value_fn)


def test_forward(solver):
    nav = solver.lqr

    T = 10
    x0 = np.random.normal(size=(nav.state_size, 1)).astype("f")

    policy, _ = solver.backward(T)

    x, u, c = solver.forward(policy, x0, T)
    assert len(x) == len(u) + 1 == len(c)
    assert np.allclose(x[0], x0, atol=1e-4)

    F_t = nav.F.numpy()
    f_t = nav.f.numpy()
    C_t = nav.C.numpy()
    c_t = nav.c.numpy()

    for t in range(T):
        K, k = policy[t]
        action = np.dot(K, x[t]) + k
        assert np.allclose(action, u[t], atol=1e-4)

        inputs = np.concatenate([x[t], u[t]], axis=0)
        next_state = np.dot(F_t, inputs) + f_t
        assert np.allclose(next_state, x[t + 1], atol=1e-4)

        cost = 1 / 2 * np.dot(np.dot(inputs.T, C_t), inputs) + np.dot(c_t.T, inputs)
        assert np.allclose(cost, c[t], atol=1e-4)
