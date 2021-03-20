import pytest
import tensorflow as tf


from tfmpc.envs.cartpole.swingup import CartPoleSwingUp
from tfmpc.solvers.ilqr import iLQR

from tests.conftest import sample_action, sample_state


@pytest.fixture(scope="module")
def env():
    return CartPoleSwingUp()


@pytest.fixture(scope="module")
def solver(env):
    return iLQR(env)


def test_start(solver):
    x0 = solver.env.initial_state
    T = 100
    states, actions, costs = solver.start(x0, T)

    assert states.shape == (T + 1, solver.env.state_size, 1)
    assert actions.shape == (T, solver.env.action_size, 1)
    assert costs.shape == (T + 1,)


def test_derivatives(solver):
    T = 100
    states, actions, costs = solver.start(solver.env.initial_state, T)
    models = solver.derivatives(states, actions)

    assert len(models) == 3
    transition_model, cost_model, final_cost_model = models

    assert all(tf.shape(g)[0] == T for g in transition_model)
    assert all(tf.shape(g)[0] == T for g in cost_model)

    n = solver.env.state_size
    m = solver.env.action_size
    assert transition_model.f.shape == (T, n, 1)
    assert transition_model.f_x.shape == (T, n, n)
    assert transition_model.f_u.shape == (T, n, m)


def test_backward(solver):
    pass


def test_forward(solver):
    pass
