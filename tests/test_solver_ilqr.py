import pytest
import tensorflow as tf


from tfmpc.envs.cartpole.swingup import CartPoleSwingUp
from tfmpc.solvers import ilqr

from tests.conftest import sample_action, sample_state


@pytest.fixture(scope="module")
def env():
    return CartPoleSwingUp()


@pytest.fixture(scope="module")
def solver(env):
    return ilqr.iLQR(env)


def test_start(solver):
    x0 = solver.env.initial_state
    T = 100
    trajectory, total_cost = solver.start(x0, T)

    assert trajectory.states.shape == (T + 1, solver.env.state_size, 1)
    assert trajectory.actions.shape == (T, solver.env.action_size, 1)
    assert trajectory.costs.shape == (T + 1,)

    expected_total_cost = tf.reduce_sum(trajectory.costs)
    assert tf.abs(total_cost - expected_total_cost) < 1e-4


def test_derivatives(solver):
    T = 100
    (states, actions, costs), _ = solver.start(solver.env.initial_state, T)
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
    T = 100
    (states, actions, costs), _ = solver.start(solver.env.initial_state, T)
    f_model, l_model, l_final_model = solver.derivatives(states, actions)

    n = solver.env.state_size
    m = solver.env.action_size

    controller, dV1, dV2, g_norm = solver.backward(
        T, f_model, l_model, l_final_model)

    assert controller.K.shape == (T, m, n)
    assert controller.k.shape == (T, m, 1)

    assert dV1.shape == ()
    assert dV2.shape == ()
    assert g_norm.shape == ()


def test_forward(solver):
    T = 100
    candidate, _ = solver.start(solver.env.initial_state, T)
    f_model, l_model, l_final_model = solver.derivatives(
        candidate.states, candidate.actions)

    n = solver.env.state_size
    m = solver.env.action_size

    controller, dV1, dV2, g_norm = solver.backward(
        T, f_model, l_model, l_final_model)

    alpha = 1.0
    trajectory, total_cost, residual = solver.forward(
        candidate, controller, alpha)

    assert trajectory.states.shape == (T + 1, n, 1)
    assert trajectory.actions.shape == (T, m, 1)
    assert trajectory.costs.shape == (T + 1,)
    assert total_cost.shape == ()
    assert residual.shape == ()


def test_solve(solver):
    n = solver.env.state_size
    m = solver.env.action_size

    T = 100
    x0 = solver.env.initial_state
    trajectory, num_iterations, converged = solver.solve(x0, T)

    assert trajectory.states.shape == (T + 1, n, 1)
    assert trajectory.actions.shape == (T, m, 1)
    assert trajectory.costs.shape == (T + 1,)


def test_hamiltonian(solver):
    T = 100
    (states, actions, costs), _ = solver.start(solver.env.initial_state, T)
    f_model, l_model, l_final_model = solver.derivatives(states, actions)

    n = solver.env.state_size
    m = solver.env.action_size

    V_model = ilqr.ValueModel(l_final_model.l_x, l_final_model.l_xx)

    for t in range(T):
        f_t_model = ilqr.DynamicsModel(*[m[t] for m in f_model])
        l_t_model = ilqr.CostModel(*[m[t] for m in l_model])
        Q_model = solver._hamiltonian(f_t_model, l_t_model, V_model)

        assert Q_model.Q_x.shape == (n, 1)
        assert Q_model.Q_u.shape == (m, 1)
        assert Q_model.Q_xx.shape == (n, n)
        assert Q_model.Q_uu.shape == (m, m)
        assert Q_model.Q_ux.shape == (m, n)


def test_controller(solver):
    T = 100
    (states, actions, costs), _ = solver.start(solver.env.initial_state, T)
    f_model, l_model, l_final_model = solver.derivatives(states, actions)

    n = solver.env.state_size
    m = solver.env.action_size

    V_model = ilqr.ValueModel(l_final_model.l_x, l_final_model.l_xx)
    f_t_model = ilqr.DynamicsModel(*[m[-1] for m in f_model])
    l_t_model = ilqr.CostModel(*[m[-1] for m in l_model])
    Q_model = solver._hamiltonian(f_t_model, l_t_model, V_model)

    K, k = solver._controller(Q_model.Q_uu, Q_model.Q_u, Q_model.Q_ux)

    assert K.shape == (m, n)
    assert k.shape == (m, 1)


def test_value_update(solver):
    T = 100
    (states, actions, costs), _ = solver.start(solver.env.initial_state, T)
    f_model, l_model, l_final_model = solver.derivatives(states, actions)

    n = solver.env.state_size
    m = solver.env.action_size

    V_model = ilqr.ValueModel(l_final_model.l_x, l_final_model.l_xx)
    f_t_model = ilqr.DynamicsModel(*[m[-1] for m in f_model])
    l_t_model = ilqr.CostModel(*[m[-1] for m in l_model])
    Q_model = solver._hamiltonian(f_t_model, l_t_model, V_model)

    controller = solver._controller(Q_model.Q_uu, Q_model.Q_u, Q_model.Q_ux)

    V_x, V_xx = solver._value_update(Q_model, controller)

    assert V_x.shape == (n, 1)
    assert V_xx.shape == (n, n)
