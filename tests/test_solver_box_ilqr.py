import numpy as np
import pytest
import tensorflow as tf
import tensorflow.compat.v1.logging as tf_logging

from tfmpc.envs.lqr import navigation
from tfmpc.solvers.box_ilqr import *
from tfmpc.utils.trajectory import Trajectory


tf_logging.set_verbosity(tf_logging.ERROR)


@pytest.fixture(scope="module", params=[[[5.5], [-9.0]]])
def goal(request):
    return request.param


@pytest.fixture(scope="module", params=[0.0, 5.0], ids=["beta=0.0", "beta=5.0"])
def beta(request):
    return request.param


@pytest.fixture(scope="module")
def initial_state():
    return tf.constant([[0.0], [0.0]])


@pytest.fixture(scope="module")
def horizon():
    return tf.constant(10)


@pytest.fixture(scope="module")
def solver(goal, beta):
    goal = tf.constant(goal)
    beta = tf.constant(beta)
    low, high = -1.0, 1.0
    env = navigation.NavigationLQR(goal, beta, low, high)
    return BoxILQR(env)


def test_start(solver, initial_state, horizon):
    x0 = initial_state
    T = horizon
    trajectory, total_cost = solver.start(x0, T)

    assert trajectory.states.shape == (T + 1, solver.env.state_size, 1)
    assert trajectory.actions.shape == (T, solver.env.action_size, 1)
    assert trajectory.costs.shape == (T + 1,)

    expected_total_cost = tf.reduce_sum(trajectory.costs)
    assert tf.abs(total_cost - expected_total_cost) < 1e-4


def test_derivatives(solver, initial_state, horizon):
    T = horizon
    (states, actions, costs), _ = solver.start(initial_state, T)
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


def test_backward(solver, initial_state, horizon):
    T = horizon
    trajectory, _ = solver.start(initial_state, T)
    solver._candidate = trajectory
    f_model, l_model, l_final_model = solver.derivatives(
        trajectory.states, trajectory.actions)

    n = solver.env.state_size
    m = solver.env.action_size

    controller, dV1, dV2, g_norm = solver.backward(
        T, f_model, l_model, l_final_model)

    assert controller.K.shape == (T, m, n)
    assert controller.k.shape == (T, m, 1)

    assert dV1.shape == ()
    assert dV2.shape == ()
    assert g_norm.shape == ()


def test_forward(solver, initial_state, horizon):
    T = horizon
    trajectory, _ = solver.start(initial_state, T)
    solver._candidate = trajectory
    f_model, l_model, l_final_model = solver.derivatives(
        trajectory.states, trajectory.actions)

    n = solver.env.state_size
    m = solver.env.action_size

    controller, dV1, dV2, g_norm = solver.backward(
        T, f_model, l_model, l_final_model)

    alpha = 1.0
    trajectory, total_cost, residual = solver.forward(
        trajectory, controller, alpha)

    assert trajectory.states.shape == (T + 1, n, 1)
    assert trajectory.actions.shape == (T, m, 1)
    assert trajectory.costs.shape == (T + 1,)
    assert total_cost.shape == ()
    assert residual.shape == ()


def test_solve(solver, initial_state, horizon):
    n = solver.env.state_size
    m = solver.env.action_size

    T = horizon
    x0 = initial_state
    trajectory, num_iterations, converged = solver.solve(x0, T)

    assert trajectory.states.shape == (T + 1, n, 1)
    assert trajectory.actions.shape == (T, m, 1)
    assert trajectory.costs.shape == (T + 1,)

    print()
    print(Trajectory(*trajectory))


def test_hamiltonian(solver, initial_state, horizon):
    T = horizon
    (states, actions, costs), _ = solver.start(initial_state, T)
    f_model, l_model, l_final_model = solver.derivatives(states, actions)

    n = solver.env.state_size
    m = solver.env.action_size

    V_model = ValueModel(l_final_model.l_x, l_final_model.l_xx)

    for t in range(T):
        f_t_model = DynamicsModel(*[m[t] for m in f_model])
        l_t_model = CostModel(*[m[t] for m in l_model])
        Q_model = solver._hamiltonian(f_t_model, l_t_model, V_model)

        assert Q_model.Q_x.shape == (n, 1)
        assert Q_model.Q_u.shape == (m, 1)
        assert Q_model.Q_xx.shape == (n, n)
        assert Q_model.Q_uu.shape == (m, m)
        assert Q_model.Q_ux.shape == (m, n)


def test_controller(solver, initial_state, horizon):
    T = horizon
    (states, actions, costs), _ = solver.start(initial_state, T)
    f_model, l_model, l_final_model = solver.derivatives(states, actions)

    n = solver.env.state_size
    m = solver.env.action_size

    V_model = ValueModel(l_final_model.l_x, l_final_model.l_xx)
    f_t_model = DynamicsModel(*[m[-1] for m in f_model])
    l_t_model = CostModel(*[m[-1] for m in l_model])
    Q_model = solver._hamiltonian(f_t_model, l_t_model, V_model)

    K, k = solver._controller(
        actions[-1], Q_model.Q_uu, Q_model.Q_ux, Q_model.Q_u,
    )

    assert K.shape == (m, n)
    assert k.shape == (m, 1)


def test_value_update(solver, initial_state, horizon):
    T = horizon
    (states, actions, costs), _ = solver.start(initial_state, T)
    f_model, l_model, l_final_model = solver.derivatives(states, actions)

    n = solver.env.state_size
    m = solver.env.action_size

    V_model = ValueModel(l_final_model.l_x, l_final_model.l_xx)
    f_t_model = DynamicsModel(*[m[-1] for m in f_model])
    l_t_model = CostModel(*[m[-1] for m in l_model])
    Q_model = solver._hamiltonian(f_t_model, l_t_model, V_model)

    controller = solver._controller(
        actions[-1], Q_model.Q_uu, Q_model.Q_ux, Q_model.Q_u,
    )
    V_x, V_xx = solver._value_update(Q_model, controller)

    assert V_x.shape == (n, 1)
    assert V_xx.shape == (n, n)
