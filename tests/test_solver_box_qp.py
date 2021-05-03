import time

import pytest
import tensorflow as tf

from tfmpc.solvers.box_qp import BoxQP


@pytest.fixture(scope="module")
def solver():
    return BoxQP()


@pytest.fixture(params=[
    ([0.0, 0.0], [-1.0, 0.5], [1.0, 1.0], [0.0, 0.5]),
    ([0.0, 0.0], [0.5, -1.0], [1.0, 1.0], [0.5, 0.0]),
    ([1.0, 1.0], [0.0, 1.5], [2.0, 2.0], [1.0, 1.5]),
    ([1.0, 1.0], [1.5, 0.0], [2.0, 2.0], [1.5, 1.0]),

    ([0.0, 0.0, 0.0], [-1.0, 0.5, -1.0], [1.0, 1.0, 1.0], [0.0, 0.5, 0.0]),
    ([0.0, 0.0, 0.0], [-1.0, 0.5, 0.30], [1.0, 1.0, 1.0], [0.0, 0.5, 0.30]),
])
def qp(request):
    goal, low, high, x_star = request.param

    dim = len(goal)

    H = 2 * tf.eye(dim)
    goal = tf.constant(goal, shape=(dim, 1))
    q = -2 * goal
    low = tf.constant(low, shape=(dim, 1))
    high = tf.constant(high, shape=(dim, 1))
    x_star = tf.constant(x_star, shape=(dim, 1))

    return H, q, low, high, x_star, goal


def test_solve(solver, qp):
    H, q, low, high, x_star, _ = qp
    solver.setup(H, q, low, high)
    x, Hff_llt, free, clamped = solver.solve(x_star)

    assert tf.reduce_all(tf.abs(x - x_star) < 1e-4)

    for i in range(3):
        x_0 = tf.random.uniform(tf.shape(x_star), minval=low, maxval=high)

        x, *_ = solver.solve(x_0)
        assert tf.reduce_all(tf.abs(x - x_star) < 1e-4)

        for i in range(x_0.shape[0]):
            x = tf.Variable(x_0, trainable=False)
            x[i].assign(low[i])
            x, *_ = solver.solve(x)
            assert tf.reduce_all(tf.abs(x - x_star) < 1e-4)

        for i in range(x_0.shape[0]):
            x = tf.Variable(x_0, trainable=False)
            x[i].assign(high[i])
            x, *_ = solver.solve(x)
            assert tf.reduce_all(tf.abs(x - x_star) < 1e-4)


def test_compute_gradient(solver, qp):
    H, q, low, high, _, goal = qp
    solver.setup(H, q, low, high)
    x = (low + high) / 2
    sign = tf.math.sign(goal - x)
    g = solver._compute_gradient(x)

    assert g.shape == x.shape
    assert tf.reduce_all(sign == tf.math.sign(-g))


def test_get_active_set(solver, qp):
    H, q, low, high, _, _ = qp
    solver.setup(H, q, low, high)

    # fully free
    x = (low + high) / 2
    g = solver._compute_gradient(x)
    free, clamped = solver._get_active_set(g, x)

    assert free.shape == x.shape
    assert clamped.shape == x.shape
    assert tf.reduce_all(free == True)
    assert tf.reduce_all(clamped == False)


def test_factorize(solver, qp):
    H, q, low, high, _, _ = qp
    solver.setup(H, q, low, high)

    x = (low + high) / 2
    g = solver._compute_gradient(x)
    free, clamped = solver._get_active_set(g, x)
    Hff_inv_llt = solver._factorize(free)

    n_free = tf.reduce_sum(tf.cast(free, tf.int32))
    assert Hff_inv_llt.shape == (n_free, n_free)


def test_projected_newton_qp(solver, qp):
    H, q, low, high, _, _ = qp
    solver.setup(H, q, low, high)

    x = (low + high) / 2
    g = solver._compute_gradient(x)
    free, clamped = solver._get_active_set(g, x)
    Hff_inv_llt = solver._factorize(free)

    dx = solver._projected_newton_step(x, Hff_inv_llt, free, clamped)
    assert dx.shape == x.shape
