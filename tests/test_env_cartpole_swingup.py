import pytest
import tensorflow as tf

from tfmpc.envs.cartpole.swingup import CartPoleSwingUp

from tests.conftest import sample_action, sample_state


@pytest.fixture
def env():
    return CartPoleSwingUp()


def test_initial_state(env):
    assert tf.reduce_all(env.initial_state == tf.zeros((4, 1)))


def test_transition(env):
    batch_size = 100
    state = sample_state(env, batch_size)
    action = sample_action(env, batch_size)
    next_state = env.transition(state, action)

    assert next_state.shape == (batch_size, env.state_size, 1)
    for s in next_state:
        assert s.numpy() in env.obs_space


def test_cost(env):
    batch_size = 100
    state = sample_state(env, batch_size)
    action = sample_action(env, batch_size)
    next_state = env.transition(state, action)
    cost = env.cost(state, action)

    assert cost.shape == (batch_size,)


def test_final_cost(env):
    state = env.obs_space.sample()
    cost = env.final_cost(state)

    assert cost.shape == ()

