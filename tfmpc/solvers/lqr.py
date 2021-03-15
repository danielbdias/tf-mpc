import json

import numpy as np
import tensorflow as tf

from tfmpc.utils import trajectory


class LQR:

    def __init__(self, lqr):
        self.lqr = lqr

    @tf.function
    def backward(self, T):
        policy, value_fn = [], []

        state_size = self.lqr.state_size

        F, f, C, c = self.lqr.F, self.lqr.f, self.lqr.C, self.lqr.c

        V = C[:state_size, :state_size]
        v = c[:state_size]
        const = 0.0

        value_fn.append((V, v, const))

        for t in reversed(range(T)):
            F_trans_V = tf.matmul(tf.transpose(F), V)
            Q = C + tf.matmul(F_trans_V, F)
            q = (c
                 + tf.matmul(F_trans_V, f)
                 + tf.matmul(tf.transpose(F), v))

            Q_uu = Q[state_size:, state_size:]
            Q_ux = Q[state_size:, :state_size]
            q_u = q[state_size:]

            inv_Q_uu = tf.linalg.inv(Q_uu)

            K = -tf.matmul(inv_Q_uu, Q_ux)
            k = -tf.matmul(inv_Q_uu, q_u)

            K_trans = tf.transpose(K)
            k_trans = tf.transpose(k)

            Q_xx = Q[:state_size, :state_size]
            Q_xu = Q[:state_size, state_size:]
            q_x = q[:state_size]
            K_Q_uu = tf.matmul(K_trans, Q_uu)

            V = (Q_xx
                 + tf.matmul(Q_xu, K)
                 + tf.matmul(K_trans, Q_ux)
                 + tf.matmul(K_Q_uu, K))

            v = (q_x
                 + tf.matmul(Q_xu, k)
                 + tf.matmul(K_trans, q_u)
                 + tf.matmul(K_Q_uu, k))

            V_, v_, _ = value_fn[-1]

            F_trans = tf.transpose(F)
            f_trans = tf.transpose(f)
            V_f = tf.matmul(V_, f)

            W = C + tf.matmul(F_trans, tf.matmul(V_, F))
            w = c + tf.matmul(F_trans, V_f) + tf.matmul(F_trans, v_)
            W_uu = W[state_size:, state_size:]
            w_u = w[state_size:]

            const1 = 1 / 2 * tf.matmul(k_trans, tf.matmul(W_uu, k))
            const2 = tf.matmul(k_trans, w_u)
            const3 = 1/ 2 * tf.matmul(f_trans, V_f) + tf.matmul(f_trans, v_)
            const += (const1 + const2 + const3)

            policy.append((K, k))
            value_fn.append((V, v, const))

        policy = list(reversed(policy))
        value_fn = list(reversed(value_fn[1:]))

        return policy, value_fn

    @tf.function
    def forward(self, policy, x0, T):
        states = [x0]
        actions = []
        costs = []

        state = x0

        for t in range(T):
            K, k = policy[t]
            action = tf.matmul(K, state) + k

            next_state = self.lqr.transition(state, action)
            cost = self.lqr.cost(state, action)

            state = next_state

            states.append(next_state)
            actions.append(action)
            costs.append(cost)

        final_cost = self.lqr.final_cost(state)
        costs.append(final_cost)

        states = tf.stack(states, axis=0)
        actions = tf.stack(actions, axis=0)
        costs = tf.stack(costs, axis=0)

        return states, actions, costs

    def solve(self, x0, T):
        policy, value_fn = self.backward(T)
        states, actions, costs = self.forward(policy, x0, T)
        return trajectory.Trajectory(states, actions, costs)
