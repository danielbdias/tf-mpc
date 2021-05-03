from collections import namedtuple

import tensorflow as tf

from tfmpc.solvers.box_qp import BoxQP
from tfmpc.utils import trajectory


Trajectory = namedtuple("Trajectory", "states actions costs")
Controller = namedtuple("Controller", "K k")
DynamicsModel = namedtuple("DynamicsModel", "f f_x f_u")
CostModel = namedtuple("CostModel", "l l_x l_u l_xx l_uu l_ux l_xu")
FinalCostModel = namedtuple("FinalCostModel", "l l_x l_xx")
ValueModel = namedtuple("ValueModel", "V_x V_xx")
QModel = namedtuple("QModel", "Q_x Q_u Q_xx Q_uu Q_ux")


class BoxILQR:

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
            c2=1.0,
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

        self._box_qp = BoxQP()

    def start(self, x0, T):
        """Returns an initial feasible candidate trajectory.

        Args:
            x0: the initial state.
            T: the horizon of the task.

        Returns:
            (Trajectory, total_cost): the integrated trajectory and its total cost.
        """
        states = [x0]
        costs = []
        actions = []

        total_cost = 0.0

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
            total_cost += cost

            actions.append(action)
            states.append(state)
            costs.append(cost)

        final_cost = self.env.final_cost(state)
        total_cost += final_cost
        costs.append(final_cost)

        states = tf.stack(states, axis=0)
        actions = tf.stack(actions, axis=0)
        costs = tf.stack(costs, axis=0)

        return Trajectory(states, actions, costs), total_cost

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

    @tf.function
    def backward(self, T, f_model, l_model, l_final_model):
        """Implements the DDP backward pass.

        Args:
            T: the horizon of the task.
            f_model: dynamics approximation model.
            l_model: running cost approximation model.
            l_final_model: final cost approximation model.

        Returns:
            (controller, dV1, dV2, g_norm): the controller, the corresponding
            expected value improvement terms, and the total gradient norm.
        """
        action_size = self.env.action_size
        state_size = self.env.state_size

        K = tf.TensorArray(
            size=T, dtype=tf.float32, element_shape=(action_size, state_size)
        )
        k = tf.TensorArray(
            size=T, dtype=tf.float32, element_shape=(action_size, 1)
        )

        dV1 = tf.TensorArray(size=T, dtype=tf.float32, element_shape=())
        dV2 = tf.TensorArray(size=T, dtype=tf.float32, element_shape=())
        g_norm = tf.TensorArray(size=T, dtype=tf.float32, element_shape=())

        V_model = ValueModel(l_final_model.l_x, l_final_model.l_xx)

        for t in tf.range(T - 1, -1, -1):
            # evaluation
            f_t_model = DynamicsModel(*[m[t] for m in f_model])
            l_t_model = CostModel(*[m[t] for m in l_model])
            Q_model = self._hamiltonian(f_t_model, l_t_model, V_model)

            # optimization
            controller = tf.py_function(
                self._controller,
                inp=[
                    self._candidate.actions[t],
                    Q_model.Q_uu,
                    Q_model.Q_ux,
                    Q_model.Q_u,
                ],
                Tout=[tf.float32, tf.float32],
            )
            controller = Controller(*controller)

            K = K.write(t, controller.K)
            k = k.write(t, controller.k)

            # value update
            V_model = self._value_update(Q_model, controller)

            # expected improvement terms
            dV1 = dV1.write(t, tf.squeeze(tf.matmul(controller.k, Q_model.Q_u, transpose_a=True)))
            dV2 = dV2.write(t,
                tf.squeeze(
                    tf.matmul(
                        tf.matmul(controller.k, Q_model.Q_uu, transpose_a=True),
                        controller.k)))

            # gradient norm (stopping criteria)
            g_norm = g_norm.write(t, tf.reduce_sum(Q_model.Q_u ** 2))

        K = K.stack()
        k = k.stack()

        dV1 = tf.reduce_sum(dV1.stack())
        dV2 = tf.reduce_sum(dV2.stack())
        g_norm = tf.reduce_sum(g_norm.stack())

        return Controller(K, k), dV1, dV2, g_norm

    @tf.function
    def forward(self, candidate, controller, alpha):
        """Implements the DDP forward pass.

        Args:
            candidate: the current candidate trajectory.
            controller: the (K, k) gain matrices of the recent controller.
            alpha: the line-search parameter.

        Returns:
            (trajectory, total_cost, residual): A new candidate trajectory and
            its total cost and residual.
        """
        x = candidate.states
        u = candidate.actions

        T = tf.shape(u)[0]

        state_size = self.env.state_size
        action_size = self.env.action_size
        low = self.env.action_space.low
        high = self.env.action_space.high

        states = tf.TensorArray(
            size=T+1, dtype=tf.float32, element_shape=(state_size, 1)
        )
        actions = tf.TensorArray(
            size=T, dtype=tf.float32, element_shape=(action_size, 1)
        )
        costs = tf.TensorArray(
            size=T+1, dtype=tf.float32, element_shape=()
        )

        states = states.write(0, x[0])

        total_cost = 0.0
        residual = 0.0

        x_t = x[0]

        for t in tf.range(T):
            K_t = controller.K[t]
            k_t = controller.k[t]

            delta_x = x_t - x[t]
            delta_u = alpha * k_t + tf.matmul(K_t, delta_x)

            u_t = tf.clip_by_value(u[t] + delta_u, low, high)

            x_t = tf.expand_dims(x_t, axis=0)
            u_t = tf.expand_dims(u_t, axis=0)

            c_t = tf.squeeze(self.env.cost(x_t, u_t), axis=0)
            x_t = tf.squeeze(self.env.transition(x_t, u_t), axis=0)
            u_t = tf.squeeze(u_t, axis=0)

            states = states.write(t+1, x_t)
            actions = actions.write(t, u_t)
            costs = costs.write(t, c_t)

            total_cost += c_t
            residual = tf.math.maximum(residual, tf.reduce_max(tf.abs(delta_u)))

        c_T = self.env.final_cost(x_t)
        costs = costs.write(T, c_T)
        total_cost += c_T

        states = states.stack()
        actions = actions.stack()
        costs = costs.stack()

        return Trajectory(states, actions, costs), total_cost, residual

    def solve(self, x0, T, show_progress=True):
        """Solves the task by iteratively running backward-forward DDP passes.

        Initializes a feasible solution and incrementally improves it
        until convergence or the maximum number of iterations is achieved.

        Args:
            x0: the initial state.
            T: the horizon of the task.
            show_progress: the boolean flag for progress monitoring.

        Returns:
            (trajectory, num_iterations, converged): the best trajectory,
            the number of iterations used, and the convergence flag.
        """

        self._candidate, self._current_total_cost = self.start(x0, T)

        converged = False

        T = tf.constant(T)

        for n_iter in range(1, self.max_iterations + 1):

            # local approximation
            f_model, l_model, l_final_model = self.derivatives(
                self._candidate.states,
                self._candidate.actions
            )

            # backward
            backward_done = False

            while not backward_done:
                try:
                    controller, dV1, dV2, g_norm = self.backward(
                        T, f_model, l_model, l_final_model
                    )

                    backward_done = True

                except tf.errors.InvalidArgumentError as e: # Q_uu not positive-definite
                    self._increase_regularization()

                    if self._reg >= self.reg_max:
                        converged = False
                        return self._candidate, n_iter, converged

            # forward
            for k in range(self.step_length_num):
                alpha = tf.constant(self.step_length_factor ** k)

                trajectory, total_cost, residual = self.forward(
                    self._candidate, controller, alpha
                )

                dV = self._expected_improvement(dV1, dV2, alpha)

                if self._accept_criteria(total_cost, dV):
                    self._candidate = trajectory
                    self._current_total_cost = total_cost

                    break

            # regularization
            if alpha > self.step_length_max:
                self._decrease_regularization()

            elif alpha <= self.step_length_min:
                self._increase_regularization()

                if self._reg >= self.reg_max:
                    converged = False
                    return self._candidate, n_iter, converged

            # stopping criteria
            if self._stopping_criteria(g_norm, residual):
                converged = True
                return self._candidate, n_iter, converged

        return self._candidate, n_iter, converged

    def _hamiltonian(self, f_model, l_model, V_model):
        """Returns the Hamiltonian model.

        Args:
            f_model: the dynamics approximated model.
            l_model: the running cost approximated model.
            V_model: the next state Value function model.

        Returns:
            (QModel): the regularized model of the Hamiltonian.
        """
        f_x = f_model.f_x
        f_u = f_model.f_u
        l_x = l_model.l_x
        l_u = l_model.l_u
        l_xx = l_model.l_xx
        l_xu = l_model.l_xu
        l_uu = l_model.l_uu
        V_x = V_model.V_x
        V_xx = V_model.V_xx

        f_x_trans = tf.transpose(f_x)
        f_u_trans = tf.transpose(f_u)

        Q_x = l_x + tf.matmul(f_x_trans, V_x)
        Q_u = l_u + tf.matmul(f_u_trans, V_x)

        f_x_trans_V_xx = tf.matmul(f_x_trans, V_xx)
        f_u_trans_V_xx = tf.matmul(f_u_trans, V_xx)
        f_u_trans_V_xx_reg = tf.matmul(
            f_u_trans, V_xx + self._reg * tf.eye(self.env.state_size))

        Q_xx = l_xx + tf.matmul(f_x_trans_V_xx, f_x)
        Q_uu = l_uu + tf.matmul(f_u_trans_V_xx, f_u)
        Q_ux = tf.transpose(l_xu) + tf.matmul(f_u_trans_V_xx, f_x)

        Q_uu_reg = l_uu + tf.matmul(f_u_trans_V_xx_reg, f_u)
        Q_ux_reg = tf.transpose(l_xu) + tf.matmul(f_u_trans_V_xx_reg, f_x)

        return QModel(Q_x, Q_u, Q_xx, Q_uu_reg, Q_ux)

    def _controller(self, u, Q_uu, Q_ux, Q_u):
        """Returns the controller for the Hamiltonian action-based components.

        Args:
            Q_uu: the action Hessian of the Q-function approximation.
            Q_u: the action gradient of the Q-function approximation.
            Q_ux: the action-state Hessian of the Q-function approximation.

        Returns:
            (Controller): the gains (K, k) of the controller.
        """
        low = self.env.action_space.low - u
        high = self.env.action_space.high - u

        #k_0 = tf.zeros_like(u)
        k_0 = (low + high) / 2

        self._box_qp.setup(Q_uu, Q_u, low, high)
        k, Hff_inv_llt, free, clamped = self._box_qp.solve(k_0)

        action_dim = self.env.action_size
        state_dim = self.env.state_size
        K = tf.Variable(tf.zeros([action_dim, state_dim]))

        n_free = tf.math.count_nonzero(tf.squeeze(free))
        if n_free > 0:
            free = tf.squeeze(free)
            Q_ux_f = Q_ux[free]
            indices = tf.where(free)
            values = - tf.linalg.cholesky_solve(Hff_inv_llt, Q_ux_f)
            K.scatter_nd_update(indices, values)

        K = tf.constant(K.numpy())

        return Controller(K, k)

    def _value_update(self, Q_model, controller):
        """Returns the improved Value update for the current controller.

        Args:
            Q_model: the regularized model of the Hamiltonian.
            controller: the (K, K) controller.

        Returns:
            (ValueModel): the (V_x, V_xx) optimal value approximation model.
        """
        Q_x = Q_model.Q_x
        Q_u = Q_model.Q_u
        Q_xx = Q_model.Q_xx
        Q_uu = Q_model.Q_uu
        Q_ux = Q_model.Q_ux
        K, k = controller

        K_trans = tf.transpose(K)
        k_trans = tf.transpose(k)
        K_trans_Q_uu = tf.matmul(K_trans, Q_uu)

        V_x = tf.reshape(
            (Q_x
             + tf.matmul(Q_ux, k, transpose_a=True)
             + tf.matmul(K_trans, Q_u)
             + tf.matmul(K_trans_Q_uu, k)),
            shape=[self.env.state_size, 1]
        )

        V_xx = tf.reshape(
            (Q_xx
             + tf.matmul(Q_ux, K, transpose_a=True)
             + tf.matmul(K_trans, Q_ux)
             + tf.matmul(K_trans_Q_uu, K)),
            shape=[self.env.state_size, self.env.state_size]
        )
        V_xx = 1 / 2 * (V_xx + tf.transpose(V_xx))

        return ValueModel(V_x, V_xx)

    def _increase_regularization(self):
        """Increases the current regularization parameter."""
        self._reg = min(self.reg_max, self._reg * self.reg_factor)

    def _decrease_regularization(self):
        """Decreases the current regularization parameter."""
        self._reg = max(self.reg_min, self._reg / self.reg_factor)

    def _expected_improvement(self, dV1, dV2, alpha):
        """Returns the expected value improvement."""
        return -alpha * (dV1 + 0.5 * alpha * dV2)

    def _accept_criteria(self, total_cost, dV):
        """Returns True iff the candidate acceptance criteria is met."""
        z = (self._current_total_cost - total_cost) / dV
        return self.c1 < z

    def _stopping_criteria(self, g_norm, residual):
        """Returns True iif stopping criteria is met."""
        return (g_norm < self.g_norm_threshold) or (residual < self.residual_threshold)
