import tensorflow as tf


class BoxQP:
    """BoxQP implements the projected Newton QP solution.

    For details please see:

    >> Control-Limited Differential Dynamic Programming
    >> Tassa, Mansard, and Todorov (2014)
    """

    def __init__(
            self,
            max_iterations=100,
            step_threshold=0.1,
            g_norm_threshold=1e-9,
            step_length_factor=0.5,
            step_length_num=10,
    ):
        self.max_iterations = max_iterations

        # convergence criteria
        self.step_threshold = step_threshold
        self.g_norm_threshold = g_norm_threshold

        # line-search parameters
        self.step_length_factor = step_length_factor
        self.step_length_num = step_length_num

    def setup(self, H, q, low, high):
        # QP specification
        self._H = H
        self._q = q
        self._low = low
        self._high = high

        # QP solution
        self._solution = None
        self._Hff_inv_llt = None
        self._free = None
        self._clamped = None

    def solve(self, x_init):
        # enforce feasible warm start
        x = tf.clip_by_value(x_init, self._low, self._high)

        self._solution = x
        self._clamped = tf.zeros_like(x)

        for n_iter in range(self.max_iterations):
            old_clamped = self._clamped

            # compute gradient
            grad = self._compute_gradient(self._solution)

            # find constraint active set
            self._free, self._clamped = self._get_active_set(grad, self._solution)

            # factorization
            if n_iter == 0 or tf.reduce_any(old_clamped != self._clamped):
                self._Hff_inv_llt = self._factorize(self._free)

            # check convergence
            g_norm = tf.norm(grad[self._free])
            if g_norm < self.g_norm_threshold:
                break

            # get projected Newton descent direction
            dx = self._projected_newton_step(
                self._solution, self._Hff_inv_llt, self._free, self._clamped)

            # line search over descent direction
            self._solution = self._line_search(self._solution, dx, grad)

        return self._solution, self._Hff_inv_llt, self._free, self._clamped

    def _compute_gradient(self, x):
        return self._q + tf.matmul(self._H, x)

    def _get_active_set(self, g, x, eps=1e-8):
        clamped = tf.logical_or(
            tf.logical_and(tf.abs(x - self._low) < eps, g > 0),
            tf.logical_and(tf.abs(self._high - x) < eps, g < 0)
        )
        free = tf.logical_not(clamped)
        return free, clamped

    def _factorize(self, free):
        free_int = tf.cast(free, tf.int32)
        dim_f = tf.math.count_nonzero(free_int)
        ff = tf.cast(tf.matmul(free_int, free_int, transpose_b=True), tf.bool)
        H_ff = tf.reshape(self._H[ff], [dim_f, dim_f])
        Hff_inv_llt = tf.linalg.cholesky(H_ff)
        return Hff_inv_llt

    def _projected_newton_step(self, x, Hff_inv_llt, free, clamped):
        grad_clamped = self._q + tf.matmul(self._H, x * tf.cast(clamped, tf.float32))
        indices = tf.where(free)
        x_free = tf.expand_dims(x[free], axis=-1)
        grad_clamped_free = tf.expand_dims(grad_clamped[free], axis=-1)

        dx_free = -tf.linalg.cholesky_solve(Hff_inv_llt, grad_clamped_free) - x_free
        dx = tf.Variable(tf.zeros_like(x))
        dx_free = tf.reshape(dx_free, [-1])
        dx.scatter_nd_update(indices, dx_free)
        return dx

    def _line_search(self, x, dx, grad):
        old_value = self._qp_evaluation(x)
        step_length = 1.0
        for _ in range(self.step_length_num):
            new_x = tf.clip_by_value(x + step_length * dx, self._low, self._high)
            new_value = self._qp_evaluation(new_x)
            expected_improvement = tf.matmul(grad, (x - new_x), transpose_a=True)
            value_change = old_value - new_value
            if value_change > self.step_threshold * expected_improvement:
                x = new_x
                break
            step_length *= self.step_length_factor
        return x

    def _qp_evaluation(self, x):
        quadratic = tf.matmul(x, tf.matmul(self._H, x), transpose_a=True)
        linear = tf.matmul(self._q, x, transpose_a=True)
        return tf.squeeze(1 / 2 * quadratic + linear)
