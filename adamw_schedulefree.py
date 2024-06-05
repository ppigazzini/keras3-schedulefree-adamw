import tensorflow.compat.v2 as tf
from tensorflow.keras.optimizers import Optimizer


class AdamWScheduleFree(Optimizer):
    r"""
    This is a tensorflow backed keras implementation of Schedule-Free AdamW optimizer
    [Defazio et al., 2024](https://arxiv.org/abs/2405.15682),

    As the name suggests, no scheduler is needed with this optimizer.
    To add warmup, rather than using a learning rate schedule you can just
    set the warmup_steps parameter.

    This optimizer requires that .set_in_eval_mode() and .set_in_eval_mode() be called
    on model variables before the  beginning of training and evaluation respectively.
    The optimizer should also be placed in eval mode when saving checkpoints.

    Args:
        learning_rate: A `tf.Tensor`, floating point value. The learning rate. Defaults to `0.0025`.
        beta_1: A float value. The momentum parameter for interpolation between Polyak-Ruppert
         averaging (0.0) and Primal averaging (1.0) Defaults to `0.9`.
        beta_2: A float value. The exponential decay rate for the 2nd moment estimates.
            Defaults to `0.999`.
        epsilon: A float value added to the denominator outside the root operation to
            improve numerical stability. Defaults to '1e-8'.
        weight_decay: A float value for weight decay, i.e. a L2 penalty. Defaults to '0.0'.
        warmup_steps: A int value for linear learning rate warmup. Defaults to '0'.
        r: A float value used for polynomial weighting in the average with power r.
            Defaults to '0.0'
        weight_lr_power: A float value. During warmup, the weights in the average will
            be equal to lr raised to this power. Set to 0 for no weighting. Defaults to '2.0'
        {{base_optimizer_keyword_args}}



    Reference:
        - [Defazio et al., 2024](https://arxiv.org/abs/2405.15682)

    Notes:

    According to authors, the optimal learning rate for SFAdamW is 1-10x times higher than
    learning rate for Adam. The initial 0.0025 value for learning rate  is used in original
    PyTorch implementation of SFAdamW.
    Note that while the algorithm provided in the peper needs three sequences (x, y, z) to
    be stored, this implementation computes x on the fly, matching the memory requirements
    of original AdamW.


    """

    def __init__(
            self,
            learning_rate=0.0025,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8,
            weight_decay=0,
            warmup_steps=0,
            r=0.0,
            weight_lr_power=2.0,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            jit_compile=True,
            name="SFAdamW",
            **kwargs,
    ):
        super().__init__(
            name=name,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            jit_compile=jit_compile,
            **kwargs
        )
        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.r = r
        self.warmup_steps = warmup_steps
        self.weight_sum = tf.Variable(initial_value=0.0, dtype=tf.float32)
        self.lr_max = tf.Variable(initial_value=-1.0, dtype=tf.float32)
        self.weight_lr_power = weight_lr_power
        self.train_mode = tf.Variable(initial_value=True, dtype=tf.bool)

    def build(self, var_list):
        """Initialize optimizer variables.

        SFAdamW optimizer has 2 types of variables denoted as z (primary iterates) and v (velocities).

        Args:
            var_list: list of model variables to build SFAdamW variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self._built = True
        self._z = []
        self._v = []
        for var in var_list:
            self._z.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="z", initial_value=var
                )
            )
            self._v.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        gradient = tf.cast(gradient, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        warmup_steps = tf.cast(self.warmup_steps, variable.dtype)
        schedule = tf.cond(tf.greater_equal(warmup_steps, local_step),
                           lambda: local_step / tf.cast(self.warmup_steps, variable.dtype),
                           lambda: 1.0)

        bias_correction2 = 1 - tf.pow(
            tf.cast(self.beta_2, variable.dtype), local_step)

        lr = tf.multiply(lr, schedule)
        lr = tf.multiply(tf.sqrt(bias_correction2), lr)
        self.lr_max.assign(tf.maximum(self.lr_max, lr))

        weight = tf.multiply(tf.pow(local_step, self.r), tf.pow(self.lr_max, self.weight_lr_power))
        self.weight_sum.assign_add(weight)

        ckp1 = tf.math.divide_no_nan(weight, self.weight_sum)

        var_key = self._var_key(variable)
        y = variable  # Notation to match theory
        z = self._z[self._index_dict[var_key]]
        v = self._v[self._index_dict[var_key]]

        v.assign(tf.add(tf.multiply(v, self.beta_2), tf.multiply(1 - self.beta_2, tf.square(gradient))))
        denominator = tf.add(tf.sqrt(v), self.epsilon)
        gradient_normalized = tf.divide(gradient, denominator)

        # Weight decay calculated at y
        if self.weight_decay > 0:
            if self._use_weight_decay(variable):
                tf.add(gradient_normalized, tf.multiply(y, self.weight_decay))

        # These operations update y in-place,
        # without computing x explicitly.
        y.assign_add(tf.multiply(ckp1, tf.subtract(z, y)))
        alpha = tf.multiply(lr, tf.multiply(self.beta_1, (1 - ckp1)) - 1)
        y.assign_add(tf.multiply(alpha, gradient_normalized))

        # z step
        z.assign_sub(tf.multiply(gradient_normalized, lr))

    def set_in_eval_mode(self, var_list):
        if self.train_mode:
            weight = 1 - 1 / self.beta_1
            for var in var_list:
                var_key = self._var_key(var)
                z = self._z[self._index_dict[var_key]]
                var.assign_add(tf.multiply(weight, tf.subtract(z, var)))
            self.train_mode.assign(False)

    def set_in_train_mode(self, var_list):
        if not self.train_mode:
            weight = 1 - self.beta_1
            for var in var_list:
                var_key = self._var_key(var)
                z = self._z[self._index_dict[var_key]]
                var.assign_add(tf.multiply(weight, tf.subtract(z, var)))
            self.train_mode.assign(True)

    def _apply_weight_decay(self, variables):
        # overwriting base optimizers' weight_decay logic as it is already
        # implemented in .update_step()
        return
