"""Schedule-Free AdamW optimizer for Keras 3.x with backend-agnostic support."""

from __future__ import annotations

import keras
from keras import callbacks, ops
from keras.optimizers import Optimizer
from keras.utils import register_keras_serializable


@register_keras_serializable(package="keras_adamw_sf")
class AdamWScheduleFree(Optimizer):
    """A Keras 3.x backend-agnostic implementation of the Schedule-Free AdamW optimizer.

    See: https://arxiv.org/abs/2405.15682

    Args:
        learning_rate: The learning rate. Defaults to 0.0025.
        beta_1: Interpolation momentum parameter. Must be > 0. Defaults to 0.9.
        beta_2: Exponential decay rate for 2nd moment estimates. Defaults to 0.999.
        epsilon: Small constant for numerical stability. Defaults to 1e-8.
        weight_decay: Decoupled weight decay (L2) applied to y-parameters.
            Defaults to 0.0.
        warmup_steps: Steps for linear learning-rate warmup. Defaults to 0.
        r: Power for polynomial weighting in the update schedule. Defaults to 0.0.
        weight_lr_power: Power of lr-max used during warmup weighting.
            Defaults to 2.0.
        state_dtype: Optional dtype for state variables, e.g., "float32".
            Defaults to None.
        name: Optimizer name. Defaults to "AdamWScheduleFree".
        **kwargs: Additional Optimizer keyword arguments.

    Raises:
        ValueError: If `beta_1` is not strictly positive.

    """

    # ruff: noqa: PLR0913
    def __init__(
        self,
        learning_rate: float = 0.0025,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        state_dtype: str | None = None,
        name: str = "AdamWScheduleFree",
        **kwargs: object,  # type: ignore[ANN003]
    ) -> None:
        """Initialize the optimizer with hyperparameters.

        Args:
            learning_rate: The learning rate.
            beta_1: Interpolation momentum parameter. Must be > 0.
            beta_2: Exponential decay rate for 2nd moment estimates.
            epsilon: Small constant for numerical stability.
            weight_decay: Decoupled weight decay (L2) applied to y-parameters.
            warmup_steps: Steps for linear learning-rate warmup.
            r: Power for polynomial weighting in the schedule.
            weight_lr_power: Power of lr-max used during warmup weighting.
            state_dtype: Optional dtype for state variables, e.g., "float32".
            name: Optimizer name.
            **kwargs: Additional Optimizer keyword arguments.

        Raises:
            ValueError: If `beta_1` is not strictly positive.

        """
        if not (beta_1 > 0.0):
            msg = "beta_1 must be strictly positive for Schedule-Free evaluation."
            raise ValueError(msg)
        super().__init__(
            learning_rate=learning_rate,
            name=name,
            weight_decay=0.0,  # handled decoupled manually
            **kwargs,
        )
        self.beta_1 = float(beta_1)
        self.beta_2 = float(beta_2)
        self.epsilon = float(epsilon)
        self.decoupled_weight_decay = float(weight_decay)
        self.warmup_steps = int(warmup_steps)
        self.r = float(r)
        self.weight_lr_power = float(weight_lr_power)
        self.state_dtype = state_dtype  # e.g., "float32" to stabilize accumulators

    def build(self, var_list: list[object]) -> None:
        """Initialize optimizer state variables for each model variable.

        Args:
            var_list: List of model variables to build optimizer state for.

        Raises:
            ValueError: If `var_list` is empty.

        """
        if self.built:
            return
        if not var_list:
            msg = "var_list must be non-empty to build optimizer state."
            raise ValueError(msg)
        super().build(var_list)

        # Resolve a numerically stable dtype for optimizer state.
        if self.state_dtype is not None:
            state_dtype = ops.convert_to_tensor(0, dtype=self.state_dtype).dtype
        else:
            # Prefer at least float32 precision for accumulators.
            state_dtype = ops.convert_to_tensor(0, dtype="float32").dtype

        # Create per-variable state using the reference dtype/shape
        self._z: list = [self.add_variable_from_reference(v, "z") for v in var_list]
        self._exp_avg_sq: list = self.add_optimizer_variables(var_list, "exp_avg_sq")

        # Per-step accumulators
        self._weight_sum = self.add_variable(
            shape=(),
            name="weight_sum",
            dtype=state_dtype,
        )
        self._lr_max = self.add_variable(
            shape=(),
            name="lr_max",
            initializer=keras.initializers.Constant(-1.0),
            dtype=state_dtype,
        )
        # Per-step schedule and mixing coefficient
        self._schedule = self.add_variable(
            shape=(),
            name="schedule",
            dtype=state_dtype,
        )
        self._c = self.add_variable(
            shape=(),
            name="c",
            dtype=state_dtype,
        )

    def _prepare_step(self, trainable_variables: list[object]) -> None:
        """Compute per-step globals and lazily initialize z on step 0.

        Computes globals: schedule, lr_max, weight_sum, c.

        Args:
            trainable_variables: Current step's trainable variables.

        """

        def _init() -> None:
            for var in trainable_variables:
                # Use index lookup to ensure correct z-variable pairing
                # and handle cases where trainable_variables is a subset or reordered.
                var_index = self._get_variable_index(var)
                z = self._z[var_index]
                self.assign(z, var)

        ops.cond(self.iterations == 0, _init, lambda: None)

        dtype = trainable_variables[0].dtype
        k = ops.cast(self.iterations + 1, dtype)  # 1-based like PyTorch

        # Linear warmup to 1.0 matching torch/jax (k/warmup_steps, clipped at 1).
        if self.warmup_steps > 0:
            warm = ops.minimum(
                k / ops.cast(self.warmup_steps, dtype),
                ops.cast(1.0, dtype),
            )
            sched = warm
        else:
            sched = ops.cast(1.0, dtype)
        self.assign(self._schedule, sched)

        # Scheduled LR and running max for weighting
        lr = ops.cast(self.learning_rate, dtype)
        lr_scheduled = lr * sched
        self.assign(self._lr_max, ops.maximum(self._lr_max, lr_scheduled))

        # Weight and cumulative sum; then c_{k+1}
        weight = ops.power(k, ops.cast(self.r, dtype)) * ops.power(
            self._lr_max,
            ops.cast(self.weight_lr_power, dtype),
        )
        self.assign_add(self._weight_sum, weight)
        weight_sum = ops.cast(self._weight_sum, dtype)
        c_kp1 = ops.where(weight_sum > 0, weight / weight_sum, ops.cast(0.0, dtype))
        self.assign(self._c, c_kp1)

    def _backend_apply_gradients(
        self,
        grads: list,
        trainable_variables: list,
    ) -> None:  # type: ignore[override]
        """Prepare step-wise scalars once per optimizer step.

        Args:
            grads: Gradients list aligned with `trainable_variables`.
            trainable_variables: List of variables to update.

        """
        if trainable_variables:
            self._prepare_step(trainable_variables)
        super()._backend_apply_gradients(grads, trainable_variables)

    def update_step(
        self,
        gradient: object,
        variable: object,
        learning_rate: float,
    ) -> None:
        """Perform a single optimization step for a given variable.

        The update equations are:
        grad_effective = grad_norm + weight_decay * y
        y = y + c * (z - y) + gamma * (beta_1 * (1 - c) - 1) * grad_effective
        z = z - gamma * grad_effective

        where grad_norm is the Adam-normalized gradient.

        Args:
            gradient: Gradient tensor for the variable.
            variable: Model variable to update.
            learning_rate: Learning rate for this step (scalar or tensor).

        """
        lr = ops.cast(learning_rate, variable.dtype)
        beta_1 = ops.cast(self.beta_1, variable.dtype)
        beta_2 = ops.cast(self.beta_2, variable.dtype)
        epsilon = ops.cast(self.epsilon, variable.dtype)
        k = ops.cast(self.iterations + 1, variable.dtype)

        var_index = self._get_variable_index(variable)
        z = self._z[var_index]
        exp_avg_sq = self._exp_avg_sq[var_index]

        # Use prepared per-step scalars
        s_k = ops.cast(self._schedule, variable.dtype)
        ckp1 = ops.cast(self._c, variable.dtype)
        gamma = lr * s_k  # effective step size

        # Adam RMS normalization with bias correction
        self.assign(
            exp_avg_sq,
            exp_avg_sq * beta_2 + ops.square(gradient) * (1 - beta_2),
        )
        bias_correction2 = 1 - ops.power(beta_2, k)
        denom = ops.sqrt(exp_avg_sq / bias_correction2) + epsilon
        grad_normalized = gradient / denom

        # Fold weight decay into an effective gradient used for both y and z.
        if self.decoupled_weight_decay > 0.0:
            grad_effective = grad_normalized + self.decoupled_weight_decay * variable
        else:
            grad_effective = grad_normalized

        # Schedule-Free updates
        self.assign_add(variable, (z - variable) * ckp1)
        alpha = gamma * (beta_1 * (1 - ckp1) - 1.0)
        self.assign_add(variable, grad_effective * alpha)
        self.assign_sub(z, grad_effective * gamma)

    def get_config(self) -> dict:
        """Return the optimizer configuration."""
        config = super().get_config()
        config.update(
            {
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "epsilon": self.epsilon,
                "weight_decay": self.decoupled_weight_decay,
                "warmup_steps": self.warmup_steps,
                "r": self.r,
                "weight_lr_power": self.weight_lr_power,
            },
        )
        return config

    def get_eval_weights(self, model: object) -> list[object]:
        """Compute stateless x-weights for evaluation without in-place mutation.

        x-weights: x = (y - (1 - beta1) * z) / beta1.

        Args:
            model: A built Keras model whose trainable weights match optimizer state.

        Raises:
            ValueError: If `beta_1` is not strictly positive.

        """
        if self.beta_1 <= 0.0:
            msg = "beta_1 must be > 0 to compute evaluation weights x from (y, z)."
            raise ValueError(msg)
        x_weights: list[object] = []
        for w in model.trainable_weights:
            try:
                var_index = self._get_variable_index(w)
            except ValueError:
                # Variable not tracked by this optimizer; return y as-is.
                x_weights.append(w)
                continue

            z = self._z[var_index]
            x = (w - (1.0 - self.beta_1) * z) / self.beta_1
            x_weights.append(x)
        return x_weights


def _unwrap_optimizer(opt: object) -> object:
    """Unwrap common Keras optimizer wrappers to get the base optimizer.

    Walks known attributes such as `optimizer`, `inner_optimizer`, `base_optimizer`,
    and `_optimizer` until the base optimizer is reached.

    Args:
        opt: Possibly wrapped optimizer instance.

    Returns:
        object: The unwrapped base optimizer instance.

    """
    base = opt
    # Walk through common wrapper attributes until none match.
    while True:
        for attr in ("optimizer", "inner_optimizer", "base_optimizer", "_optimizer"):
            if hasattr(base, attr):
                base = getattr(base, attr)
                break
        else:
            break
    return base


class ScheduleFreeEvalCallback(callbacks.Callback):
    """Swap y-weights with x-weights during evaluation and prediction.

    This callback integrates schedule-free evaluation with Keras by:
      - Computing stateless x-weights via the optimizer's `get_eval_weights`.
      - Assigning them to the model's trainable variables at test/predict begin.
      - Restoring original y-weights at test/predict end.

    Requirements:
      - The model's base optimizer must implement `get_eval_weights(model)`.
      - Works with wrapped optimizers (unwraps to the base).

    Notes:
      - Only trainable variables are swapped; non-trainables remain unchanged.
      - Uses optimizer.assign(...) to remain backend-agnostic.

    Example:
      model.compile(optimizer=AdamWScheduleFree(...), ...)
      model.evaluate(x, y, callbacks=[ScheduleFreeEvalCallback()])

    """

    def __init__(self) -> None:
        """Initialize the callback with empty backup storage."""
        super().__init__()
        self._backup_weights: list[object] | None = None

    def on_test_begin(self, _logs: dict | None = None) -> None:
        """Swap to x-weights at the start of evaluation."""
        self._swap_to_eval_weights()

    def on_test_end(self, _logs: dict | None = None) -> None:
        """Restore original y-weights at the end of evaluation."""
        self._restore_backup_weights()

    def on_predict_begin(self, _logs: dict | None = None) -> None:
        """Swap to x-weights at the start of prediction."""
        self._swap_to_eval_weights()

    def on_predict_end(self, _logs: dict | None = None) -> None:
        """Restore original y-weights at the end of prediction."""
        self._restore_backup_weights()

    def _swap_to_eval_weights(self) -> None:
        """Compute x-weights and assign them to model variables for eval/predict.

        Backs up current trainable weights as NumPy arrays for reliable restore
        across backends and dtypes. If the optimizer provides `get_eval_weights`,
        it is used; otherwise, x-weights are computed from state.
        """
        if self.model is None:
            return
        base_opt = _unwrap_optimizer(self.model.optimizer)

        # Backup as numpy for reliable restore across backends/dtypes.
        self._backup_weights = [
            ops.convert_to_numpy(v) for v in self.model.trainable_weights
        ]

        # Prefer optimizer-provided method; fall back to direct computation.
        x_weights: list[object]
        if hasattr(base_opt, "get_eval_weights"):
            try:
                x_weights = base_opt.get_eval_weights(self.model)
            except Exception:  # noqa: BLE001
                x_weights = self._compute_x_from_state(base_opt)
        else:
            x_weights = self._compute_x_from_state(base_opt)

        # Assign x-weights directly to variables (backend-agnostic).
        for var, x in zip(self.model.trainable_weights, x_weights, strict=True):
            var.assign(x)

    def _compute_x_from_state(self, base_opt: object) -> list[object]:
        """Compute x-weights from optimizer state: x = (y - (1 - beta1) z) / beta1.

        Args:
            base_opt: The (possibly unwrapped) optimizer with `beta_1` and `_z` state.

        Raises:
            RuntimeError: If required optimizer attributes are missing.

        """
        beta_1 = getattr(base_opt, "beta_1", None)
        z_vars = getattr(base_opt, "_z", None)
        if beta_1 is None or z_vars is None:
            msg = (
                "ScheduleFreeEvalCallback needs either "
                "optimizer.get_eval_weights(model) "
                "or optimizer state attributes '_z' and 'beta_1' to compute x-weights."
            )
            raise RuntimeError(msg)
        return [
            (w - (1.0 - float(beta_1)) * z) / float(beta_1)
            for w, z in zip(self.model.trainable_weights, z_vars, strict=True)
        ]

    def _restore_backup_weights(self) -> None:
        """Restore previously backed-up y-weights after eval/predict concludes."""
        if self.model is None or self._backup_weights is None:
            return
        for var, w_np in zip(
            self.model.trainable_weights,
            self._backup_weights,
            strict=True,
        ):
            var.assign(ops.convert_to_tensor(w_np, dtype=var.dtype))
        self._backup_weights = None
