# Schedule-Free AdamW for Keras 3.x

Schedule-Free AdamW implementation for Keras 3.x with backend-agnostic support for all official Keras backends (`tensorflow`, `torch`, `jax`) using the new Keras 3 public API.

Original paper: [The Road Less Scheduled](https://arxiv.org/abs/2405.15682)

Authors' PyTorch implementation: https://github.com/facebookresearch/schedule_free

Authors: Aaron Defazio, Xingyu (Alice) Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, Ashok Cutkosky

This code provides `AdamWScheduleFree` optimizer as a replacement for AdamW that does not require learning rate scheduling. It closely tracks the Pareto frontier of loss of cosine scheduled AdamW for any number of steps `t`, but does not require the `t` parameter to be set in advance of training.

## When to Use and What to Expect

- **No Schedule Required:** You do not need to specify a decay schedule (cosine, linear, etc.) or the total number of training steps.
- **Performance:** Typically matches or exceeds a well-tuned cosine-decay schedule.
- **Tuning:** You still need to tune the learning rate and weight decay. In our experiments, learning rates **1x–3x larger** than optimal cosine-decayed AdamW rates often work best.
- **Warmup:** Using `warmup_steps` is highly recommended for stability.

## Algorithm

Schedule-Free replaces first-order momentum with interpolation and averaging between sequences y, z, and x. Let $g_t$ be the gradient at $y_t$ normalized Adam-style, and let $\gamma_t = s_t \cdot \text{lr}$ with a linear warmup schedule $s_t \in (0, 1]$. Define the mixing coefficient $c_{t+1}$ from a cumulative weighted average (polynomial-in-step and lr-weighted during warmup).

We implement true decoupled AdamW weight decay on the parameter $y$.

$$
\begin{aligned}
&\text{Adam RMS norm: } \hat g_t \gets \frac{g_t}{\sqrt{\frac{v_t}{1-\beta_2^{t}}} + \epsilon} \\
&\text{Decoupled weight decay: } y_t \gets y_t - \gamma_t \lambda\, y_t \\
&\text{Schedule-Free updates: } \\
&\quad y_{t+1} \gets y_t + c_{t+1}\,(z_t - y_t) + \gamma_t\big(\beta_1(1-c_{t+1}) - 1\big)\,\hat g_t \\
&\quad z_{t+1} \gets z_t - \gamma_t \hat g_t
\end{aligned}
$$

Notes:
- Warmup: $s_t$ increases linearly to 1; $\gamma_t = s_t \cdot \text{lr}$.
- Averaging: $c_{t+1}$ is computed once per step from cumulative weights; this repo supports $(t+1)^r$ and optional lr-max weighting during warmup.
- Evaluation should be performed at $x_t = \frac{y_t - (1-\beta_1) z_t}{\beta_1}$.

## Parameters

`AdamWScheduleFree(...)` key arguments and defaults:
- `learning_rate`: float. Default `2.5e-3`.
- `beta_1`: float. Default `0.9`. Must be strictly > 0 for schedule-free evaluation.
- `beta_2`: float. Default `0.999`.
- `epsilon`: float. Default `1e-8`.
- `weight_decay`: float. Decoupled AdamW decay on y. Default `0.0`.
- `warmup_steps`: int. Linear LR warmup steps. Default `0`.
- `r`: float. Power for polynomial step weighting in the average. Default `0.0`.
- `weight_lr_power`: float. Power of LR-max used for warmup weighting. Default `2.0`.

## Caveats & Best Practices

1.  **Evaluation Weights (x vs y):**
    The optimizer maintains "y" weights for training but "x" weights for evaluation. You **must** use the provided `ScheduleFreeEvalCallback` (or manually swap weights) during validation and testing to see correct performance.

2.  **BatchNorm & EMA:**
    If your model uses BatchNorm or maintains Exponential Moving Averages (EMA) of weights, be aware that these are updated based on the training "y" weights. For strict correctness, you may need to re-compute statistics using the "x" weights before final evaluation, though for many Keras models the callback swap is sufficient.

3.  **Weight Decay Scope:**
    By default, `weight_decay` applies to **all** trainable variables (including biases and normalization scales). For optimal performance on large models, consider masking weight decay for 1D parameters (biases, LayerNorm scales) if your training recipe requires it.

4.  **Beta_1 Sensitivity:**
    Schedule-Free is more sensitive to `beta_1` than standard momentum. The default `0.9` works well for most problems, but for very long training runs, `0.95` or `0.98` may be better.

## How to Use

You can specify `warmup_steps` in the optimizer initialization for learning rate warmup.

Keras 3.x backend-agnostic usage:
- Set the backend with `os.environ["KERAS_BACKEND"] = "tensorflow"` (or `"torch"`, `"jax"`) before importing Keras.
- The optimizer and model code is the same for all backends.

Unified Example Notebook:
- Use the single notebook `MNIST_test.ipynb` to run experiments on any backend.
- At the top of the notebook, set the backend you want to use:
```python
BACKEND = "jax"  # "jax", "tensorflow", "torch"
import os
os.environ["KERAS_BACKEND"] = BACKEND
```

### Recommended evaluation: automatic x-weights via callback

Use the provided `ScheduleFreeEvalCallback` so validation and evaluate run with the correct x-weights without manual weight swaps.

```python
from adamw_schedulefree import AdamWScheduleFree, ScheduleFreeEvalCallback
import keras

opt = AdamWScheduleFree(
    learning_rate=2.5e-3,
    warmup_steps=steps_per_epoch,
    weight_decay=1e-3,  # decoupled AdamW decay
)

model.compile(
    optimizer=opt,
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=["accuracy"],
)

# Validation uses x-weights each epoch
model.fit(
    x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    callbacks=[ScheduleFreeEvalCallback()],
)

# Final evaluation with x-weights
model.evaluate(x_test, y_test, callbacks=[ScheduleFreeEvalCallback()])
```

### Alternative: explicit stateless evaluation (backend-agnostic)

If you prefer the explicit, stateless pattern (JAX/Optax-style), compute x-weights and assign them temporarily. Use `keras.ops` to stay backend-agnostic:

```python
import keras

# Save current weights as NumPy arrays (backend-agnostic)
original = [keras.ops.convert_to_numpy(w) for w in model.trainable_weights]

# Compute stateless evaluation weights from the optimizer
eval_weights = model.optimizer.get_eval_weights(model)

# Assign x-weights, evaluate, then restore
for w, v in zip(model.trainable_weights, eval_weights):
    w.assign(v)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss (x-weights):", score[0])
print("Test accuracy (x-weights):", score[1])

for w, v_np in zip(model.trainable_weights, original):
    w.assign(keras.ops.convert_to_tensor(v_np, dtype=w.dtype))
```

Notes:
- `ScheduleFreeEvalCallback` is recommended for most users; it’s backend-agnostic and wrapper-safe.
- This evaluation pattern mirrors Optax’s `schedule_free_eval_params`, which evaluates at x-weights, but the training implementation in this repo follows the authors' PyTorch `AdamWScheduleFree` optimizer rather than the Optax wrapper.
- The stateless method mirrors the Optax reference:
  https://github.com/google-deepmind/optax/blob/main/optax/contrib/_schedule_free.py

---

This repository supports Keras 3.x with TensorFlow, PyTorch, and JAX backends.
For legacy Keras 2.x (TensorFlow-only), see:
https://github.com/sseltref/schedule_free_AdamW_tf-keras
