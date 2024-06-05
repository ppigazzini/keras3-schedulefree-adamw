# Schedule-Free AdamW keras
Schedule-Free AdamW implementation in tf-keras.
This implementation works only with pure TensorFlow implementation of Keras, i.e Keras 2.x.

Original paper: [The Road Less Scheduled](https://arxiv.org/abs/2405.15682)

Authors' PyTorch implementation: https://github.com/facebookresearch/schedule_free

Authors: Aaron Defazio, Xingyu (Alice) Yang, Harsh Mehta, Konstantin Mishchenko, Ahmed Khaled, Ashok Cutkosky

This code provides `AdamWScheduleFree` optimizer as a replacement for AdamW that does not require learning rate scheduling. It closely tracks the Pareto frontier of loss of cosine scheduled AdamW for any number of steps `t` , but does not require the `t` parameter do be set in advance of training.

`MNIST_test.ipynb` compares Adam and Schedule-Free Adam training results on MNIST dataset

## Algorithm
In Schedule-Free AdamW, first order momentum is replaced with combination of interpolation and averaging:

$$
\begin{align*}
y_{t} & = (1-\beta)z_{t} + \beta x_{t},\\
z_{t+1} & =z_{t}-\gamma\nabla f(y_{t}),\\
x_{t+1} & =\left(1-\frac{1}{t+1}\right)x_{t}+\frac{1}{t+1}z_{t+1},
\end{align*}
$$

Here $x$ is the sequence that evaluations of test/val loss should occur at, which differs from the primary iterates $z$ and the gradient evaluation locations $y$. The updates to $z$ correspond to Adam updates with `beta_1` set to `0`.

## How to Use

You can specify `warmup_steps` in the optimizer intialization for learning rate warmup.

Use `.set_in_train_mode()` and `.set_in_eval_mode()` to set model weights to $y$ and $x$ respectively. You can do it by adding a following callback to `model.fit()`:
```
class ChangeModeCallback(tf.keras.callbacks.Callback):
    def __init__(self, steps_per_epoch):
        super(ChangeModeCallback, self).__init__()
        self.steps_per_epoch = steps_per_epoch
        
    def on_epoch_begin(self, epoch, logs=None):
        self.model.optimizer.set_in_train_mode(self.model.trainable_variables)
    
    def on_train_batch_end(self, batch, logs=None):
        if batch == self.steps_per_epoch - 1: 
            self.model.optimizer.set_in_eval_mode(self.model.trainable_variables)
```
