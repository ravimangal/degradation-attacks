import tensorflow as tf

from tensorflow.keras.losses import SparseCategoricalCrossentropy


class RsL2Pgd(object):

    def __init__(
        self,
        epsilon,
        sigma=0.125,
        samples=100,
        steps=50,
        step_size=None,
        loss=None,
        aggregate_before_grad=False,
    ):
        self.epsilon = epsilon
        self.sigma = sigma
        self.samples = samples
        self.steps = steps

        if step_size is None:
            self.step_size = epsilon / steps * 2
        else:
            self.step_size = step_size

        if loss is None:
            self.loss = SparseCategoricalCrossentropy(from_logits=True)
        else:
            self.loss = loss

        self.aggregate_before_grad = aggregate_before_grad

    def __call__(self, model, x, y):
        n = x.shape[0]
        h = x.shape[1]
        w = x.shape[2]
        c = x.shape[3]

        x_0 = x

        for _ in range(self.steps):

            y_noised = tf.repeat(y, self.samples, axis=0)

            noise = tf.random.normal(
                (self.samples, 1, h, w, c), stddev=self.sigma)

            with tf.GradientTape() as tape:
                if self.aggregate_before_grad:
                    tape.watch(x)
                else:
                    tape.watch(x_noised)

                x_noised = tf.reshape(x[None] + noise, (-1, h, w, c))
                y_pred = model(x_noised)

                if self.aggregate_before_grad:
                    loss = self.loss(
                        y,
                        tf.reduce_mean(
                            tf.reshape(y_pred, (self.samples, n, -1)),
                            axis=0))

                    grad = tape.gradient(loss, x)

                else:
                    loss = self.loss(y_noised, y_pred)

                    grad = tf.reduce_mean(
                        tf.reshape(
                            tape.gradient(loss, x_noised),
                            (self.samples, n, h, w, c)),
                        axis=0)

            x = x + self.step_size * grad

            norm = tf.sqrt(tf.reduce_sum(
                (x - x_0)**2, axis=(1,2,3), keepdims=True))

            x = x_0 + (x - x_0) / norm * tf.minimum(self.epsilon, norm)

        # Check if the attack succeeded.
        noise = tf.random.normal((self.samples, 1, h, w, c), stddev=self.sigma)

        x_noised = tf.reshape(x[None] + noise, (-1, h, w, c))
        y_noised = model(x_noised)

        # Computes the mode prediction on the noised inputs.
        y_pred = tf.argmax(
            # Computes the number of times each class was predicted.
            tf.reduce_sum(
                tf.reshape(
                    tf.cast(
                        tf.equal(
                            y_noised,
                            tf.reduce_max(y_noised, axis=1, keepdims=True)),
                        'int32'),
                    (self.samples, n, -1)),
                axis=0),
            axis=1)

        successes = tf.not_equal(y, y_pred)

        return x, successes
