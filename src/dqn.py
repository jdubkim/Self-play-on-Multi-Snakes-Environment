import numpy as np
import tensorflow as tf

class DQN:

    def __init__(self, session: tf.Session, input: np.ndarray, output: int, name: str = "main") -> None:

        self.session = session
        self.input = input
        self.input_size = input.size
        self.output_size = output
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size = 16, l_rate = 0.001) -> None:

        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size], name = "input_x")
            x, y, z = self.input.shape

            net = tf.reshape(self._X, [-1, x, y, z]) # 224 * 256 * 3

            padding = 'SAME'
            activation = tf.nn.relu
            init = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)

            #1st cnn layer
            net = tf.layers.conv2d(inputs=net, name='layer_conv1',
                                   filters=32, kernel_size=5, strides=1,
                                   padding=padding,
                                   kernel_initializer=init, activation=activation)

            #2nd cnn layer
            net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                                   filters=32, kernel_size=5, strides=2,
                                   padding=padding,
                                   kernel_initializer=init, activation=activation)

            #3rd cnn layer
            net = tf.layers.conv2d(inputs=net, name='layer_conv3',
                                   filters=64, kernel_size=5, strides=4,
                                   padding=padding,
                                   kernel_initializer=init, activation=activation)

            net = tf.contrib.layers.flatten(net)

            # First fully-connected (aka. dense) layer.
            net = tf.layers.dense(inputs=net, name='layer_fc1', units=1024,
                                  kernel_initializer=init, activation=activation)

            net = tf.layers.dense(inputs=net, name='layer_fc2', units=1024,
                                  kernel_initializer=init, activation=activation)

            # Final fully-connected layer.
            net = tf.layers.dense(inputs=net, name='layer_fc_out', units=self.output_size,
                                  kernel_initializer=init, activation=None)

            self._Qpred = net
            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)

            self._train = tf.train.AdamOptimizer(learning_rate = l_rate).minimize(self._loss)

    def predict(self, state: np.ndarray) -> np.ndarray:

        """ Returns Q(s, a)
        """
        x = np.reshape(state, [-1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:

        feed = {
            self._X: x_stack,
            self._Y: y_stack
        }

        return self.session.run([self._loss, self._train], feed)