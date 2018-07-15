import numpy as np
import tensorflow as tf

class DQN:

    def __init__(self, session: tf.Session, input_size: np.ndarray.shape, output_size: int, name: str = "main") -> None:

        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size = 16, l_rate = 0.001) -> None:

        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [None, self.input_size.size], name="input_x")
            (x, y, z) = self.input_size.shape
            net = tf.reshape(self._X, [-1, x, y, z])  # 24 * 24 * 3

            padding = 'SAME'
            activation = tf.nn.relu
            init = tf.truncated_normal_initializer(mean=0.0, stddev=2e-2)

            # 1st cnn layer
            self.conv1 = tf.layers.conv2d(inputs=net, name='conv1',filters=32,
                                   kernel_size=(5,5), strides=1,
                                   padding=padding,
                                   kernel_initializer=init, activation=activation)
            #  variable_summaries(self.conv1)
            #print(net.shape.dims)

            # 1st pooling layer
            pool1 = tf.layers.max_pooling2d(inputs=self.conv1, name='pool1',
                                 pool_size=[2,2], strides=2)

            #  variable_summaries(pool1)
            # 2nd cnn layer
            conv2 = tf.layers.conv2d(inputs=pool1, name='conv2', filters=64,
                                   kernel_size=(5, 5), strides=1,
                                   padding=padding, kernel_initializer=init,
                                   activation=activation)
            #  variable_summaries(conv2)
            # 2nd pooling layer
            pool2 = tf.layers.max_pooling2d(inputs=conv2, name='pool1',
                                          pool_size=[2, 2], strides=2)
            #  variable_summaries(pool2)
            flatten = tf.contrib.layers.flatten(pool2)

            # fully-connected layer.
            dense1 = tf.layers.dense(inputs=flatten, name='dense1',
                                     units=10, activation=activation)
            #  variable_summaries(dense1)
            logits = tf.layers.dense(inputs=dense1, name='layer_fc_out',
                                     units=self.output_size,
                                     kernel_initializer=init, activation=tf.nn.softmax)

            self._Qpred = logits
            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)
            # v ariable_summaries(self._loss)

            self._train = tf.train.AdamOptimizer(learning_rate = l_rate).minimize(self._loss)

            correct_prediction = tf.equal(
                tf.argmax(self._Qpred, 1), tf.argmax(self._Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    """ 
    Returns Q(s, a)
    """
    def predict(self, state: np.ndarray) -> int:
        x = np.reshape(state, [-1, self.input_size.size])
        result = self.session.run(self._Qpred, feed_dict={self._X: x})

        return result

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:

        feed = {
            self._X: x_stack,
            self._Y: y_stack
        }

        return self.session.run([self._loss, self._train], feed)

    def write_summary(self, sess, merged, _X, _Y):
        _X = np.reshape(_X, (-1, 1200))
        _Y = np.reshape(_Y, (-1, 4))
        print(type(_X), " dim: ", _X.shape)

        feed = {self._X: _X,
                self._Y: _Y}

        sess.run(merged, feed)

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

