import tensorflow as tf
import os


class PolicyEstimator:
    """
    Policy function estimator, the Actor
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        scope='policy',
        learning_rate=0.01,
        summaries_dir=None
    ):
        """
        Actor - policy gradient network
        :param scope: TensorFlow variable scope
        :param learning_rate: gradient descent learning rate
        :param summaries_dir: TensorBoard summaries
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.scope = scope
        self.learning_rate = learning_rate
        self.summary_writer = None

        # build the graph
        with tf.variable_scope(self.scope):
            self._build_model()
            if summaries_dir:
                summaries_dir = os.path.join(summaries_dir, 'summaries_{}'.format(scope))
                if not os.path.exists(summaries_dir):
                    os.makedirs(summaries_dir)
                self.summary_writer = tf.summary.FileWriter(summaries_dir)

    def _build_model(self):
        """
        Build the computation graph
        :return:
        """
        # placeholder for input
        # input states of shape [...]
        self.state_ph = tf.placeholder(shape=[1, self.state_dim], dtype=tf.float32, name='state')
        # picked action
        self.action_ph = tf.placeholder(dtype=tf.int32, name='action')
        # how much better the chosen action is above the average
        self.target_ph = tf.placeholder(dtype=tf.float32, name='target')

        # mlp layer sizes
        hidden_size = 50
        action_space_size = self.action_dim
        # one hidden layer MLP
        self.hidden = tf.layers.dense(self.state_ph, units=hidden_size, activation=tf.nn.sigmoid)
        self.logits = tf.layers.dense(self.hidden, units=action_space_size, activation=None)

        # current policy
        self.logits_1d = tf.squeeze(self.logits)
        self.action_probs = tf.nn.softmax(self.logits_1d)

        # pick out the picked action's probability
        self.pick_action_prob = tf.gather(self.action_probs, self.action_ph)

        # loss, policy gradient
        self.loss = -tf.log(self.pick_action_prob) * self.target_ph

        # training operation
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        # summaries
        self.summaries = tf.summary.merge([
            tf.summary.scalar('policy_loss', self.loss)
        ])

    def predict(self, sess, state):
        """
        policy network predicts action probabilities
        :param sess: TensorFlow session
        :param state: state input of shape [7]
        :return:
        """
        return sess.run([self.action_probs, self.logits_1d], {self.state_ph: state})

    def update(self, sess, state, target, action):
        """
        train the policy network towards the target
        :param sess: TensorFlow session
        :param state: state input of shape [7]
        :param target: target input of shape [1]
        :param action: action input of shape [1]
        :return: loss of shape [1]
        """
        feed_dict = {self.state_ph: state, self.target_ph: target, self.action_ph: action}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class ValueEstimator:
    """
    Value function approximator, the Critic
    """
    def __init__(
        self,
        state_dim
        scope='value',
        learning_rate=0.001,
        summaries_dir=None
    ):
        """
        Critic, value function network
        :param scope: TensorFlow variable scope
        :param learning_rate: gradient descent learning rate
        :param summaries_dir: TensorBoard summaries
        """
        self.state_dim = state_dim
        self.scope = scope
        self.learning_rate = learning_rate
        self.summary_writer = None

        # build the graph
        with tf.variable_scope(self.scope):
            self._build_model()
            if summaries_dir:
                summaries_dir = os.path.join(summaries_dir, 'summaries_{}'.format(scope))
                if not os.path.exists(summaries_dir):
                    os.makedirs(summaries_dir)
                self.summary_writer = tf.summary.FileWriter(summaries_dir)

    def _build_model(self):
        """
        Build the computation graph
        :return:
        """
        # placeholder for input
        # input states
        self.state_ph = tf.placeholder(shape=[1, self.state_dim], dtype=tf.float32, name='state')
        # how much better the
        self.target_ph = tf.placeholder(dtype=tf.float32, name='target')

        # mlp layer sizes
        state_size = self.state_ph.get_shape()[0]
        hidden_size = 50
        # one hidden layer
        self.hidden = tf.layers.dense(self.state_ph, units=hidden_size, activation=tf.nn.sigmoid)
        self.pred = tf.layers.dense(self.hidden, units=1, activation=None)

        self.value_estimate = tf.squeeze(self.pred)

        # loss
        self.loss = tf.squared_difference(self.value_estimate, self.target_ph)

        # training operation
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())

        # summaries
        self.summaries = tf.summary.merge([
            tf.summary.scalar('value_loss', self.loss)
        ])

    def predict(self, sess, state):
        """
        value network predicts action probabilities
        :param sess: TensorFlow session
        :param state: state input of shape [7]
        :return:
        """
        return sess.run(self.value_estimate, {self.state_ph: state})

    def update(self, sess, state, target):
        """
        train the policy network towards the target
        :param sess: TensorFlow session
        :param state: state input of shape [7]
        :param target: target input of shape [1]
        :return: loss of shape [1]
        """
        feed_dict = {self.state_ph: state, self.target_ph: target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


