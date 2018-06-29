import tensorflow as tf
import os


class DQN:
    """
    Q-Value Estimator neural network.
    This network is used for both the Q-Network and the Target Network.
    """
    def __init__(
    		self, 
    		state_dim,
            action_dim,
    		scope='estimator',
    		lr_init=0.0001,
    		lr_dacay_step=10000,
    		lr_decay_rate=0.9, 
    		summaries_dir=None
    ):
    	self.state_dim = state_dim
    	self.action_dim = action_dim
        self.scope = scope
        self.lr_init = lr_init
        self.lr_dacay_step = lr_dacay_step
        self.lr_decay_rate = lr_decay_rate
        # Writes Tensorboard summaries
        self.summary_writer = None
        with tf.variable_scope(scope):
            # build the graph
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
        self.states_ph = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name='states')
        # the TD target values
        self.TD_targets_ph = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='y_TD_targets')
        # selected actions
        self.actions_ph = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name='actions')
        # state-action pairs
        self.state_action_pairs = tf.concat([self.states_ph, self.actions_ph], 1)  # [batch, d_state + d_action = 57]

        state_action_size = self.state_action_pairs.get_shape()[1]

        # deep Q network, could be any neural nets. here a three layer MLP as an example.
        mlp1 = tf.layers.dense(self.state_action_pairs, units=100, activation=tf.nn.tanh)
        mlp2 = tf.layers.dense(mlp1, units=30, activation=tf.nn.tanh)
        mlp3 = tf.layers.dense(mlp2, units=10, activation=tf.nn.tanh)

        # prediction
        self.prediction = tf.layers.dense(mlp3, units=1, activation=None)

        # loss
        self.losses = tf.squared_difference(self.TD_targets_ph, self.prediction)
        self.loss = tf.reduce_mean(self.losses)
        self.learning_rate = tf.train.exponential_decay(self.lr_init,  # Base learning rate.
                                                        tf.train.get_global_step(),  
                                                        self.lr_dacay_step,  # Decay step.
                                                        self.lr_decay_rate,  # Decay rate.
                                                        staircase=False)
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
 
        # summaries
        self.summaries = tf.summary.merge([
            tf.summary.scalar('loss', self.loss),
            tf.summary.histogram('loss_hist', self.losses),
            tf.summary.histogram('q_values_hist', self.prediction),
            tf.summary.scalar('max_q_value', tf.reduce_max(self.prediction)),
            tf.summary.scalar('learning_rate', self.learning_rate)
        ])

    def predict(self, sess, s, a):
        """
        Predict action values
        :param sess: TensorFlow session
        :param s: state input of shape [batch_size, self.state_dim]
        :param a: action input of shape [batch_size, self.action_dim]
        :return: action values of shape [batch_size]
        """
        return sess.run(self.prediction, {self.states_ph: s, self.actions_ph: a})

    def update(self, sess, s, a, q, save):
        """
        Updates the estimator towards the given targets
        :param sess: TensorFlow session
        :param s: state input of shape [batch_size, self.state_dim]
        :param a: selected actions of shape[batch_size, self.action_dim]
        :param q: target values of shape [batch_size]
        :param save: boolean, save or not
        :return: the calculated loss on the batch
        """
        feed_dict = {self.states_ph: s, self.TD_targets_ph: q, self.actions_ph: a}
        summaries, global_step, _, loss = sess.run(
            [self.summaries, tf.train.get_global_step(), self.train_op, self.loss],
            feed_dict
        )
        if self.summary_writer and save:
            self.summary_writer.add_summary(summaries, global_step)
        return loss