import tensorflow as tf
import os


class PPO:
    """
    Proximal Policy Optimization
    """
    def __init__(
            self,
            state_dim,
            action_dim,
            epsilon,
            critic_lr,
            actor_lr,
            critic_update_steps,
            actor_update_steps,
            summaries_dir=None
    ):
        """
        PPO constructor
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param epsilon: control surrogate clipping
        :param critic_lr: gradient descent learning rate for Critic
        :param actor_lr: gradient descent learning rate for Actor
        :param critic_update_steps: Critic gradient descent steps for one update
        :param actor_update_steps: Actor gradient descent steps for one update
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.critic_update_steps = critic_update_steps
        self.actor_update_steps = actor_update_steps
        self.averaged_reward_global_steps = 0
        self.actor_global_steps = 0
        self.critic_global_steps = 0

        # TensorFlow summary
        self.summary_writer = None
        if summaries_dir:
            summaries_dir = os.path.join(summaries_dir, 'summaries_{}'.format('ppo'))
            if not os.path.exists(summaries_dir):
                os.makedirs(summaries_dir)
            self.summary_writer = tf.summary.FileWriter(summaries_dir)

        # placeholders
        self._create_placeholders()

        # build Critic including network, loss, and train_op
        with tf.variable_scope('critic'):
            self._build_critic()

        # build Actor
        with tf.variable_scope('actor'):
            pi, pi_params, _ = self._build_actor_network(scope='pi', trainable=True)
            oldpi, oldpi_params, _ = self._build_actor_network(scope='oldpi', trainable=False)

            # define pi sampling operation
            with tf.variable_scope('sample_action'):
                self.sample_op = tf.squeeze(pi.sample(sample_shape=1), axis=0)
                # print('sample_op shape', self.sample_op.get_shape())

            # define old pi update operation
            with tf.variable_scope('update_oldpi'):
                self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

            # define Actor loss
            with tf.variable_scope('loss'):
                # Here we use surrogate, not KL divergence (https://arxiv.org/abs/1707.06347)
                with tf.variable_scope('surrogate'):
                    ratio = pi.prob(self.actions_ph) / oldpi.prob(self.actions_ph)
                    surrogate = ratio * self.advantages_ph
                    # then define the loss using clipping
                    self.actor_loss = -tf.reduce_mean(
                        tf.minimum(surrogate,
                                   tf.clip_by_value(ratio,
                                                    1. - self.epsilon,
                                                    1 + self.epsilon) * self.advantages_ph
                                   )
                    )
            # define Actor train operation
            self.actor_train_op = tf.train.AdamOptimizer(self.actor_lr).minimize(
                self.actor_loss,
                global_step=tf.train.get_global_step()
            )
            self.actor_loss_summary = tf.summary.scalar('actor_loss', self.actor_loss)

        self.averaged_reward = tf.summary.scalar('averaged_reward', self.averaged_reward_ph)

    def _create_placeholders(self):
        """
        TensorFlow placeholders
        :return:
        """
        self.states_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.state_dim], name='states')
        self.actions_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim], name='actions')
        self.discounted_reward_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='discounted_reward')
        self.advantages_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='advantages')
        # averaged reward summary
        self.averaged_reward_ph = tf.placeholder(dtype=tf.float32, name='averaged_reward')

    def _build_critic(self):
        """
        Build Critic - three hidden layers MLP
        :return:
        """
        hid1 = tf.layers.dense(self.states_ph, 150, activation=tf.tanh)
        hid2 = tf.layers.dense(hid1, 200, activation=tf.tanh)
        hid3 = tf.layers.dense(hid2, 150, activation=tf.tanh)
        self.values = tf.layers.dense(hid3, 1, activation=None)
        self.advantages = self.discounted_reward_ph - self.values
        self.critic_loss = tf.reduce_mean(tf.square(self.advantages))
        self.critic_train_op = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)
        self.critic_loss_summary = tf.summary.scalar('critic_loss', self.critic_loss)

    def _build_actor_network(self, scope, trainable):
        """
        Build Actor - three hidden layers MLP
        :param scope: TensorFlow variable scope
        :param trainable: boolean, the network to build is trainable or not
        :return:
        """
        with tf.variable_scope(scope):
            hid1 = tf.layers.dense(self.states_ph, 150, activation=tf.tanh, trainable=trainable)
            hid2 = tf.layers.dense(hid1, 200, activation=tf.tanh, trainable=trainable)
            hid3 = tf.layers.dense(hid2, 150, activation=tf.tanh, trainable=trainable)
            logits = tf.layers.dense(hid3, self.action_dim, activation=None, trainable=trainable)
            multinomial_dist = tf.distributions.Multinomial(total_count=1., logits=logits)
        params = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        return multinomial_dist, params, logits

    def update(self, sess, states, actions, discounted_rewards):
        """
        Update both Actor and Critic
        :param sess: TensorFlow session
        :param states: states
        :param actions: actions
        :param discounted_rewards: discounted rewards
        :return:
        """
        # copy current policy network params to old policy network params
        sess.run(self.update_oldpi_op)
        # calculate advantages
        advantages = sess.run(
            self.advantages,
            {self.states_ph: states, self.discounted_reward_ph: discounted_rewards}
        )

        # update Actor, clipping method according to OpenAI's paper
        for _ in range(self.actor_update_steps):
            _, actor_loss_summary = sess.run(
                [self.actor_train_op, self.actor_loss_summary],
                {self.states_ph: states, self.actions_ph: actions, self.advantages_ph: advantages}
            )
            if self.summary_writer:
                self.summary_writer.add_summary(actor_loss_summary, self.actor_global_steps)
                self.actor_global_steps += 1

        # update Critic
        for _ in range(self.critic_update_steps):
            _, critic_loss_summary = sess.run(
                [self.critic_train_op, self.critic_loss_summary],
                {self.states_ph: states, self.discounted_reward_ph: discounted_rewards}
            )
            if self.summary_writer:
                self.summary_writer.add_summary(critic_loss_summary, self.critic_global_steps)
                self.critic_global_steps += 1

    def choose_action(self, sess, states):
        """
        Choose action by Actor
        :param sess: TensorFlow session
        :param states: states
        :return: action of the shape [action_dim]
        """
        return sess.run(self.sample_op, {self.states_ph: states})[0]

    def get_value(self, sess, states):
        """
        Get value of the state
        :param sess: TensorFlow session
        :param states: states
        :return:
        """
        return sess.run(self.values, {self.states_ph: states})[0, 0]

    def summary(self, sess, averaged_reward):
        """
        Observe averaged reward by TensorBoard during training
        :param sess: TensorFlow session
        :param averaged_reward: averaged reward after each episode
        :return:
        """
        summaries = sess.run(
            self.averaged_reward,
            {self.averaged_reward_ph: averaged_reward}
        )
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step=self.averaged_reward_global_steps)
            self.averaged_reward_global_steps += 1
