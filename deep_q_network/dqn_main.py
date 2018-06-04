import tensorflow as tf
import os
import time

from dqn_env import Environment
from dqn_utils import StateProcessor


# global variables
STATE_DIM = 0
ACTION_DIM = 0


if __name__ == '__main__':
    # choose GPU
    GPU_ID = 0
    # reset Tensorflow computation graph
    tf.reset_default_graph()

    # where to save the checkpoints
    time_stamp = time.ctime()
    time_stamp = time_stamp.replace(':', '_')
    time_stamp = time_stamp.replace(' ', '_')
    experiment_dir = os.path.abspath('./experiments/{}'.format(time_stamp))

    with tf.device('/device:GPU:' + str(GPU_ID)):
        # create a TensorFlow global step variable
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # create estimators
        q_estimator = DQN(
            state_dim=STATE_DIM, 
            action_dim=ACTION_DIM, 
            scope='q', 
            summaries_dir=experiment_dir
        )
        target_estimator = DQN(
            tate_dim=STATE_DIM, 
            action_dim=ACTION_DIM,
            scope='target_q'
        )

        # create environment
        env = Environment()

        # create state processor
        state_processor = StateProcessor()

        # training
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            deep_q_learning(
                sess=sess,
                env=env,
                q_estimator=q_estimator,
                target_estimator=target_estimator,
                state_processor=state_processor,
                num_episodes=20000,
                num_iter=1000,
                experiment_dir=experiment_dir,
                state_dim=STATE_DIM,
                action_dim=ACTION_DIM,
                replay_memory_size=500000,
                replay_memory_init_size=10000,
                update_target_estimator_every=500,
                discount_factor=0.9,
                epsilon_start=1.0,
                epsilon_end=0.1,
                epsilon_decay_steps=50000,
                batch_size=2048
            )