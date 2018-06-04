import tensorflow as tf
import numpy as np
import os
import itertools
import time

from env import Environment
from a2c_utils import StateProcessor

# global variables
STATE_DIM = 0
ACTION_DIM = 0


def valid_action_gen(sess, env, state, policy_estimator):
    """
    Generate valid actions
    :param sess: TensorFlow session
    :param env: RL environment
    :param state: RL state
    :param policy_estimator: an policy_estimator instance
    :return:
    """
    action_probs, _ = policy_estimator.predict(sess, state)
    action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    # keep sampling action until valid
    while not env.valid_action(action):
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
    return action


def a2c_train(sess,
                 env,
                 policy_estimator,
                 value_estimator,
                 state_processor,
                 num_episodes,
                 num_iter,
                 experiment_dir,
                 discount_factor=0.99
                 ):
    """
    Actor Critic algorithm
    :param sess: TensorFlow session
    :param env: Environment
    :param policy_estimator: policy network - Actor
    :param value_estimator: value network - Critic
    :param num_episodes: number of episodes to run
    :param state_processor: process raw state from env to 1D array
    :param num_iter: number of maximum iteration for each episode
    :param discount_factor: discount factor for future values
    :param experiment_dir: directory to save TensorFlow summaries in
    :param discount_factor: long-term reward sum discount factor, Gammar
    :return:
    """

    # statistics for total reward of each episode
    episode_reward_list = []
    episode_length_list = []

    # create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'model')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # model saver
    saver = tf.train.Saver()

    # load a previous checkpoint if there is one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print('loading model checkpoint {}...\n'.format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # get the current time step
    total_step = sess.run(tf.train.get_global_step())

    # start training
    for i_episode in range(num_episodes):
        # initialized total reward and episode length
        total_reward = 0
        episode_length = 1

        # save the current model
        saver.save(tf.get_default_session(), checkpoint_path)

        policy_loss = None
        value_loss = None

        # reset environment
        state = env.reset()
        state = state_processor.process(state)

        for t in itertools.count():
            # print steps
            print('\rStep {} ({}) @ Episode {}/{}, policy_loss: {}, value_loss: {}'.format(
                t, total_step, i_episode + 1, num_episodes, policy_loss, value_loss), end='')

            # sample a valid action
            action = valid_action_gen(sess, env, state, policy_estimator)

            # take one step
            next_state, reward, done = env.step(action_picked)
            next_state = state_processor.process(next_state)

            # update statistics of total reward for this episode
            episode_length += 1
            total_reward += reward

            # calculate TD target
            value_next = value_estimator.predict(sess, next_state)
            td_target = reward + discount_factor * value_next
            td_error = td_target - value_estimator.predict(sess, state)

            # update value estimator, the Critic
            value_loss = value_estimator.update(sess, state, td_target)

            # update policy estimator, the Actor, by using the TD error as the advantage estimate
            policy_loss = policy_estimator.update(sess, state, td_error, action_idx)

            if done:
                break

            if t >= num_iter:
                break

            # next round
            state = next_state
            total_step += 1

        # collect statistics of total reward for the above episode
        episode_reward_list.append(total_reward)
        episode_length_list.append(episode_length)

        print('\nAveraged reward for Episode {}'.format(i_episode), ':', total_reward/episode_length)


# main function
if __name__ == '__main__':
    # reset TensorFlow computation graph
    tf.reset_default_graph()

    # where to save the checkpoints
    time_stamp = time.ctime()
    time_stamp = time_stamp.replace(':', '_')
    time_stamp = time_stamp.replace(' ', '_')
    experiment_dir = os.path.abspath('./experiments/{}'.format(time_stamp))

    # create a TensorFlow global step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # create the Actor and the Critic
    actor = PolicyEstimator(
        state_dim=STATE_DIM, 
        action_dim=ACTION_DIM, 
        scope='policy', 
        summaries_dir=experiment_dir
    )
    critic = ValueEstimator(state_dim=STATE_DIM, scope='value', summaries_dir=experiment_dir)

    # create environment
    env = Environment()

    # create state processor
    state_processor = StateProcessor()

    # training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        a2c_train(
            sess=sess,
            env=env,
            policy_estimator=actor,
            value_estimator=critic,
            state_processor=state_processor,
            num_episodes=20000,
            num_iter=1000,
            experiment_dir=experiment_dir,
        )