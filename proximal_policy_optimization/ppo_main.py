import tensorflow as tf
import numpy as np

import os
import time

from ppo_environment import Environment, StateProcessor
from ppo_model import PPO


# global variables
STATE_DIM = 100
ACTION_DIM = 100
EPSILON = 0.2  # clipped surrogate objective controlling
GAMMA = 0.9  # reward discount
CRITIC_LR = 1e-4
ACTOR_LR = 1e-4
CRITIC_UPDATE_STEPS = 5
ACTOR_UPDATE_STEPS = 5
BATCH_SIZE = 128  # batch size for training
EPISODE_MAX = 200000  # max episode number
EPISODE_LEN = 1000  # max episode length
MODEL_SAVE_EPISODES = 10


def valid_action_gen(sess, env, state, ppo):
    """
    Generate valid actions
    :param sess: TensorFlow session
    :param env: RL environment
    :param state: RL state
    :param ppo: an PPO instance
    :return:
    """
    action_ppo = ppo.choose_action(sess=sess, states=state)
    # keep sampling action until valid
    while not env.valid_action(action_ppo):
        action_ppo = ppo.choose_action(sess=sess, states=state)
    return action_ppo


def train(
        sess,
        env,
        ppo,
        state_processor,
        experiment_dir,
):
    """
    Train the model
    :param sess: TensorFlow session
    :param env: RL environment
    :param ppo: proximal policy optimization instance
    :param state_processor: state processor
    :param experiment_dir: directory to save TensorFlow checkpoints
    :return:
    """
    # create directories for checkpoints
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'model')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # TensorFlow saver
    saver = tf.train.Saver()

    # load a previous checkpoint if there is one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        print('loading model checkpoint {}...\n'.format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    total_step = 0
    for ep in range(EPISODE_MAX):
        # save the model every MODEL_SAVE_EPISODES
        if ep % MODEL_SAVE_EPISODES == 0:
            saver.save(tf.get_default_session(), checkpoint_path)

        # empty the buffers
        buffer_state, buffer_action, buffer_reward = [], [], []
        episode_reward = 0
        episode_length = 1

        # reset environment
        state = env.reset()
        state = state_processor.process(state)

        for t in range(EPISODE_LEN):
            # choose action
            action_ppo = valid_action_gen(sess=sess, state=state_array, ppo=ppo)
            # TODO: some processing of the state and action

            # take one step in the environment
            next_state, reward, done = env.condense_step(state, allocation)
            next_state = state_processor.process(next_state)

            # TODO: some processing of the next_state

            # accumulate episode reward
            episode_reward += reward
            episode_length += 1

            # record the trajectory
            buffer_state.append(state)
            buffer_action.append(action_ppo)
            buffer_reward.append(reward)

            # update ppo if it is the time
            if (t+1) % BATCH_SIZE == 0 or t == EPISODE_LEN-1:
                value_ = ppo.get_value(sess=sess, states=next_state_array)
                # print('value:', value_)
                discounted_reward = []
                for r in buffer_reward[::-1]:
                    value_ = r + GAMMA * value_
                    discounted_reward.append(value_)
                discounted_reward.reverse()

                # fabricate batch for training
                batch_state, batch_action, batch_discounted_reward = np.vstack(buffer_state),\
                                                                     np.vstack(buffer_action),\
                                                                     np.vstack(buffer_reward)
                buffer_state, buffer_action, buffer_reward = [], [], []

                # update the actor and critic networks
                ppo.update(
                    sess=sess,
                    states=batch_state,
                    actions=batch_action,
                    discounted_rewards=batch_discounted_reward
                )

            # print steps
            print('\rStep {} ({}) @ Episode {}/{}, reward: {}'.format(
                t, total_step, ep + 1, EPISODE_MAX, reward), end='')

            if done:
                break

            total_step += 1

        ppo.summary(sess=sess, averaged_reward=episode_reward / episode_length)
        print('\nAveraged reward for Episode {}'.format(ep), ':', episode_reward / episode_length)


# main function
if __name__ == "__main__":
    # reset TensorFlow computation graph
    tf.reset_default_graph()

    # create a TensorFlow global step variable
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # where to save the checkpoints
    time_stamp = time.ctime()
    time_stamp = time_stamp.replace(':', '_')
    time_stamp = time_stamp.replace(' ', '_')
    experiment_dir = os.path.abspath('./experiments/{}'.format(time_stamp))

    # create a PPO instance
    ppo = PPO(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        epsilon=EPSILON,
        critic_lr=CRITIC_LR,
        actor_lr=ACTOR_LR,
        critic_update_steps=CRITIC_UPDATE_STEPS,
        actor_update_steps=ACTOR_UPDATE_STEPS,
        summaries_dir=experiment_dir
    )

    # create environment
    env = Environment()

    # create state processor
    state_processor = StateProcessor()

    # start training...
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # reinforcement learning
        train(
            sess=sess,
            env=env,
            ppo=ppo,
            state_processor=state_processor,
            experiment_dir=experiment_dir
        )