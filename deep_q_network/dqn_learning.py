import tensorflow as tf
import numpy as np
import random
import itertools
import os
from collections import namedtuple

from dqn_utils import ReplayMemory

def valid_action_gen(sess, env, state):
    """
    Generate valid actions list
    :param sess: TensorFlow session
    :param env: RL environment
    :param state: RL state
    :param ppo: an PPO instance
    :return: valid actions list containing all valid actions for the passed in state
    """
    # TODO: generate all valid actions for the passed in state
    return valid_action_list


def copy_model_params(sess, estimator_from, estimator_to):
    """
    Copies the model params of one model to another.
    :param sess: TensorFlow session
    :param estimator_from: estimator to copy the params from
    :param estimator_to: estimator to copy the params to
    :return:
    """
    params_from = [t for t in tf.trainable_variables() if t.name.startswith(estimator_from.scope)]
    params_from = sorted(params_from, key=lambda v: v.name)
    params_to = [t for t in tf.trainable_variables() if t.name.startswith(estimator_to.scope)]
    params_to = sorted(params_to, key=lambda v: v.name)

    update_ops = []
    for v_from, v_to in zip(params_from, params_to):
        op = v_to.assign(v_from)
        update_ops.append(op)

    sess.run(update_ops)


def make_epsilon_greedy_policy(estimator):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon
    :param estimator: An estimator that returns q values for a given state
    :return: A function that takes the (sess, observation, epsilon) as an argument and returns the probabilities 
    		for each action in the form of a numpy array of length nA.
    """
    def policy_fn(sess, observation, action_list, epsilon):
        """
        Policy function
        :param sess: TensorFlow session
        :param observation: state for which to choose action
        :param action_list: list of 1d array indicating all valid actions
        :param epsilon: exploitation & exploration trade off
        :return: 1D array containing the possibilities for each action
        """
        # initialize policy
        array = np.ones(len(action_list), dtype=float) * epsilon / len(action_list)

        # convert state and actions to the data type for TensorFlow
        valid_action_size = len(action_list)
        states = np.tile(observation, [valid_action_size, 1])
        actions = np.array(action_list)

        # use the current DQN to evaluate the q values
        q_values = estimator.predict(sess, states, actions)
        best_action = np.argmax(q_values)
        array[best_action] += (1.0 - epsilon)
        return array
    return policy_fn


def deep_q_learning(
		sess,
        env,
        q_estimator,
        target_estimator,
        state_processor,
        num_episodes,
        num_iter,
        experiment_dir,
        state_dim,
        action_dim,
        replay_memory_size=500000,
        replay_memory_init_size=5000,
        update_target_estimator_every=10000,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=5000000,
        batch_size=32,
        save_ckp_every=5,
        max_ckp_save=20,
        refresh_exp_mem_every=30,
        record_every=10
):
    """
    Q-Learning algorithm for off-policy TD control using Function Approximation.
    Finds the optimal greedy policy while following an epsilon-greedy policy.
    :param sess: TensorFlow session object
    :param env: Environment
    :param q_estimator: DQN object used for the q values
    :param target_estimator: DQN object used for the targets
    :param state_processor: StateProcessor object
    :param num_episodes: number of episodes to run
    :param num_iter: number of maximum iteration for each episode
    :param experiment_dir: directory to save TensorFlow summaries in
    :param state_dim: RL state dimension
    :param action_dim: RL action dimension
    :param replay_memory_size: size of the replay memory
    :param replay_memory_init_size: Number of random experiences to sample when initializing the reply memory
    :param update_target_estimator_every: Copy parameters from the Q estimator to the target estimator every N steps
    :param discount_factor: Gamma discount factor
    :param epsilon_start: Epsilon for exploration
    :param epsilon_end: Epsilon for exploration
    :param epsilon_decay_steps: Epsilon for exploration
    :param batch_size: Size of batches to sample from the replay memory
    :param save_ckp_every: save checkpoint every N episodes
    :param max_ckp_save: max number of checkpoints to keep
    :param refresh_exp_mem_every: refresh experience replay memory every k episodes
    :param record_every: Record something every N episodes
    :return:
    """
    saved_args = locals()
    print(saved_args)

    # define RL transition as a namedtuple
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

    # the replay memory
    replay_memory = ReplayMemory(memory_size=replay_memory_size, pop_num=replay_memory_init_size)

    # statistics for total reward of each episode
    episode_reward_list = []
    episode_length_list = []

    # create directories for checkpoints and summaries
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    checkpoint_path = os.path.join(checkpoint_dir, 'model')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(os.path.join(experiment_dir, 'param.txt'), 'w') as f:
        for arg, val in saved_args.items():
            f.write(str(arg) + '\t' + str(val) + '\n')

    saver = tf.train.Saver(max_to_keep=max_ckp_save)

    # load a previous checkpoint if there is one
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    if latest_checkpoint:
        print('loading model checkpoint {}...\n'.format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)

    # get the current time step
    total_step = sess.run(tf.train.get_global_step())

    # the epsilon decay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    # init epsilon at beginning, will be updated as training steps go
    epsilon = epsilons[0]

    # the policy we are following
    policy = make_epsilon_greedy_policy(q_estimator)

    def refresh_exp_mem(eps):
        replay_memory.clean()
        # populate the replay memory with initial experiences
        print('Populating replay memory...')
        state = state_processor.process(env.reset())
        for i in range(replay_memory_init_size):
            # print('Populating replay memory...: {:2.2f}%'.format(i / replay_memory_init_size * 100))
            if i % (replay_memory_init_size/10) == 0:
                print('Populating replay memory...: {:2.2f}%'.format(i / replay_memory_init_size * 100))
            valid_actions_list = valid_actions_gen(sess, env, state)
            action_probs = policy(sess, state, valid_actions_list, eps)
            action_idx = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            action_chosen = valid_actions_list[action_idx]
            next_state, reward, done = env.step(action_chosen)
            next_state = state_processor.process(next_state)

            # if the episode is over, reset the environment
            if done:
                state = state_processor.process(env.reset())
            else:
                replay_memory.append(Transition(state, action_chosen, reward, next_state, done))
                state = next_state

    # start training DQN
    for i_episode in range(num_episodes):

        # initialized total reward and episode length
        total_reward = 0
        episode_length = 0

        # save the current model
        if i_episode % save_ckp_every == 0:
          saver.save(tf.get_default_session(), checkpoint_path, global_step=tf.train.get_global_step())

        # refresh the current experience memory
        if i_episode % refresh_exp_mem_every == 0:
            refresh_exp_mem(epsilon)

        loss = None

        # reset environment
        state = state_processor.process(env.reset())

        for t in itertools.count():

            # epsilon for this time step
            epsilon = epsilons[min(total_step, epsilon_decay_steps-1)]

            # check if update the target network
            if total_step % update_target_estimator_every == 0:
                copy_model_params(sess, q_estimator, target_estimator)
                print('\nCopied madel params to target network')

            # print steps
            print('\rStep {} ({}) @ Episode {}/{}, memo size: {}, eps: {:2.2f}, loss: {}'.format(
                t, total_step, i_episode+1, num_episodes, len(valid_actions_gen.state_action), epsilon, loss), end='')

            # take one step in the environment
            # generate all valid actions
            valid_actions_list = valid_actions_gen(sess, env, state)

            # choose action with policy
            action_probs = policy(sess, state, valid_actions_list, epsilon)
            action_idx = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            action_chosen = valid_actions_list[action_idx]

            # take one step
            next_state, reward, done = env.step(action_chosen)
            next_state = state_processor.process(next_state)

            # update statistics of total reward for this episode
            episode_length += 1
            total_reward += reward

            # TODO: there may be some post processing after one step in the environment

            # if replay memory is full, pop the first element
            if replay_memory.is_full():
                replay_memory.pop()

            # save transition to replay memory
            replay_memory.append(Transition(state, action_chosen, reward, next_state, done))

            # sample a mini-batch from the replay memory to train the DQN
            samples = random.sample(replay_memory.replay_memory, batch_size)
            states_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array, zip(*samples))
            reward_batch = reward_batch.reshape((-1, 1))

            # calculate q values and targets
            # double Q
            next_state_valid_actions_list = list(map(valid_actions_gen.valid_actions_array, next_state_batch))
            # list as length of batch size, each item is an array of [num valid acts, ACTION_DIM]

            next_state_valid_actions_array = np.concatenate(next_state_valid_actions_list, axis=0)
            # shape: [sum of num acts, ACTION_DIM]
            next_state_number_valid_actions = list(map(len, next_state_valid_actions_list))
            # length is batch size

            next_state_repeat_batch = []

            for next_single_state, n_acts in zip(next_state_batch, next_state_number_valid_actions):
                next_state_repeat_batch.append(np.tile(next_single_state, [n_acts, 1]))

            next_state_repeat_batch_array = np.concatenate(next_state_repeat_batch, axis=0)
            # shape: [sum of num acts, STATE_DIM]

            next_state_all_q_values = q_estimator.predict(sess,
                                                          next_state_repeat_batch_array,
                                                          next_state_valid_actions_array)
            # shape: [sum of num acts, 1]

            next_state_best_actions = []

            prev_ind = 0
            for ind in next_state_number_valid_actions:
                forw_ind = prev_ind + ind
                next_state_best_actions.append(
                    next_state_valid_actions_array[np.argmax(next_state_all_q_values[prev_ind:forw_ind]) + prev_ind])
                prev_ind = forw_ind

            next_state_best_actions = np.array(next_state_best_actions)
            q_values_next_target = target_estimator.predict(sess, next_state_batch, next_state_best_actions)
            targets_batch = reward_batch + discount_factor * q_values_next_target

            # update DQN with gradient decent
            states_batch = np.array(states_batch)
            save_summary = True if t % 10 == 0 else False
            loss = q_estimator.update(sess, states_batch, action_batch, targets_batch, save_summary)

            if done:
                break

            if t >= num_iter:
                break

            state = next_state
            total_step += 1

        # collect statistics of total reward for the above episode
        episode_reward_list.append(total_reward)
        episode_length_list.append(episode_length)

        print('\nAveraged reward for Episode {}'.format(i_episode + 1), ':', total_reward/episode_length)

        # save reached states
        if i_episode % record_every == 0:
            with open(os.path.join(experiment_dir, 'reached_states.txt'), 'w') as f:
                f.write("\n".join(str(state) for state in valid_actions_gen.state_action.keys()))
