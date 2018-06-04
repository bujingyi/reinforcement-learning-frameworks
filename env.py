class Environment:
    """
    Reinforcement learning environment
    """
    def __init__(self):
        # TODO: Environment initializer
        # initialize the environment
        self.init_state = self.reset()

    def initialize_state(self, state_dim):
        """
        Randomly generate initial state.
        :param state_dim: state dimension
        :return: random init state of shape 
        """
        # TODO: initialize a random state for the environment
        return init_state

    def reset(self):
        """
        Reset the environment
        :return:
        """
        # initialize a random state
        init_state = self.initialize_state(state_dim)

        # TODO: reset the environment
        return init_state

    def step(self):
        """
        Take one step in the environment
        :return: next_state, reward, done
        """
        # TODO: take one step in the environment
        return next_state, reward, done

    def valid_action(self, action):
        """
        judge an action is valid or not
        :param action: an action in the RL environment
        :return: boolean. True if action is valid; False otherwise
        """
        # TODO: judge the passed in action is valid or not
        return valid


class StateProcessor:
    """
    Process a state from environment. Convert it to an RL state.
    """
    def __init__(self):
        # TODO: StateProcessor initializer

    def process(self, state_env):
        """
        convert raw environment state to 
        :param state_env: raw state from environm
        :return: an RL state of shape...
        """
        # TODO: convert the raw state
        return RL_state