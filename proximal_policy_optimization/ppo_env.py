
class Environment:
    """
    Reinforcement learning environment
    """
    def __init__(self):
        # TODO: Environment initializer

        self.state_processor = StateProcessor()

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


class StateProcessor:
    """
    Process a state from environment. Convert it to an RL state.
    """
    def __init__(self):
        # TODO: StateProcessor initializer

    def process(self, state_dict):
        """
        convert original environment state to 
        :return: an RL state of shape...
        """
        # TODO: convert the original state
        return RL_state