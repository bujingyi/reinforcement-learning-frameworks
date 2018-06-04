class ReplayMemory:
    """
    Experience replay
    """
    def __init__(self, memory_size, pop_num=1):
        """
        Replay memory class constructor
        :param memory_size:
        :param memory_init_size:
        """
        self.memory_size = memory_size
        self.pop_num = pop_num
        self.replay_memory = []

    def is_full(self):
        """
        Check if replay memory is full
        :return: True if full, False if else
        """
        return len(self.replay_memory) == self.memory_size

    def pop(self):
        """
        If the replay memory is full, pop the first element
        :param pop_num: number of items to pop out
        :return:
        """
        for _ in range(self.pop_num):
            self.replay_memory.pop(0)

    def append(self, transition):
        """
        Save transition to replay memory
        :param transition: namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])
        :return:
        """
        self.replay_memory.append(transition)

    def clean(self):
        self.replay_memory = []


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