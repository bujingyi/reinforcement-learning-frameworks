# Distributed Proximal Policy Optimization

Combine [PPO](https://github.com/bujingyi/reinforcement-learning-frameworks/tree/master/proximal_policy_optimization) with [distributed TensorFlow](https://github.com/bujingyi/distributed-tensorflow-framework/blob/master/distributed_tensorflow_framework.ipynb) will get dppo.

Model will be saved in `parameter severs (ps)`. Each `worker` calculate gradients asynchronously. See DeepMind's [paper](https://arxiv.org/abs/1707.02286).