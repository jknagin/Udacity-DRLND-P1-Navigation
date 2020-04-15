[DQN]: images/DQN.png
[scores]: images/scores.png
[Udacity's Deep Reinforcement Learning Nanodegree]: https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893

# Udacity Deep Reinforcement Learning Nanodegree Project 1: Navigation

## Introduction
The purpose of this project is to train a Deep Q-Learning agent to navigate a virtual world to collect yellow bananas and avoid blue bananas.

## Environment
In this environment, state space is continuous, the action space is discrete, and we are not supplied with a state transition model. The state space has 37 dimensions and contains variable associated with the agent's velocity and the location and color of bananas with respect to the agent. The action space is discrete with four possible actions:
* `[0]` - move forward
* `[1]` - move backward
* `[2]` - turn left
* `[3]` - turn right

The agent receives a reward of +1 and -1 for every yellow and blue banana collected in an episode, respectively. The environment is considered to be solved when the agent achieves an average score of at least 13 over 100 episodes.

## Deep Q-Learning Algorithm
Q-Learning is an appropriate reinforcement learning algorithm for this use case because the state transition dynamics are unknown, the dynamics will not be learned over time, the agent is capable of learning with bootstrapping in the middle of an episode, and the action space is discrete. Deep Q-Learning in particular is used in this project because the state space is continuous. A deep neural network is used to map elements of the continuous state space to action-value vectors with dimensionality equal to the number of possible actions. Since the action space is discrete, the number of possible actions is finite, and these vectors are of finite dimension. An epsilon-greedy policy is then taken to choose the action.

The screenshot below is a description of the algorithm in pseudocode. The screenshot was taken from the lecture materials of [Udacity's Deep Reinforcement Learning Nanodegree].

![DQN]

## Implementation
The code in this submission is based on an implementation of Deep Q-Learning in the Udacity Deep Reinforcement Learning Nanodegree for the [Open AI Gym Lunar Lander V2 Environment](https://gym.openai.com/envs/LunarLander-v2/). The code has been slightly modified for use with the Banana environment in this project.

### Descriptions of each file
#### `dqn_agent.py`
Implements the Deep Q-Learning agent class, which provides the following methods:
* `__init__`
  * State space and action set dimensions are provided as arguments
  * Initializes deep neural network for Q-value estimation
  * Initializes experience replay buffer to an empty deque
* `act()`
  * Returns epsilon-greedy action based on agent's current Q-value network and epsilon value
* `hard_update()`
  * Sets the weights of the agent's target network to be equal to the weights of the agent's local network
* `learn()`
  * Updates weights of agent's local network by learning from batches of experiences from the replay buffer
  * Updates weights of agent's target network using `soft_update()`
* `load()`
  * Loads weights from existing weights file into agent's local and target networks
* `save()`
  * Saves weights of agent's local network to a file
* `soft_update()`
  * Updates weights of agent's target network based on agent's local network and parameter `TAU`
* `step()`
  * Stores current (S, A, R, S, done) tuple in experience replay buffer
  * Every `UPDATE_EVERY` episode:
    * Samples from the experience replay buffer
    * Calls `learn()` on experience replay buffer sample to update the local and target weights

#### `replay_buffer.py`
Implements an experience replay buffer as a deque with random sampling. The experience replay buffer class contains the following methods:
* `__init__`
  * Initializes the replay buffer with a batch size, a maximum buffer size, and a seed
* `__len__()`
  * Returns the number of elements currently stored in the internal deque
* `add()`
  * Adds a (S, A, R, S, done) tuple to the internal deque
* `sample()`
  * Randomly samples a batch of unique (S, A, R, S, done) tuples from internal deque

#### `model.py`
Implements a simple neural network with one hidden layer to approximate Q-values of state-action pairs.

#### `Navigation.ipynb`
Main notebook for running the code. The notebook loads the Banana environment, instantiates the agent, trains the agent, saves the agent's weight file, and plots the the agent's score per episode during training. The notebook can also be used to load weight files into an agent and play the environment to see how well the agent performs.

### Hyperparameters
| Hyperparameter | Value | Description | Defined In|
|-               |-      | -           | -         |
|`eps_start` | `1.0` | Starting value of epsilon for epsilon-greedy exploration | `Navigation.ipynb` |
|`eps_decay` | `0.995` | Epsilon decay rate to reduce exploration after each episode | `Navigation.ipynb` |
|`eps_end` | `0.01`| Minimum value of epsilon, nonzero to encourage exploration always | `Navigation.ipynb` |
|`BUFFER_SIZE` | `100000` | Number of (S, A, R, S, done) tuples to store in experience replay buffer | `dqn_agent.py` |
|`BATCH_SIZE` | `64` | Number of (S, A, R, S, done) tuples to sample from experience replay buffer and process during learning | `dqn_agent.py` |
|`GAMMA` | `0.99` | Discount factor for bootstrapping future rewards | `dqn_agent.py` |
|`TAU` | `0.001` | Soft update parameter for updating target network weights based on local network weights during training | `dqn_agent.py` |
|`LR` | `0.0005` | Learning rate for Adam optimizer | `dqn_agent.py` |
|`UPDATE_EVERY`| `4` | Number of episodes to finish before sampling from experience replay buffer and learning | `dqn_agent.py` |


### Network architecture

The network architecture is defined fully in `model.py`.

| Layer    | Input Dim | Output Dim | Activation |                    Notes                      |
| ---------| --------- | ---------- | ---------- | --------------------------------------------- |
| `FC 1 `    |     `37`    |     `64`     |    `ReLU`    | Input dimension is the state space dimension  |
| `FC 2`     |     `64`    |     `64`     |    `ReLU`    |                                               |
| `FC 3`     |     `64`   |     `2`      |    `None`    | Output dimension is the number of actions     |

### Running `Navigation.ipynb`
To run the notebook, follow these steps:

1. Ensure that the following Python libraries are installed:
  * `numpy`
  * `pandas`
  * `matplotlib`
  * `pytorch`

  Also ensure that the Banana environment is installed following the instructions in the README.
1. Run the first code cell to import all necessary libraries.
1. Update the `file_name` argument to the `UnityEnvironment` function with the location of the Banana environment (`env = UnityEnvironment(file_name=...)`)
1. Run the second code cell to load the Banana environment and get some information about the state and action space. This information is used to instantiate the agent. This cell will print out the dimension of the state space and action set, and will also print out an example state. The number of actions is 4 and the states have a length of 37.
1. Run the third code cell to train the agent. The code will loop until the maximum number of episodes have been played (specified by `n_episodes`) or the agent achieves an average score of 13.0 or greater over the 100 most recent episodes. If the agent achieves such a score, this cell will save the agent's weights to a file called `solution.pth` and the list of scores to a file called `scores.npy`.
1. After training the agent, run the next code cell to plot the score and the running average of the 100 most recent scores.
1. The next two code cells are used to load an existing `solution.pth` file into the agent and watch it perform in the Banana environment.
1. The final code cell closes the environment.

## Results
### Score plot
![scores]

The above image shows the plot of the agent's score for each episode (blue) and the running average of the scores of the previous 100 episodes (red). **The agent achieves an average score greater than or equal to 13.0 for 100 episodes after episode 500.**

## Future work
The following suggestions for future work have been taken from the lecture materials of [Udacity's Deep Reinforcement Learning Nanodegree].

* Double DQN
  > Deep Q-Learning [tends to overestimate](https://www.ri.cmu.edu/pub_files/pub1/thrun_sebastian_1993_1/thrun_sebastian_1993_1.pdf) action values. [Double Q-Learning](https://arxiv.org/abs/1509.06461) has been shown to work well in practice to help with this.

* Prioritized Experience Replay
  > Deep Q-Learning samples experience transitions uniformly from a replay memory. [Prioritized experienced replay](https://arxiv.org/abs/1511.05952) is based on the idea that the agent can learn more effectively from some transitions than from others, and the more important transitions should be sampled with higher probability.


* Dueling DQN
  > Currently, in order to determine which states are (or are not) valuable, we have to estimate the corresponding action values for each action. However, by replacing the traditional Deep Q-Network (DQN) architecture with a [dueling architecture](https://arxiv.org/abs/1511.06581), we can assess the value of each state, without having to learn the effect of each action.
