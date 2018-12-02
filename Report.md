[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://github.com/shashanktyagi/p3_collab-compet/blob/master/training_scores.png 

# Project 3: Collaboration and Competition

### Introduction

For this project, we work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

**Environment solved criterion:** The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### 2. Learning Algorithm
We train the network using DDPG algorithm. For the Actor, we use a three layer MLP with 128 and 128 neurons respectively in hidden layers. The state vector is the 8 dimensional vector described in section 1. The output vector is of size 2. We use same Actor and Critic networks for both the players. We add the experiences of both the players to the same replay buffer and sample from it to compute the loss.
We train the network using Adam optimizer with an actor learning rate of 0.001, critic learning rate of 0.001 and batch size of 128. We use a discount factor of 0.99.

### 3. Results
The figure below shows average rewards per episode as the agent is being trained. The training is terminated when the average reward per episode reaches 0.5. We were able to solve the environement in 2052 episodes.

![Rewards per episode][image2]

### 4. Future Work
We trained a the environment using DDPG algorithm. In future we can explore other algorithms like MADDPG. We can also tune the hyperparameters further to solve the environment in fewer number of episodes. 


  
### References
DDPG paper: https://arxiv.org/abs/1509.02971
