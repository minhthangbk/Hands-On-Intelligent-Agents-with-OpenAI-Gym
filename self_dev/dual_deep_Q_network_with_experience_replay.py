# DQN with Replay buffer and fixed Q-target to address instabilities of DQN
import os
from typing import Dict, List, Tuple
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output


# Typically, people implement replay buffers with one of the following three data structures: collections.deque,
# list, or numpy.ndarray
# deque is very easy to handle once you initialize its maximum length (e.g. deque(# maxlen=buffer_size)).
# However, the indexing operation of deque gets terribly slow as it grows up because it is
# internally doubly linked list. On the other hands, list is an array, so it is relatively faster than deque when you
# sample batches at every step. Its amortized cost of Get item is O(1). Last but not least, let's see numpy.ndarray.
# numpy.ndarray is even faster than list due to the fact that it is a homogeneous array of fixed-size items,
# so you can get the benefits of locality of reference. Whereas list is an array of pointers to objects,
# even when all of them are of the same type. Here, we are going to implement a replay buffer using numpy.ndarray.

class ReplayBuffer:
    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)  # observation buffer
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)  # next observation buffer
        self.acts_buf = np.zeros([size], dtype=np.float32)  # action buffer
        self.rews_buf = np.zeros([size], dtype=np.float32)  # rewards buffer
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def _store(self,
              obs: np.ndarray,
              act: np.ndarray,
              rew: float,
              next_obs: np.ndarray,
              done: bool,
              ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size  # Update the ptr
        self.size = min(self.size + 1, self.max_size)  # Update the size

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)  # sample the batch uniformly
        return dict(obs=self.next_obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


# use a simple network architecture with three fully connected layers and two non-linearity functions (ReLU).
class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Defines the computation performed at every call.
        # Should be overridden by all subclasses.
        return self.layers(x)


# Implement DQN Agent
class DQNAgent:
    def __init__(
            self,
            env: gym.Env,
            memory_size: int,  # replay memory to store transitions
            batch_size: int,  # batch size for sampling
            target_update: int,  # period for target model's hard update
            epsilon_decay: float,
            max_epsilon: float = 1.0,  # max value of epsilon for greedy policy
            min_epsilon: float = 0.1,  # min value of epsilon for greedy policy
            gamma: float = 0.99,  # discount factor
    ):
        obs_dim = env.observation_space.shape[0]  # the dimension of observation
        action_dim = env.action_space.n # the dimension of action

        self.env = env
        self.memory = ReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"  # Select GPU when available
        )
        print(self.device)

        self.dqn = Network(obs_dim, action_dim).to(self.device)  # translate tensor data to GPU/CPU-based data
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())  # torch.nn.Module.load_state_dict: Loads a model’s parameter dictionary using a deserialized state_dict.
        self.dqn_target.eval()  # model.eval() to set dropout and batch normalization layers to evaluation mode before running inference. Failing to do this will yield inconsistent inference results

        self.optimizer = optim.Adam(self.dqn.parameters())

        self.transition = list()  # store transition as a list in memory
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:  # return NUMPY array
        if self.epsilon > np.random.random():
            selected_action = self.env.action_space.sample()  # choose the action randomly
        else:
            selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()  # Select the action based on maximum action-value
            # convert to GPU Tensor, and select the maximum value (based on ARGMAX function of NUMPY)
            # ARGMAX() Returns the indices of the maximum values along an axis.
            # .detach(): Returns a new Tensor, detached from the current computation graph
            # .cpu() Returns a copy of this object in CPU memory.
            selected_action = selected_action.detach().cpu().numpy()  # convert GPU-Tensor to CPU-Tensor for numpy

        if not self.is_test:
            self.transition = [state, selected_action]  # save the transition with tuple(state,selected_action)
        return selected_action

    def update_model(self) -> torch.Tensor:  # return loss as Tensor data
        samples = self.memory.sample_batch()  # random choose samples from ReplayBuffer memory
        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()  # Clear the old gradient before computing the new ones
        loss.backward()  # Computes the gradient of current tensor w.r.t. graph leaves.
        self.optimizer.step()  # update parameters with the gradients using ADAM optimizer

        return loss.item()  # loss is a single value stored in a Tensor shape, so loss.item() gets the scalar value held in the Tensor.

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:  # return torch Tensor data type
        """Return dqn loss"""
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        # reshape Tensor to 1-D vertical array
        action = torch.LongTensor(samples["acts"].reshape(-1,1)).to(self.device)  # .to() Performs Tensor dtype and/or device conversion. convert 64-bit long signed to GPU Tensor
        reward = torch.FloatTensor(samples["rews"].reshape(-1,1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1,1)).to(self.device)

        curr_q_value = self.dqn(state).gather(1, action)  # torch.Tensor.gather(dim, index): Gathers values along an axis specified by dim select q values based on selected actions

        selected_action_from_dqn = self.dqn(next_state).argmax(dim=1, keepdim=True)  # Let the DQN (first DQN) selects the action
        next_q_value_target = self.dqn_target(next_state).gather(1, selected_action_from_dqn).detach()  # Let the target DQN (second DQN) selects the Q value of the selected action

        mask = 1 - done
        target = (reward + self.gamma * next_q_value_target * mask).to(self.device)

        loss = F.smooth_l1_loss(curr_q_value, target)  # known as Huber loss
        # It is less sensitive to outliers than the mean square error loss and in some cases prevents exploding gradients.
        # In mean square error loss, we square the difference which results in a number which is much larger than the original number.
        # These high values result in exploding gradients.
        # This is avoided here as for numbers greater than 1, the numbers are not squared.

        return loss

    def _step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        next_state, reward, done, _ = self.env.step(action)  # gym environment return next state (observation), reward, done

        if not self.is_test:
            self.transition += [reward, next_state, done]  # [current state, selected action] + [reward, next_state, done] = [state, selected action, reward, next_state, done]
            self.memory._store(*self.transition)  # store the transition into the ReplayBuffer memory for batch-based update

        return next_state, reward, done

    def train(self, num_frames: int, plotting_interval: int = 200):
        self.is_test = False
        state = self.env.reset()  # Reset the environment, and get the initial observation (state)
        update_cnt = 0
        epsilons = []
        losses = []
        scores = []
        score = 0

        for frame_idx in range(1, num_frames+1):
            action = self.select_action(state)
            next_state, reward, done = self._step(action) # interact to the environment and receive the feedback

            state = next_state
            score += reward

            if done:
                state = self.env.reset()
                scores.append(score)
                score = 0

            if len(self.memory) >= self.batch_size:
                loss = self.update_model()  # Update parameters when a batch of ReplayBuffer memory is filled up
                losses.append(loss)  # for plotting
                update_cnt += 1

                # Reduce epsilon for exploration after a batch update
                self.epsilon = max(self.min_epsilon, self.epsilon - (self.max_epsilon - self.min_epsilon) * self.epsilon_decay)

                epsilons.append(self.epsilon)  # for plotting

                if update_cnt % self.target_update == 0:  # update the target dqn periodically to avoid fluctuation and divergence
                    self._target_hard_update()

            if frame_idx % plotting_interval == 0:  # auto plot periodically
                self._plot(frame_idx, scores, losses, epsilons)

        self.env.close()

    def _target_hard_update(self):
        # state_dict: is simply a Python dictionary object that maps each layer to its parameter tensor
        # Because state_dict objects are Python dictionaries, they can be easily saved, updated, altered, and restored, adding a great deal of modularity to PyTorch models and optimizers.
        self.dqn_target.load_state_dict(self.dqn.state_dict())  # torch.nn.Module.load_state_dict: Loads a model’s parameter dictionary using a deserialized state_dict.

    def _test(self) -> None:
        self.is_test = True

        state = self.env.reset()
        done = False
        score = 0

        while not done:
            self.env.render()
            action = self.select_action(state)
            next_state, reward, done = self._step(action)

            state = next_state
            score += reward

        print("score: ", score)
        self.env.close()

    def _plot(self, frame_idx: int, scores: List[float], losses: List[float], epsilons: List[float]):
        clear_output(True)
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilon')
        plt.plot(epsilons)
        plt.show()




env_id = "CartPole-v0"
env = gym.make(env_id)
seed = 777
def seed_torch(seed):
    torch.manual_seed(seed)
    # if torch.backends.cudnn.enabled:
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True

np.random.seed(seed)
seed_torch(seed)
env.seed(seed)

num_frames = 20000
memory_size = 1000
batch_size = 32
target_update = 100
epsilon_decay = 1 / 2000

agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)
agent.train(num_frames)
agent.env = gym.wrappers.Monitor(env, "videos", force=True)
agent._test()



