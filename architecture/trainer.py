from architecture.transition import Transition
from architecture.network import Network
from architecture.environment import Environment
from architecture.memory import Memory

import math
import random
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.nn.functional as F
import torch.optim as optim

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


class Trainer:

  def __init__(self, memory = Memory(10000), plot = True):
    self.steps_done = 0
    self.environment = Environment()

    _, _, self.screen_height, self.screen_width = self.environment.get_screen().shape
    self.action_space = self.environment.gym.action_space.n
    self.device = self.environment.device

    self.policy_net = Network(self.screen_height, self.screen_width, self.action_space).to(self.device)
    self.target_net = Network(self.screen_height, self.screen_width, self.action_space).to(self.device)
    
    self.episode_durations = []
    self.memory = memory

    self.optimizer = optim.RMSprop(self.policy_net.parameters())
    
    self.plot = plot

  def select_action(self, state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
    self.steps_done += 1
  
    if sample > eps_threshold:
      with torch.no_grad():
        # t.max(1) will return largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return self.policy_net(state).max(1)[1].view(1, 1)
    else:
      return torch.tensor([[random.randrange(self.action_space)]], device = self.device, dtype = torch.long)

  def plot_durations(self):
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(self.episode_durations, dtype = torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
      means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
      means = torch.cat((torch.zeros(99), means))
      plt.plot(means.numpy())
    
    plt.pause(0.001)  # pause a bit so that plots are updated

  def optimize(self):
    if len(self.memory) < BATCH_SIZE:
      return
    
    transitions = self.memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device = self.device,
                                  dtype = torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device = self.device)
    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    
    for param in self.policy_net.parameters():
      param.grad.data.clamp_(-1, 1)
    
    self.optimizer.step()
    
    return loss

  def train(self, num_episodes: int, plot = True):
    plt.ion()
    plt.figure()
    plt.imshow(self.environment.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation = 'none')
    plt.title('Example extracted screen')
    plt.show()

    for i_episode in range(num_episodes):
      # Initialize the environment and state
      self.environment.gym.reset()
      last_screen = self.environment.get_screen()
      current_screen = self.environment.get_screen()
      state = current_screen - last_screen
      mean_loss = 0
    
      for t in count():
        # Select and perform an action
        action = self.select_action(state)
        _, reward, done, _ = self.environment.gym.step(action.item())
        reward = torch.tensor([reward], device = self.device)
      
        # Observe new state
        last_screen = current_screen
        current_screen = self.environment.get_screen()
        if not done:
          next_state = current_screen - last_screen
        else:
          next_state = None
      
        # Store the transition in memory
        self.memory.push(state, action, next_state, reward)
      
        # Move to the next state
        state = next_state
      
        # Perform one step of the optimization (on the target network)
        loss = self.optimize()
        if loss is not None:
          mean_loss += loss.item()
      
        if done:
          self.episode_durations.append(t + 1)
          if plot:
            self.plot_durations()
          break
    
      print(mean_loss / BATCH_SIZE)
    
      # Update the target network, copying all weights and biases in Network
      if i_episode % TARGET_UPDATE == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())


    print('Complete')
    self.environment.gym.render()
    self.environment.gym.close()
    
    plt.ioff()
    plt.show()