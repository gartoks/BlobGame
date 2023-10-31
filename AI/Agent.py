import torch
import random, numpy as np
from pathlib import Path

from Model import Model
from collections import deque


class Agent:
    def __init__(self, image_dim, additional_dim, action_dim, save_dir, checkpoint=None):
        self.image_dim = image_dim
        self.additional_dim = additional_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=100_000)
        self.batch_size = 32

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.burnin = 1e5  # min. experiences before training
        self.learn_every = 3   # no. of experiences between updates to Q_online
        self.sync_every = 1e4   # no. of experiences between Q_target & Q_online sync

        self.save_every = 5000   # no. of experiences between saving Mario Net
        self.save_dir = save_dir

        self.use_cuda = torch.cuda.is_available()

        if (self.use_cuda):
            print("Using GPU")
            self.device = torch.device('cuda')
        else:
            print("Using CPU")
            self.device = torch.device("cpu")

        # Mario's DNN to predict the most optimal action - we implement this in the Learn section
        self.net = Model(self.image_dim, self.additional_dim, self.action_dim).float()
        self.net.to(self.device)
        
        if checkpoint is not None:
            self.load(checkpoint)


        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def reset_feedback(self):
        self.net.reset_feedback("online")
        self.net.reset_feedback("target")

    def act(self, original_state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            image_state = torch.FloatTensor(original_state[0]).to(self.device)
            image_state = image_state.unsqueeze(0)
            additional_state = torch.cat((torch.FloatTensor(original_state[1]), torch.FloatTensor(original_state[2]), torch.FloatTensor(original_state[3]))).to(self.device)
            action_values = self.net(image_state, additional_state, model='online')
            action_idx = torch.argmax(action_values, axis=0).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        action = torch.LongTensor([action])
        reward = torch.DoubleTensor([reward])
        done = torch.BoolTensor([done])

        self.memory.append( (state, next_state, action, reward, done,) )


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = zip(*batch)
        image_state = [torch.FloatTensor(state[i][0]).to(self.device) for i in range(self.batch_size)]
        additioal_state = [torch.cat((torch.FloatTensor(state[i][1]), torch.FloatTensor(state[i][2]), torch.FloatTensor(state[i][3]))).to(self.device) for i in range(self.batch_size)]
        next_image_state = [torch.FloatTensor(next_state[i][0]).to(self.device) for i in range(self.batch_size)]
        next_additioal_state = [torch.cat((torch.FloatTensor(next_state[i][1]), torch.FloatTensor(next_state[i][2]), torch.FloatTensor(next_state[i][3]))).to(self.device) for i in range(self.batch_size)]
        return (torch.stack(image_state), torch.stack(additioal_state)), (torch.stack(next_image_state), torch.stack(next_additioal_state)), torch.stack(action).squeeze(), torch.stack(reward).squeeze(), torch.stack(done).squeeze()


    def td_estimate(self, state, data, action):
        current_Q = self.net(state, data, model='online', feedback=torch.zeros((self.batch_size, self.net.feedback_dim)).to(self.device))[np.arange(0, self.batch_size), action] # Q_online(s,a)
        return current_Q


    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(*next_state, model='online', feedback=torch.zeros((self.batch_size, self.net.feedback_dim)).to(self.device))
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(*next_state, model='target', feedback=torch.zeros((self.batch_size, self.net.feedback_dim)).to(self.device))[np.arange(0, self.batch_size), best_action]
        next_Q = next_Q.to("cpu")
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()


    def update_Q_online(self, td_estimate, td_target) :
        loss = self.loss_fn(td_estimate, td_target.to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(*state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)


    def save(self):
        save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate
            ),
            save_path
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")


    def load(self, load_path):
        if not load_path.exists():
            raise ValueError(f"{load_path} does not exist")

        ckp = torch.load(load_path, map_location=self.device)
        exploration_rate = ckp.get('exploration_rate')
        state_dict = ckp.get('model')

        print(f"Loading model at {load_path} with exploration rate {exploration_rate}")
        self.net.load_state_dict(state_dict)
        self.exploration_rate = exploration_rate