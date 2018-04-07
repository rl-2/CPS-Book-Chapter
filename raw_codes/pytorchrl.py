import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
from torch.distributions import Categorical
import torchvision
import numpy as np
import math
import utilities as util
import ipdb
import math 

class NetCNN(nn.Module):
    def __init__(self, num_actions):
        super(NetCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(21472, 120)
        #self.fc1 = nn.Linear(624, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_actions)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        #Pdb().set_trace()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(-1, 21472)
        #x = x.view(-1, 624)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def forward2(self, x):
        #Pdb().set_trace()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) 
        x = x.view(-1, 21472)
        #x = x.view(-1, 624)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class NetMLP(nn.Module):
    def __init__(self, num_pixels, num_actions):
        super(NetMLP, self).__init__()
        self.fc1 = nn.Linear(num_pixels, 50)
        self.fc2 = nn.Linear(50, num_actions)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal(m.weight.data)

    def forward(self, x):
        Pdb().set_trace()
        x = Variable(torch.Tensor(x).view(-1))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=0)

class Agent():
    def __init__(self, model, num_pixels, num_actions, max_steps, discount_factor, learning_rate, scheduler_step_size, scheduler_gamma):
        if model == "MLP":
            self.policy = NetMLP(num_pixels, num_actions)
            self.model = "MLP"
        elif model == "CNN":
            self.policy = NetCNN(num_actions)
            self.policy.cuda()
            self.model = "CNN"

        self.discount_factor = discount_factor
        print("Using {} model".format(self.model))
        print("discount_factor {}".format(self.discount_factor))
        print("Initial learning rate {}".format(learning_rate))
        self.num_actions = num_actions

        # Initalize baseline to zeros 
        self.max_steps = max_steps
        self.baseline = np.zeros(self.max_steps)
        self.baseline_update_cnt = 1

        self.optimizer = optim.SGD(self.policy.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma) # step_size was 10/.99

    def select_action(self, state, greedy=False):
        #Pdb().set_trace()
        if self.model == "CNN":
            # input must be (minibatch x in_channels x iH x iW)
            #state = torch.Tensor(state) # iH x iW
            #state = state.unsqueeze(0)  # 1 x iH x iW
            #state = state.unsqueeze(0)  # 1 x 1 x iH x iW
            probs = self.policy.forward(Variable(state.cuda()))
        elif self.model == "MLP":
            probs = self.policy.forward(state)

        #print(probs)
        
        if greedy:     
            _, action = probs.data.max(1)
            action = int(action)
        else:
            action = Categorical(probs).sample().data[0]

        return action, probs

    def fit(self, states, actions, rewards, advantage_type):
        success = True

        actions = Variable(torch.LongTensor(actions).cuda())
        #actions = actions.cuda()
        rewards = np.array(rewards)

        #if self.model == "CNN":
        #    states = torch.cat(states)
            #S = S.cuda()

        #Pdb().set_trace()

        if advantage_type == 0:
            R = 0
            discounted = []
            for r in rewards[::-1]:
                R = r + self.discount_factor * R
                discounted.insert(0, R)
            discounted = torch.Tensor(discounted)

            if len(rewards) > 10:
                discounted = (discounted - discounted.mean()) / (discounted.std() + np.finfo(np.float32).eps)
            
            if len(discounted.numpy()) == 1:
                print("A glitch was handled in Agent.fit()")
                success = False
            else:
                print(discounted.numpy())

        elif advantage_type == 1:
            discounted = []
            H = len(rewards)
            for t in range(H):
                R = .2*(t-H-1)+2
                discounted.append(R)  
            print(discounted)
        
        elif advantage_type == 2:

            R = 0
            discounted = []
            for r in rewards[::-1]:
                R = r + self.discount_factor * R
                discounted.insert(0, R)
            discounted = torch.Tensor(discounted)

            #if len(rewards) > 10:
            #    discounted = (discounted - discounted.mean()) / (discounted.std() + np.finfo(np.float32).eps)
            
            if len(discounted.numpy()) == 1:
                print("A glitch was handled in Agent.fit()")
                success = False
            else:
                print(discounted.numpy())

        # Exponentially penalized returns
        elif advantage_type == 3:
            #Pdb().set_trace()

            # Sometimes drone spawns right on box, take care of that
            rewards[0:2] = 0
            if sum(rewards)==0:
                print("Skipping network update")
                return


            R = 0
            discounted = []
            max_box = 3

            collected = float(sum(rewards))
            
            # linear reward
            # collection_ratio = collected/max_box
            
            # exponential reward
            collection_ratio = math.exp(collected-max_box)

            for r in rewards[::-1]:
                R = r + self.discount_factor * R
                discounted.insert(0, R)
            discounted = torch.Tensor(discounted)
            discounted = discounted*collection_ratio

            #if len(rewards) > 10:
            #    discounted = (discounted - discounted.mean()) / (discounted.std() + np.finfo(np.float32).eps)
            
            if len(discounted.numpy()) == 1:
                print("A glitch was handled in Agent.fit()")
                success = False
            else:
                print(discounted.numpy())

        # baselines
        elif advantage_type == 4:

            # Sometimes drone spawns right on box, take care of that
            rewards[0:3] = 0
           
            # remove the below three lines to make a penalty for missing for three boxes 
            # --penalty--
            # if sum(rewards)==0:
            #     print("Skipping network update")
            #     return

            R = 0
            discounted = []
            for r in rewards[::-1]:
                R = r + self.discount_factor * R
                discounted.insert(0, R)

            if len(discounted) == 1:
                print("A glitch was handled in Agent.fit()")
                success = False

            # Make discounted and self.baseline the same length
            discounted = np.append(discounted,np.zeros(len(self.baseline)-len(discounted)))
            
            # The original version of discounted is needed in the baseline update
            discounted_copy = discounted - self.baseline            

            print(discounted_copy)

            # Update baseline based on discounted return
            self.baseline = (self.baseline*(self.baseline_update_cnt-1) + discounted)/self.baseline_update_cnt
            self.baseline_update_cnt += 1
            
            # Move discounted_copy to discounted
            discounted = torch.Tensor(discounted_copy)


        policy_loss = []
        for (state,action,reward) in zip(states, actions, discounted):
            if self.model == "CNN":
                #state = state.unsqueeze(0)
                #probs = self.policy.forward(Variable(util.processImage(state)))
                probs = self.policy.forward(Variable(state.cuda()))
            elif self.model == "MLP":
                probs = self.policy.forward(state)

            m = Categorical(probs)
            log_prob = m.log_prob(action)
            policy_loss.append(-log_prob*reward)
            
        #Pdb().set_trace()            
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.scheduler.step()
        print("Using learning rate {}".format(self.optimizer.param_groups[0]['lr']))
        self.optimizer.step()
        return success

if __name__ == '__main__':
    agent = Agent(3)