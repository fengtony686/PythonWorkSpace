import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n  # 所有动作的个数，这个例子是2
N_STATES = env.observation_space.shape[0]  # 一个状态的维度数,这个例子是4


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.memory_counter = 0  # 记录内存存了多少了
        self.learn_step_counter = 0  # 记录学了多少步了
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))  # 形状是MEMORY_CAPACITY行，N_STATES*2+2列，这里是因为每一步需要存两个状态和一个动作一个回馈
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() <= EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]  # 返回的是一个动作的序号
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))  # 形状是1行12列
        index = self.memory_counter % MEMORY_CAPACITY  # 判断满了没，满了就归0
        self.memory[index, :] = transition  # 放到新的一行里面
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 随便抽一个数据，每一列是一个数据
        b_memory = self.memory[sample_index, :]  # 取出batch行
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])  # 相当于第0个到第3个，初始状态
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))  # 第4,5个是动作
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        q_eval = self.eval_net(b_s).gather(1, b_a)  # 1表示横向，b_a是索引即选哪个动作 (batch, 1)
        q_next = self.target_net(b_s_).detach()  # q_next 不进行反向传递误差, 所以 detach (batch ,2)
        q_target = b_r + GAMMA * q_next.max(1)[0]  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

for i_episode in range(400):
    s = env.reset()
    while True:
        env.render()  # 显示动画
        a = dqn.choose_action(s)  # 根据环境s得到动作a
        s_, r, done, info = env.step(a)  # 得到新状态s_，反馈r
        print(r)
        x, x_dot, theta, theta_dot = s_  # 把状态拿出来
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8  # 定义状态r1
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5  # 定义状态r2
        r = r1 + r2
        dqn.store_transition(s, a, r, s_)
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()  # 记忆库满了就进行学习

        if done:  # 如果回合结束, 进入下回合
            break

        s = s_
