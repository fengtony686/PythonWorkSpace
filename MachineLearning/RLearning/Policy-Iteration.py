import gym
import numpy as np

GAMMA = 1.0
ENV = gym.make('FrozenLake-v0')
THRESHOLD = 1e-20  # 状态估计函数的的优化范围
NUM_OF_ITERATIONS = 200000  # 最多训练次数


# 计算value table
def compute_value_function(policy, gamma=GAMMA, threshold=THRESHOLD):
    value_table = np.zeros(ENV.nS)  # 相当于学习到的知识
    while True:
        updated_value_table = np.copy(value_table)
        for state in range(ENV.nS):
            action = policy[state]  # 动作根据状态
            value_table[state] = sum([trans_prob * (reward_prob + gamma * updated_value_table[next_state])
                                     for trans_prob, next_state, reward_prob, _ in ENV.P[state][action]])
            # policy iteration的公式， 更新value table
        if np.sum((np.fabs(updated_value_table-value_table))) <= threshold:
            break
    return value_table


# 从策略集Q table中选取策略，返回一个策略值
def extract_policy(value_table, gamma=GAMMA):
    policy = np.zeros(ENV.observation_space.n)
    for state in range(ENV.observation_space.n):
        Q_table = np.zeros(ENV.action_space.n)
        for action in range(ENV.action_space.n):
            for next_sr in ENV.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
        policy[state] = np.argmax(Q_table)  # 取出最大的一个
    return policy


# 交替优化策略函数和状态值估计
def policy_iteration(env, gamma=GAMMA, num_of_iterations=NUM_OF_ITERATIONS):
    random_policy = np.zeros(env.observation_space.n)
    for i in range(num_of_iterations):
        new_value_function = compute_value_function(random_policy, gamma)  # 估计状态值
        new_policy = extract_policy(new_value_function, gamma)  # 生成新的策略
        if np.all(random_policy == new_policy):
            print('Policy-Iteration converged at step %d.' % (i+1))
            break
        random_policy = new_policy  # 优化策略
    return new_policy


if __name__ == '__main__':
    print(policy_iteration(ENV))

