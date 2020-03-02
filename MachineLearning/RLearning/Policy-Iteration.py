import gym
import numpy as np

GAMMA = 1.0
env = gym.make('FrozenLake-v0')
THRESHOLD = 1e-20


def compute_value_function(policy, gamma = GAMMA, threshold = THRESHOLD):
    value_table = np.zeros(env.nS)
    while True:
        updated_value_table = np.copy(value_table)
        for state in range(env.nS):
            action = policy[state]
            value_table[state] = sum([trans_prob * (reward_prob + gamma * updated_value_table[next_state])
                                       for trans_prob, next_state, reward_prob, _ in env.P[state][action]])
            if np.sum((np.fabs(updated_value_table-value_table))) <= threshold:
                break
        return value_table


def extract_policy(value_table, gamma = GAMMA):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))
        policy[state] = np.argmax(Q_table)
    return policy


def policy_iteration(env, gamma=GAMMA, no_of_iterations=200000):
    random_policy = np.zeros(env.observation_space.n)
    for i in range(no_of_iterations):
        new_value_function = compute_value_function(random_policy, gamma)
        new_policy = extract_policy(new_value_function, gamma)
        if np.all(random_policy == new_policy):
            print('Policy-Iteration converged at step %d.' %(i+1))
            break
        random_policy = new_policy
    return new_policy

print(policy_iteration(env))

