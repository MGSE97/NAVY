
"""
Description:
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
Source:
    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
Observation:
    Type: Box(4)
    Num	Observation                 Min         Max
    0	Cart Position             -4.8            4.8
    1	Cart Velocity             -Inf            Inf
    2	Pole Angle                 -24 deg        24 deg
    3	Pole Velocity At Tip      -Inf            Inf

Actions:
    Type: Discrete(2)
    Num	Action
    0	Push cart to the left
    1	Push cart to the right

    Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
Reward:
    Reward is 1 for every step taken, including the termination step
Starting State:
    All observations are assigned a uniform random value in [-0.05..0.05]
Episode Termination:
    Pole Angle is more than 12 degrees
    Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
    Episode length is greater than 200
    Solved Requirements
    Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
"""
import time

import gym
import numpy as np

from Model import QNet

cuda = True

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda:0" if cuda and torch.cuda.is_available() else "cpu")

from CheeseGame.QLearn import Agent

env = gym.make('CartPole-v0')


def check_bound(value, *break_points):
    """
    Converts to range (lower, in range, higher)
    """
    r = break_points
    if not isinstance(break_points[0], int) and not isinstance(break_points[0], float):
        r = break_points[0]
    for i, p in enumerate(r):
        if value <= p:
            return i

    return len(break_points)


def round_observation(observation):
    """
    Converts observation to ranges indexes for agent

    :param observation: Raw observation
    :return: Range indexes
    """

    return [
        # int(check_bound(observation[0], np.arange(-4.8, 4.8, 0.1))),
        # 0	Cart Position             -4.8            4.8
        int(check_bound(observation[1], np.arange(-0.88, 0.88, 0.08))),
        # 1	Cart Velocity             -Inf            Inf
        int(check_bound(np.degrees(observation[2]), np.arange(-11, 11, 1))),
        # 2	Pole Angle                 -24 deg        24 deg
        int(check_bound(observation[3], np.arange(-0.88, 0.88, 0.08)))
        # 3	Pole Velocity At Tip      -Inf            Inf
    ]


# Create Agent
actions = range(env.action_space.n)
agent = Agent(None, (25, 25, 25), actions)
temp_agent = agent.__copy__()

# Create Network
net_size = 128
net = QNet(env.observation_space.shape[0], env.action_space.n, net_size, device).to(device)
optimizer = optim.Adam(net.parameters(), lr=1e-3)
net.train()

ok = False
guts = 0
i_episode = 0
total = 0
loss = 0
guts_required = 100
guts_print_div = 10
big_data = [[], []]
print("Learning...")
while not ok:
    # Agent learning
    while guts < guts_required:
        observation = env.reset()
        temp_agent.Position = agent.Position = round_observation(observation)
        score = 0
        observations = []
        done = False
        while not done:
            i, action = temp_agent.decide()
            observation, reward, done, info = env.step(action)
            score += reward

            # learn
            obs = torch.Tensor(observation)
            obs.unsqueeze(0)
            observations.append([
                [
                    obs,
                    torch.FloatTensor([1.0, 0.0] if i == 1 else [0.0, 1.0])
                ]
                , round_observation(observation), i])

            observation = round_observation(observation)
            temp_agent.remember(observation, i, reward)
            if done and guts % (guts_required / guts_print_div) == 0:
                print("\r{:3d} %: Episode {}, Avg. score: {},  loss: {}".format(int((guts % guts_required) / guts_required * 100), i_episode, total / 10, loss), end="")


        for observation in observations:
            big_data[0].append(observation[0][0])
            big_data[1].append(observation[0][1])


        if score >= 200:
            # Main Agent learning
            for observation in observations:
                agent.remember(observation[1], observation[2], 1)
                '''
                big_data[0].append(observation[0][0])
                big_data[1].append(observation[0][1])
                '''
            guts += 1
        else:
            temp_agent = agent.__copy__()

        i_episode += 1
        #if guts == guts_required:
            #i_episode = 0

    # Net learnig
    #for o in range(0, 10):
    loss = net.train_model(big_data, optimizer)
    loss = abs(loss / len(big_data))
    print("\r{:3d} %: Episode {}, Avg. score: {},  loss: {}".format(int(100), i_episode, total / 10, loss), end="")

    # Test
    total = 0
    for e in range(0, 10):
        observation = env.reset()
        score = 0
        done = False
        while not done:
            #env.render()
            with torch.no_grad():
                observation = torch.from_numpy(observation).float().to(device)
                observation = observation.unsqueeze(0)
                #result = net(observation).detach().cpu().numpy()
                result = net.get_action(observation)

            observation, reward, done, info = env.step(result)
            score += reward
        total += score

    ok = total >= 1500 # or i_episode >= 10
    print("\r{:3d} %: Episode {}, Avg. score: {},  loss: {}".format(100, i_episode, total / 10, loss), end="")

    if not ok:
        guts = 0
        big_data = [[],[]]

    i_episode += 1

# Visualize results
print("\nShowcase ...")
i_episode = 0
while True:
    observation = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        observation = torch.from_numpy(observation).float().to(device)
        observation = observation.unsqueeze(0)
        #action = net(observation).detach().cpu().numpy()
        action = net.get_action(observation)
        observation, reward, done, info = env.step(action)
        score += reward

    i_episode += 1
    print("\r{:3d} %: Episode {} score {}".format(int((guts % guts_required) / guts_required * 100), i_episode, score), end="")
    time.sleep(1)

env.close()
