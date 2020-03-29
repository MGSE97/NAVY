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
        # int(check_bound(observation[0], np.arange(-4.8, 4.8, 0.1))),        # 0	Cart Position             -4.8            4.8
        int(check_bound(observation[1], np.arange(-0.88, 0.88, 0.08))),       # 1	Cart Velocity             -Inf            Inf
        int(check_bound(np.degrees(observation[2]), np.arange(-11, 11, 1))),  # 2	Pole Angle                 -24 deg        24 deg
        int(check_bound(observation[3], np.arange(-0.88, 0.88, 0.08)))        # 3	Pole Velocity At Tip      -Inf            Inf
    ]


# Create Agent
actions = range(env.action_space.n)
agent = Agent(None, (25, 25, 25), actions)
temp_agent = agent.__copy__()

ok = False
guts = 0
i_episode = 0
guts_required = 1000    # Number of successful samples before testing
guts_print_div = 100    # Print divider
test_samples = 100      # Number of test samples after training
total = 0
print("Learning ...")
while not ok:
    # Agent learning
    observation = env.reset()
    temp_agent.Position = agent.Position = round_observation(observation)
    score = 0
    observations = []
    done = False
    while not done:
        # Trial Learning
        i, action = temp_agent.decide()
        observation, reward, done, info = env.step(action)
        observation = round_observation(observation)

        score += reward
        observations.append([observation, i])

        temp_agent.remember(observation, i, reward)
        if done and guts % (guts_required / guts_print_div) == 0:
            print("\r{:3d} %: Episode {}, Avg. score: {}".format(int((guts % guts_required) / guts_required * 100), i_episode, total), end="")

    if score >= 200:
        # Main Agent learning
        for observation in observations:
            agent.remember(observation[0], observation[1], 1)
        guts += 1
    else:
        # Reset temp_agent on fail
        temp_agent = agent.__copy__()

    # Test
    if guts > guts_required:
        total = 0
        for e in range(0, test_samples):
            # Testing agent on environment
            observation = env.reset()
            score = 0
            done = False
            while not done:
                # env.render()
                agent.Position = round_observation(observation)
                i, action = agent.decide()
                observation, reward, done, info = env.step(action)
                score += reward
            total += score

        # If all test samples passed, stop learning
        ok = total >= 200 * test_samples
        total = total / test_samples
        print("\r{:3d} %: Episode {}, Avg. score: {}".format(100, i_episode, total), end="")

        if not ok:
            # Learn next batch and try again
            guts = 0

    i_episode += 1

observations.clear()
temp_agent = None

# Visualization of results
i_episode = 0
print("\nShowcase ...")
while True:
    observation = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        agent.Position = round_observation(observation)
        i, action = agent.decide()
        observation, reward, done, info = env.step(action)
        score += reward

    i_episode += 1
    print("\rEpisode {} score {}".format(i_episode, score), end="")
    time.sleep(1)

env.close()
