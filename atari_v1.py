import gym
import numpy as np

env = gym.make('Pong-v0')

for i_episode in range(2):
    observation = env.reset()
    for t in range(100):
        env.render()
        #print(observation)
        action = env.action_space.sample()
        print(action)
        observation, reward, done, info = env.step(4)
        #print(np.array(observation).shape)
        print(reward)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

