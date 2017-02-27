#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 09:56:10 2017

@author: shengx
"""

#%%
import tensorflow as tf
import numpy as np
import random
import gym
import matplotlib.pyplot as plt

DEBUG = True


RENDER = False  # if displaying game graphics real time
LEARNING_RATE = 0.001 

IMG_X, IMG_Y = 80, 80

#%% Deep Q-Network Structure
class DQNet():
    def __init__(self,input_size = (80, 80, 1), action_space = 3):

        self.input_x, self.input_y, self.input_frame= input_size
        self.action_space = action_space

    def build_nn(self):
        

        # [batch, in_height, in_width, in_channels]
        # assuming input to be batch_size*84*84*4
        self.dqn_input = tf.placeholder(tf.float32, shape=[None, self.input_x, self.input_y, self.input_frame])


        self.W1 = tf.Variable(tf.truncated_normal([6400, 512], stddev = 0.1))
        self.b1 = tf.Variable(tf.truncated_normal([1, 512], stddev = 0.1))
        self.hidden = tf.nn.relu(tf.matmul(tf.reshape(self.dqn_input,[-1, 6400]), self.W1) + self.b1)
        self.W2 = tf.Variable(tf.truncated_normal([512, 3], stddev = 0.1))
        self.b2 = tf.Variable(tf.truncated_normal([1, 3], stddev = 0.1))
        
        self.output = tf.matmul(self.hidden, self.W2) + self.b2
        ###########################################################
        #[filter_height, filter_width, in_channels, out_channels]
        # conv layer 1, 8*8*32 filters, 4 stride
        #        self.dqn_conv1_W = tf.Variable(tf.truncated_normal([8, 8, 1, 32], 
        #                                                           stddev = 0.1))
        #        self.dqn_conv1_strides = [1, 4, 4, 1]
        #        #output 20*20*32 
        #        self.dqn_conv1_out = tf.nn.conv2d(self.dqn_input, 
        #                                          self.dqn_conv1_W, 
        #                                          self.dqn_conv1_strides, 
        #                                          padding = 'SAME')
        #        self.dqn_conv1_out = tf.nn.relu(self.dqn_conv1_out)
        #        
        #        
        #        ###########################################################
        #        # conv layer 2, 4*4*64 filters, 2 stride
        #        self.dqn_conv2_W = tf.Variable(tf.truncated_normal([4, 4, 32, 64],
        #                                                           stddev = 0.1))
        #        self.dqn_conv2_strides = [1, 2, 2, 1]
        #        # output 9*9*64
        #        self.dqn_conv2_out = tf.nn.conv2d(self.dqn_conv1_out, 
        #                                          self.dqn_conv2_W, 
        #                                          self.dqn_conv2_strides, 
        #                                          padding = 'VALID')
        #        self.dqn_conv2_out = tf.nn.relu(self.dqn_conv2_out)
        #        
        #        
        #        ###########################################################
        #        # conv layer 3, 3*3*64 filters
        #        self.dqn_conv3_W = tf.Variable(tf.truncated_normal([3, 3, 64, 64],
        #                                                           stddev = 0.1))
        #        self.dqn_conv3_strides = [1, 1, 1, 1]
        #        # output 7*7*64
        #        self.dqn_conv3_out = tf.nn.conv2d(self.dqn_conv2_out, 
        #                                          self.dqn_conv3_W, 
        #                                          self.dqn_conv3_strides, 
        #                                          padding = 'VALID')
        #        self.dqn_conv3_out = tf.nn.relu(self.dqn_conv3_out)
        #        
        #        
        #        ###########################################################
        #        # fully connected layer 1, (7*7*64 = 3136) * 512
        #        self.dqn_ff1_input = tf.reshape(self.dqn_conv3_out, [-1, 3136])
        #        self.dqn_ff1_W = tf.Variable(tf.truncated_normal([3136, 512],
        #                                                         stddev = 0.1))
        #        self.dqn_ff1_b = tf.Variable(tf.truncated_normal([1, 512],
        #                                                         stddev = 0.1))
        #        # output batch_size * 512
        #        self.dqn_ff1_out = tf.matmul(self.dqn_ff1_input, self.dqn_ff1_W) + self.dqn_ff1_b
        #        self.dqn_ff1_out = tf.nn.relu(self.dqn_ff1_out)
        #        
        #        
        #        ###########################################################
        #        # fully connected layer 2, 
        #        self.dqn_ff2_W = tf.Variable(tf.truncated_normal([ 512, self.action_space],
        #                                                         stddev = 0.1))
        #        self.dqn_ff2_b = tf.Variable(tf.truncated_normal([ 1, self.action_space],
        #                                                         stddev = 0.1))
        #        # final output, batch_size * action_space
        #        self.dqn_ff2_out = tf.matmul(self.dqn_ff1_out, self.dqn_ff2_W) + self.dqn_ff2_b
        
        
        ###########################################################
        # prediction, loss, and update
        
        self.predict = tf.argmax(self.output, 1)
        
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.actions_onehot = tf.one_hot(self.actions, self.action_space, dtype=tf.float32)
        
        self.Q = tf.reduce_sum((self.output * self.actions_onehot), 
                               reduction_indices=1)
        
        self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))

        self.update = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(self.loss)

#%% replay memory class

class replayMemory():
    
    def __init__(self, size):
        # [i, :, :, 0:4] is the current state
        # [i, :, :, 1:5] is the next state
        self.frames = np.zeros((size, IMG_X, IMG_Y, 2), dtype = 'float32')
        self.actions = np.zeros((size), dtype = 'int32')
        self.rewards = np.zeros((size), dtype = 'float32')  
        self.done = np.zeros((size), dtype = 'int32')
        self.__counter = 0
        self.__size = size
        
    def add(self, state, action, reward, done):
        
        self.frames[self.__counter, :, :, : ] = state
        self.actions[self.__counter] = action
        self.rewards[self.__counter] = reward
        self.done[self.__counter] = done
        
        self.__counter += 1
        self.__counter = self.__counter % self.__size
     
    def makeBatch(self, idx):  
        return (self.frames[idx, :, :, 0], self.frames[idx, :, :, 1], self.actions[idx], self.rewards[idx], self.done[idx])

#%% utility functions

def process_frame(frame):
    # input a single frame
    # crop & downsample & average over 3 color channels 
    return np.mean(frame[34: 194 : 2, 0: 160 : 2, :], axis = 2, dtype = 'float32') > 100
 
#%% initialize and running the model 

###################################################################
# initialize parameter
max_episode = 2
max_frame = 1000

frame_skip = 2
reward_decay = 0.99

###################################################################
# initialize replay memory buffer

buffer_size = 10000

###################################################################
# Set the game
env = gym.make("Pong-v0")

#%%
###################################################################
# pre-training, fill the replay memory buffer with 10,000 random examples
memory_buffer = replayMemory(buffer_size)
buffer_counter = 0
state_input = np.zeros((IMG_X, IMG_Y, 2), dtype = 'float32')
while True:
    
    # reset the game environment, take a initial screen shot
    observation = env.reset()
    # the state of current game play, 0:2 is 3 previous frame,
    # 3 is the current frame, 4 is the frame after action
    state = np.zeros((IMG_X, IMG_Y, 5), dtype = 'float32')
    state[:,:,-1] = process_frame(observation)
      
    for t in range(buffer_size):
        
        env.render()
        action = env.action_space.sample()
        
        # run the game with same action for a few frames
        for _ in range(frame_skip):
            observation, reward, done, info = env.step(action)
            if done:
                break
        
        state = np.roll(state, -1, axis = 2)
        # effective area [34:194, 0:168] with 2*2 downsampling -> 160/2 * 130/2 matrix
        state[:,:,-1] = process_frame(observation) 
        state_input[:,:,0] = state[:,:,-2] - state[:,:,-3]
        state_input[:,:,1] = state[:,:,-1] - state[:,:,-2]
        memory_buffer.add(state_input, action, reward, done)
        buffer_counter += 1
                
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    
    if buffer_counter > buffer_size:
        break
        

env.close()
  
#%%
###################################################################
# training 
action_space = 3  # possible action = 1, 2, 3; still, up, down

max_episode = 21
max_frame = 1000
batch_size = 32
running_reward = None
future_reward_discount = 0.99
random_action_prob = 0.9
rand_prob_step = (0.9 - 0.1)/5000


save_frequency = 100
save_path = "/home/shengx/Documents/CheckpointData/"



tf.reset_default_graph()
Atari_AI = DQNet()
Atari_AI.build_nn()


init_op = tf.global_variables_initializer()
sess = tf.Session()
saver = tf.train.Saver()

sess.run(init_op)

try:
    ckpt = tf.train.get_checkpoint_state(save_path)
    load_path = ckpt.model_checkpoint_path
    saver.restore(sess, load_path)
    print("Session restored...")
except:
    print("Nothing to restore...")


i_episode = 0


while True:
    
    i_episode += 1
    
    observation = env.reset()
    
    state = np.zeros((IMG_X, IMG_Y, 5), dtype = 'float32')
    state[:,:,-1] = process_frame(observation)
    reward_sum = 0
    
    for t in range(max_frame):
        if RENDER:
            env.render()
        # select an action based on the action-value function Q
        if np.random.random_sample() < random_action_prob:
            # use model to predict action
            state_input[:,:,0] = state[:,:,-2] - state[:,:,-3]
            action = sess.run(Atari_AI.predict,
                              feed_dict = {Atari_AI.dqn_input: np.reshape(state[:,:,-1] - state[:,:,-2], [1, 80, 80, 1])})[0]
        else: 
            # random action
            action = random.randint(1, 3) # random sample action from 1 to 3
        random_action_prob -= rand_prob_step
        random_action_prob = max(random_action_prob, 0.1)
        # excute the action for a few steps
        for _ in range(frame_skip):
            observation, reward, done, info = env.step(action)
            reward_sum += reward
            if done:
                break
        
        # update the new state and reward and memory buffer
        state = np.roll(state, -1, axis = 2)
        state[:,:,-1] = process_frame(observation) 
        
        state_input[:,:,0] = state[:,:,-2] - state[:,:,-3]
        state_input[:,:,1] = state[:,:,-1] - state[:,:,-2]
        memory_buffer.add(state_input, action, reward, done)
        
        # randomly sample minibatch from memory
        batch_sample_index = random.sample(range(buffer_size), batch_size)
        state_current, state_future, actions, current_rewards, end_game = memory_buffer.makeBatch(batch_sample_index)
        future_rewards = sess.run(Atari_AI.output,
                                  feed_dict = {Atari_AI.dqn_input: np.expand_dims(state_future, axis = 3)})
        targetQ = current_rewards + future_reward_discount * (1 - end_game) * np.amax(future_rewards, axis = 1)
        
        # update the target-value function Q
        sess.run(Atari_AI.update, feed_dict = {
                Atari_AI.dqn_input: np.expand_dims(state_current, axis = 3),
                Atari_AI.actions: actions,
                Atari_AI.targetQ: targetQ})
        
        # every C step reset Q' = Q
        
        
        # save the model after every 200 updates       
        if done:
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            
            if i_episode % 10 == 0:
                print('ep {}: reward: {}, mean reward: {:3f}'.format(i_episode, reward_sum, running_reward))
            else:
                print('\tep {}: reward: {}'.format(i_episode, reward_sum))
                
                
            if i_episode % save_frequency == 0:
                saver.save(sess, save_path+'model-'+str(i_episode)+'.cptk')
                print("SAVED MODEL #{}".format(i_episode))   
                
            
            break
