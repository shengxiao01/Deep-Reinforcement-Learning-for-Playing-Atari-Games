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
import pickle

DEBUG = True

RENDER = False  # if displaying game graphics real time

LEARNING_RATE = 0.00025

IMG_X, IMG_Y = 80, 80

action_space = 3  # possible action = 1, 2, 3; still, up, down

if DEBUG:
    LEARNING_RATE = 0.0025
    max_episode = 21
    max_frame = 100000
    batch_size = 32
    running_reward = None
    future_reward_discount = 0.99
    random_action_prob = 0.9
    rand_prob_step = (0.9 - 0.1)/100000
    buffer_size = 100000
    frame_skip = 2
    sync_freq = 2000
    update_freq = 5
    save_freq = 100
else:
    max_episode = 21
    max_frame = 10000
    batch_size = 32
    running_reward = None
    future_reward_discount = 0.99
    random_action_prob = 0.9
    rand_prob_step = (0.9 - 0.1)/1000000
    buffer_size = 500000
    frame_skip = 2
    sync_freq = 2000
    update_freq = 4
    save_freq = 200


save_path = "./"

#%% Deep Q-Network Structure
class DQNet():
    def __init__(self,input_size = (80, 80, 4), action_space = 3):

        self.input_x, self.input_y, self.input_frame= input_size
        self.action_space = action_space

    def build_nn(self):
        

        # [batch, in_height, in_width, in_channels]
        # assuming input to be batch_size*84*84*4
        self.input = tf.placeholder(tf.float32, shape=[None, self.input_x, self.input_y, self.input_frame])

        ##########################################################
        #[filter_height, filter_width, in_channels, out_channels]
        # conv layer 1, 8*8*32 filters, 4 stride
        self.conv1_W = tf.Variable(tf.truncated_normal([8, 8, self.input_frame, 32], 
                                                           stddev = 0.1))
        self.conv1_b = tf.Variable(tf.truncated_normal([1, 20, 20, 32], 
                                                           stddev = 0.1))
        self.conv1_strides = [1, 4, 4, 1]
        #output 20*20*32 
        self.conv1_out = tf.nn.conv2d(self.input, 
                                          self.conv1_W, 
                                          self.conv1_strides, 
                                          padding = 'SAME') + self.conv1_b
        self.conv1_out = tf.nn.relu(self.conv1_out)
        
        
        ###########################################################
        # conv layer 2, 4*4*64 filters, 2 stride
        self.conv2_W = tf.Variable(tf.truncated_normal([4, 4, 32, 64],
                                                           stddev = 0.1))
        self.conv2_b = tf.Variable(tf.truncated_normal([1, 9, 9, 64], 
                                                           stddev = 0.1))
        self.conv2_strides = [1, 2, 2, 1]
        # output 9*9*64
        self.conv2_out = tf.nn.conv2d(self.conv1_out, 
                                          self.conv2_W, 
                                          self.conv2_strides, 
                                          padding = 'VALID') + self.conv2_b
        self.conv2_out = tf.nn.relu(self.conv2_out)
        
        
        ###########################################################
        # conv layer 3, 3*3*64 filters
        self.conv3_W = tf.Variable(tf.truncated_normal([3, 3, 64, 64],
                                                           stddev = 0.1))
        self.conv3_b = tf.Variable(tf.truncated_normal([1, 7, 7, 64], 
                                                           stddev = 0.1))
        self.conv3_strides = [1, 1, 1, 1]
        # output 7*7*64
        self.conv3_out = tf.nn.conv2d(self.conv2_out, 
                                          self.conv3_W, 
                                          self.conv3_strides, 
                                          padding = 'VALID') + self.conv3_b
        self.conv3_out = tf.nn.relu(self.conv3_out)

        ###########################################################
        # fully connected layer 1, (7*7*64 = 3136) * 512
        self.ff1_input = tf.reshape(self.conv3_out, [-1, 3136])
        self.ff1_W = tf.Variable(tf.truncated_normal([3136, 512],
                                                         stddev = 0.1))
        self.ff1_b = tf.Variable(tf.truncated_normal([1, 512],
                                                         stddev = 0.1))
        # output batch_size * 512
        self.ff1_out = tf.matmul(self.ff1_input, self.ff1_W) + self.ff1_b
        self.ff1_out = tf.nn.relu(self.ff1_out)
        
        
        self.advantage_input,self.value_input = tf.split(self.ff1_out, 2, axis = 1)
        
        self.advantage_W = tf.Variable(tf.truncated_normal([256, self.action_space], 
                                                  stddev = 0.1))
        self.value_W = tf.Variable(tf.truncated_normal([256, 1],
                                                  stddev = 0.1))
        
        self.advantage_out = tf.matmul(self.advantage_input,self.advantage_W)
        
        self.value_out = tf.matmul(self.value_input,self.value_W)
        
        #Then combine them together to get our final Q-values.
        self.output = self.value_out + self.advantage_out - tf.reduce_mean(self.advantage_out,reduction_indices=1,keep_dims=True)
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

    def variable_list(self):
        return [self.conv1_W, self.conv1_b, self.conv2_W, self.conv2_b, self.conv3_W, self.conv3_b, self.ff1_W, self.ff1_b, self.advantage_W, self.value_W]

#%% utility functions

class replayMemory():
    
    def __init__(self, size):
        # [i, :, :, 0:4] is the current state
        # [i, :, :, 1:5] is the next state
        self.frames = np.zeros((size, IMG_X, IMG_Y, 5), dtype = 'float32')
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
        return (self.frames[idx, :, :, 0:4], self.frames[idx, :, :, 1:5], self.actions[idx], self.rewards[idx], self.done[idx])


def process_frame(frame):
    # input a single frame
    # crop & downsample & average over 3 color channels 
    return np.mean(frame[34: 194 : 2, 0: 160 : 2, :], axis = 2, dtype = 'float32') > 100
 
def copy_variables(from_nn, to_nn, sess):   
    for i in range(len(from_nn)):
        op = to_nn[i].assign(from_nn[i].value())
        sess.run(op)


#%%
###################################################################
# pre-training, fill the replay memory buffer with 10,000 random examples
memory_buffer = replayMemory(buffer_size)
buffer_counter = 0
state_input = np.zeros((IMG_X, IMG_Y, 2), dtype = 'float32')

env = gym.make("Pong-v0")
while True:
    
    # reset the game environment, take a initial screen shot
    observation = env.reset()
    # the state of current game play, 0:2 is 3 previous frame,
    # 3 is the current frame, 4 is the frame after action
    state = np.zeros((IMG_X, IMG_Y, 5), dtype = 'float32')
    state[:,:,-1] = process_frame(observation)
      
    for t in range(buffer_size):

        action = random.randint(0,2)
        
        # run the game with same action for a few frames
        for _ in range(frame_skip):
            observation, reward, done, info = env.step(action+1)
            if done:
                break
        
        state = np.roll(state, -1, axis = 2)
        # effective area [34:194, 0:168] with 2*2 downsampling -> 160/2 * 130/2 matrix
        state[:,:,-1] = process_frame(observation) 
        memory_buffer.add(state, action, reward, done)
        buffer_counter += 1
                
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    
    if buffer_counter > buffer_size:
        break
env.close()
  
#%%
###################################################################
# Initialize environment

env = gym.make("Pong-v0")


tf.reset_default_graph()
Atari_AI_primary = DQNet()
Atari_AI_primary.build_nn()

Atari_AI_target = DQNet()
Atari_AI_target.build_nn()

init_op = tf.global_variables_initializer()
reward_log = []

sess = tf.Session()
sess.run(init_op)

# Initialize saver
saver = tf.train.Saver()

try:
    ckpt = tf.train.get_checkpoint_state(save_path)
    load_path = ckpt.model_checkpoint_path
    saver.restore(sess, load_path)
    f = open(save_path + 'reward_log.cptk','rb')
    reward_log = pickle.load(f)
    f.close()
    random_action_prob = 0.1
    print("Session restored...")
except:
    primary_variables = Atari_AI_primary.variable_list()
    target_variables = Atari_AI_target.variable_list()
    copy_variables(primary_variables, target_variables, sess)    
    
    
    print("Nothing to restore...")


# start training
i_episode = 0
steps = 0
updates = 0
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
        if np.random.random_sample() > random_action_prob:
            # use model to predict action
            action = sess.run(Atari_AI_primary.predict,
                              feed_dict = {Atari_AI_primary.input: np.expand_dims(state[:,:,1:5], axis = 0)})[0]
        else: 
            # random action
            action = random.randint(0, 2) # random sample action from 1 to 3
            

        # excute the action for a few steps
        for _ in range(frame_skip):
            observation, reward, done, info = env.step(action+1)
            reward_sum += reward
            if done:
                break
        
        # update the new state and reward and memory buffer
        state = np.roll(state, -1, axis = 2)
        state[:,:,-1] = process_frame(observation) 
        
        memory_buffer.add(state, action, reward, done)
        
        
        updates += 1
        if updates % update_freq == 0:
            
            if random_action_prob > 0.1:    
                random_action_prob -= rand_prob_step
            steps += 1
            # randomly sample minibatch from memory
            batch_sample_index = random.sample(range(buffer_size), batch_size)
            state_current, state_future, actions, current_rewards, end_game = memory_buffer.makeBatch(batch_sample_index)
            future_rewards = sess.run(Atari_AI_target.output,
                                      feed_dict = {Atari_AI_target.input: state_future})
            targetQ = current_rewards + future_reward_discount * (1 - end_game) * np.amax(future_rewards, axis = 1)
            
            # update the target-value function Q
            sess.run(Atari_AI_primary.update, feed_dict = {
                    Atari_AI_primary.input: state_current,
                    Atari_AI_primary.actions: actions,
                    Atari_AI_primary.targetQ: targetQ})
                    # every C step reset Q' = Q
            if steps % sync_freq == 0:
                primary_variables = Atari_AI_primary.variable_list()
                target_variables = Atari_AI_target.variable_list()
                copy_variables(primary_variables, target_variables, sess)
        
        
        # save the model after every 200 updates       
        if done:
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

            if DEBUG:
                if i_episode % 10 == 0:
                    print('ep {}: reward: {}, mean reward: {:3f}'.format(i_episode, reward_sum, running_reward))
                else:
                    print('\tep {}: reward: {}'.format(i_episode, reward_sum))
             
            # saving results    
            if i_episode % 10 == 0:
                reward_log.append(running_reward)
                
            if i_episode % save_freq == 0:
                saver.save(sess, save_path+'model-'+str(i_episode)+'.cptk')
                f = open(save_path + 'reward_log.cptk','wb')
                pickle.dump(reward_log, f)
                f.close()
                
            break