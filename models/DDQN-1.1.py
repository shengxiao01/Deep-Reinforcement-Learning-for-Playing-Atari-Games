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
import threading

DEBUG = False

RENDER = False  # if displaying game graphics real time

LEARNING_RATE = 0.00025 

IMG_X, IMG_Y = 80, 80

action_space = 3  # possible action = 1, 2, 3; still, up, down

if DEBUG:
    LEARNING_RATE = 0.001
    max_episode = 21
    max_frame = 1000
    batch_size = 32
    running_reward = None
    max_reward = None
    future_reward_discount = 0.99
    random_action_prob = 0.9
    rand_prob_step = (0.9 - 0.1)/2000
    buffer_size = 1000
    frame_skip = 2
    sync_freq = 10
    update_freq = 15
    save_freq = 5
else:
    max_episode = 21
    max_frame = 10000
    batch_size = 32
    running_reward = None
    max_reward = None
    future_reward_discount = 0.99
    random_action_prob = 0.9
    rand_prob_step = (0.9 - 0.1)/1000000
    buffer_size = 500000
    frame_skip = 2
    sync_freq = 5000
    update_freq = 4
    save_freq = 200


save_path = "./"

#%% Deep Q-Network Structure
class DQNet():
    def __init__(self, scope, action_space, input_size = (80, 80, 4)):

        self.input_x, self.input_y, self.input_frame= input_size
        self.action_space = action_space
        self.__scope = scope

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

    def sync_variables(self, sess, from_scope):
        # adding scope to network        
        from_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope.name)
        
        for from_var, to_var in zip(from_variables, self.variables):
            op = to_var.assign(from_var.value())
            sess.run(op)

    def train(self, sess, state, action, targetQ):
        sess.run(self.update, feed_dict = {
                 self.input: state,
                 self.actions: action,
                 self.targetQ: targetQ})

    def estimateQ(self, sess, state):
        Q_out = sess.run(self.output, 
                         feed_dict = {self.input: state})
        return Q_out

    def predict_act(self, sess, state):
        # 1X80X80X4 single image
        action = sess.run(self.predict,
                          feed_dict = {self.input: state})
        return action
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
     
    def makeBatch(self, batch_size):  
        # randomly sample a batch 
        idx = random.sample(range(self.__size), batch_size)
        return (self.frames[idx, :, :, 0:4], self.frames[idx, :, :, 1:5], self.actions[idx], self.rewards[idx], self.done[idx])


def process_frame(frame):
    # input a single frame
    # crop & downsample & average over 3 color channels 
    return np.mean(frame[34: 194 : 2, 0: 160 : 2, :], axis = 2, dtype = 'float32') > 100
 
def copy_variables(from_nn, to_nn, sess):   
    for i in range(len(from_nn)):
        op = to_nn[i].assign(from_nn[i].value())
        sess.run(op)

#%%  Logger class
class Logger(object):
    def __init__(self, sess, saver):
        # RawValue because we don't need it to create a Lock:
        self.global_episode = 0
        self.running_reward = None
        
        self.save_freq = 100
        self.save_path = './'
        self.__saver = sess
        self.__sess = saver
        self.reward_log = []

    def log(self, reward_sum):
        

        self.running_reward = reward_sum if self.running_reward is None else self.running_reward * 0.99 + reward_sum * 0.01
        self.global_episode += 1
        if self.global_episode % 10 == 0:
	        self.reward_log.append(self.running_reward)
	        print('Ep {}: reward: {}, mean reward: {:3f}'.format(self.global_episode, reward_sum, self.running_reward))
        else:
	        print('\tEp {}: reward: {}'.format(self.global_episode, reward_sum)) 
	
        if self.global_episode % 200 == 0 and self.__saver is not None:
            # separate saving model and saving scores
	        self.save()
                
    def save(self):
        
        self.__saver.save(self.__sess, self.save_path+'model-'+str(self.global_episode)+'.cptk')
        f = open(self.save_path + 'reward_log.cptk','wb')
        pickle.dump(self.reward_log, f)
        f.close()    

    def restore(self): 
        
        try:
            ckpt = tf.train.get_checkpoint_state(self.save_path)
            load_path = ckpt.model_checkpoint_path
            saver.restore(sess, load_path)
            f = open(self.save_path + 'reward_log.cptk','rb')
            self.reward_log = pickle.load(f)
            # taking the average of last 100 episode...
            self.running_reward = np.mean(self.reward_log[-10:])
            f.close()           
            print('Network variables restored!')
            return True
        
        except:
            print('Cannot restore variables')
            return False
        


#%% Agent class  
class Agent():
    def __init__(self):

        self.env = gym.make('Pong-v0')
        
        # setting up parameters
        self.action_space = self.env.action_space.n
        self.batch_size = 32
        self.buffer_size = 5000
        self.random_action_prob = 0.9
        self.frame_skip = 2
        self.update_freq = 4
        self.sync_freq = 2000
        self.rand_prob_step = (0.9 - 0.1)/2000
        self.reward_discount = 0.99
        
        # setting up utilities
        self.memory_buffer = replayMemory(self.buffer_size)
        
        
        # building networks
        with tf.variable_scope('primary') as scope:
            self.primary_nn = DQNet(scope, self.action_space)
            self.primary_nn.build_nn()
            self.__primary_scope = scope
            
        with tf.variable_scope('target') as scope:
            self.target_nn = DQNet(scope, self.action_space)
            self.target_nn.build_nn()
            self.__target_scope = scope 
            
        # initialize variables    
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        
        # restore variables
        self.logger = Logger(self.sess, self.saver)
        self.restored = self.logger.restore()

    def reset_game(self):
        pass
        
    def init_memory(self):
        # reset the game environment, take a initial screen shot
        observation = self.env.reset()
        buffer_counter = 0
        # the state of current game play, 0:2 is 3 previous frame,
        # 3 is the current frame, 4 is the frame after action
        state = np.zeros((1, IMG_X, IMG_Y, 5), dtype = 'float32')
        state[0, :, :, -1] = self.process_frame(observation)
          
        for t in range(self.buffer_size):
            if self.restored:
                if np.random.random_sample() > random_action_prob:
                    # use model to predict action
                    action = self.primary_nn.predict_act(self.sess, state[:, :, :, 1:5])
                else: 
                    # random action
                    action = np.random.randint(self.action_space) # random sample action from 1 to 3
            else:
                action = np.random.randint(self.action_space)
            
            # run the game with same action for a few frames
            for _ in range(frame_skip):
                observation, reward, done, info = self.env.step(action)
                if done:
                    break
            
            state = np.roll(state, -1, axis = 3)
            # effective area [34:194, 0:168] with 2*2 downsampling -> 160/2 * 130/2 matrix
            state[0, :, :, -1] = self.process_frame(observation) 
            self.memory_buffer.add(state, action, reward, done)
            buffer_counter += 1
                    
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        
            if buffer_counter > self.buffer_size:
                break      

    def process_frame(self, frame):
        return np.mean(frame[34: 194 : 2, 0: 160 : 2, :], axis = 2, dtype = 'float32') > 100
    
    def discount_rewards(self):
        pass

    def run(self):
        
        steps = 0
        updates = 0
        while True:
            reward_sum = 0
            observation = self.env.reset()
            
            state = np.zeros((1, IMG_X, IMG_Y, 5), dtype = 'float32')
            state[0, :,:,-1] = self.process_frame(observation)
            
            for t in range(100000):
                # select an action based on the action-value function Q
                if np.random.random_sample() > self.random_action_prob:
                    # use model to predict action
                    action = self.primary_nn.predict_act(self.sess, state[:, :, :, 1:5])
                else: 
                    # random action
                    action = np.random.randint(self.action_space) # random sample action from 1 to 3
                    
                # excute the action for a few steps
                for _ in range(self.frame_skip):
                    observation, reward, done, info = self.env.step(action)
                    reward_sum += reward
                    if done:
                        break
                
                # update the new state and reward and memory buffer
                state = np.roll(state, -1, axis = 3)
                state[0, :, :, -1] = self.process_frame(observation) 
                
                # add current state to the memory buffer
                self.memory_buffer.add(state, action, reward, done)
                
                steps += 1
                # update the network after few steps
                if steps % self.update_freq == 0:
                    
                    if self.random_action_prob > 0.1:    
                        self.random_action_prob -= self.rand_prob_step
                    updates += 1
                    
                    # randomly sample minibatch from memory
                    state_current, state_future, actions, current_rewards, end_game = self.memory_buffer.makeBatch(self.batch_size)
                    
                    next_Q = self.target_nn.estimateQ(self.sess, state_future)
                    
                    targetQ = current_rewards + self.reward_discount * (1 - end_game) * np.amax(next_Q, axis = 1)
                    
                    # update the target-value function Q
                    self.primary_nn.train(self.sess, state_current, actions, targetQ)
                            
                    # # every C step reset Q' = Q
                    if updates % sync_freq == 0:
                        self.target_nn.sync_variables(self.sess, self.__primary_scope)
                # save the model after every 200 updates       
                if done:
                    #save and log
                    self.logger.log(reward_sum)
                    
                    break
    def test():
        pass
#%%


a = Agent()
a.init_memory()
a.run()
