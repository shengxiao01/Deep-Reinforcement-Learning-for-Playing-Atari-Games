#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:21:30 2017

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
    LEARNING_RATE = 0.001
    max_episode = 21
    max_frame = 5000
    batch_size = 32
    temporal_length = 8
    running_reward = None
    max_reward = None
    future_reward_discount = 0.99
    random_action_prob = 0.9
    rand_prob_step = (0.9 - 0.1)/10000
    buffer_size = 80
    frame_skip = 4
    sync_freq = 500
    update_freq = 8
    save_freq = 100
else:
    max_episode = 21
    max_frame = 10000
    batch_size = 32
    temporal_length = 8
    running_reward = None
    max_reward = None
    future_reward_discount = 0.99
    random_action_prob = 0.9
    rand_prob_step = (0.9 - 0.1)/1000000
    buffer_size = 5000
    frame_skip = 4
    sync_freq = 1000
    update_freq = 24
    save_freq = 200


save_path = "./"

#%% Deep Q-Network Structure
class DQNet():
    def __init__(self, scope, input_size = (80, 80), action_space = 3):

        self.input_x, self.input_y= input_size
        self.action_space = action_space
        self.__scope = scope

    def build_nn(self):
        

        # [batch, in_height, in_width, in_channels]
        # assuming input to be (batch_size * frames)*84*84*4
        self.input = tf.placeholder(tf.float32, shape=[None, self.input_x, self.input_y, 1])

        ##########################################################
        #[filter_height, filter_width, in_channels, out_channels]
        # conv layer 1, 8*8*32 filters, 4 stride
        self.conv1_W = tf.Variable(tf.truncated_normal([8, 8, 1, 32], 
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
        
        ###########################################################
        # recurrent layer
        self.batch_size = tf.placeholder(tf.int32)
        self.rnn_in = tf.reshape(self.ff1_out,[self.batch_size, -1, 512])
        self.rnn_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(num_units = 512)
        self.rnn_state_in = self.rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state_out = tf.nn.dynamic_rnn(inputs = self.rnn_in,
                                                         cell = self.rnn_cell,
                                                         dtype = tf.float32,
                                                         initial_state = self.rnn_state_in)
        
        self.rnn_out = tf.reshape(self.rnn,shape=[-1,512])
        #self.rnn_out = self.ff1_out
        
        ############################################################
        # split the network into two paths
        self.advantage_input,self.value_input = tf.split(self.rnn_out, 2, axis = 1)
        
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
        
        self.loss_mask = tf.placeholder(shape=[None], dtype=tf.float32)
        
        self.loss = tf.square(self.targetQ - self.Q) * self.loss_mask

        self.loss_clipped = tf.clip_by_value(self.loss, -1, 1)
        
        self.loss_sum = tf.reduce_mean(self.loss_clipped)
        
        self.update = tf.train.RMSPropOptimizer(learning_rate = LEARNING_RATE, momentum = 0.95).minimize(self.loss_sum)

    def variable_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.__scope.name)

#%% utility functions

class replayMemory():
    
    def __init__(self, size):
        # [i, :, :, 0:4] is the current state
        # [i, :, :, 1:5] is the next state
        self.frames = [None] * size
        self.actions = [None] * size
        self.rewards = [None] * size
        self.done = [None] * size
        self.__counter = 0
        self.__size = size
        
    def add(self, state, action, reward, done):
        
        self.frames[self.__counter] = state
        self.actions[self.__counter] = action
        self.rewards[self.__counter] = reward
        self.done[self.__counter] = done
        self.__counter += 1
        self.__counter = self.__counter % self.__size
     
    def makeBatch(self, batch_size, temporal_length):  
        
        idx = np.random.randint(0, self.__size, (batch_size))
        
        current_frame_sample = np.zeros((batch_size, temporal_length, IMG_X, IMG_Y)).astype('float32')
        next_frame_sample = np.zeros((batch_size, temporal_length, IMG_X, IMG_Y)).astype('float32')
        action_sample = np.zeros((batch_size, temporal_length)).astype('int32')
        reward_sample = np.zeros((batch_size, temporal_length)).astype('float32')
        done_sample = np.zeros((batch_size, temporal_length)).astype('int32')
        
        for i, c_idx in enumerate(idx):
            
            sequence_start_idx = np.random.randint(0, len(self.actions[c_idx]) - temporal_length - 1)
            sequence_idx = np.array(range(sequence_start_idx, sequence_start_idx + temporal_length))
            
            current_frame_sample[i, :, :, :] = self.frames[c_idx][sequence_idx, :, :]
            next_frame_sample[i, :, :, :] = self.frames[c_idx][sequence_idx + 1, :, :]
            action_sample[i, :] = self.actions[c_idx][sequence_idx]
            reward_sample[i, :] = self.rewards[c_idx][sequence_idx]
            done_sample[i, :] = self.done[c_idx][sequence_idx]
        
        return (current_frame_sample, next_frame_sample, action_sample, reward_sample, done_sample)


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

env = gym.make("Pong-v0")
for i in range(buffer_size):
    
    # reset the game environment, take a initial screen shot
    observation = env.reset()
    
    frame_sequence = np.zeros((max_frame, IMG_X, IMG_Y)).astype('float32')
    action_sequence = np.zeros((max_frame)).astype('int32')
    reward_sequence = np.zeros((max_frame)).astype('float32')
    done_sequence = np.zeros((max_frame)).astype('int32')
    
    current_frame = process_frame(observation)
      
    for t in range(max_frame):

        action = random.randint(0,2)
        
        # run the game with same action for a few frames
        reward = 0
        for _ in range(frame_skip):
            observation, reward_temp, done, info = env.step(action+1)
            reward += reward_temp
            if done:
                break
            
        frame_sequence[t, :, :] = current_frame
        action_sequence[t] = action
        reward_sequence[t] = reward 
        done_sequence[t] = done
        # effective area [34:194, 0:168] with 2*2 downsampling -> 160/2 * 130/2 matrix
        current_frame = process_frame(observation) 

        if done:
            memory_buffer.add(frame_sequence[0:t+1, :, :], action_sequence[0:t+1], reward_sequence[0:t+1], done_sequence[0:t+1])
            
            print("Episode {} finished after {} timesteps".format(i, t+1))
            break

env.close()
  
#%%
###################################################################
# Initialize environment

env = gym.make("Pong-v0")


tf.reset_default_graph()
with tf.variable_scope("Primary", initializer=tf.truncated_normal_initializer(stddev = 0.1)) as scope:
    Atari_AI_primary = DQNet(scope)
    Atari_AI_primary.build_nn()
with tf.variable_scope("Target", initializer=tf.truncated_normal_initializer(stddev = 0.1)) as scope:
    Atari_AI_target = DQNet(scope)
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
#%%
while True:
    
    i_episode += 1
    # reset the game environment
    observation = env.reset()
    
    # initialize parameters
    reward_sum = 0
    rnn_state = (np.zeros([1,512]),np.zeros([1,512]))
    
    frame_sequence = np.zeros((max_frame, IMG_X, IMG_Y)).astype('float32')
    action_sequence = np.zeros((max_frame)).astype('int32')
    reward_sequence = np.zeros((max_frame)).astype('float32')
    done_sequence = np.zeros((max_frame)).astype('int32')
    
    #state = np.zeros((IMG_X, IMG_Y, 5), dtype = 'float32')
    current_frame = process_frame(observation)
    
    
    for t in range(max_frame):
        if RENDER:
            env.render()
        # select an action based on the action-value function Q
        if np.random.random_sample() > random_action_prob:
            # use model to predict action
            action, rnn_state = sess.run([Atari_AI_primary.predict, Atari_AI_primary.rnn_state_out],
                              feed_dict = {Atari_AI_primary.input: np.reshape(current_frame, (1, IMG_X, IMG_Y, 1)),
                                           Atari_AI_primary.rnn_state_in: rnn_state,
                                           Atari_AI_primary.batch_size: 1})
            action = action[0]
            
        else: 
            # random action
            action = random.randint(0, 2) # random sample action from 1 to 3
            # get rnn cell state
            rnn_state = sess.run(Atari_AI_primary.rnn_state_out,
                              feed_dict = {Atari_AI_primary.input: np.reshape(current_frame, (1, IMG_X, IMG_Y, 1)),
                                           Atari_AI_primary.rnn_state_in: rnn_state,
                                           Atari_AI_primary.batch_size: 1})
            

        # excute the action for a few steps
        reward = 0
        for _ in range(frame_skip):
            observation, reward_temp, done, info = env.step(action+1)
            reward += reward_temp
            if done:
                break
        
        # append the new state and reward
        reward_sum += reward
        frame_sequence[t, :, :] = current_frame
        action_sequence[t] = action
        reward_sequence[t] = reward
        done_sequence[t] = done

        current_frame = process_frame(observation)
        
        
        updates += 1
        if updates % update_freq == 0:
            
            if random_action_prob > 0.1:    
                random_action_prob -= rand_prob_step
            steps += 1
            # randomly sample minibatch from memory
            current_frames, next_frames, actions, current_rewards, end_game = memory_buffer.makeBatch(batch_size, temporal_length)
            current_frames = np.reshape(current_frames, (-1, IMG_X, IMG_Y, 1))
            next_frames = np.reshape(next_frames, (-1, IMG_X, IMG_Y, 1))
            actions = actions.flatten()
            current_rewards = current_rewards.flatten()
            end_game = end_game.flatten()
            
            
            # get target Q from target network
            state_init = (np.zeros([batch_size,512]),np.zeros([batch_size,512])) 
            future_rewards = sess.run(Atari_AI_target.output,
                                      feed_dict = {Atari_AI_target.input: next_frames,
                                                   Atari_AI_target.rnn_state_in: state_init,
                                                   Atari_AI_target.batch_size: batch_size})
            targetQ = current_rewards + future_reward_discount * (1 - end_game) * np.amax(future_rewards, axis = 1)
            
            # update the target-value function Q
            
            loss_mask = np.array((0, 1))
            loss_mask = np.kron(loss_mask, np.ones(int(temporal_length/2)))  # temporal_length must be even
            loss_mask = np.tile(loss_mask, batch_size)
            
            sess.run(Atari_AI_primary.update, feed_dict = {
                    Atari_AI_primary.input: current_frames,
                    Atari_AI_primary.actions: actions,
                    Atari_AI_primary.targetQ: targetQ,
                    Atari_AI_primary.rnn_state_in: state_init,
                    Atari_AI_primary.batch_size: batch_size,
                    Atari_AI_primary.loss_mask: loss_mask})
                    # every C step reset Q' = Q
            if steps % sync_freq == 0:
                primary_variables = Atari_AI_primary.variable_list()
                target_variables = Atari_AI_target.variable_list()
                copy_variables(primary_variables, target_variables, sess)
        
        
        # save the model after every 200 updates       
        if done:
            # an episode is done, update the memory buffer
            memory_buffer.add(frame_sequence[0:t+1,:,:], action_sequence[0:t+1], reward_sequence[0:t+1], done_sequence[0:t+1])
            
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
                max_reward = running_reward if max_reward is None else max(max_reward, running_reward)
                if max_reward == running_reward:
                    saver.save(sess, save_path+'model-'+str(i_episode)+'.cptk')
                    f = open(save_path + 'reward_log.cptk','wb')
                    pickle.dump(reward_log, f)
                    f.close()
                
            break