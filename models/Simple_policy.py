#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 18:57:02 2017

@author: shengx
"""


#%%
import tensorflow as tf
import numpy as np
import gym
import pickle

DEBUG = True

RENDER = False  # if displaying game graphics real time
LEARNING_RATE = 0.0001 

IMG_X, IMG_Y = 80, 80

# saver
save_frequency = 100
save_path = "./"

# initializing parameters
action_space = 3  # possible action = 1, 2, 3; still, up, down

max_episode = 21
max_frame = 10000
frame_skip = 2
running_reward = None
future_reward_discount = 0.99
#%%################################################################
# Simple Policy Network Class
class PNet():
    def __init__(self,input_size = (IMG_X * IMG_Y, 1), action_space = 3):

        self.input_size, self.input_frame= input_size
        self.action_space = action_space

    def build_nn(self):
        

        # [batch, in_height, in_width, in_channels]
        # assuming input to be batch_size*84*84*4
        
        self.input = tf.placeholder(tf.float32, shape = [None, self.input_size])
        self.W1 = tf.Variable(tf.truncated_normal([self.input_size, 512], stddev = 0.1))
        self.b1 = tf.Variable(tf.truncated_normal([1, 512], stddev = 0.1))
        self.hidden = tf.nn.relu(tf.matmul(self.input, self.W1) + self.b1)
        self.W2 = tf.Variable(tf.truncated_normal([512, self.action_space], stddev = 0.1))
        self.b2 = tf.Variable(tf.truncated_normal([1, self.action_space], stddev = 0.1))
        
        self.output = tf.nn.softmax(tf.matmul(self.hidden, self.W2) + self.b2)
        
        self.predict_action = tf.argmax(self.output, axis = 1)
        
        self.advantage = tf.placeholder(tf.float32, shape=[None])
        self.actions = tf.placeholder(tf.int32, shape=[None])
        
        self.actions_onehot = tf.one_hot(self.actions, self.action_space, dtype=tf.float32)
        self.neg_log_prob =  tf.multiply(-self.actions_onehot * tf.log(self.output + 1e-10), tf.reshape(self.advantage, [-1, 1]))
        self.loss = tf.reduce_mean(self.neg_log_prob)
        
        self.update = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(self.loss)
        
#%%################################################################
# utility functions

def process_frame(frame):
    # input a single frame
    # crop & downsample & average over 3 color channels 
    # Convert the image to binary. Foreground = 1, baackground = 0
    return np.mean(frame[34: 194 : 2, 0: 160 : 2, :], axis = 2, dtype = 'float32') > 100


def discount_rewards(r):
  
  discounted_r = np.zeros((len(r)))
  running_add = 0
  for t in reversed(range(0, len(r))):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * future_reward_discount + r[t]
    discounted_r[t] = running_add
  return discounted_r


#%%################################################################
# starting tensorflow and game environment
tf.reset_default_graph()
env = gym.make("Pong-v0")

Atari_AI = PNet()
Atari_AI.build_nn()

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
    
    print("Session restored...")
except:
    print("Nothing to restore...")


# Training

i_episode = 0
state = np.zeros((80, 80, 2),dtype = 'float32')
state_sequence = np.zeros((max_frame, 6400)).astype('float32')
action_sequence = np.zeros((max_frame)).astype('int32')
reward_sequence = np.zeros((max_frame)).astype('float32')


while True:
    
    i_episode += 1
    
    state[:,:,0] = process_frame(env.reset())
    
    reward_sum = 0
    
    for t in range(max_frame):
        if RENDER:
            env.render()
        # select an action based on policy network output probability
        diff_frame = np.reshape(state[:,:,1] - state[:,:,0], (1, 6400))
        nn_prob = sess.run(Atari_AI.output, feed_dict={
                Atari_AI.input: diff_frame})
    
        action = np.random.choice(3, p=np.squeeze(nn_prob))
        # excute the action for a few steps
        for _ in range(frame_skip):
            observation, reward, done, info = env.step(action + 1)
            reward_sum += reward
            if done:
                break
        
        # update the new state and reward and memory buffer
        state[:,:,0] = state[:,:,1]
        state[:,:,1] = process_frame(observation) 
        
        state_sequence[t, :] = diff_frame
        action_sequence[t] = action
        reward_sequence[t] = reward
        
        
        # save the model after every 200 updates       
        if done:
            
            decay_reward = discount_rewards(reward_sequence[0:t+1])
            
            # normalize the reward
            decay_reward -= np.mean(decay_reward)
            decay_reward /= np.std(decay_reward)

            
            sess.run(Atari_AI.update, feed_dict={
                                                 Atari_AI.input: state_sequence[0:t+1, :] ,
                                                 Atari_AI.actions: action_sequence[0:t+1],
                                                 Atari_AI.advantage:  decay_reward})

            # reset sequence memory
            state_sequence = np.zeros((max_frame, 6400))
            action_sequence = np.zeros((max_frame))
            reward_sequence = np.zeros((max_frame))
            
            # claculate and display the moving average of rewards
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01 
            
            if DEBUG:
                if i_episode % 10 == 0:
                    print('ep {}: reward: {}, mean reward: {:3f}'.format(i_episode, reward_sum, running_reward))
                else:
                    print('\tep {}: reward: {}'.format(i_episode, reward_sum))
             
            # saving results    
            if i_episode % 10 == 0:
                reward_log.append(running_reward)
                
            if i_episode % save_frequency == 0:
                saver.save(sess, save_path+'model-'+str(i_episode)+'.cptk')
                f = open(save_path + 'reward_log.cptk','wb')
                pickle.dump(reward_log, f)
                f.close()
            break
