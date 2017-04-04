#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:12:38 2017
@author: shengx
"""

#%%
import tensorflow as tf
import numpy as np
import random
import gym
import pickle
import threading
import time

DEBUG = True

RENDER = False  # if displaying game graphics real time

LEARNING_RATE = 0.00025 

IMG_X, IMG_Y = 80, 80

action_space = 3  # possible action = 1, 2, 3; still, up, down

if DEBUG:
    LEARNING_RATE = 0.001
    max_episode = 21
    max_frame = 3000
    batch_size = 32
    running_reward = None
    max_reward = None
    future_reward_discount = 0.99
    random_action_prob = 0.9
    rand_prob_step = (0.9 - 0.1)/10000
    buffer_size = 10000
    frame_skip = 2
    sync_freq = 200
    update_freq = 4
    save_freq = 100
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
    sync_freq = 2000
    update_freq = 4
    save_freq = 200


save_path = "./"

#%% Deep Q-Network Structure
class DQNet():
    def __init__(self, scope, input_size = (80, 80, 4), action_space = 3):

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
                                                           stddev = 0.01))
        self.conv1_b = tf.Variable(tf.truncated_normal([1, 20, 20, 32], 
                                                           stddev = 0.01))
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
                                                           stddev = 0.01))
        self.conv2_b = tf.Variable(tf.truncated_normal([1, 9, 9, 64], 
                                                           stddev = 0.01))
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
                                                           stddev = 0.01))
        self.conv3_b = tf.Variable(tf.truncated_normal([1, 7, 7, 64], 
                                                           stddev = 0.01))
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
                                                         stddev = 0.01))
        self.ff1_b = tf.Variable(tf.truncated_normal([1, 512],
                                                         stddev = 0.01))
        # output batch_size * 512
        self.ff1_out = tf.matmul(self.ff1_input, self.ff1_W) + self.ff1_b
        self.ff1_out = tf.nn.relu(self.ff1_out)
        
        
        self.policy_W = tf.Variable(tf.truncated_normal([512, self.action_space],
                                                         stddev = 0.01))
        self.policy_b = tf.Variable(tf.truncated_normal([1, self.action_space],
                                                         stddev = 0.01))

        self.policy_out = tf.nn.softmax(tf.matmul(self.ff1_out, self.policy_W) + self.policy_b)
        
        self.value_W = tf.Variable(tf.truncated_normal([512, 1],
                                                         stddev = 0.01))
        self.value_out = tf.matmul(self.ff1_out, self.value_W)        
        
        ###########################################################
        # prediction, loss, and update
        
        self.predict = tf.argmax(self.policy_out, 1)
        
        #self.targetV = tf.placeholder(shape=[None],dtype=tf.float32)
        
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.R = tf.placeholder(shape= [None], dtype=tf.float32)
        
        self.actions_onehot = tf.one_hot(self.actions, self.action_space, dtype=tf.float32)
        
        self.action_policy = tf.reduce_sum(self.policy_out * self.actions_onehot, axis = 1)
        
        self.policy_entropy = tf.reduce_sum(self.policy_out * tf.log(self.policy_out + 1e-10))
        
        self.policy_loss = -tf.log(self.action_policy + 1e-6) * (self.R - tf.stop_gradient(tf.reshape(self.value_out,[-1])))
        
        self.V_loss = 0.5 * tf.square(self.R - tf.reshape(self.value_out,[-1]))
        
        self.entropy = self.policy_out * tf.log(self.policy_out + 1e-6)
        
        self.total_loss = tf.reduce_sum(self.policy_loss) +tf.reduce_sum(self.V_loss) + 0.01 * tf.reduce_sum(self.entropy)
        
        ##########################################################
        # updates
        
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.__scope.name)
        
        self.grad = tf.gradients(self.total_loss, self.variables)
        

    def variable_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.__scope.name)
        
        
    def sync_variables(self, sess, from_scope):
        
        from_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope.name)
        
        for from_var, to_var in zip(from_variables, self.variables):
            op = to_var.assign(from_var.value())
            sess.run(op)
            
    def return_scope(self):
        
        return self.__scope
      
###################################################################
#%%
class agent():
    def __init__(self, id, scope_name, logger, optimizer, global_agent = None):
        
        self.__thread_id = id
        self.__scope_name = scope_name
        self.__logger = logger
        self.exit = False

        self.running_reward = None
        self.episode = 0
        self.update_freq = 5
        
        self.env = gym.make("Pong-v0")
        self.reward_sum = 0
        self.state = np.zeros((IMG_X, IMG_Y, 5), dtype = 'float32')
        
        
        with tf.variable_scope(scope_name) as scope:
            self.local_nn = DQNet(scope)
            self.local_nn.build_nn()
            self.__scope = scope
            
        if global_agent is not None:
            self.global_vars = global_agent.local_nn.variable_list()
            self.apply_gradient = optimizer.apply_gradients(zip(self.local_nn.grad, self.global_vars))
            
            self.sync_op = []
            for from_var, to_var in zip(self.global_vars, self.local_nn.variables):
                self.sync_op.append(to_var.assign(from_var.value()))
                    

    def process_frame(self, frame):
        # input a single frame
        # crop & downsample & average over 3 color channels 
        return np.mean(frame[34: 194 : 2, 0: 160 : 2, :], axis = 2, dtype = 'float32') > 100
    
    def reset_game(self):
        #self.log()
        if self.episode != 0:
            self.__logger.log(self.__thread_id, self.reward_sum)
        
        #self.local_nn.sync_variables(sess, global_agent.local_nn.return_scope())
        sess.run(self.sync_op)
        observation = self.env.reset()
        self.state.fill(0)
        self.state[:,:,-1] = self.process_frame(observation)
        self.reward_sum = 0


    def log(self):        
        if True:
            self.running_reward = self.reward_sum if self.episode == 0 else self.running_reward * 0.99 + self.reward_sum * 0.01
            if self.episode % 10 == 0:
                print('Thread {}: ep {}: reward: {}, mean reward: {:3f}'.format(self.__thread_id, self.episode, self.reward_sum, self.running_reward))
            else:
                print('\tThread {}: ep {}: reward: {}'.format(self.__thread_id, self.episode, self.reward_sum))        
    
    def train(self, sess, graph, coord, reward_decay):

        # initialize environment      
        self.reset_game()


        frame_sequence = np.zeros((self.update_freq, IMG_X, IMG_Y, 4), dtype = 'float32')
        action_sequence = np.zeros((self.update_freq), dtype = 'float32')
        reward_sequence = np.zeros((self.update_freq), dtype = 'float32')
        done_sequence = np.zeros((self.update_freq), dtype = 'float32')
        R_sequence = np.zeros((self.update_freq + 1), dtype = 'float32')
        #while True:
        while not self.exit:

            # running game
            for t in range(self.update_freq):
                policy_prob= sess.run(self.local_nn.policy_out, 
                                       feed_dict = {
                                       self.local_nn.input: np.expand_dims(self.state[:,:,1:5], axis = 0)})
                # choose an action according to policy
                action = np.random.choice(3, p=np.squeeze(policy_prob))
                
                # take this action for certain steps and record the reward
                reward = 0
                for _ in range(frame_skip):
                    observation, reward_temp, done, info = self.env.step(action + 1)
                    reward += reward_temp
                    if done:
                        break
                self.reward_sum += reward
                
                # record game progress
                frame_sequence[t, :, :, :] = self.state[:,:,1:5]
                action_sequence[t] = action
                reward_sequence[t] = reward
                done_sequence[t] = done

                # update next game state
                self.state = np.roll(self.state, -1, axis = 2)
                self.state[:,:,-1] = self.process_frame(observation)
                if done:
                    break
                
            R_sequence[t+1] = 0 if done else sess.run(self.local_nn.value_out,
                                      feed_dict = {
                                      self.local_nn.input: np.expand_dims(self.state[:,:,1:5], axis = 0)})
            R_sequence[t+1] = np.clip(R_sequence[t+1], -1, 1)
    
            for idx in range(t, -1, -1):
                if reward_sequence[idx] != 0:
                    R_sequence[idx] = reward_sequence[idx]
                else:
                    R_sequence[idx] = reward_decay * R_sequence[idx+1]            

            sess.run(self.apply_gradient,
                            feed_dict = {
                            self.local_nn.input: frame_sequence[0:t+1, :, :, :],
                            self.local_nn.actions: action_sequence[0:t+1],
                            self.local_nn.R: R_sequence[0:t+1]})
            if done:
                self.reset_game()
                self.episode += 1
                    
#%%
class Logger(object):
    def __init__(self):
        # RawValue because we don't need it to create a Lock:
        self.global_episode = 0
        self.running_reward = 0
        self.lock = threading.Lock()
        
        self.save_freq = 100
        self.save_path = './'
        self.__saver = None
        self.__sess = None   
        self.reward_log = []

    def log(self, thread_id, reward_sum):
        with self.lock:
            self.running_reward = reward_sum if self.global_episode == 0 else self.running_reward * 0.99 + reward_sum * 0.01
            self.global_episode += 1
            if self.global_episode % 10 == 0:
                self.reward_log.append(self.running_reward)
                print('Ep {}: thread {}: reward: {}, mean reward: {:3f}'.format(self.global_episode, thread_id, reward_sum, self.running_reward))
            else:
                print('\tEp {}: thread {}: reward: {}'.format(self.global_episode, thread_id, reward_sum)) 
                
            if self.global_episode % 2000 == 0 and self.__saver is not None:
                self.save()
                
    def save(self):
        
        self.__saver.save(self.__sess, self.save_path+'model-'+str(self.global_episode)+'.cptk')
        f = open(self.save_path + 'reward_log.cptk','wb')
        pickle.dump(self.reward_log, f)
        f.close()     
        
    def add_saver(self, sess, saver):
        self.__saver = saver
        self.__sess = sess 


#%%
# initialize tensorflow
tf.reset_default_graph()
graph = tf.Graph()
sess = tf.Session(graph=graph)
coord = tf.train.Coordinator()
logger = Logger()
optimizer = tf.train.RMSPropOptimizer(
                learning_rate=0.00025,
                decay=0.99,
                momentum=0,
                epsilon=0.1,
                use_locking = True)


with graph.as_default():
    # saver
    
    
    # initialize global network
    global_agent = agent(-1, 'global', logger, optimizer)
    # initialize local networks
    THREAD_NUM = 4
    local_agent = []
    for thread_id in range(THREAD_NUM):
        local_scope = 'local'+str(thread_id)
        local_agent.append(agent(thread_id, local_scope, logger, optimizer, global_agent))

    # initialize tensorflow 
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    
sess.run(init_op)
logger.add_saver(sess, saver)


try:
    ckpt = tf.train.get_checkpoint_state(save_path)
    load_path = ckpt.model_checkpoint_path
    saver.restore(sess, load_path)
    f = open(save_path + 'reward_log.cptk','rb')
    reward_log = pickle.load(f)
    f.close()
    print("Session restored...")
except:
    # sync variables for all local network

    print("Nothing to restore...")


training_thread = []
for id, agent in enumerate(local_agent):
    t = threading.Thread(target = agent.train, args = (sess, graph, coord, future_reward_discount))
    t.start()
    time.sleep(1)
    training_thread.append(t)
#%%

for (agent, agent_thread) in zip(local_agent, training_thread):
    agent.exit = True
    time.sleep(0.5)
    agent_thread.join(1)






