#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 18:59:12 2017

@author: shengx
"""

import numpy as np
from multiprocessing import Process, Queue, Value, RawValue, Lock
from threading import Thread
import pickle
import tensorflow as tf
import gym
import time
from matplotlib import pyplot as plt
SAVE = False
RESTORE = False
#########################################################
#%%
# A class for A3C network
class A3CNet():
    def __init__(self, scope, input_size = (80, 80, 4), action_space = 3):

        self.input_x, self.input_y, self.input_frame= input_size
        self.action_space = action_space
        self.__scope = scope
        
        self.build_nn()

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
        '''
        ##########################################################
        #[filter_height, filter_width, in_channels, out_channels]
        # conv layer 1, 8*8*16 filters, 4 stride
        self.conv1_W = tf.Variable(tf.truncated_normal([8, 8, self.input_frame, 16], 
                                                           stddev = 0.01))
        self.conv1_b = tf.Variable(tf.truncated_normal([1, 20, 20, 16], 
                                                           stddev = 0.01))
        self.conv1_strides = [1, 4, 4, 1]
        #output 20*20*32 
        self.conv1_out = tf.nn.conv2d(self.input, 
                                          self.conv1_W, 
                                          self.conv1_strides, 
                                          padding = 'SAME') + self.conv1_b
        self.conv1_out = tf.nn.relu(self.conv1_out)
        
        
        ###########################################################
        # conv layer 2, 4*4*32 filters, 2 stride
        self.conv2_W = tf.Variable(tf.truncated_normal([4, 4, 16, 32],
                                                           stddev = 0.01))
        self.conv2_b = tf.Variable(tf.truncated_normal([1, 9, 9, 32], 
                                                           stddev = 0.01))
        self.conv2_strides = [1, 2, 2, 1]
        # output 9*9*32
        self.conv2_out = tf.nn.conv2d(self.conv1_out, 
                                          self.conv2_W, 
                                          self.conv2_strides, 
                                          padding = 'VALID') + self.conv2_b
        self.conv2_out = tf.nn.relu(self.conv2_out) 
        
        ###########################################################
        # fully connected layer 1, (9*9*32 = 2596) * 512
        self.ff1_input = tf.reshape(self.conv2_out, [-1, 2592])
        self.ff1_W = tf.Variable(tf.truncated_normal([2592, 256],
                                                         stddev = 0.01))
        self.ff1_b = tf.Variable(tf.truncated_normal([1, 256],
                                                         stddev = 0.01))
        # output batch_size * 512
        self.ff1_out = tf.matmul(self.ff1_input, self.ff1_W) + self.ff1_b
        self.ff1_out = tf.nn.relu(self.ff1_out)
        
        '''
        self.policy_W = tf.Variable(tf.truncated_normal([512, self.action_space],
                                                         stddev = 0.01))
        self.policy_b = tf.Variable(tf.truncated_normal([1, self.action_space],
                                                         stddev = 0.01))        
        
        
        ###########################################################
        self.policy_out = tf.nn.softmax(tf.matmul(self.ff1_out, self.policy_W) + self.policy_b)
        #self.policy_out = tf.clip_by_value(self.policy_out, 0.05, 1)
        #self.policy_out = tf.div(self.policy_out, tf.reduce_sum(self.policy_out, axis = 1, keep_dims = True))
        
        self.value_W = tf.Variable(tf.truncated_normal([512, 1],
                                                         stddev = 0.01))
        self.value_b = tf.Variable(tf.truncated_normal([1, 1],
                                                         stddev = 0.01))
        self.value_out = tf.matmul(self.ff1_out, self.value_W) + self.value_b        
               
        #Then combine them together to get our final Q-values.
        #self.output = self.value_out + self.advantage_out - tf.reduce_mean(self.advantage_out,reduction_indices=1,keep_dims=True)
        ###########################################################
        # prediction, loss, and update
        
        self.predict = tf.argmax(self.policy_out, 1)
        
        #self.targetV = tf.placeholder(shape=[None],dtype=tf.float32)
        
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.R = tf.placeholder(shape= [None], dtype=tf.float32)
        
        self.actions_onehot = tf.one_hot(self.actions, self.action_space, dtype=tf.float32)
        
        self.action_policy = tf.reduce_sum(self.policy_out * self.actions_onehot, axis = 1)
        
        self.policy_loss = -tf.log(self.action_policy + 1e-6) * (self.R - tf.stop_gradient(tf.reshape(self.value_out,[-1])))
        
        self.V_loss = tf.square(self.R - tf.reshape(self.value_out,[-1]))
        
        self.entropy_loss = self.policy_out * tf.log(self.policy_out + 1e-6)
        
        self.total_loss = tf.reduce_sum(self.policy_loss) + 0.5 * tf.reduce_sum(self.V_loss) + 0.01 * tf.reduce_sum(self.entropy_loss)

        ##########################################################
        # updates
                       
        self.update = tf.train.RMSPropOptimizer(
                learning_rate=0.00025,
                decay=0.99,
                momentum=0,
                epsilon=0.1).minimize(self.total_loss)
        
    def train(self, sess, states, actions, rewards):
        sess.run(self.update, feed_dict = {
                              self.input: states,
                              self.actions: actions,
                              self.R: rewards})

    def predict_pv(self, sess, states):
        policy, value = sess.run([self.policy_out, self.value_out], feed_dict = {
                                                                    self.input: states})
        return (policy, value)

    def predict_action(self, sess, states):
        actions = sess.run(self.predict, feed_dict = {
                                         self.input: states})
        return actions

    def variable_list(self):
        
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.__scope.name)
    
    def apply_gradients(self, sess, gradients):
        
        self.gradients = [tf.Variable(tf.zeros_like(var), trainable=False) for var in self.variables]
        
        grad_var_pair = [(self.gradients[i].assign(grad), var) for i, (grad,var) in enumerate(zip(gradients, self.variables))]                
        
        sess.run(self.optimizer.apply_gradients(grad_var_pair))
        
        
    def sync_variables(self, sess, from_scope):
        
        from_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope.name)
        
        for from_var, to_var in zip(from_variables, self.variables):
            op = to_var.assign(from_var.value())
            sess.run(op)
            
    def return_scope(self):
        
        return self.__scope

#%%
#########################################################
# A class for game agent
class Agent():
    def __init__(self, id, prediction_q, training_q, log_q):
        self.__id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.log_q = log_q
        self.exit = False
        self.episode = 0

        self.action_q = Queue(maxsize = 1)
        
        self.action_num = 3
        self.update_freq = 5
        self.frame_skip = 1 
        self.IMG_X = 80
        self.IMG_Y = 80 
        
        self.env = gym.make("Pong-v0")
        self.state_current = np.zeros((1, self.IMG_X, self.IMG_Y, 4), dtype = 'float32')
        self.reward_sum = 0

    def reset_game(self):
        observation = self.env.reset()
        self.state_current.fill(0)
        self.state_current[0, :, :, -1] = self.process_frame(observation)
        self.reward_sum = 0
        
    def process_frame(self, frame):
        return np.mean(frame[34: 194 : 2, 0: 160 : 2, :], axis = 2, dtype = 'float32') > 100
        #frame_gray = np.dot(frame[34:194:2, 0:160:2, :], [0.299, 0.587, 0.114]).astype(np.float32)
        #return frame_gray/128 - 1

    def run(self):

        state_sequence = np.zeros((self.update_freq, self.IMG_X, self.IMG_Y, 4), dtype = 'float32')
        action_sequence = np.zeros((self.update_freq), dtype = 'int32')
        reward_sequence = np.zeros((self.update_freq), dtype = 'int32')
        done_sequence = np.zeros((self.update_freq), dtype = 'int32')
        R_sequence = np.zeros((self.update_freq + 1), dtype = 'float32')

        self.reset_game()

        while not self.exit:

            for t in range(self.update_freq):
                # put current state into a queue and wait for prediction
                self.prediction_q.put((self.__id, self.state_current))
                #print(np.max(self.state_current))
                current_policy, current_value = self.action_q.get()
                # randomly sample action according to the policy
                action = np.random.choice(self.action_num, p=current_policy)
                # take the action for a few steps
                reward = 0
                for _ in range(self.frame_skip):
                    observation, reward_temp, done, info = self.env.step(action + 1)
                    reward += reward_temp
                    if done:
                        break

                self.reward_sum += reward
                # log current states
                state_sequence[t, :, :, :] = self.state_current
                action_sequence[t] = action
                reward_sequence[t] = reward
                done_sequence[t] = done
                # ready for next states
                self.state_current = np.roll(self.state_current, -1, axis = 3)
                self.state_current[0, :, :, -1] = self.process_frame(observation)
                if done:
                    break

            # calculate discounted reward_sequence
            if done: 
                R_sequence[t+1] = 0
            else:
                self.prediction_q.put((self.__id, self.state_current))
                current_policy, current_value = self.action_q.get()
                R_sequence[t+1] = np.clip(current_value, -1, 1)
            for idx in range(t, -1, -1):
                if reward_sequence[idx] != 0:
                    R_sequence[idx] = reward_sequence[idx]
                else:
                    R_sequence[idx] = 0.99 * R_sequence[idx+1]
            
            # batch samples and send to training Queue
            training_q.put((state_sequence[0:t+1, :, :, :], action_sequence[0:t+1], R_sequence[0:t+1]))

            # reset and save experience if done
            if done:
                
                self.episode += 1
                log_q.put((self.__id, self.episode, self.reward_sum))
                self.reset_game()
#%%
#########################################################
# A class for fetching game data and return policy/value
class Predictor():
    def __init__(self, id):
        self.__id = id
        self.exit = False
        self.prediction_batch_size = 12

    def start(self, sess, global_nn, agents, prediction_q):
        ids = np.zeros(self.prediction_batch_size, dtype='int32')
        frames = np.zeros(
            (self.prediction_batch_size, 80, 80, 4),
            dtype='float32')
        
        while not self.exit:
            ids[0], frames[0, :, :, :] = prediction_q.get() 
            t = 1
            # deque frames from prediction_q, log thread id number
            while t < self.prediction_batch_size and not prediction_q.empty(): 
                ids[t], frames[t, :, :, :] = prediction_q.get() 
                t += 1
            # send batch into global_nn and get policy/value back
            policy, value = global_nn.predict_pv(sess, frames[0:t])
           
            # queue policy/value into each agent's action queue
            for idx, agent_id in enumerate(ids[0:t]):
                agents[agent_id].action_q.put((policy[idx], value[idx])) 

#%%
#########################################################
# A class for batching game data and training network
class Trainer():
    def __init__(self, id):
        self.__id = id
        self.batch_size = 35
        self.exit = False

    def start(self, sess, global_nn, training_q, prediction_q):

        while not self.exit:
            size = 0
            while size < self.batch_size:
                # deque batches from training_q until batch_size
                states, actions, rewards = training_q.get()
                if size == 0:
                    states_train = states
                    actions_train = actions
                    rewards_train = rewards
                else:
                    states_train = np.concatenate((states_train, states), axis = 0)
                    actions_train = np.concatenate((actions_train, actions), axis = 0)
                    rewards_train = np.concatenate((rewards_train, rewards), axis= 0)
                size += actions.shape[0]
            # update network
            #print('Training queue ', str(training_q.qsize()), ' Prediction Queue ', str(prediction_q.qsize()))
            global_nn.train(sess, states_train, actions_train, rewards_train)
#%%
class Logger():
    def __init__(self, id, sess, saver, log_q):
        self.exit = False
        self.global_episode = 0
        self.save_freq = 100
        self.save_path = './'
        self.reward_log = []
        
        self.__id = id
        self.__sess = sess
        self.__saver = saver
        self.__log_q = log_q
        
    def restore(self):
        
        try:
            ckpt = tf.train.get_checkpoint_state(self.save_path)
            load_path = ckpt.model_checkpoint_path
            saver.restore(sess, load_path)
            
            f = open(self.save_path + 'reward_log.cptk','rb')
            self.reward_log = pickle.load(f)
            f.close()           
            print('Network variables restored!')
        except:
            print('Cannot restore variables')
            
    def save(self):
        
        self.__saver.save(self.__sess, self.save_path+'model-'+str(self.global_episode)+'.cptk')
        f = open(self.save_path + 'reward_log.cptk','wb')
        pickle.dump(self.reward_log, f)
        f.close()
    
    def start(self):
        if RESTORE:
            self.restore()
        
        agent_id, agent_ep, agent_reward = self.__log_q.get()
        self.running_reward = agent_reward
        self.global_episode += 1
        print('ep ', str(self.global_episode), ' reward ', str(agent_reward))

        while not self.exit:
            agent_id, agent_ep, agent_reward = self.__log_q.get()
            self.global_episode += 1
            self.running_reward = agent_reward * 0.01 + self.running_reward * 0.99
            if self.global_episode % 10 != 0:
                print('ep ', str(self.global_episode), ' reward ', str(agent_reward))
            else: 
                print('ep ', str(self.global_episode), ' reward ', str(agent_reward), ' running_reward ', str(self.running_reward))
                self.reward_log.append(self.running_reward)
                
            if SAVE and self.global_episode % self.save_freq == 0:
                self.save()
            
                
#%%
#########################################################


tf.reset_default_graph()
# initialize global network
global_nn = A3CNet('global')
# initialize tensorflow 
init_op = tf.global_variables_initializer()
sess = tf.Session()
coord = tf.train.Coordinator() 
sess.run(init_op)
saver = tf.train.Saver()


# setup global queues
prediction_q = Queue(maxsize = 128)
training_q = Queue(maxsize = 100)
log_q = Queue(maxsize = 128)


#%%
# setup Agent
AGENT_NUM = 16
agents = []
agents_process = []
for agent_id in range(AGENT_NUM):
    agents.append(Agent(agent_id, prediction_q, training_q, log_q))
    p = Process(target = agents[-1].run, args = ())
    agents_process.append(p)
    p.start()  
#%%
# threads
train_thread = Trainer(0)
predict_thread = Predictor(0)
log_thread = Logger(0, sess, saver, log_q)
train_thread2 = Trainer(0)
predict_thread2 = Predictor(0)

t1 = Thread(target = train_thread.start, args = (sess, global_nn, training_q, prediction_q))
t1.Daemon = True
t1.start()

t4 = Thread(target = train_thread2.start, args = (sess, global_nn, training_q, prediction_q))
t4.Daemon = True
t4.start()

t2 = Thread(target = predict_thread.start, args = (sess, global_nn, agents, prediction_q))
t2.Daemon = True
t2.start()

t5 = Thread(target = predict_thread2.start, args = (sess, global_nn, agents, prediction_q))
t5.Daemon = True
t5.start()


t3 = Thread(target = log_thread.start, args = ( ))
t3.Daemon = True
t3.start()



#%%

for (agent, p) in zip(agents, agents_process):
    agent.exit = True
    time.sleep(0.5)
    p.terminate()
train_thread.exit = True
predict_thread.exit = True
log_thread.exit = True
train_thread2.exit = True
predict_thread2.exit = True
t1.join(1)
t2.join(1)
t3.join(1)
t4.join(1)
t5.join(1)

#sess.close()

