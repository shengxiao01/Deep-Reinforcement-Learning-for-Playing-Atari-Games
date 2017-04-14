import tensorflow as tf
import numpy as np
import gym

from ddqn_lstm import DDQNet
from logger import Logger
from memory import replayMemory
from params import Params

  
class Agent():
    
    def __init__(self):

        self.env = gym.make('Pong-v0')
        
        # setting up parameters
        self.batch_size = Params['BATCH_SIZE']
        self.buffer_size = Params['MEMORY_BUFFER_SIZE']
        self.random_action_prob = Params['RANDOM_ACTION_PROB_START']
        self.random_action_prob_end = Params['RANDOM_ACTION_PROB_END']
        self.frame_skip = Params['FRAME_SKIP']
        self.update_freq = Params['UPDATE_FREQ']
        self.sync_freq = Params['SYNC_FREQ']
        self.rand_prob_step = (self.random_action_prob - self.random_action_prob_end)/Params['ANNEALING_STEP']
        self.reward_discount = Params['REWARD_DISCOUNT']
        self.IMG_X = Params['IMG_X']
        self.IMG_Y = Params['IMG_Y']
        self.rnn_h_units = Params['RNN_H_UNIT']
        self.rnn_seq_len = Params['RNN_SEQUENCE_LENGTH']
        
        self.action_space = self.env.action_space.n
        self.updates = 0
        
        # setting up utilities
        self.memory_buffer = replayMemory(self.IMG_X, self.IMG_Y, self.buffer_size)
        
        self.nn = DDQNet(self.action_space)

        # initialize variables    
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        
        # restore variables
        self.logger = Logger(self.sess, self.saver)
        self.random_action_prob = self.random_action_prob_end if self.logger.restore() else self.random_action_prob
           
                       
    def init_memory(self):
        # reset the game environment, take a initial screen shot
        buffer_counter = 0
        
        while True:
            observation = self.env.reset()
            # the state of current game play, 0:2 is 3 previous frame,
            # 3 is the current frame, 4 is the frame after action
            state = np.zeros((1, self.IMG_X, self.IMG_Y, 1), dtype = 'float32')
            rnn_state = (np.zeros([1,512]),np.zeros([1,512]))
            state[0, :, :, 0] = self.process_frame(observation)
            
            state_sequence = []
            action_sequence = []
            reward_sequence = []
            done_sequence = []
            
            while True:

                observation, action, reward, done, rnn_state = self.take_action(state, rnn_state, 1)
                
                # effective area [34:194, 0:168] with 2*2 downsampling -> 160/2 * 130/2 matrix
                
                state_sequence.append(state[0, :, :, :])
                action_sequence.append(action)
                reward_sequence.append(reward)
                done_sequence.append(done)
                
                state[0, :, :, 0] = self.process_frame(observation)
                if done:
                    self.memory_buffer.add(state_sequence, action_sequence, reward_sequence, done_sequence)
                    buffer_counter += 1
                    print("Episode finished after {} timesteps".format(buffer_counter+1))
                    break
        
            if buffer_counter > self.buffer_size:
                break      

    def run(self):
        # initialize memory buffer
        self.init_memory()
        
        steps = 0
        while True:
            reward_sum = 0
            observation = self.env.reset()
            rnn_state = (np.zeros([1,512]),np.zeros([1,512]))
            
            state_sequence = []
            action_sequence = []
            reward_sequence = []
            done_sequence = []
            
            state = np.zeros((1, self.IMG_X, self.IMG_Y, 1), dtype = 'float32')
            state[0, :,:,0] = self.process_frame(observation)
            
            while True:
                # select an action based on the action-value function Q
                observation, action, reward, done, rnn_state = self.take_action(state, rnn_state, 1)
                reward_sum += reward

                # add current state to the memory buffer
                state_sequence.append(state[0, :, :, :])
                action_sequence.append(action)
                reward_sequence.append(reward)
                done_sequence.append(done)
                
                # update the new state and reward and memory buffer
                state[0, :, :, 0] = self.process_frame(observation)                 
                steps += 1
                # update the network after few steps
                if steps % self.update_freq == 0:
                    
                    if self.random_action_prob > 0.1:    
                        self.random_action_prob -= self.rand_prob_step
                    
                    self.update_nn()
                            
                # save the model after every 200 updates       
                if done:
                    self.memory_buffer.add(state_sequence, action_sequence, reward_sequence, done_sequence)
                    self.logger.log(reward_sum)                    
                    break
                

    def take_action(self, current_state, rnn_state_in, batch_size):
        
        # e-greedy algorithm for taking an action
        if np.random.random_sample() > self.random_action_prob:
            # use model to predict action
            action, rnn_state = self.nn.predict_act(self.sess, current_state, rnn_state_in, batch_size)
        else: 
            # random action
            action = np.random.randint(self.action_space) # random sample action from 1 to 3
            rnn_state = self.nn.return_rnn_state(self.sess, current_state, rnn_state_in, batch_size)
            
        # excute the action for a few steps
        reward = 0
        for _ in range(self.frame_skip):
            observation, reward_temp, done, info = self.env.step(action)
            reward += reward_temp
            if done:
                break  
        return (observation, action, reward, done, rnn_state)
    
    def update_nn(self):
        # randomly sample minibatch from memory
        state_current, state_future, actions, current_rewards, end_game = self.memory_buffer.makeBatch(self.batch_size, self.rnn_seq_len)
        
        rnn_state_init = (np.zeros([self.batch_size, self.rnn_h_units]),np.zeros([self.batch_size, self.rnn_h_units])) 
        
        self.nn.train(self.sess, state_current, state_future, actions, current_rewards, end_game, rnn_state_init, self.batch_size, self.rnn_seq_len)
        
        # # every C step reset Q' = Q
        self.updates += 1
        if self.updates % self.sync_freq == 0:
            self.nn.sync_variables(self.sess)
            
    def test(self):
        while True:
            
            observation = self.env.reset()
            
            state = np.zeros((1, self.IMG_X, self.IMG_Y, 1), dtype = 'float32')
            state[0, :,:,-1] = self.process_frame(observation)
            
            while True:
                self.env.render()
                # select an action based on the action-value function Q
                observation, action, reward, done = self.take_action(state)
                # update the new state
                state[0, :, :, 0] = self.process_frame(observation) 
 
                if done:
                    break
                
    def process_frame(self, frame):
        #return np.mean(frame[34: 194 : 2, 0: 160 : 2, :], axis = 2, dtype = 'float32') > 100
        return np.mean(frame[::2,::2], axis = 2, dtype = 'float32') / 128 - 1
    
    
    def reset_game(self):
        pass
