import tensorflow as tf
import numpy as np
import gym

from a3c import A3CNet
from logger import Logger
from params import Params

  
class Agent():
    
    def __init__(self, id, scope_name, sess, logger, optimizer, global_scope = None):
        
        self.__thread_id = id
        self.__scope_name = scope_name
        self.__sess = sess
        self.__opt = optimizer
        self.__logger = logger
        self.exit = False
        self.env = gym.make("Pong-v0")

        self.IMG_X = Params['IMG_X']
        self.IMG_Y = Params['IMG_Y']
        self.IMG_Z = Params['IMG_Z']
        self.frame_skip = Params['FRAME_SKIP']
        self.reward_discount = Params['REWARD_DISCOUNT']
        self.update_freq = Params['UPDATE_FREQ']
        self.rnn_h_units = Params['RNN_H_UNITS']
        self.action_space = self.env.action_space.n
        
        
        self.reward_sum = 0
        self.state = np.zeros((self.IMG_X, self.IMG_Y, 1), dtype = 'float32')
        self.rnn_state = (np.zeros([1, self.rnn_h_units]),np.zeros([1, self.rnn_h_units]))
        self.rnn_state_init = (np.zeros([1, self.rnn_h_units]),np.zeros([1, self.rnn_h_units]))
         
        self.local_nn = A3CNet(self.__scope_name, self.action_space, self.__sess, self.__opt, global_scope)
           

    def run(self):
        
        # initialize environment      
        self.reset_game()

        frame_sequence = np.zeros((self.update_freq, self.IMG_X, self.IMG_Y, self.IMG_Z), dtype = 'float32')
        action_sequence = np.zeros((self.update_freq), dtype = 'float32')
        reward_sequence = np.zeros((self.update_freq), dtype = 'float32')
        done_sequence = np.zeros((self.update_freq), dtype = 'float32')
        R_sequence = np.zeros((self.update_freq + 1), dtype = 'float32')
        #while True:
        while not self.exit:

            # running game
            for t in range(self.update_freq):
                # take actions
                observation, reward, action, done, self.rnn_state = self.take_action(self.state, self.rnn_state)
                
                self.reward_sum += reward
                
                # record game progress
                frame_sequence[t, :, :, :] = self.state
                action_sequence[t] = action
                reward_sequence[t] = reward
                done_sequence[t] = done

                # update next game state
                self.state[:,:,-1] = self.process_frame(observation)
                if done:
                    break

            R_sequence[t+1] = 0 if done else self.get_value(self.state, self.rnn_state)
            R_sequence[t+1] = np.clip(R_sequence[t+1], -1, 1)
    
            for idx in range(t, -1, -1):
                if reward_sequence[idx] != 0:
                    R_sequence[idx] = reward_sequence[idx]
                else:
                    R_sequence[idx] = self.reward_discount * R_sequence[idx+1]            

            self.update_nn(frame_sequence[0:t+1, :, :, :], action_sequence[0:t+1], R_sequence[0:t+1], self.rnn_state_init)
            self.rnn_state_init = self.rnn_state
    
            if done:
                self.reset_game()

    def take_action(self, current_state, rnn_state):
    
        policy_prob, rnn_state= self.local_nn.predict_policy(np.expand_dims(current_state, axis = 0), rnn_state)
        # choose an action according to policy
        action = np.random.choice(self.action_space, p=np.squeeze(policy_prob))
        
        # take this action for certain steps and record the reward
        reward = 0
        for _ in range(self.frame_skip):
            observation, reward_temp, done, info = self.env.step(action)
            reward += reward_temp
            if done:
                break
        return observation, reward, action, done, rnn_state
    
    def get_value(self, state, rnn_state):
        return self.local_nn.predict_value(np.expand_dims(state, axis = 0), rnn_state)
    
    def update_nn(self, states, actions, rewards, rnn_state_init):
        
        self.local_nn.update_global(states, actions, rewards, rnn_state_init)

    def test(self):
        pass

                
    def process_frame(self, frame):
        # output shape 105X80
        return np.mean(frame[::2,::2], axis = 2, dtype = 'float32') / 128 - 1
    
    
    def reset_game(self):
        #self.log()
        self.__logger.log(self.__thread_id, self.reward_sum)
        
        #self.local_nn.sync_variables(sess, global_agent.local_nn.return_scope())
        self.local_nn.sync_variables()
        observation = self.env.reset()
        self.state.fill(0)
        self.rnn_state[0].fill(0)
        self.rnn_state[1].fill(0)
        self.rnn_state_init[0].fill(0)
        self.rnn_state_init[1].fill(0)
        self.state[:,:,-1] = self.process_frame(observation)
        self.reward_sum = 0
