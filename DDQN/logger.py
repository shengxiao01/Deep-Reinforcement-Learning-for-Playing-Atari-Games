import pickle
import os
import tensorflow as tf
import numpy as np
from params import Params

class Logger(object):
    def __init__(self, sess, saver):
        # RawValue because we don't need it to create a Lock:
        self.global_episode = 0
        self.running_reward = None

        
        self.save_freq = Params['SAVE_FREQ']
        self.save_path = Params['SAVE_PATH']
        self.__saver = saver
        self.__sess = sess
        self.reward_log = []
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def log(self, reward_sum):
        

        self.running_reward = reward_sum if self.running_reward is None else self.running_reward * 0.99 + reward_sum * 0.01
        
        self.global_episode += 1
        if self.global_episode % 10 == 0:
	        self.reward_log.append(self.running_reward)
	        print('Ep {}: reward: {}, running average: {:3f}'.format(self.global_episode, reward_sum, self.running_reward))
        else:
	        print('Ep {}: reward: {}'.format(self.global_episode, reward_sum)) 
	
        if self.global_episode % self.save_freq == 0:
	        self.save()
                
    def save(self):
        
        
        f = open(self.save_path + 'reward_log.cptk','wb')
        pickle.dump(self.reward_log, f)
        f.close()    
        
        self.__saver.save(self.__sess, self.save_path+'model-'+str(self.global_episode)+'.cptk')

    def restore(self): 
        
        try:
            ckpt = tf.train.get_checkpoint_state(self.save_path)
            load_path = ckpt.model_checkpoint_path
            self.__saver.restore(self.__sess, load_path)
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
