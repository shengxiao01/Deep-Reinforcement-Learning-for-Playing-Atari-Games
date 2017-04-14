import numpy as np

    
class replayMemory():
    
    def __init__(self, IMG_X, IMG_Y, size):
        # [i, :, :, 0:4] is the current state
        # [i, :, :, 1:5] is the next state
        self.IMG_X = IMG_X
        self.IMG_Y = IMG_Y
        self.frames = [None] * size
        self.actions = [None] * size
        self.rewards = [None] * size
        self.done = [None] * size
        self.__counter = 0
        self.__size = size
        
    def add(self, state, action, reward, done):
        
        self.frames[self.__counter] = np.array(state)
        self.actions[self.__counter] = np.array(action)
        self.rewards[self.__counter] = np.array(reward)
        self.done[self.__counter] =np.array(done)
        self.__counter += 1
        self.__counter = self.__counter % self.__size
     
    def makeBatch(self, batch_size, temporal_length):  
        
        idx = np.random.randint(0, self.__size, (batch_size))
        
        current_frame_sample = np.zeros((batch_size, temporal_length, self.IMG_X, self.IMG_Y, 1)).astype('float32')
        next_frame_sample = np.zeros((batch_size, temporal_length, self.IMG_X, self.IMG_Y, 1)).astype('float32')
        action_sample = np.zeros((batch_size, temporal_length)).astype('int32')
        reward_sample = np.zeros((batch_size, temporal_length)).astype('float32')
        done_sample = np.zeros((batch_size, temporal_length)).astype('int32')
        
        i = 0
        for c_idx in idx:
            
            sequence_start_idx = np.random.randint(0, len(self.actions[c_idx]) - temporal_length - 1)
            sequence_idx = np.array(range(sequence_start_idx, sequence_start_idx + temporal_length))
            
            current_frame_sample[i, :, :, :, :] = self.frames[c_idx][sequence_idx,:,:,:]
            next_frame_sample[i, :, :, :, :] = self.frames[c_idx][sequence_idx + 1,:,:,:]
            action_sample[i, :] = self.actions[c_idx][sequence_idx]
            reward_sample[i, :] = self.rewards[c_idx][sequence_idx]
            done_sample[i, :] = self.done[c_idx][sequence_idx]
            i += 1
        current_frame_sample = np.reshape(current_frame_sample, (batch_size*temporal_length, self.IMG_X, self.IMG_Y, 1))
        next_frame_sample = np.reshape(next_frame_sample, (batch_size*temporal_length, self.IMG_X, self.IMG_Y, 1))
        action_sample = np.reshape(action_sample, (-1))
        reward_sample = np.reshape(reward_sample, (-1))
        done_sample = np.reshape(done_sample, (-1))
        return (current_frame_sample, next_frame_sample, action_sample, reward_sample, done_sample)