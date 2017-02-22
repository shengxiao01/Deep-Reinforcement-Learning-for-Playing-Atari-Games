#%%
import tensorflow as tf
import numpy as np
import gym

DEBUG = True


RENDER = True  # if displaying game graphics real time
LEARNING_RATE = 0.0001 


#%% Deep Q-Network Structure
class DQNet():
    def __init__(self,input_size = (84, 84, 4), action_space = 2):

        self.input_x, self.input_y, self.input_frame= input_size
        self.action_space = action_space

    def build_nn(self):
        

        # [batch, in_height, in_width, in_channels]
        # assuming input to be batch_size*84*84*4
        self.dqn_input = tf.placeholder(tf.float32, shape=[None, self.input_x, self.input_y, self.input_frame])


        ###########################################################
        #[filter_height, filter_width, in_channels, out_channels]
        # conv layer 1, 8*8*32 filters, 4 stride
        self.dqn_conv1_W = tf.Variable(tf.truncated_normal([8, 8, 4, 32], 
                                                           stddev = 0.1))
        self.dqn_conv1_strides = [1, 4, 4, 1]
        #output 20*20*32 
        self.dqn_conv1_out = tf.nn.conv2d(self.dqn_input, 
                                          self.dqn_conv1_W, 
                                          self.dqn_conv1_strides, 
                                          padding = 'VALID')
        self.dqn_conv1_out = tf.nn.relu(self.dqn_conv1_out)
        
        
        ###########################################################
        # conv layer 2, 4*4*64 filters, 2 stride
        self.dqn_conv2_W = tf.Variable(tf.truncated_normal([4, 4, 32, 64],
                                                           stddev = 0.1))
        self.dqn_conv2_strides = [1, 2, 2, 1]
        # output 9*9*64
        self.dqn_conv2_out = tf.nn.conv2d(self.dqn_conv1_out, 
                                          self.dqn_conv2_W, 
                                          self.dqn_conv2_strides, 
                                          padding = 'VALID')
        self.dqn_conv2_out = tf.nn.relu(self.dqn_conv2_out)
        
        
        ###########################################################
        # conv layer 3, 3*3*64 filters
        self.dqn_conv3_W = tf.Variable(tf.truncated_normal([3, 3, 64, 64],
                                                           stddev = 0.1))
        self.dqn_conv3_strides = [1, 1, 1, 1]
        # output 7*7*64
        self.dqn_conv3_out = tf.nn.conv2d(self.dqn_conv2_out, 
                                          self.dqn_conv3_W, 
                                          self.dqn_conv3_strides, 
                                          padding = 'VALID')
        self.dqn_conv3_out = tf.nn.relu(self.dqn_conv3_out)
        
        
        ###########################################################
        # fully connected layer 1, (7*7*64 = 3136) * 512
        self.dqn_ff1_input = tf.reshape(self.dqn_conv3_out, [-1, 3136])
        self.dqn_ff1_W = tf.Variable(tf.truncated_normal([3136, 512],
                                                         stddev = 0.1))
        self.dqn_ff1_b = tf.Variable(tf.truncated_normal([1, 512],
                                                         stddev = 0.1))
        # output batch_size * 512
        self.dqn_ff1_out = tf.matmul(self.dqn_ff1_input, self.dqn_ff1_W) + self.dqn_ff1_b
        self.dqn_ff1_out = tf.nn.relu(self.dqn_ff1_out)
        
        
        ###########################################################
        # fully connected layer 2, 
        self.dqn_ff2_W = tf.Variable(tf.truncated_normal([ 512, self.action_space],
                                                         stddev = 0.1))
        self.dqn_ff2_b = tf.Variable(tf.truncated_normal([ 1, self.action_space],
                                                         stddev = 0.1))
        # final output, batch_size * action_space
        self.dqn_ff2_out = tf.matmul(self.dqn_ff1_out, self.dqn_ff2_W) + self.dqn_ff2_b
        
        
        ###########################################################
        # prediction, loss, and update
        self.predict = tf.argmax(self.dqn_ff2_out, 1)
        
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.actions_onehot = tf.one_hot(self.actions, 2, dtype=tf.float32)
        
        self.Q = tf.reduce_sum((self.dqn_ff2_out * self.actions_onehot), 
                               reduction_indices=1)
        
        self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))

        self.update = tf.train.AdamOptimizer(learning_rate = LEARNING_RATE).minimize(self.loss)
        
 
#%% initialize and running the model 
tf.reset_default_graph()
Atari_AI = DQNet()
Atari_AI.build_nn()

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()


with tf.Session() as sess:
  sess.run(init_op)
  try:
      saver.restore(sess, "./model.ckpt")
      print("Session restored...")
  except:
      save_path = saver.save(sess, "./model.ckpt")
      print("Session saved...")
  
  sess.close()
#%% running OpenAI Gym environment


env = gym.make("Pong-v0")


for i_episode in range(2):
    observation = env.reset()
    for t in range(100):
        if RENDER:
            env.render()
        if DEBUG:
            print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break


