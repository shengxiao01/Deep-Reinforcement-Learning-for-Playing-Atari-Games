#%%
import tensorflow as tf
import numpy as np

#%%
class DQNet():
    def __init__(self,input_size = (80, 80, 4), action_space = 2):

        self.input_x, self.input_y, self.input_frame= input_size
        self.action_space = action_space

    def build_nn(self):
        

        # [batch, in_height, in_width, in_channels]
        # assuming input to be batch_size*84*84*4
        self.dqn_input = tf.placeholder(tf.float32, shape=[None, self.input_x, self.input_y, self.input_frame])

        #[filter_height, filter_width, in_channels, out_channels]
        # conv layer 1, 8*8*32 filters, 4 stride
        self.dqn_conv1_W = tf.Variable(tf.truncated_normal([8, 8, 4, 32], 
                                                           stddev = 0.1))
        self.dqn_conv1_strides = [1, 4, 4, 1]
        #output 20*20*32 
        self.dqn_conv1_out = tf.nn.conv2d(self.dqn_input, 
                                          self.dqn_conv1_W, 
                                          self.dqn_conv1_strides, 
                                          padding = 'SAME')
        self.dqn_conv1_out = tf.nn.relu(self.dqn_conv1_out)
        
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
        
        
        # fully connected layer 1, (7*7*64) * 512
        self.dqn_ff1_input = tf.reshape(self.dqn_conv3_out, [-1, 3136])
        self.dqn_ff1_W = tf.Variable(tf.truncated_normal([3136, 512],
                                                         stddev = 0.1))
        self.dqn_ff1_b = tf.Variable(tf.truncated_normal([1, 512],
                                                         stddev = 0.1))
        # output batch_size * 512
        self.dqn_ff1_out = tf.matmul(self.dqn_ff1_input, self.dqn_ff1_W) + self.dqn_ff1_b
        self.dqn_ff1_out = tf.nn.relu(self.dqn_ff1_out)
        
        # fully connected layer 2, 
        self.dqn_ff2_W = tf.Variable(tf.truncated_normal([ 512, self.action_space],
                                                         stddev = 0.1))
        self.dqn_ff2_b = tf.Variable(tf.truncated_normal([ 1, self.action_space],
                                                         stddev = 0.1))
        # final output, batch_size * action_space
        self.dqn_ff2_out = tf.matmul(self.dqn_ff1_out, self.dqn_ff2_W) + self.dqn_ff2_b
        
        
        # predicted action
        self.predict = tf.argmax(self.dqn_ff2_out, 1)
        
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.target_label = tf.placeholder(shape=[32, 2],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.actions_onehot = tf.one_hot(self.actions, 2, dtype=tf.float32)
        
        self.Q = tf.reduce_sum((self.dqn_ff2_out * self.actions_onehot), 
                               reduction_indices=1)
        
        #self.loss = tf.reduce_mean(tf.square(self.targetQ - self.Q))
        self.loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=self.target_label, logits=self.dqn_ff2_out))
        self.train = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update = self.train.minimize(self.loss)
        
 #%%       
batch_size = 32

input_img = np.random.random_sample((batch_size, 80, 80, 4)).astype('float32')

true_label = np.random.randint(0, 2, size = (batch_size)).astype('float32')
action = np.random.randint(0, 2, size = (batch_size)).astype('int32')
true_label_onehot = np.zeros((32, 2))
true_label_onehot[np.arange(32), true_label.astype('int32')] = 1
#%%
tf.reset_default_graph()
mainQN = DQNet()
mainQN.build_nn()

#%%
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#%%
for i in range(1):
    sess.run(mainQN.update, feed_dict={
            mainQN.dqn_input: input_img + 0.05* np.random.random_sample((batch_size, 80, 80, 4)).astype('float32'),
            mainQN.target_label: true_label_onehot,
            mainQN.actions: action})
    predicted_label = sess.run(mainQN.predict, feed_dict={mainQN.dqn_input: input_img})
    
    print(np.linalg.norm(predicted_label - true_label))
#%%
sess.close()

