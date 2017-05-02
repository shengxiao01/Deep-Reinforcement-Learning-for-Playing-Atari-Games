import tensorflow as tf
from params import Params

class DPNet():
    def __init__(self, action_space):
        
        self.IMG_X = Params['IMG_X']
        self.IMG_Y = Params['IMG_Y']
        self.IMG_Z = Params['IMG_Z']
        self.entropy_penalty = Params['ENTROPY_PENALTY']
        self.action_space = action_space
        self.learning_rate = Params['LEARNING_RATE']
        
        self.reward_discount = 0.99
        
        self.policy_nn()
        


    def policy_nn(self):
        
        self.state_in, self.policy_out = self.build_nn()
        
        self.predict_action = tf.argmax(self.policy_out, axis = 1)
        
        self.exp_reward = tf.placeholder(tf.float32, shape=[None])
        
        self.actions = tf.placeholder(tf.int32, shape=[None])
        
        r_mean, r_var = tf.nn.moments(self.exp_reward, axes = [0])
        
        normalized_reward = (self.exp_reward  - r_mean)/tf.sqrt(r_var)
        
        #normalized_reward = self.exp_reward
        
        actions_onehot = tf.one_hot(self.actions, self.action_space, dtype=tf.float32)
        
        neg_log_prob =  -tf.multiply( actions_onehot * tf.log(self.policy_out + 1e-6), tf.reshape(normalized_reward, [-1, 1]))
        
        entropy = self.policy_out * tf.log(self.policy_out + 1e-6)
        
        loss = tf.reduce_mean(neg_log_prob) + self.entropy_penalty * tf.reduce_mean(entropy)
        
        self.update = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss) 
              
            
        
        
    def build_nn(self):
        

        # [batch, in_height, in_width, in_channels]
        # assuming input to be batch_size*84*84*4
        state_in = tf.placeholder(tf.float32, shape=[None, self.IMG_X, self.IMG_Y, self.IMG_Z])
        state_resized = tf.image.resize_images(state_in, [80, 80])

        ##########################################################
        #[filter_height, filter_width, in_channels, out_channels]
        # conv layer 1, 8*8*32 filters, 4 stride
        conv1_W = tf.Variable(tf.truncated_normal([8, 8, self.IMG_Z, 32], stddev = 0.01))
        conv1_b = tf.Variable(tf.truncated_normal([1, 20, 20, 32], stddev = 0.01))
        conv1_strides = [1, 4, 4, 1]
        #output 20*20*32 
        conv1_out = tf.nn.conv2d(state_resized, conv1_W, conv1_strides, 
                                          padding = 'SAME') + conv1_b
        conv1_out = tf.nn.relu(conv1_out)
        
        
        ###########################################################
        # conv layer 2, 4*4*64 filters, 2 stride
        conv2_W = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.01))
        conv2_b = tf.Variable(tf.truncated_normal([1, 9, 9, 64], stddev = 0.01))
        conv2_strides = [1, 2, 2, 1]
        # output 9*9*64
        conv2_out = tf.nn.conv2d(conv1_out, conv2_W, conv2_strides, 
                                          padding = 'VALID') + conv2_b
        conv2_out = tf.nn.relu(conv2_out)
        
        
        ###########################################################
        # fully connected layer 1, (7*7*64 = 3136) * 512
        ff1_input = tf.reshape(conv2_out, [-1, 5184])
        ff1_W = tf.Variable(tf.truncated_normal([5184, 256], stddev = 0.01))
        ff1_b = tf.Variable(tf.truncated_normal([1, 256], stddev = 0.01))
        # output batch_size * 512
        ff1_out = tf.matmul(ff1_input, ff1_W) + ff1_b
        ff1_out = tf.nn.relu(ff1_out)
        
        
        ##################################################################
        ff2_W = tf.Variable(tf.truncated_normal([ 256, self.action_space],
                                                         stddev = 0.01))
        ff2_b = tf.Variable(tf.truncated_normal([ 1, self.action_space],
                                                         stddev = 0.01))        
        # final output, batch_size * action_space
        ff2_out = tf.matmul(ff1_out, ff2_W) + ff2_b
        
        policy_out = tf.nn.softmax(ff2_out)
        
        return state_in, policy_out

            
    def train(self, sess, state, action, reward):
        
        sess.run(self.update, feed_dict={self.state_in: state,
                                         self.actions: action,
                                         self.exp_reward: reward})

    def predict_policy(self, sess, state):
        # 1X80X80X4 single image
        policy = sess.run(self.policy_out,
                          feed_dict = {self.state_in: state})
        return policy
