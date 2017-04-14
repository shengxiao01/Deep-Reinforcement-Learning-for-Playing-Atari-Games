import tensorflow as tf
from params import Params

class DDQNet():
    def __init__(self, action_space):

        self.IMG_X = Params['IMG_X']
        self.IMG_Y = Params['IMG_Y']
        self.action_space = action_space
        self.learning_rate = Params['LEARNING_RATE']
        self.rnn_h_units = Params['RNN_H_UNIT']
        self.primary_scope = 'primary'
        self.target_scope = 'target'
        
        self.reward_discount = 0.99
        
        self.dueling_nn()
        


    def dueling_nn(self):
        
        with tf.variable_scope(self.primary_scope) as scope:
            self.primary_in, self.primary_out, self.primary_rnn_in, self.primary_rnn_out, self.primary_bz = self.build_nn()
            
        with tf.variable_scope(self.target_scope) as scope:
            self.target_in, self.target_out, self.target_rnn_in, self.target_rnn_out, self.target_bz = self.build_nn()
            

        self.end_game = tf.placeholder(shape=[None],dtype=tf.float32)
        self.current_reward = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.trainLength = tf.placeholder(tf.int32)
        
        next_Q = tf.reduce_max(self.target_out, axis = 1)
        
        targetQ = self.current_reward + self.reward_discount * tf.multiply(1 - self.end_game, next_Q)
        
        targetQ = tf.stop_gradient(targetQ)
                
        actions_onehot = tf.one_hot(self.actions, self.action_space, dtype=tf.float32)
        
        Q = tf.reduce_sum((self.primary_out * actions_onehot), reduction_indices=1)
        
        
        maskA = tf.zeros([self.primary_bz, self.trainLength//2])
        maskB = tf.ones([self.primary_bz, self.trainLength//2])
        mask = tf.concat([maskA,maskB],1)
        mask = tf.reshape(mask,[-1])
        
        loss = tf.reduce_mean(tf.square(targetQ - Q))
        
        # training
        self.update = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(loss)
        # predict action according to the target network
        self.predict = tf.argmax(self.primary_out, axis = 1)
        
        # synchronize two networks
        from_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.primary_scope)
        to_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.target_scope)
        self.sync_op = []
        for from_var, to_var in zip(from_variables, to_variables):
            self.sync_op.append(to_var.assign(from_var.value()))
        
        
    def build_nn(self):
        

        # [batch, in_height, in_width, in_channels]
        # assuming input to be batch_size*84*84*4
        state_in = tf.placeholder(tf.float32, shape=[None, self.IMG_X, self.IMG_Y, 1])
        state_resized = tf.image.resize_images(state_in, [80, 80])

        ##########################################################
        #[filter_height, filter_width, in_channels, out_channels]
        # conv layer 1, 8*8*32 filters, 4 stride
        conv1_W = tf.Variable(tf.truncated_normal([8, 8, 1, 32], stddev = 0.01))
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
        # conv layer 3, 3*3*64 filters
        conv3_W = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.01))
        conv3_b = tf.Variable(tf.truncated_normal([1, 7, 7, 64], stddev = 0.01))
        conv3_strides = [1, 1, 1, 1]
        # output 7*7*64
        conv3_out = tf.nn.conv2d(conv2_out, conv3_W, conv3_strides, 
                                          padding = 'VALID') + conv3_b
        conv3_out = tf.nn.relu(conv3_out)

        ###########################################################
        # fully connected layer 1, (7*7*64 = 3136) * 512
        ff1_input = tf.reshape(conv3_out, [-1, 3136])
        ff1_W = tf.Variable(tf.truncated_normal([3136, self.rnn_h_units], stddev = 0.01))
        ff1_b = tf.Variable(tf.truncated_normal([1, self.rnn_h_units], stddev = 0.01))
        # output batch_size * 512
        ff1_out = tf.matmul(ff1_input, ff1_W) + ff1_b
        ff1_out = tf.nn.relu(ff1_out)
        
        ############################################################
        # recurrent layer
        batch_size = tf.placeholder(tf.int32)
        
        rnn_in = tf.reshape(ff1_out, [batch_size, -1, self.rnn_h_units])
        
        rnn_cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(num_units = self.rnn_h_units)
        
        rnn_state_in = rnn_cell.zero_state(batch_size, tf.float32)
        
        rnn, rnn_state_out = tf.nn.dynamic_rnn(inputs=rnn_in, 
                                                         cell=rnn_cell, 
                                                         dtype=tf.float32, 
                                                         initial_state=rnn_state_in)
        rnn_out = tf.reshape(rnn, [-1, self.rnn_h_units])
        
        ##############################################################
        advantage_in, value_in = tf.split(rnn_out, 2, axis = 1)
        
        advantage_W = tf.Variable(tf.truncated_normal([256, self.action_space], stddev = 0.01))
        value_W = tf.Variable(tf.truncated_normal([256, 1], stddev = 0.01))
        
        advantage_out = tf.matmul(advantage_in, advantage_W)
        
        value_out = tf.matmul(value_in, value_W)
        
        #Then combine them together to get our final Q-values.
        Q_out = value_out + advantage_out - tf.reduce_mean(advantage_out,reduction_indices=1,keep_dims=True)
        
        return state_in, Q_out, rnn_state_in, rnn_state_out, batch_size


    def sync_variables(self, sess):

        # adding scope to network        
        sess.run(self.sync_op)
            
    def train(self, sess, state_current, state_future, action, reward, end_game, rnn_state_in, batch_size, rnn_seq_len):
        
        sess.run(self.update, feed_dict={self.target_in: state_future,
                                         self.primary_in: state_current,
                                         self.actions: action,
                                         self.current_reward: reward,
                                         self.end_game: end_game,
                                         self.primary_rnn_in: rnn_state_in,
                                         self.primary_bz: batch_size,
                                         self.target_bz: batch_size,
                                         self.trainLength: rnn_seq_len})

    def predict_act(self, sess, state, rnn_state_in, batch_size):
        # 1X80X80X4 single image
        action, rnn_state_out = sess.run([self.predict, self.primary_rnn_out],
                          feed_dict = {self.primary_in: state,
                                       self.primary_rnn_in: rnn_state_in,
                                       self.primary_bz: batch_size})
        return action, rnn_state_out
    
    def return_rnn_state(self, sess, state, rnn_state_in, batch_size):
        # 1X80X80X4 single image
        rnn_state_out = sess.run(self.primary_rnn_out,
                          feed_dict = {self.primary_in: state,
                                       self.primary_rnn_in: rnn_state_in,
                                       self.primary_bz: batch_size})
        return rnn_state_out