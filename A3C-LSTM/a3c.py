import tensorflow as tf
from params import Params

class A3CNet():
    def __init__(self, scope, action_space, session, optimizer, global_scope = None):
        
        self.IMG_X = Params['IMG_X']
        self.IMG_Y = Params['IMG_Y']
        self.IMG_Z = Params['IMG_Z']
        self.entropy_penalty = Params['ENTROPY_PENALTY']
        self.learning_rate = Params['LEARNING_RATE']
        self.entropy_reg = Params['ENTROPY_PENALTY']
        self.rnn_h_units = Params['RNN_H_UNITS']
        
        self.__scope = scope
        self.__sess = session
        self.action_space = action_space
        
        with tf.variable_scope(self.__scope):  
            self.local_dict = self.build_nn()
        
        # apply gradients and sync variables
        if global_scope is not None:
            
            global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=global_scope)
            self.apply_gradient = optimizer.apply_gradients(zip(self.local_dict['gradients'], global_vars))
            
            self.sync_op = []
            for from_var, to_var in zip(global_vars, self.local_dict['variables']):
                
                self.sync_op.append(to_var.assign(from_var.value()))            
        
    def build_nn(self):
        
        # [batch, in_height, in_width, in_channels]
        # assuming input to be batch_size*84*84*4
        state_in = tf.placeholder(tf.float32, shape=[None, self.IMG_X, self.IMG_Y, self.IMG_Z])
        
        state_resized = tf.image.resize_images(state_in, [80, 80])

        ##########################################################
        #[filter_height, filter_width, in_channels, out_channels]
        # conv layer 1, 8*8*32 filters, 4 stride
        conv1_W = tf.Variable(tf.truncated_normal([8, 8, self.IMG_Z, 32], 
                                                           stddev = 0.01))
        conv1_b = tf.Variable(tf.truncated_normal([1, 20, 20, 32], 
                                                           stddev = 0.01))
        conv1_strides = [1, 4, 4, 1]
        #output 20*20*32 
        conv1_out = tf.nn.conv2d(state_resized, conv1_W, conv1_strides, padding = 'SAME') + conv1_b
        conv1_out = tf.nn.relu(conv1_out)
        
        
        ###########################################################
        # conv layer 2, 4*4*64 filters, 2 stride
        conv2_W = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev = 0.01))
        conv2_b = tf.Variable(tf.truncated_normal([1, 9, 9, 64], stddev = 0.01))
        conv2_strides = [1, 2, 2, 1]
        # output 9*9*64
        conv2_out = tf.nn.conv2d(conv1_out, conv2_W, conv2_strides, padding = 'VALID') + conv2_b
        conv2_out = tf.nn.relu(conv2_out)
        
        
        ###########################################################
        # conv layer 3, 3*3*64 filters
        conv3_W = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.01))
        conv3_b = tf.Variable(tf.truncated_normal([1, 7, 7, 64], stddev = 0.01))
        conv3_strides = [1, 1, 1, 1]
        # output 7*7*64
        conv3_out = tf.nn.conv2d(conv2_out, conv3_W, conv3_strides, padding = 'VALID') + conv3_b
        conv3_out = tf.nn.relu(conv3_out)

        ###########################################################
        # fully connected layer 1, (7*7*64 = 3136) * 512
        ff1_input = tf.reshape(conv3_out, [-1, 3136])
        ff1_W = tf.Variable(tf.truncated_normal([3136, 512], stddev = 0.01))
        ff1_b = tf.Variable(tf.truncated_normal([1, 512], stddev = 0.01))
        # output batch_size * 512
        ff1_out = tf.matmul(ff1_input, ff1_W) + ff1_b
        ff1_out = tf.nn.relu(ff1_out)
        
        ############################################################
        # recurrent layer
        
        rnn_in = tf.reshape(ff1_out, [1, -1, self.rnn_h_units])
        
        rnn_cell = tf.contrib.rnn.core_rnn_cell.LSTMCell(num_units = self.rnn_h_units)
        
        rnn_c_in = tf.placeholder(shape=[None, self.rnn_h_units],dtype=tf.float32)
        rnn_h_in = tf.placeholder(shape=[None, self.rnn_h_units],dtype=tf.float32)
        rnn_state_in = tf.contrib.rnn.LSTMStateTuple(rnn_c_in, rnn_h_in)
        
        rnn, rnn_state_out = tf.nn.dynamic_rnn(inputs=rnn_in, 
												 cell=rnn_cell, 
												 dtype=tf.float32, 
												 initial_state=rnn_state_in)
        rnn_out = tf.reshape(rnn, [-1, self.rnn_h_units])  
              
        ############################################################
        # output layer        
        policy_W = tf.Variable(tf.truncated_normal([self.rnn_h_units, self.action_space], stddev = 0.01))
        policy_b = tf.Variable(tf.truncated_normal([1, self.action_space], stddev = 0.01))

        policy_out = tf.nn.softmax(tf.matmul(rnn_out, policy_W) + policy_b)
        
        value_W = tf.Variable(tf.truncated_normal([self.rnn_h_units, 1], stddev = 0.01))
        value_b = tf.Variable(tf.truncated_normal([1, 1], stddev = 0.01))
        value_out = tf.matmul(rnn_out, value_W) + value_b   
        
        ###########################################################
        # prediction, loss, and update
        
        actions = tf.placeholder(shape=[None],dtype=tf.int32)
        
        R = tf.placeholder(shape= [None], dtype=tf.float32)
        
        actions_onehot = tf.one_hot(actions, self.action_space, dtype=tf.float32)
        
        action_policy = tf.reduce_sum(policy_out * actions_onehot, axis = 1)
        
        policy_loss = -tf.log(action_policy + 1e-6) * (R - tf.stop_gradient(tf.reshape(value_out,[-1])))
        
        V_loss = 0.5 * tf.square(R - tf.reshape(value_out,[-1]))
        
        entropy = policy_out * tf.log(policy_out + 1e-6)
        
        total_loss = tf.reduce_sum(policy_loss) + tf.reduce_sum(V_loss) + self.entropy_reg * tf.reduce_sum(entropy)
        
        ##########################################################
        # updates
        
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.__scope)
        
        grad = tf.gradients(total_loss, variables)
        
        model_dict = {'state_in': state_in, 'action_in': actions, 'R_in': R, 
                      'rnn_c_in': rnn_c_in, 'rnn_h_in': rnn_h_in,
                      'policy_out': policy_out, 'value_out': value_out, 'rnn_state_out': rnn_state_out,
                      'gradients': grad, 'variables': variables}
        
        return model_dict

            
    def variable_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.__scope.name)
    
    def predict_value(self, state, rnn_state):
        value = self.__sess.run(self.local_dict['value_out'],
                          feed_dict = {self.local_dict['state_in']: state,
						  self.local_dict['rnn_c_in']: rnn_state[0],
						  self.local_dict['rnn_h_in']: rnn_state[1]})
        return value
        
    def predict_policy(self, state, rnn_state):
        # 1X80X80X4 single image
        policy, rnn_state = self.__sess.run([self.local_dict['policy_out'], self.local_dict['rnn_state_out']],
                          feed_dict = {self.local_dict['state_in']: state,
                          self.local_dict['rnn_c_in']: rnn_state[0],
                          self.local_dict['rnn_h_in']: rnn_state[1]})
        return policy, rnn_state
    
    def sync_variables(self):
        self.__sess.run(self.sync_op)
        
    def update_global(self, state, action, R, rnn_state):

        self.__sess.run(self.apply_gradient,
                feed_dict = {
                self.local_dict['state_in']: state,
                self.local_dict['action_in']: action,
                self.local_dict['R_in']: R,
                self.local_dict['rnn_c_in']: rnn_state[0],
                self.local_dict['rnn_h_in']: rnn_state[1]})
