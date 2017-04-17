
from agent import Agent
from logger import Logger
from params import Params
import tensorflow as tf
import threading
import time

def main():

    tf.reset_default_graph()
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    logger = Logger()
    optimizer = tf.train.RMSPropOptimizer(
                    learning_rate=Params['LEARNING_RATE'],
                    decay=0.99,
                    momentum=0,
                    epsilon=0.1,
                    use_locking = True)
    
    with graph.as_default():
        # initialize global network
        global_agent = Agent(-1, 'global', sess, logger, optimizer)
        # initialize local networks
        THREAD_NUM = Params['THREAD_NUM']
        local_agent = []
        for thread_id in range(THREAD_NUM):
            local_scope = 'local'+str(thread_id)
            local_agent.append(Agent(thread_id, local_scope, sess, logger, optimizer, 'global'))
    
        # initialize tensorflow 
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
    sess.run(init_op)
    logger.add_saver(sess, saver)
    logger.restore()
    
    
    training_thread = []
    for id, agent in enumerate(local_agent):
        t = threading.Thread(target = agent.run, args = ())
        t.start()
        time.sleep(1)
        training_thread.append(t)
    
    return local_agent, training_thread
        
if __name__ == "__main__":
    local_agent, training_thread = main()    
#%%
for (agent, agent_thread) in zip(local_agent, training_thread):
    agent.exit = True
    time.sleep(0.5)
    agent_thread.join(1)