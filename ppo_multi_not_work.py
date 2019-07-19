"""
The Vrep env cannot work with multithreading, one env has to be in one process！
image-based observation for learning: image of the tactip, size 256*256 = 65536, encoded to be 2048 dim
task: tactip following the cylinder to reach the ball target
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
from gym_unity.envs import UnityEnv
import argparse
from PIL import Image
import time

from reacher_sawyer_env import ReacherEnv


EP_MAX = 1000000
EP_LEN = 20
N_WORKER = 2                # parallel workers
GAMMA = 0.9                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0002               # learning rate for critic
MIN_BATCH_SIZE = 10         # minimum batch size for updating PPO
UPDATE_STEP = 8            # loop update operation n-steps
EPSILON = 0.2               # for clipping surrogate objective

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()


# env=ReacherEnv(headless=True)
# S_DIM=env.observation_space.shape[0]
# A_DIM= env.action_space.shape[0] 

S_DIM=17
A_DIM=7



class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        

        # critic
        l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
        l2 = tf.layers.dense(l1, 100, tf.nn.relu)
        l3 = tf.layers.dense(l2, 100, tf.nn.relu)
        self.v = tf.layers.dense(l3, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    # def maxpool2d(x, k=2):
    #     # MaxPool2D wrapper
    #     return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    
    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]

                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available
    
    # def encoder(self, input):
    #     model = tf.keras.models.Sequential(name='encoder')
    #     model.add(Conv2D(filters=8, kernel_size=(2,2), padding='same', activation='relu'))
    #     model.add(MaxPooling2D(pool_size = (2,2)))
    #     model.add(Conv2D(filters=4, kernel_size=(2,2), padding='same', activation='relu'))
    #     model.add(MaxPooling2D(pool_size = (2,2)))
    #     model.add(Conv2D(filters=2, kernel_size=(2,2), padding='same', activation='relu'))
    #     model.add(MaxPooling2D(pool_size = (2,2)))
    #     model.add(Flatten()) # latent dim 2048
    #     return model(input)


        

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.tanh, trainable=trainable)
            l2 = tf.layers.dense(l1, 200, tf.nn.tanh, trainable=trainable)
            l3 = tf.layers.dense(l2, 200, tf.nn.tanh, trainable=trainable)
            action_scale = 1.0
            mu = action_scale * tf.layers.dense(l3, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l3, A_DIM, tf.nn.softplus, trainable=trainable)
            sigma +=1e-1 # without this line, 0 value sigma may cause NAN action
            # print('mu,sig: ', mu, sigma)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :] # np.newaxis is to increase dim, [] -> [[]]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :] 
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


    def save(self, path):
            self.saver.save(self.sess, path)

    def load(self, path):
            self.saver.restore(self.sess, path)




class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env=ReacherEnv(headless=True) # for multi-process the headless has to be True
        self.ppo = GLOBAL_PPO

        self.pins_x=[]
        self.pins_y=[]

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        step_set=[]
        epr_set=[]
        step=0

        while not COORD.should_stop():
            s=self.env.reset()
            step+=1
            
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            self.pins_x=[]
            self.pins_y=[]
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data
                a = self.ppo.choose_action(s)
                # print('a: ', a)
                s_, r, done = self.env.step(a)

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)                    # normalize reward, find to be useful
                s  = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1                      # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []                           # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break

                    

            if GLOBAL_EP%100==0 and GLOBAL_EP>0:
                self.ppo.save(model_path)
            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_r)
            else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+ep_r*0.1)
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP/EP_MAX*100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r,)
            step_set.append(step)
            # print(step)
            epr_set.append(ep_r)
            if step % 10==0:  # plot every N episode; some error about main thread for plotting
                plt.plot(step_set,epr_set)  # no moving average
                try:
                    plt.savefig('./ppo_multi.png')
                except:
                    print('writing conflict!')
                
        self.env.shutdown()


if __name__ == '__main__':
    model_path = './model/ppo_multi'
    if args.train:
        GLOBAL_PPO = PPO()
        UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
        UPDATE_EVENT.clear()            # not update now
        ROLLING_EVENT.set()             # start to roll out
        workers = [Worker(wid=i) for i in range(N_WORKER)]
        
        GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
        GLOBAL_RUNNING_R = []
        COORD = tf.train.Coordinator()
        QUEUE = queue.Queue()           # workers putting data in this queue
        threads = []
        for worker in workers:          # worker threads
            t = threading.Thread(target=worker.work, args=())
            t.daemon = True             # kill the main thread, the sub-threads die as well
            t.start()                   # training
            threads.append(t)
        # add a PPO updating thread
        threads.append(threading.Thread(target=GLOBAL_PPO.update,))
        threads[-1].start()  # start the updating thread as well
        COORD.join(threads)  # waiting to finish all threads 

        GLOBAL_PPO.save(model_path)

    if args.test:
        env=ReacherEnv(headless=True)
        env.reset()
        GLOBAL_PPO = PPO()
        GLOBAL_PPO.load(model_path)
        test_steps = 200
        test_episode = 10
        

        for _ in range(test_episode):
            s, info = env.reset()
 
            for t in range(test_steps):
                s, r, done, info = env.step(GLOBAL_PPO.choose_action(s))
     
        env.shutdown()