"""
The environment of Sawyer Arm + Baxter Gripper for graping object.
With a bounding box of the arange that the gripper cannot move outside.
"""
from os.path import dirname, join, abspath
from pyrep import PyRep

import numpy as np
import matplotlib.pyplot as plt
from reacher_sawyer_env_boundingbox import ReacherEnv
import pickle
import argparse




parser = argparse.ArgumentParser(description='Load Test')
parser.add_argument('--load', dest='load', action='store_true', default=False)

args = parser.parse_args()


if __name__ == '__main__':
    if args.load: 
        data_file=open('./demons_data/demon_data.pickle', "rb")
        data = pickle.load(data_file)
        print(len(data), data[5])
        data_file.close()

    else:

        data_file=open('demon_data.pickle', "wb")

        env=ReacherEnv(headless=False, control_mode='end_position')

        max_epi=10
        max_steps=30
        offset=np.array([0,0,0.1])  # move on top of the object
        action_range = 0.1
        demons_buffer=[]
        for ep in range(max_epi):
            state = env.reset()
            for step in range(max_steps):
                target_pos=env.target.get_position()
                gripper_pos=env.gripper.get_position()
                pos_diff=np.array(target_pos)+offset-np.array(gripper_pos)
                action = np.concatenate((np.clip(pos_diff ,-action_range, action_range), np.random.uniform(-0.1,0.1,1)))  # clip the action range to be valid
                print('Eps: ', ep, 'Step: ', step)
                # action=np.random.uniform(-0.1,0.1,4)  #  4 dim control for 'end_position': 3 positions and 1 rotation (z-axis)
                try:
                    # state, action, reward, next_state, done
                    next_state, reward, done, _ = env.step(action)
                    demons_buffer.append([state, action, reward, next_state, done])
                    # print(state, action, reward, next_state, done)
                    state = next_state
                except KeyboardInterrupt:
                    print('Shut Down!')
                    pickle.dump(demons_buffer, data_file)
                    env.shutdown()
        pickle.dump(demons_buffer, data_file)
        env.shutdown()
