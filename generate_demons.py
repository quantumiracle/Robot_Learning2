"""
The environment of Sawyer Arm + Baxter Gripper for graping object.
With a bounding box of the arange that the gripper cannot move outside.
"""
from os.path import dirname, join, abspath
from pyrep import PyRep

import numpy as np
import matplotlib.pyplot as plt
from reacher_sawyer_env_boundingbox import ReacherEnv


if __name__ == '__main__':
    env=ReacherEnv(headless=False, control_mode='end_position')

    max_epi=10
    max_steps=30
    offset=np.array([0,0,0.1])  # move on top of the object
    for ep in range(max_epi):
        env.reset()

        for step in range(max_steps):
            target_pos=env.target.get_position()
            gripper_pos=env.gripper.get_position()
            pos_diff=np.array(target_pos)+offset-np.array(gripper_pos)
            action = np.concatenate((pos_diff , np.random.uniform(-0.1,0.1,1)))
            print(step)
            # action=np.random.uniform(-0.1,0.1,4)  #  4 dim control for 'end_position': 3 positions and 1 rotation (z-axis)
            try:
                env.step(action)
            except KeyboardInterrupt:
                print('Shut Down!')
                env.shutdown()

    env.shutdown()
