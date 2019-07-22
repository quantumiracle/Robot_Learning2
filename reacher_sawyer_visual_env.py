"""
An example of how one might use PyRep to create their RL environments.
In this case, the Franka Panda must reach a randomly placed target.
This script contains examples of:
    - RL environment example.
    - Scene manipulation.
    - Environment resets.
    - Setting joint properties (control loop disabled, motor locked at 0 vel)

following common format of Openai gym
"""
from os.path import dirname, join, abspath
from pyrep import PyRep
from pyrep.robots.arms.sawyer import Sawyer
from pyrep.robots.end_effectors.baxter_gripper import BaxterGripper
from pyrep.objects.proximity_sensor import ProximitySensor
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.shape import Shape
import numpy as np
import matplotlib.pyplot as plt

SCENE_FILE = join(dirname(abspath(__file__)), './scenes/sawyer_reacher_rl_new.ttt')
POS_MIN, POS_MAX = [0.1, -0.3, 0.8], [0.45, 0.3, 0.8]  # valid position range of target object 


class ReacherEnv(object):

    def __init__(self, headless, visual_control=False):
        '''
        :visual_control: bool, controlled by visual state or not (vector state).
        '''
        self.visual_control = visual_control
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()
        self.agent = Sawyer()
        self.gripper = BaxterGripper()
        self.gripper_left_pad = Shape('BaxterGripper_leftPad')  # the pad on the gripper finger
        self.proximity_sensor = ProximitySensor('BaxterGripper_attachProxSensor')  # need the name of the sensor here
        self.vision_sensor = VisionSensor('Vision_sensor')  # need the name of the sensor here
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('target')  # object
        # self.table = Shape('diningTable')
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()
        self.action_space = np.zeros(7)  # 7 DOF velocity control
        self.observation_space = np.zeros(17)  # position and velocity of 7 joints + position of the target

    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        return np.array(self.agent.get_joint_positions() +
                self.agent.get_joint_velocities() +
                self.target.get_position())

    def _get_visual_state(self):
        # Return a numpy array of size (width, height, 3)
        return self.vision_sensor.capture_rgb()  # A numpy array of size (width, height, 3)

    def _is_holding(self):
        # Return is holding the target or not, return bool

        # Nothe that the collision check is not always accurate all the time, 
        # for continuous conllision, maybe only the first 4-5 frames of collision can be detected
        pad_collide_object = self.gripper_left_pad.check_collision(self.target)
        if  pad_collide_object and self.proximity_sensor.is_detected(self.target)==True:
            return True 
        else:
            return False

    def reset(self):
        # Get a random position within a cuboid and set the target position
        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.target.set_position(pos)
        # changing the color or texture for domain randomization
        self.target.set_color(np.random.uniform(low=0, high=1, size=3).tolist()) # set [r,g,b] 3 channel values of object color
        self.agent.set_joint_positions(self.initial_joint_positions)
        # self.table.set_collidable(True)
        self.gripper_left_pad.set_collidable(True)  # set the pad on the gripper to be collidable, so as to check collision
        self.target.set_collidable(True)
        if self.visual_control:
            return self._get_visual_state
        else:
            return self._get_state()

    def step(self, action):
        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation
        # ax, ay, az = self.agent_ee_tip.get_position()
        ax, ay, az = self.gripper.get_position()
        tx, ty, tz = self.target.get_position()
        # Reward is negative distance to target
        distance = (ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2
        done=False
        # print(self.proximity_sensor.is_detected(self.target))
        current_vision = self.vision_sensor.capture_rgb()  # capture a screenshot of the view with vision sensor
        plt.imshow(current_vision)
        plt.savefig('./img/vision.png')

        # close the gripper if close enough to the object and the object is detected with the proximity sensor
        if distance<0.05 and self.proximity_sensor.is_detected(self.target)== True: 
            self.gripper.actuate(0, velocity=0.04)  # if done, close the hand, 0 for close and 1 for open.
            self.pr.step()  # Step the physics simulation

            if self._is_holding():
                # reward for hold here!
                done=True

        reward = -np.sqrt(distance)
        if self.visual_control:
            return self._get_visual_state, reward, done
        else:
            return self._get_state(), reward, done

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

