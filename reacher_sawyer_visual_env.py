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
from pyrep.objects.dummy import Dummy

import numpy as np
import matplotlib.pyplot as plt

POS_MIN, POS_MAX = [0.1, -0.3, 0.8], [0.45, 0.3, 0.8]  # valid position range of target object 


class ReacherEnv(object):

    def __init__(self, headless, control_mode='end_position', visual_control=False):
        '''
        :visual_control: bool, controlled by visual state or not (vector state).
        '''
        self.reward_offset=10.0
        self.reward_range = self.reward_offset # reward range for register gym env when using vectorized env wrapper
        self.fall_down_offset = 0.1 # for judging the target object fall off the table
        self.metadata=[]  # gym env format
        self.visual_control = visual_control
        self.control_mode = control_mode
        self.pr = PyRep()
        if control_mode == 'end_position':  # need to use different scene, the one with all joints in inverse kinematics mode
            SCENE_FILE = join(dirname(abspath(__file__)), './scenes/sawyer_reacher_rl_new_ik.ttt')
        elif control_mode == 'joint_velocity': # the scene with all joints in force/torche mode for forward kinematics
            SCENE_FILE = join(dirname(abspath(__file__)), './scenes/sawyer_reacher_rl_new.ttt')
        self.pr.launch(SCENE_FILE, headless=headless)
        self.pr.start()
        self.agent = Sawyer()
        self.gripper = BaxterGripper()
        self.gripper_left_pad = Shape('BaxterGripper_leftPad')  # the pad on the gripper finger
        self.proximity_sensor = ProximitySensor('BaxterGripper_attachProxSensor')  # need the name of the sensor here
        self.vision_sensor = VisionSensor('Vision_sensor')  # need the name of the sensor here
        if control_mode == 'end_position':
            self.agent.set_control_loop_enabled(True)  # if false, won't work
            self.action_space = np.zeros(4)  # 3 DOF end position control + 1 rotation of gripper
        elif control_mode == 'joint_velocity':
            self.agent.set_control_loop_enabled(False)
            self.action_space = np.zeros(8)  # 7 DOF velocity control + 1 rotation of gripper
        else:
            raise NotImplementedError
        if self.visual_control == False:
            self.observation_space = np.zeros(17)  # position and velocity of 7 joints + position of the target
        else:
            self.observation_space = np.zeros(100)  # dim of img!
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('target')  # object
        self.tip_target = Dummy('Sawyer_target')   # the target point of the tip to move towards
        # self.table = Shape('diningTable')
        self.agent_ee_tip = self.agent.get_tip()
        self.tip_pos = self.agent_ee_tip.get_position()
        self.tip_quat=self.agent_ee_tip.get_quaternion()  # tip rotation as quaternion, if not control the rotation
        
        # set a proper initial gesture/tip position
        if control_mode == 'end_position': 
            # agent_position=self.agent.get_position()
            # initial_pos_offset = [0.4, 0.3, 0.2]  # initial relative position of gripper to the whole arm
            # initial_pos = [(a + o) for (a, o) in zip(agent_position, initial_pos_offset)]
            initial_pos = [0.3, 0.1, 0.9]
            self.tip_target.set_position(initial_pos)
            # one big step for rotation setting is enough, with reset_dynamics=True, set the rotation instantaneously
            self.tip_target.set_orientation([0,3.1415,1.5707], reset_dynamics=True)  # make gripper face downwards
            self.pr.step()  
        elif control_mode == 'joint_velocity':
            self.initial_joint_positions = [0.001815199851989746, -1.4224984645843506, 0.704303503036499, 2.54307222366333, 2.972468852996826, -0.4989511966705322, 4.105560302734375]
            self.agent.set_joint_positions(self.initial_joint_positions)

        self.initial_joint_positions = self.agent.get_joint_positions()
        self.initial_tip_positions = self.agent_ee_tip.get_position()
        self.initial_target_positions = self.target.get_position()

    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        return np.array(self.agent.get_joint_positions() +
                self.agent.get_joint_velocities() +
                self.target.get_position())
    
    def _is_holding(self):
        '''
         Return is holding the target or not, return bool.
        '''

        # Nothe that the collision check is not always accurate all the time, 
        # for continuous conllision, maybe only the first 4-5 frames of collision can be detected
        pad_collide_object = self.gripper_left_pad.check_collision(self.target)
        if  pad_collide_object and self.proximity_sensor.is_detected(self.target)==True:
            return True 
        else:
            return False

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

    # def _move(self, action):
    #     ''' 
    #     Move the tip according to the action with inverse kinematics for 'end_position' control;
    #     with control of tip target in inverse kinematics mode instead of using .solve_ik() in forward kinematics mode.
    #     '''
    #     robot_moving_unit=0.01  # the amount of single step move of robot, not accurate
    #     moving_loop_itr=int(np.sum(np.abs(action[:3]))/robot_moving_unit)+1  # adaptive number of moving steps, with minimal of 1 step. 
    #     # print(moving_loop_itr)
    #     small_step = list(1./moving_loop_itr*np.array(action))  # break the action into small steps, as the robot cannot move to the target position within one frame
    #     pos=self.agent_ee_tip.get_position()

    #     ''' 
    #     there is a mismatch between the object set_orientation() and get_orientation():
    #     the (x,y,z) in set_orientation() will be (y,x,-z) in get_orientation().
    #     '''
    #     ori_z=-self.agent_ee_tip.get_orientation()[2] # the minus is because the mismatch between the set and get
    #     assert len(small_step) == len(pos)+1  # 3 values for position, 1 value for rotation

    #     for _ in range(moving_loop_itr):
    #         for idx in range(len(pos)):
    #             pos[idx] += small_step[idx]
    #         self.tip_target.set_position(pos)
    #         self.pr.step()
    #         ''' deprecated! no need to use small steps for the rotation with reset_dynamics=True'''
    #         # ori_z+=small_step[3]  # change the orientation along z-axis with a small step
    #         # self.tip_target.set_orientation([0,3.1415,ori_z], reset_dynamics=True)  # make gripper face downwards
    #         # self.pr.step()
    #     ''' one big step for z-rotation is enough, with reset_dynamics=True, set the rotation instantaneously '''
    #     ori_z+=action[3]
    #     self.tip_target.set_orientation([0,3.1415,ori_z], reset_dynamics=True)  # make gripper face downwards
    #     self.pr.step()

    def _move(self, action):
        ''' 
        Move the tip according to the action with inverse kinematics for 'end_position' control;
        with control of tip target in inverse kinematics mode instead of using .solve_ik() in forward kinematics mode.
        Mode 2: a close-loop control, using ik.
        '''
        pos=self.gripper.get_position()
        bounding_offset=0.1
        step_factor=0.2  # small step factor mulitplied on the gradient step calculated by inverse kinematics
        max_itr=20  # maximum moving iterations
        max_error=0.1  # upper bound of distance error for movement at each call
        rotation_norm =5. # factor for normalization of rotation values
        # check if state+action will be within of the bounding box, if so, move normally; else no action.
        #  x_min < x < x_max  and  y_min < y < y_max  and  z > z_min
        if pos[0]+action[0]>POS_MIN[0]-bounding_offset and pos[0]+action[0]<POS_MAX[0]+bounding_offset  \
            and pos[1]+action[1] > POS_MIN[1]-bounding_offset and pos[1]+action[1] < POS_MAX[1]+bounding_offset  \
            and pos[2]+action[2] > POS_MIN[2]-bounding_offset:

            ''' 
            there is a mismatch between the object set_orientation() and get_orientation():
            the (x,y,z) in set_orientation() will be (y,x,-z) in get_orientation().
            '''
            ori_z=-self.agent_ee_tip.get_orientation()[2] # the minus is because the mismatch between the set and get
            target_pos = np.array(self.agent_ee_tip.get_position())+np.array(action[:3])
            diff=1
            itr=0
            # print('before: ', ori_z)
            while np.sum(np.abs(diff))>max_error and itr<max_itr:
                itr+=1
                # set pos in small step
                cur_pos = self.agent_ee_tip.get_position()
                diff=target_pos-cur_pos
                pos = cur_pos+step_factor*diff
                self.tip_target.set_position(pos.tolist())
                self.pr.step()

            ''' one big step for z-rotation is enough '''
            ori_z+=rotation_norm*action[3]
            self.tip_target.set_orientation([0, np.pi, ori_z])  # make gripper face downwards
            self.pr.step()

        else:
            # print("Potential Movement Out of the Bounding Box!")
            pass # no action if potentially out of the bounding box

    def reset(self):
        # Get a random position within a cuboid and set the target position
        max_itr=10
        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.target.set_position(pos)
        self.target.set_orientation([0,0,0])
        self.pr.step()
        # changing the color or texture for domain randomization
        self.target.set_color(np.random.uniform(low=0, high=1, size=3).tolist()) # set [r,g,b] 3 channel values of object color
        # set end position to be initialized
        if self.control_mode == 'end_position':  # JointMode.IK
            self.agent.set_control_loop_enabled(True)
            self.tip_target.set_position(self.initial_tip_positions)  # cannot set joint positions directly due to in ik mode nor force/torch mode
            self.pr.step()
            # prevent stuck case
            itr=0
            while np.sum(np.abs(np.array(self.agent_ee_tip.get_position()-np.array(self.initial_tip_positions))))>0.1 and itr<max_itr:
                itr+=1
                self.step(np.random.uniform(-0.2,0.2,4))  # take random actions for preventing the stuck cases
                self.pr.step()
        elif self.control_mode == 'joint_velocity': # JointMode.FORCE
            self.agent.set_joint_positions(self.initial_joint_positions)  # sometimes the gripper is stuck, cannot get back to initial
            self.pr.step()
        
        # self.table.set_collidable(True)
        self.gripper_left_pad.set_collidable(True)  # set the pad on the gripper to be collidable, so as to check collision
        self.target.set_collidable(True)
        if np.sum(self.gripper.get_open_amount())<1.5:
            self.gripper.actuate(1, velocity=0.5)  # open the gripper
            self.pr.step()
        if self.visual_control:
            return self._get_visual_state()
        else:
            return self._get_state()

    def step(self, action):
        '''
        Move the robot arm according to the action.
        If control_mode=='joint_velocity', action is 7 dim of joint velocity values + 1 dim of gripper rotation.
        if control_mode=='end_position', action is 3 dim of tip (end of robot arm) position values + 1 dim of gripper rotation.
        '''
        if self.control_mode == 'end_position':
            if action is None or action.shape[0]!=4:
                action = list(np.random.uniform(-0.1, 0.1, 4))  # random
            self._move(action)
        elif self.control_mode == 'joint_velocity':
            if action is None or action.shape[0]!=7:
                print('No actions or wrong action dimensions!')
                action = list(np.random.uniform(-0.1, 0.1, 7))  # random
            self.agent.set_joint_target_velocities(action)  # Execute action on arm
            self.pr.step()
        else:
            raise NotImplementedError

        ax, ay, az = self.gripper.get_position()
        tx, ty, tz = self.target.get_position()
        # Reward is negative distance to target
        distance = (ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2
        done=False

        # print(self.proximity_sensor.is_detected(self.target))
        current_vision = self.vision_sensor.capture_rgb()  # capture a screenshot of the view with vision sensor
        plt.imshow(current_vision)
        plt.savefig('./img/vision.png')

        reward = 0
        # close the gripper if close enough to the object and the object is detected with the proximity sensor
        if distance<0.1 and self.proximity_sensor.is_detected(self.target)== True: 
            # make sure the gripper is open before grasping
            self.gripper.actuate(1, velocity=0.5)
            self.pr.step()
            
            self.gripper.actuate(0, velocity=0.5)  # if done, close the hand, 0 for close and 1 for open.
            self.pr.step()  # Step the physics simulation

            if self._is_holding():
                # reward for hold here!
                reward += 10
                done=True

            else:
                self.gripper.actuate(1, velocity=0.5)
                self.pr.step()

        elif np.sum(self.gripper.get_open_amount())<1.5: # if gripper is closed due to collision or esle, open it; .get_open_amount() return list of gripper joint values
            self.gripper.actuate(1, velocity=0.5)
            self.pr.step()

        else:
            pass


        reward -= np.sqrt(distance)

        if tz < self.initial_target_positions[2]-self.fall_down_offset:  # the object fall off the table
            done = True
            reward = -self.reward_offset

        if self.visual_control:
            return self._get_visual_state(), reward, done, {}
        else:
            return self._get_state(), reward, done, {}

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


if __name__ == '__main__':
    env=ReacherEnv(headless=False, control_mode='end_position', visual_control=False)
    env.reset()
    for step in range(1000):
        print(step)
        '''  
        4 dim control for 'end_position': 3 positions (-0.1~0.1) for arm and 1 rotation (-3.14~3.14, z-axis) for gripper
        7 dim control for 'joint_velocity': 7 velocities for arm 
        '''
        action=np.random.uniform(-0.1,0.1,4)  
        try:
            env.step(action)
        except KeyboardInterrupt:
            print('Shut Down!')
            env.shutdown()

    env.shutdown()