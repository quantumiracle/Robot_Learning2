"""
The environment of Sawyer Arm + Baxter Gripper for graping object.
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

    def __init__(self, headless, control_mode='end_position'):
        '''
        :headless: bool, if True, no visualization, else with visualization.
        :control mode: str, 'end_position' or 'joint_velocity'.
        '''
        self.reward_offset=10.0  # reward of achieving the grasping object
        self.metadata=[]  # gym env format
        self.control_mode = control_mode  # the control mode of robotic arm: 'end_position' or 'joint_velocity'
        self.pr = PyRep()
        if control_mode == 'end_position':  # need to use different scene, the one with all joints in inverse kinematics mode
            SCENE_FILE = join(dirname(abspath(__file__)), './scenes/sawyer_reacher_rl_new_ik.ttt')
        elif control_mode == 'joint_velocity': # the scene with all joints in force/torche mode for forward kinematics
            SCENE_FILE = join(dirname(abspath(__file__)), './scenes/sawyer_reacher_rl_new.ttt')
        self.pr.launch(SCENE_FILE, headless=headless)  # lunch the scene, headless means no visualization
        self.pr.start()       # start the scene
        self.agent = Sawyer()  # get the robot arm in the scene
        self.gripper = BaxterGripper()  # get the gripper in the scene
        self.gripper_left_pad = Shape('BaxterGripper_leftPad')  # the pad on the gripper finger
        self.proximity_sensor = ProximitySensor('BaxterGripper_attachProxSensor')  # need the name of the sensor here
        self.vision_sensor = VisionSensor('Vision_sensor')  # need the name of the sensor here
        if control_mode == 'end_position':  # control the robot arm by the position of its end using inverse kinematics
            self.agent.set_control_loop_enabled(True)  # if false, won't work
            self.action_space = np.zeros(4)  # 3 DOF end position control + 1 rotation of gripper
        elif control_mode == 'joint_velocity':  # control the robot arm by directly setting velocity values on each joint, forward kinematics
            self.agent.set_control_loop_enabled(False)
            self.action_space = np.zeros(8)  # 7 DOF velocity control + 1 rotation of gripper
        else:
            raise NotImplementedError
        self.observation_space = np.zeros(17)  # position and velocity of 7 joints + position of the target
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('target')  # get the target object
        self.agent_ee_tip = self.agent.get_tip()  # a part of robot as the end of inverse kinematics chain for controlling
        self.tip_target = Dummy('Sawyer_target')   # the target point of the tip (end of the robot arm) to move towards
        self.tip_pos = self.agent_ee_tip.get_position()  # tip x,y,z position
        self.tip_quat=self.agent_ee_tip.get_quaternion()  # tip rotation as quaternion, if not control the rotation
        
        # set a proper initial gesture/tip position
        agent_position=self.agent.get_position()
        initial_pos_offset = [0.6, 0.2, 0.4]  # initial relative position of gripper to the whole arm
        initial_pos = [(a + o) for (a, o) in zip(agent_position, initial_pos_offset)]
        self.tip_target.set_position(initial_pos)
        self.tip_target.set_orientation([0,3.1415,1.5708])  # make gripper face downwards
        self.pr.step()
        self.initial_joint_positions = self.agent.get_joint_positions()
        self.initial_gripper_positions = self.gripper.get_position()

    def _get_state(self):
        '''
         Return state containing arm joint angles/velocities & target position.
         '''
        return np.array(self.agent.get_joint_positions() +  # list
                self.agent.get_joint_velocities() +  # list
                self.target.get_position())  # list

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

    # def _move(self, action):
    #     # Deprecated, always fail the ik! Move the tip according to the action with inverse kinematics for 'end_position' control.
    #     action=list(action)
    #     pos=self.agent_ee_tip.get_position()
    #     assert len(action) == len(pos)
    #     for idx in range(len(pos)):
    #         pos[idx] += action[idx]
    #     print('pos: ', pos)
    #     new_joint_angles = self.agent.solve_ik(pos, quaternion=self.tip_quat)
    #     self.agent.set_joint_target_positions(new_joint_angles)

    def _move(self, action):
        ''' 
        Move the tip according to the action with inverse kinematics for 'end_position' control;
        with control of tip target in inverse kinematics mode instead of using .solve_ik() in forward kinematics mode.
        '''
        robot_moving_unit=0.01  # the amount of single step move of robot, not accurate; the smaller the value, the smoother the movement.
        moving_loop_itr=int(np.sum(np.abs(action[:3]))/robot_moving_unit)+1  # adaptive number of moving steps, with minimal of 1 step; the larger it is, the more accurate for each movement.
        small_step = list(1./moving_loop_itr*np.array(action))  # break the action into small steps, as the robot cannot move to the target position within one frame
        pos=self.agent_ee_tip.get_position()

        ''' 
        there is a mismatch between the object set_orientation() and get_orientation():
        the (x,y,z) in set_orientation() will be (y,x,-z) in get_orientation().
        '''
        ori_z=-self.agent_ee_tip.get_orientation()[2] # the minus is because the mismatch between the set and get
        assert len(small_step) == len(pos)+1  # 3 values for position, 1 value for rotation

        # print('before: ',self.agent_ee_tip.get_position())
        for _ in range(moving_loop_itr):
            for idx in range(len(pos)):
                pos[idx] += small_step[idx]
            self.tip_target.set_position(pos)
            self.pr.step()
            ori_z+=small_step[3]  # change the orientation along z-axis with a small step
            self.tip_target.set_orientation([0,3.1415,ori_z])  # make gripper face downwards
            self.pr.step()
        # print(self.tip_target.get_orientation())
        # print(self.agent_ee_tip.get_orientation())
        # print('target: ', pos)
        # print('after: ', self.agent_ee_tip.get_position())

    def reset(self):
        '''
         Get a random position within a cuboid and set the target position.
         '''
        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.target.set_position(pos)
        self.agent.set_joint_positions(self.initial_joint_positions)
        # set collidable, for collision detection
        self.gripper_left_pad.set_collidable(True)  # set the pad on the gripper to be collidable, so as to check collision
        self.target.set_collidable(True)
        return self._get_state()

    def step(self, action):
        '''
        Move the robot arm according to the action.
        If control_mode=='joint_velocity', action is 7 dim of joint velocity values;
        if control_mode=='end_position', action is 3 dim of tip (end of robot arm) position values.
        '''
        if self.control_mode == 'end_position':
            if action is None or action.shape[0]!=4:
                action = list(np.random.uniform(-0.1, 0.1, 4))  # random
            # print(action, self.agent_ee_tip.get_position())
            self._move(action)

        elif self.control_mode == 'joint_velocity':
            self.agent.set_joint_target_velocities(action)  # Execute action on arm
            self.pr.step()
            ori_z=-self.agent_ee_tip.get_orientation()[2] # the minus is because the mismatch between the set and get
            ori_z+=action[7]  # change the orientation along z-axis
            self.tip_target.set_orientation([0,3.1415,ori_z])  # change orientation
            self.pr.step()

        else:
            raise NotImplementedError

        ax, ay, az = self.gripper.get_position()
        tx, ty, tz = self.target.get_position()

        distance = (ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2  # distance between the gripper and the target object
        done=False
        
        current_vision = self.vision_sensor.capture_rgb()  # capture a screenshot of the view with vision sensor
        plt.imshow(current_vision)
        plt.savefig('./img/vision.png')
        
        reward=0
        # close the gripper if close enough to the object and the object is detected with the proximity sensor
        if distance<0.1 and self.proximity_sensor.is_detected(self.target)== True: 
            # make sure the gripper is open before grasping
            self.gripper.actuate(1, velocity=0.5)
            self.pr.step()

            self.gripper.actuate(0, velocity=0.5)  # if done, close the hand, 0 for close and 1 for open; velocity 0.5 ensures the gripper to close with in one frame
            self.pr.step()  # Step the physics simulation

            if self._is_holding():
                # reward for hold here!
                reward += self.reward_offset  # extra reward for grasping the object
                done=True
            else:
                self.gripper.actuate(1, velocity=0.5)
                self.pr.step()

        elif np.sum(self.gripper.get_open_amount())<1.5: # if gripper is closed (not fully open) due to collision or esle, open it; .get_open_amount() return list of gripper joint values
            self.gripper.actuate(1, velocity=0.5)
            self.pr.step()

        else:
            pass

        reward -= np.sqrt(distance) # Reward is negative distance to target
        return self._get_state(), reward, done, {}

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()

if __name__ == '__main__':
    env=ReacherEnv(headless=False, control_mode='end_position')
    env.reset()
    for step in range(1000):
        print(step)
        action=np.random.uniform(-0.1,0.1,4)  #  4 dim control for 'end_position': 3 positions and 1 rotation (z-axis)
        try:
            env.step(action)
        except KeyboardInterrupt:
            print('Shut Down!')
            env.shutdown()

    env.shutdown()
