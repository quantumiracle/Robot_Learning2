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
from pyrep.objects.shape import Shape
import numpy as np

SCENE_FILE = join(dirname(abspath(__file__)), './scenes/sawyer_reacher_rl.ttt')
POS_MIN, POS_MAX = [0.2, 0.2, 0.8], [0.6, 0.6, 0.8]
EPISODES = 5
EPISODE_LENGTH = 200


class ReacherEnv(object):

    def __init__(self):
        self.pr = PyRep()
        self.pr.launch(SCENE_FILE, headless=False)
        self.pr.start()
        # self.agent = Panda()
        self.agent = Sawyer()
        self.gripper = BaxterGripper()
        self.agent.set_control_loop_enabled(False)
        self.agent.set_motor_locked_at_zero_velocity(True)
        self.target = Shape('target')
        self.agent_ee_tip = self.agent.get_tip()
        self.initial_joint_positions = self.agent.get_joint_positions()


    def _get_state(self):
        # Return state containing arm joint angles/velocities & target position
        return (self.agent.get_joint_positions() +
                self.agent.get_joint_velocities() +
                self.target.get_position())

    def reset(self):
        # Get a random position within a cuboid and set the target position
        pos = list(np.random.uniform(POS_MIN, POS_MAX))
        self.target.set_position(pos)
        self.agent.set_joint_positions(self.initial_joint_positions)
        self.pr.step()
        return self._get_state()

    def step(self, action):
        self.agent.set_joint_target_velocities(action)  # Execute action on arm
        self.pr.step()  # Step the physics simulation
        ax, ay, az = self.agent_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()
        # Reward is negative distance to target
        distance = (ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2
        done=False
        if distance<5:
            done=True
            self.gripper.actuate(0, velocity=0.04)  # if done, close the hand, 0 for close and 1 for open.
            self.pr.step()  # Step the physics simulation

        reward = -np.sqrt(distance)
        return self._get_state(), reward, done

    def shutdown(self):
        self.pr.stop()
        self.pr.shutdown()


class Agent(object):

    def act(self, state):
        del state
        return list(np.random.uniform(-1.0, 1.0, size=(7,)))

    def learn(self, replay_buffer):
        del replay_buffer
        pass


env = ReacherEnv()
agent = Agent()
replay_buffer = []

for e in range(EPISODES):

    print('Starting episode %d' % e)
    state = env.reset()
    for i in range(EPISODE_LENGTH):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        replay_buffer.append((state, action, reward, next_state))
        state = next_state
        agent.learn(replay_buffer)

print('Done!')
env.shutdown()
