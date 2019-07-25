""" Env wrappers
Note that this file is adapted from `https://pypi.org/project/gym-vec-env` and
`https://github.com/openai/baselines/blob/master/baselines/common/*_wrappers.py`
"""
from multiprocessing import Process, Pipe

import gym
import numpy as np


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env._reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv(object):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.num_envs = len(env_fns)

        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        zipped_args = zip(self.work_remotes, self.remotes, env_fns)
        self.ps = [
            Process(target=_worker,
                    args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zipped_args
        ]

        for p in self.ps:
            # if the main process crashes, we should not cause things to hang
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

    def _step_async(self, actions):
        """
            Tell all the environments to start taking a step
            with the given actions.
            Call step_wait() to get the results of the step.
            You should not call this if a step_async run is
            already pending.
            """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def _step_wait(self):
        """
            Wait for the step taken with step_async().
            Returns (obs, rews, dones, infos):
             - obs: an array of observations, or a tuple of
                    arrays of observations.
             - rews: an array of rewards
             - dones: an array of "episode done" booleans
             - infos: a sequence of info objects
            """
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        """
            Reset all the environments and return an array of
            observations, or a tuple of observation arrays.
            If step_async is still doing work, that work will
            be cancelled and step_wait() should not be called
            until step_async() is invoked again.
            """
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def _reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def __len__(self):
        return self.nenvs

    def step(self, actions):
        self._step_async(actions)
        return self._step_wait()


class Monitor(gym.Wrapper):
    def __init__(self, env):
        super(Monitor, self).__init__(env)
        self._monitor_rewards = None

    def reset(self, **kwargs):
        self._monitor_rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        o_, r, done, info = self.env.step(action)
        self._monitor_rewards.append(r)
        if done:
            info['episode'] = {
                'r': sum(self._monitor_rewards),
                'l': len(self._monitor_rewards)}
        return o_, r, done, info


class TimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


# def make_env(max_steps, seed):
#     from osim.env import L2RunEnv  # load the env
#     env = L2RunEnv(visualize=False)
#     env.seed(seed)
#     return Monitor(TimeLimit(env, max_steps))

def make_env(max_steps, seed):
    from reacher_sawyer_env import ReacherEnv
    # from reacher_sawyer_visual_env import ReacherEnv
    env = ReacherEnv(headless=True, control_mode='end_position')
    return Monitor(TimeLimit(env, max_steps))

def make_vec_env(nenv, max_steps):
    """ Make vectorized env """
    from functools import partial
    env = SubprocVecEnv([partial(make_env, max_steps, i) for i in range(nenv)])
    return env


