""" Env wrappers
Note that this file is adapted from `https://pypi.org/project/gym-vec-env` and
`https://github.com/openai/baselines/blob/master/baselines/common/*_wrappers.py`
"""
from multiprocessing import Process, Queue
import os
import gym
import numpy as np


def _worker(exclusive_queue, shared_queue, env_fn_wrapper):
    env = env_fn_wrapper.x()
    pid = os.getpid()
    while True:
        cmd, data = exclusive_queue.get()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            shared_queue.put(((ob, reward, done, info), pid))
        elif cmd == 'reset':
            ob = env.reset()
            shared_queue.put(((ob, 0, False, {}), pid))
        elif cmd == 'close':
            exclusive_queue.close()
            break
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
    def __init__(self, env_fns, menv):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.menv = menv  # env number in send buffer
        self.num_envs = len(env_fns)  # all env in sample buffer

        self.closed = False
        nenvs = len(env_fns)
        self.nenvs = nenvs
        env_queues = [Queue() for _ in range(nenvs)]
        self.shared_queue = Queue()
        self.ps = [
            Process(target=_worker,
                    args=(env_queues[i], self.shared_queue, CloudpickleWrapper(env_fns[i])))
            for i in range(nenvs)
        ]

        for p in self.ps:
            # if the main process crashes, we should not cause things to hang
            p.daemon = True
            p.start()

        self.env_queues = dict()
        for p, queue in zip(self.ps, env_queues):
            self.env_queues[p.pid] = queue
        self.current_pids = None

    def _step_async(self, actions):
        """
            Tell all the environments to start taking a step
            with the given actions.
            Call step_wait() to get the results of the step.
            You should not call this if a step_async run is
            already pending.
            """
        for pid, action in zip(self.current_pids, actions):
            self.env_queues[pid].put(('step', action))

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
        results = []
        self.current_pids = []
        while len(results) < self.menv:
            data, pid = self.shared_queue.get()
            results.append(data)
            self.current_pids.append(pid)
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
        for queue in self.env_queues.values():  # initialize all
            queue.put(('reset', None))
        results = []
        self.current_pids = []
        while len(results) < self.menv:
            data, pid = self.shared_queue.get()
            results.append(data[0])
            self.current_pids.append(pid)
        return np.stack(results)

    def close(self):
        if self.closed:
            return
        for queue in self.env_queues.values():
            queue.put(('close', None))
        self.shared_queue.close()
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


def make_env(max_steps, seed):
    # from reacher_sawyer_env import ReacherEnv
    from reacher_sawyer_env_boundingbox import ReacherEnv
    # from reacher_sawyer_visual_env import ReacherEnv
    env = ReacherEnv(headless=True, control_mode='end_position')
    return Monitor(TimeLimit(env, max_steps))

def make_vec_env(nenv, max_steps, menv):
    """ Make vectorized env """
    from functools import partial
    env = SubprocVecEnv([partial(make_env, max_steps, i) for i in range(nenv)], menv)
    return env


