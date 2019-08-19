# Robot_Learning2

Sawyer robot learning to reach the target (Reacher) controlled with PPO algorithm, using PyRep for Sawyer robot simulation and game building. The environment is wrapped into Openai Gym format in `./reacher_sawyer_env.py`.


## To Run：
* First check the environment can run successfully:

    `python reacher_sawyer_env_boundingbox.py`

    If it works properly with VRep called to run a scene, then go to next step; otherwise check the `requirements.txt` for necessary packages and versions.

* Run a single-process PPO for testing: 

  `python ppo_single.py`
  
* Run multi-process SAC for distributed training:

  `python sac_multi.py --train` for non-vectorized environment;

  `python sac_multi_vec2.py --train` for vectorized environment (acceleration).
