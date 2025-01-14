import gym
import babyai_text
import babyai.utils as utils
from babyai.paral_env_simple import ParallelEnv
from macprotocol import MacProtocolEnv
class UDTSEnv:
    def __init__(self, config_dict):
        self.n_parallel = config_dict["number_envs"]
        # self._action_space = [a.replace("_", " ") for a in config_dict["action_space"]]
        envs = []
        for i in range(config_dict["number_envs"]):
            # env = gym.make(config_dict["task"])
            env = MacProtocolEnv(config_dict["ue_num"][i])
            env.is_training = False if config_dict["test"] else True
            env.seed(100 * config_dict["seed"] + i)
            envs.append(env)

        self._env = ParallelEnv(envs)
    def reset(self):
        obs, infos = self._env.reset()
        return obs, infos
    def step(self, actions_id):
        obs, rews, dones, infos = self._env.step(actions_id)
        return  obs,rews, dones, infos
    # def get_statistical_info(self):
    #     goodput = [e.get_Goodput() for e in self._env.envs]
    #     colli_rate = [e.get_collision_rate() for e in self._env.envs]
    #     arri_rate = [e.get_packet_arrival_rate() for e in self._env.envs]
    #     return goodput, colli_rate, arri_rate

if __name__ == "__main__":
    import numpy as np
    config_dict = {
        "task": "Debug",
        "number_envs": 2,
        "seed": 12,
        "ue_num": [2,3]
    }
    envs = UDTSEnv(config_dict)
    obs, infos = envs.reset()

    while True:
        o,r,done,infos = envs.step([[np.random.randint(0, 3, 2),np.random.randint(0, 1, 2),np.random.randint(0, 1, 2)], [np.random.randint(0, 3, 3),np.random.randint(0, 1, 3),np.random.randint(0, 1, 3)]])
        print("observation:{}".format(o))
        # print("info_ue:",info)
        print("reward:{}".format(r))
        if all(done):
            print(f"env0--goodput:{infos[0]['goodput']}, colli_rate:{infos[0]['colli_num']}, arri_rate:{infos[0]['arri_num']}")
            print(f"env1--goodput:{infos[1]['goodput']}, colli_rate:{infos[1]['colli_num']}, arri_rate:{infos[1]['arri_num']}")
            break

    