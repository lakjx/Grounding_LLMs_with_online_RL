import argparse
from html import parser
import os
import itertools
import copy
import numpy as np
import torch
import gym
from gym import spaces

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# sys.path.append(parent_dir)
class DotDic(dict):
	__getattr__ = dict.get
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__

	def __deepcopy__(self, memo=None):
		return DotDic(copy.deepcopy(dict(self), memo=memo))
class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """

    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        #random_array = prng.np_random.rand(self.num_discrete_space)
        random_array = np.random.rand(self.num_discrete_space)

        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)

class MacProtocolEnv():
    def __init__(self, ue_num, discrete_action=True):
        # self.args = args
        self.env_name = 'MacPro_Env'
        self.is_training = True
        self.rho = 1.5
        self.UE_num = ue_num
        self.p_SDU_arrival = 0.5
        self.tbl_error_rate = 1e-3
        self.TTLs = 24
        self.UE_txbuff_len = 20
        self.recent_k = 0
        self.need_comm = True

        self.collision_count = 0
        self.gen_data_count = 0
        self.UE_act_space = DotDic({
            'Do Nothing': 0,
            'Transmit': 1,
            'Delete': 2
        })
        # UE_obs \in [0,|B|]
        self.UE_obs_space = spaces.Discrete(self.UE_txbuff_len + 1)
        # BS_obs \in [0,|U|+1]
        self.BS_obs_space = spaces.Discrete(self.UE_num + 2)

        self.BS_msg_space = DotDic({
            'Null': 0,
            'SG': 1,
            'ACK': 2
        })
        self.BS_msg_total_space = list(itertools.product(range(len(self.BS_msg_space)), repeat=self.UE_num))
        self.UE_msg_space = DotDic({
            'Null': 0,
            'SR': 1
        })

        self.agents = ['UE_' + str(i) for i in range(self.UE_num)] + ['BS']
        self.num_agents = len(self.agents)
        self.discrete_action_space = discrete_action
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for agent in self.agents:
            total_action_space = []
            #physical action
            u_action_space = spaces.Discrete(len(self.UE_act_space))
            if agent != 'BS':
                total_action_space.append(u_action_space)
            # elif agent == 'BS' and self.args.need_comm == False:
            #     total_action_space.append(spaces.Discrete(2))
            #communication action
            if self.need_comm:
                if agent != 'BS':
                    c_action_space = spaces.Discrete(len(self.UE_msg_space))
                    total_action_space.append(c_action_space)
                else:
                    c_action_space = spaces.Discrete(len(self.BS_msg_space)**self.UE_num)
                    total_action_space.append(c_action_space)
            
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    action_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    raise NotImplementedError
                self.action_space.append(action_space)
            elif len(total_action_space) == 1:
                self.action_space.append(total_action_space[0])
            else:
                self.action_space.append([])
            # observation space
            if agent != 'BS':
                obs_dim = 4*(self.recent_k+1) if self.need_comm else 2*(self.recent_k+1)
                self.observation_space.append(spaces.Discrete(obs_dim))
            else:
                obs_dim = self.recent_k+1 + self.UE_num*2*(self.recent_k+1) if self.need_comm else self.recent_k+1
                self.observation_space.append(spaces.Discrete(obs_dim))
            share_obs_dim += obs_dim
        self.share_observation_space = [spaces.Discrete(share_obs_dim)] * self.num_agents

        self.reset()

    def reset(self):
        self.step_count = 0
        self.collision_count = 0
        self.gen_data_count = 0

        self.UEs = [UE(i,self.UE_txbuff_len) for i in range(self.UE_num)]

        # # self.UE_SDU_Generate()
        self.UE_obs = np.zeros((self.UE_num,), dtype=np.int32)
        self.UE_actions = np.zeros((self.UE_num,), dtype=np.int32)
        self.BS_obs = np.zeros((1,), dtype=np.int32)
        self.BS_msg = np.zeros((self.UE_num,), dtype=np.int32)
        self.UE_msg = np.zeros((self.UE_num,), dtype=np.int32)

        self.trajact_UE_obs = [copy.deepcopy(self.UE_obs) for _ in range(self.recent_k + 1)]
        self.trajact_UE_actions = [copy.deepcopy(self.UE_actions) for _ in range(self.recent_k + 1)]
        self.trajact_BS_obs = [copy.deepcopy(self.BS_obs) for _ in range(self.recent_k + 1)]
        self.trajact_BS_msg = [copy.deepcopy(self.BS_msg) for _ in range(self.recent_k + 1)]
        self.trajact_UE_msg = [copy.deepcopy(self.UE_msg) for _ in range(self.recent_k + 1)]

        
        self.sdus_received = []
        self.data_channel = []
        self.rewards = 0
        self.done = False
        # record observations for each agent
        obs_n = []
        info_n = {}
        # for agent in self.agents:
        #     if agent == 'BS':
        #         obs_n.append(self.get_bs_internal_stat())
        #     else:
        #         tar_ue_idx = int(agent.split('_')[1])
        #         obs_n.append(self.get_ue_internal_stat(tar_ue_idx))
        # return obs_n
        for agent in self.agents:
            if agent == 'BS':
                bs_obs = self.get_bs_internal_stat()
                obs_n.append(bs_obs)
                # info_n["BS_des"] = self.BS_Prompt(bs_obs)
                info_n["BS_des"] = str(bs_obs)
            else:
                tar_ue_idx = int(agent.split('_')[1])
                ue_obs = self.get_ue_internal_stat(tar_ue_idx)
                obs_n.append(ue_obs)
                # info_n["UE{}_des".format(tar_ue_idx)] = self.UE_Prompt(tar_ue_idx,ue_obs)
                info_n["UE{}_des".format(tar_ue_idx)] = str(ue_obs)
        info_n["num_UE"] = self.UE_num
        return obs_n, info_n

    def step(self, action_n,UCM=None,DCM=None):
        # UE_actions = [act[0] for (i, act) in enumerate(action_n) if self.agents[i] != 'BS']
        # UCM = [act[1] for (i, act) in enumerate(action_n) if self.agents[i] != 'BS'] if self.args.need_comm else None
        # DCM = self.BS_msg_total_space[action_n[-1][0]] if self.args.need_comm else None
        # UE_actions = action_n
        UE_actions , UCM, DCM = action_n[0], action_n[1], action_n[2]
        #测试状态下，打印每个UE的buffer状态
        if not self.is_training:
            for UE in self.UEs:
                print(UE.name,UE.buff)         

        if isinstance(UE_actions, list):
            UE_actions = np.array(UE_actions)
        elif isinstance(UE_actions, torch.Tensor):
            UE_actions = UE_actions.cpu().numpy()
        
        #随机生成UE的SDU
        new_data_list = self.UE_SDU_Generate()
        print('new_data_list:',new_data_list) if not self.is_training else None

        self.UE_actions = UE_actions
        self.UE_Signaling_policy(np.array(UCM)) if UCM is not None else self.UE_Signaling_policy()
        self.BS_Signaling_policy(np.array(DCM)) if DCM is not None else self.BS_Signaling_policy()
        error_del = 0
        self.data_channel = []
        for UE in self.UEs:
            if len(UE.buff) > 0:
                if UE_actions[UE.name_id] == self.UE_act_space.Transmit and UE.buff[0] != new_data_list[UE.name_id]:
                    data = UE.transmit_SDU()
                    self.data_channel.append(data)
                
                elif UE_actions[UE.name_id] == self.UE_act_space.Delete and UE.buff[0] != new_data_list[UE.name_id]:
                    del_data = UE.delete_SDU()
                    if del_data not in self.sdus_received:
                        error_del = error_del + 1
                    else:
                        error_del = error_del - 1
            else:
                pass
        self.check_channel(error_del)                 
    
        self.trajact_UE_obs.append(copy.deepcopy(np.array([UE.get_obs() for UE in self.UEs])))
        self.trajact_UE_actions.append(copy.deepcopy(self.UE_actions))
        self.trajact_BS_obs.append(copy.deepcopy(self.BS_obs) if isinstance(self.BS_obs, np.ndarray) else np.array([self.BS_obs]))
        self.trajact_BS_msg.append(copy.deepcopy(self.BS_msg))
        self.trajact_UE_msg.append(copy.deepcopy(self.UE_msg))
        if len(self.trajact_UE_obs) > self.recent_k+1:
            self.trajact_UE_obs.pop(0)
            self.trajact_UE_actions.pop(0)
            self.trajact_BS_obs.pop(0)
            self.trajact_BS_msg.pop(0)
            self.trajact_UE_msg.pop(0)

        self.done = self.step_count >= self.TTLs
        self.step_count += 1
        if not self.is_training:
            print('step:',self.step_count,'UE_act:',UE_actions,'datachannel:',self.data_channel,'rewards:',self.rewards)
            print('BS_recieved:',self.sdus_received)
        
        obs_n, reward_n, done_n, info_n = [], [], [], {}
        for agent in self.agents:
            if agent == 'BS':
                bs_obs = self.get_bs_internal_stat()
                obs_n.append(bs_obs)
                # info_n["BS_des"] = self.BS_Prompt(bs_obs)
                info_n["BS_des"] = str(bs_obs)
            else:
                tar_ue_idx = int(agent.split('_')[1])
                ue_obs = self.get_ue_internal_stat(tar_ue_idx)
                obs_n.append(ue_obs)
                # info_n["UE{}_des".format(tar_ue_idx)] = self.UE_Prompt(tar_ue_idx,ue_obs)
                info_n["UE{}_des".format(tar_ue_idx)] = str(ue_obs)
            # reward_n.append(self.get_rwd())
            # done_n.append(self.done)
        info_n["num_UE"] = self.UE_num
        info_n["goodput"] = self.get_Goodput()
        info_n["colli_num"] = self.get_collision_rate()
        info_n["arri_num"] = self.get_packet_arrival_rate()
        return obs_n, self.get_rwd(), self.done, info_n
    def UE_Prompt(self,idx,observ):
        act_mapping = {0:'nothing',1:'transmit',2:'delete'}
        uplink_signal_mapping = {0:'null',1:'schedule_request'}
        downlink_signal_mapping = {0:'null',1:'schedule_grant',2:'acknowledge'}
        # uplink_signal_mapping = {0:'Msg_#0',1:'Msg_#1',2:'Msg_#2',3:'Msg_#3'}
        # downlink_signal_mapping = {0:'Msg_#0',1:'Msg_#1',2:'Msg_#2',3:'Msg_#3', 4:'Msg_#4',5:'Msg_#5'}
        identify = f"You are the {idx}-th UE in wireless network."
        goal = "Goal: Decide buffer management action and uplink signaling based on observations." 
        if observ[1] == None or observ[2] == None or observ[3] == None:
            print(f"UE {idx} observations is not valid, {observ}")
            raise ValueError("Action is not valid.")
        obs = f"Observations: You have {observ[0]} SDUs in buffer, You select {act_mapping.get(observ[1])} last time, You send {uplink_signal_mapping.get(observ[2])} last time,  You receive {downlink_signal_mapping.get(observ[3])} last time."
        act_candidate = "Action candidates: nothing, transmit, delete."
        
        ue_signal_candidate = "Signaling candidates: " + ", ".join(uplink_signal_mapping.values()) + "."
        prompt = "\n".join([identify,goal,obs,act_candidate,ue_signal_candidate]) + "\nAction and signaling:"
        return prompt
    def BS_Prompt(self,observ):
        ucm_msg = observ[1:1+self.UE_num]
        dcm_msg = observ[-self.UE_num:]
        uplink_signal_mapping = {0:'null',1:'schedule_request'}
        downlink_signal_mapping = {0:'null',1:'schedule_grant',2:'acknowledge'}
        # uplink_signal_mapping = {0:'Msg_#0',1:'Msg_#1',2:'Msg_#2',3:'Msg_#3'}
        # downlink_signal_mapping = {0:'Msg_#0',1:'Msg_#1',2:'Msg_#2',3:'Msg_#3', 4:'Msg_#4',5:'Msg_#5'}

        identify = "You are BS in wireless network."
        goal = [f"Goal: Decide downlink signaling to {i}-th UE based on observations." for i in range(self.UE_num)]
        if observ[0] != 0 and observ[0] != self.UE_num+1:
            channel_stat = f"You see {observ[0]-1}-th UE is transmitting data,"
        elif observ[0] == 0:
            channel_stat = "You see channel is idle,"
        else:
            channel_stat = "You see channel is congested,"
        ucmdcm_stat = ""
        for i in range(self.UE_num):
            if downlink_signal_mapping.get(dcm_msg[i]) == None or uplink_signal_mapping.get(ucm_msg[i]) == None:
                print(f"Error: {i}-th UE's downlink signaling is {dcm_msg[i]} and uplink signaling is {ucm_msg[i]}.")
                raise ValueError("Downlink signaling is not valid.")
            ucmdcm_stat += f"For {i}-th UE, you receive {uplink_signal_mapping.get(ucm_msg[i])} and send {downlink_signal_mapping.get(dcm_msg[i])} last time, "

        bs_signal_candidate = "Signaling candidates: "+ ", ".join(downlink_signal_mapping.values()) + "."
        obs = f"Observations: {channel_stat} {ucmdcm_stat}"
        prompt = []
        for i in range(self.UE_num):
            prompt.append("\n".join([identify,goal[i],obs,bs_signal_candidate]) + "\nAction and signaling:")
        return prompt
       
    def get_ue_internal_stat(self,tar_ue_idx):
        # x =(o,a,n,m) o: UE_obs, a: UE_actions, n: UE_msg, m: BS_msg
        # o = np.transpose([UE.get_obs() for UE in self.UEs])
        # o = UE.get_obs()
        # a = self.UE_actions[UE.name_id]
        # n = self.UE_msg[UE.name_id]
        # m = self.BS_msg[UE.name_id]
        # return np.array([o,a,n,m])
        if len(self.trajact_UE_obs) < self.recent_k + 1:
            # 填充缺失的轨迹数据，使用最近的观测值
            gap = self.recent_k + 1 - len(self.trajact_UE_obs)
            o = [self.trajact_UE_obs[0][tar_ue_idx]] * gap + [self.trajact_UE_obs[i][tar_ue_idx] for i in range(len(self.trajact_UE_obs))]
            a = [self.trajact_UE_actions[0][tar_ue_idx]] * gap + [self.trajact_UE_actions[i][tar_ue_idx] for i in range(len(self.trajact_UE_actions))]
            n = [self.trajact_UE_msg[0][tar_ue_idx]] * gap + [self.trajact_UE_msg[i][tar_ue_idx] for i in range(len(self.trajact_UE_msg))]
            m = [self.trajact_BS_msg[0][tar_ue_idx]] * gap + [self.trajact_BS_msg[i][tar_ue_idx] for i in range(len(self.trajact_BS_msg))]
        else:
            o = [self.trajact_UE_obs[i][tar_ue_idx] for i in range(self.recent_k + 1)]
            a = [self.trajact_UE_actions[i][tar_ue_idx] for i in range(self.recent_k + 1)]
            n = [self.trajact_UE_msg[i][tar_ue_idx] for i in range(self.recent_k + 1)]
            m = [self.trajact_BS_msg[i][tar_ue_idx] for i in range(self.recent_k + 1)]

        if self.need_comm:
            return np.concatenate((o, a, n, m), axis=0).flatten()
        else:
            return np.concatenate((o, a), axis=0).flatten()
            
            

    def get_bs_internal_stat(self):
        #x=(o_b,n_all,m_all)
        #检查BS_obs的数据类型
        assert self.BS_obs_space.contains(self.BS_obs[0] if isinstance(self.BS_obs,np.ndarray) else self.BS_obs)
        
        if len(self.trajact_BS_obs) < self.recent_k + 1:
            # 填充缺失的轨迹数据，使用最近的观测值
            gap = self.recent_k + 1 - len(self.trajact_BS_obs)
            self.trajact_BS_obs = [self.trajact_BS_obs[0]] * gap + self.trajact_BS_obs
            self.trajact_BS_msg = [self.trajact_BS_msg[0]] * gap + self.trajact_BS_msg
            self.trajact_UE_msg = [self.trajact_UE_msg[0]] * gap + self.trajact_UE_msg

        if self.need_comm:
            return np.concatenate((self.trajact_BS_obs, self.trajact_UE_msg, self.trajact_BS_msg), axis=1).flatten()
        else:
            return np.array(self.trajact_BS_obs).flatten()
    
    def get_rwd(self):
        return self.rewards
    
    def check_channel(self,error_del):
        # rho = 3
        #查看data通道上是否有冲突  e.g. data_channel = ['UE0_1', 'UE1_0', 'UE2_1']
        if len(self.data_channel) == 1: # 正常数传
            data = self.data_channel[0]
            self.BS_obs = int(data.split('_')[0][2:])+1
            if np.random.rand() > self.tbl_error_rate: #正确接收
                if data not in self.sdus_received:
                    self.sdus_received.append(data)
                    self.rewards = 3*self.rho
                else:
                    # self.rewards = -self.rho
                    self.rewards = 0
        elif self.data_channel == []: # 空闲
            self.BS_obs = 0
            self.rewards = -self.rho
        else:
            self.collision_count += 1
            self.BS_obs = self.UE_num + 1
            self.rewards = -self.rho

        self.rewards = self.rewards - error_del*self.rho
        #判断BS_obs是否合法
        assert self.BS_obs_space.contains(self.BS_obs)
                                   
    def UE_SDU_Generate(self):
        gen_data_list = []
        for UE in self.UEs:
            if np.random.rand() < self.p_SDU_arrival:
                cur_gen_data = UE.generate_SDU()
                self.gen_data_count += 1
            else:
                cur_gen_data = None
            gen_data_list.append(cur_gen_data)
        return gen_data_list

    def BS_Signaling_policy(self,DCM=None):
        # BS can send one control message to each UE
        if DCM is None:
            DCM = np.random.randint(0, len(self.BS_msg_space), self.UE_num)
            self.BS_msg = DCM
        else:
            self.BS_msg = DCM
    def UE_Signaling_policy(self,UCM=None):
        # each UE can send one control message to BS
        if UCM is None:
            UCM = np.random.randint(0, len(self.UE_msg_space), self.UE_num)
            self.UE_msg = UCM
        else:
            self.UE_msg = UCM
    
    def get_Goodput(self):
        return len(self.sdus_received)
    def get_collision_rate(self):
        return self.collision_count 
    def get_buffer_occupancy(self):
        return [UE.get_obs()/UE.buff_size for UE in self.UEs]
    def get_packet_arrival_rate(self):
        return len(self.sdus_received)
    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

class UE():
    def __init__(self, name_id , UE_txbuff_len):
        self.name_id = name_id
        self.name = 'UE' + str(name_id)
        self.buff_size = UE_txbuff_len
        self.buff = []
        self.datacount = 0
        self.SG = False
        self.ACK = False

    def generate_SDU(self):
        gen_data = None
        if len(self.buff) < self.buff_size:
            gen_data = self.name + '_' + str(self.datacount)
            self.buff.append(gen_data)
            self.datacount += 1
        return gen_data
    
    def delete_SDU(self):
        if len(self.buff) > 0:
            del_data = self.buff.pop(0)
            return del_data
        else:
            print('Delete_SDU error!'+ self.name + ' buffer is empty!')
            return None      
    
    def transmit_SDU(self):
        if len(self.buff) > 0:
            return self.buff[0]
        else:
            print('Transmit_SDU error!'+ self.name + ' buffer is empty!')
            return None
    
    def get_obs(self):
        return len(self.buff)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MacProtocolEnv')
    parser.add_argument('--rho', type=int, default=3)
    parser.add_argument('--recent_k', type=int, default=0)
    parser.add_argument('--UE_num', type=int, default=2)
    parser.add_argument('--UE_txbuff_len', type=int, default=20)
    # parser.add_argument('--UE_max_generate_SDUs', type=int, default=2)
    parser.add_argument('--p_SDU_arrival', type=float, default=0.5)
    parser.add_argument('--tbl_error_rate', type=float, default=1e-3)
    parser.add_argument('--TTLs', type=int, default=24)
    parser.add_argument('--UCM', type=int, default=None)
    parser.add_argument('--DCM', type=int, default=None)
    parser.add_argument('--need_comm', type=bool, default=True)
    np.random.seed(1113)
    args = parser.parse_args()
    env = MacProtocolEnv(args)
    env.is_training = False
    print("init obs:",env.reset())
    t=0
    while env.done == False:
        UE_actions = np.random.randint(0, 3, env.UE_num)
        o,r,_,info =env.step(UE_actions)
        print("observation:{}".format(o))
        print("info_ue:",info)
        print("reward:{}".format(r))
    print('Goodput:',env.get_Goodput())
    print('collision rate:',env.get_collision_rate())
    print('buffer occupancy:',np.average(env.get_buffer_occupancy()))
    print('packet arrival rate:',env.get_packet_arrival_rate())
        