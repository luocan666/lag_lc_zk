import random

import numpy as np
from envs.JSBSim.reward_functions.reward_function_base import BaseRewardFunction
import math
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.utils.utils import LLA2NEU, get_AO_TA_R


class EnergyReward(BaseRewardFunction):
    def __init__(self, config):
        super().__init__(config)
        self.v1 = 2
        self.v2 = 1

    def get_reward(self, task, env, agent_id):

        agent = env.agents[agent_id]
        v = agent.get('velocities/mach')*340
        h = agent.get('position/h-sl-ft') * 0.3048

        SE = self.cal_SE(v,h)
        dealt_SE = SE - env.Pre_SE[agent_id]
        # print(f'{agent_id},v {v},h {h},dealt_Se{dealt_SE}')
        if abs(dealt_SE) > 3000:
            dealt_SE = 0

        energy = math.exp(-abs(SE - self.cal_SE(300, 10000)) / self.cal_SE(300, 10000))
        if energy < 0.8:
            reward = max(dealt_SE / 10, -1) * self.v1
        else:
            reward = max(dealt_SE / 10, -1) * self.v2

        # if agent_id == "A0100":
        #     env.worksheet.write(env.current_step, 7, SE)
        #     env.worksheet.write(env.current_step, 8, dealt_SE)
        #     env.worksheet.write(env.current_step, 9, energy)
        #     env.worksheet.write(env.current_step, 10, reward)
        # print(f'{agent_id}energy_reward,{reward}')
        return reward

    def cal_SE(self, v, h):
        return (v ** 2) / 19.62 + h
