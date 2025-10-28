import numpy as np
from .reward_function_base import BaseRewardFunction

class FastExplorationReward(BaseRewardFunction):
    """
        与能量奖励（0-4）平衡的经纬度探索奖励函数：
        - 探索奖励范围控制在与能量奖励相当的水平
        - 所有参数直接赋值，无需外部配置
        """

    def __init__(self, config):
        super().__init__(config)

        # 经纬度探索范围（固定值）
        self.lat_limit = (-0.9032379, 0.9032379)  # 纬度范围 (min, max)
        self.lon_limit = (-0.8971999, 0.8971999)  # 经度范围 (min, max)

        # 探索核心参数（直接赋值，控制奖励范围）
        self.grid_resolution = 0.005  # 网格分辨率（度）
        self.base_new_grid_reward = 1.5  # 新网格基础奖励（核心值，控制整体量级）
        self.repetition_penalty = -0.5  # 重复访问惩罚（绝对值小于基础奖励）
        self.direction_bonus = 0.8  # 向未探索区域移动奖励
        self.movement_reward = 0.3  # 移动奖励（鼓励持续移动）
        self.out_of_bounds_penalty = -2.0  # 超出范围惩罚（适中强度）
        self.max_idle_time = 10.0  # 最长允许未发现新区域的时间（秒）
        self.idle_penalty_rate = -0.2  # 每超时1秒的惩罚（缓慢累积）

        # 状态记录
        self.explored_grids = {}  # {agent_id: set of (grid_lat, grid_lon)}
        self.last_new_grid_time = {}  # {agent_id: 上次发现新网格的时间}
        self.last_position = {}  # {agent_id: (lat, lon)}

    def get_reward(self, task, env, agent_id):
        agent = env.agents[agent_id]
        current_time = env.current_step
        current_lat = agent.get_geodetic()[0] # 智能体纬度
        current_lon = agent.get_geodetic()[1]  # 智能体经度

        # 初始化状态
        if agent_id not in self.explored_grids:
            self.explored_grids[agent_id] = set()
            self.last_new_grid_time[agent_id] = current_time
            self.last_position[agent_id] = (current_lat, current_lon)

        # 1. 边界检查
        lat_min, lat_max = self.lat_limit
        lon_min, lon_max = self.lon_limit
        in_bounds = (lat_min <= current_lat <= lat_max) and (lon_min <= current_lon <= lon_max)
        if not in_bounds:
            return self._process(self.out_of_bounds_penalty, agent_id)

        # 2. 经纬度转网格坐标
        grid_lat = int((current_lat - lat_min) // self.grid_resolution)
        grid_lon = int((current_lon - lon_min) // self.grid_resolution)
        current_grid = (grid_lat, grid_lon)

        # 3. 新网格奖励（核心奖励，控制在低量级）
        new_grid_reward = 0.0
        if current_grid not in self.explored_grids[agent_id]:
            # 时间差加成（间隔越短奖励略高，但整体幅度小）
            time_since_last = current_time - self.last_new_grid_time[agent_id]
            speed_bonus = 1.0 / (1.0 + time_since_last)  # 最大加成1倍（间隔0秒时）
            new_grid_reward = self.base_new_grid_reward * (1 + speed_bonus)  # 最大≈3.0

            # 更新状态
            self.explored_grids[agent_id].add(current_grid)
            self.last_new_grid_time[agent_id] = current_time

        # 4. 重复访问惩罚
        repetition_penalty = self.repetition_penalty if current_grid in self.explored_grids[agent_id] else 0.0

        # 5. 方向奖励
        direction_reward = 0.0
        last_lat, last_lon = self.last_position[agent_id]
        last_grid_lat = int((last_lat - lat_min) // self.grid_resolution)
        last_grid_lon = int((last_lon - lon_min) // self.grid_resolution)
        last_grid = (last_grid_lat, last_grid_lon)

        if last_grid != current_grid:
            # 检查8个相邻网格的未探索比例
            adjacent_grids = [
                (grid_lat + dlat, grid_lon + dlon)
                for dlat in (-1, 0, 1)
                for dlon in (-1, 0, 1)
                if (dlat, dlon) != (0, 0)
            ]
            valid_adjacent = []
            for (g_lat, g_lon) in adjacent_grids:
                adj_lat = g_lat * self.grid_resolution + lat_min
                adj_lon = g_lon * self.grid_resolution + lon_min
                if (lat_min <= adj_lat <= lat_max) and (lon_min <= adj_lon <= lon_max):
                    valid_adjacent.append((g_lat, g_lon))
            unexplored_count = sum(1 for g in valid_adjacent if g not in self.explored_grids[agent_id])
            direction_reward = (unexplored_count / len(
                valid_adjacent)) * self.direction_bonus if valid_adjacent else 0.0

        # 6. 移动奖励
        lat_diff = current_lat - last_lat
        lon_diff = current_lon - last_lon
        movement = np.sqrt(lat_diff ** 2 + lon_diff ** 2)
        movement_reward = self.movement_reward if movement > 1e-6 else 0.0

        # 7. 超时惩罚
        idle_time = current_time - self.last_new_grid_time[agent_id]
        idle_penalty = 0.0
        if idle_time > self.max_idle_time:
            idle_penalty = self.idle_penalty_rate * (idle_time - self.max_idle_time)
            # 限制最大超时惩罚（避免过大负值）
            idle_penalty = max(idle_penalty, -1.5)

        # 更新上一位置
        self.last_position[agent_id] = (current_lat, current_lon)

        # 总奖励（各项求和后范围与能量奖励0-4匹配）
        total_reward = (new_grid_reward +
                        repetition_penalty +
                        direction_reward +
                        movement_reward +
                        idle_penalty)

        # 最终限制奖励范围（确保与能量奖励平衡）
        total_reward = np.clip(total_reward, -3.0, 3.5)

        return self._process(total_reward, agent_id)